"""Per-visitor state isolation for the public RINTIK demo.

Design (deliberately small):

* The frozen runtime (checkpoint, sidecar, boosters) and the pinned synthetic
  fixtures are loaded once per process and shared read-only by every session.
* Each visitor session owns one SQLite file.  That file holds the official
  replay, the synthetic monitoring/recovery tables and the October 2026
  operational simulation, exactly like the legacy single database did.
* A request without ``X-Rintik-Session`` uses the legacy ``default`` database
  (``RINTIK_DATABASE_PATH``), so local launchers, tests and QA scripts keep
  working unchanged.
* At most ``max_sessions`` session databases are kept.  The least recently used
  one is deleted when a new session would exceed the cap; sessions that are
  serving a request at that moment are never deleted.

The raw session id is never stored or echoed: files and log-friendly labels use
its SHA-256.  A session id is a capability, not an identity; it only prevents
visitors from reading or resetting each other's demo state by accident or by
casual tampering.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from daily_batch import BatchError, DailyBatchService
from monitoring import MonitoringService
from operational_simulation import OperationalSimulationService

LOGGER = logging.getLogger("rintik.sessions")

SESSION_HEADER = "X-Rintik-Session"
DEFAULT_SESSION_LABEL = "default"
_SESSION_ID = re.compile(r"[A-Za-z0-9_-]{16,64}")
_DB_SUFFIXES = ("", "-wal", "-shm", "-journal")


def session_digest(session_id: str) -> str:
    return hashlib.sha256(session_id.encode("ascii")).hexdigest()


def parse_session_header(value: str | None) -> str | None:
    """Return the validated session id, ``None`` for the default session."""
    if value is None:
        return None
    if not _SESSION_ID.fullmatch(value):
        raise BatchError(
            "INVALID_SESSION",
            f"{SESSION_HEADER} must be 16-64 characters of A-Z, a-z, 0-9, '_' or '-'.",
            400,
        )
    return value


@dataclass
class Session:
    """State handles for one visitor; services share the frozen runtime."""

    scoped: bool
    id_hash: str | None
    daily: DailyBatchService
    monitoring: MonitoringService | None
    operational: OperationalSimulationService | None
    key: str | None = None
    database_path: Path | None = None
    in_use: int = field(default=0, repr=False)

    def describe(self) -> dict[str, object]:
        return {"id_hash": self.id_hash, "scoped": self.scoped}


class SessionRegistry:
    def __init__(self, directory: str | Path, max_sessions: int = 64):
        if max_sessions < 1:
            raise ValueError("max_sessions must be at least 1")
        self.directory = Path(directory)
        self.max_sessions = int(max_sessions)
        self._lock = threading.Lock()
        # key -> Session, or None for a database found on disk but not opened yet.
        self._entries: OrderedDict[str, Session | None] = OrderedDict()
        self._runtime: object | None = None
        self._scanned = False

    # -- internals (call with self._lock held) ---------------------------------
    def _path_for(self, key: str) -> Path:
        return self.directory / f"{key}.sqlite3"

    def _scan_existing(self) -> None:
        """Adopt databases left by a previous process so the cap still holds."""
        if self._scanned:
            return
        self._scanned = True
        self.directory.mkdir(parents=True, exist_ok=True)
        found = sorted(self.directory.glob("*.sqlite3"), key=lambda path: path.stat().st_mtime)
        for path in found:
            self._entries.setdefault(path.stem, None)

    def _delete_files(self, key: str) -> None:
        base = self._path_for(key)
        for suffix in _DB_SUFFIXES:
            target = base.with_name(base.name + suffix)
            try:
                target.unlink(missing_ok=True)
            except OSError:
                LOGGER.warning("Could not delete evicted session file %s", target.name)

    def _evict_over_cap(self) -> None:
        while len(self._entries) > self.max_sessions:
            victim = next(
                (
                    key for key, session in self._entries.items()
                    if session is None or session.in_use == 0
                ),
                None,
            )
            if victim is None:
                return  # every session is serving a request; retry on the next creation
            self._entries.pop(victim)
            self._delete_files(victim)

    def _open(self, key: str, base: Session) -> Session:
        path = self._path_for(key)
        daily = DailyBatchService(base.daily.runtime, path)
        monitoring = None
        operational = None
        if base.monitoring is not None:
            try:
                monitoring = base.monitoring.for_daily(daily)
            except Exception:
                LOGGER.exception("Session monitoring initialization failed")
        if base.operational is not None:
            try:
                operational = base.operational.with_database(path)
            except Exception:
                LOGGER.exception("Session operational simulation initialization failed")
        return Session(
            scoped=True,
            id_hash=key[:8],
            daily=daily,
            monitoring=monitoring,
            operational=operational,
            key=key,
            database_path=path,
        )

    # -- public API -------------------------------------------------------------
    def acquire(self, session_id: str, base: Session) -> Session:
        key = session_digest(session_id)[:32]
        with self._lock:
            if self._runtime is not base.daily.runtime:
                # A different runtime (e.g. tests swapping services) invalidates
                # cached handles; databases on disk stay valid.
                self._entries = OrderedDict((k, None) for k in self._entries)
                self._runtime = base.daily.runtime
            self._scan_existing()
            session = self._entries.get(key)
            if session is None:
                session = self._open(key, base)
            self._entries[key] = session
            self._entries.move_to_end(key)
            session.in_use += 1
            self._evict_over_cap()
            return session

    def release(self, session: Session) -> None:
        if not session.scoped:
            return
        with self._lock:
            session.in_use = max(0, session.in_use - 1)
        # Eviction happens only when a session is created, so a request that just
        # finished never loses its state to a still-running older request.

    @contextmanager
    def use(self, session_id: str | None, base: Session) -> Iterator[Session]:
        if session_id is None:
            yield base
            return
        session = self.acquire(session_id, base)
        try:
            yield session
        finally:
            self.release(session)

    def active_count(self) -> int:
        with self._lock:
            return len(self._entries)
