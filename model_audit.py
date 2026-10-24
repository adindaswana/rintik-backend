"""Model-health monitoring (Q/D/P/O, Trust) and periodic accuracy audit — synthetic demo.

This module EVALUATES a CSV of observations: exogenous sensors, the P3 value RINTIK produced for each
30-minute slot, technical pipeline evidence, and (optionally) verified reference RH. It never runs
Stage 1/2, EWM or P1-P3 and never changes the frozen model. Every score, gate and decision is computed
here from raw evidence with the rules in ``audit/rules_demo_v1.json`` (demo thresholds, uncalibrated).

Principles (PPT Grand Final "Yang mengawasi RINTIK", blueprint pages 10, 11, 16):
* Trust = 0.40Q + 0.30D + 0.20P + 0.10O is operational health, not the probability of being right.
* Audit = published P3 versus verified reference RH on the same timestamp; audit is not inference.
* A critical-input failure or a failed/insufficient audit cannot be overridden by a high Trust.
* Retraining is only ever recommended for evaluation; it is never executed here.

Temporal rule: a day's scores use only that day's rows, the previous day's medians (for cumulative
counters) and the fixed training baseline. ``as_of`` drops every later row before anything is computed;
the audit exists only when the full evaluation period is complete.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

AUDIT_DIR = Path(__file__).resolve().parent / "audit"
RULES_PATH = Path(os.environ.get("RINTIK_AUDIT_RULES_PATH", AUDIT_DIR / "rules_demo_v1.json"))

TIME_COLUMN = "datetime"
SITE_COLUMN = "site_id"
SENSOR_COLUMNS = [
    "env_id", "co2", "pressure", "air_temperature", "env_battery",
    "soil_ec_line1", "soil_humidity_line1", "soil_temp_line1",
    "soil_ec_line2", "soil_humidity_line2", "soil_temp_line2",
    "soil_ec_line3", "soil_humidity_line3", "soil_temp_line3",
    "water_volume_line1", "water_volume_line2", "water_volume_line3",
    "gdd", "standard_day_degree", "daily_mean_temperature",
    "daily_max_above_Tbase", "daily_max", "daily_max_reduction", "ontario_units",
]
PREDICTION_COLUMN = "rintik_p3"
PIPELINE_COLUMNS = ["pipeline_run_id", "pipeline_stage", "pipeline_status", "pipeline_error"]
REFERENCE_COLUMNS = ["rh_reference", "rh_reference_source", "rh_reference_status", "ref_site_id"]
SCHEMA = [TIME_COLUMN, SITE_COLUMN, *SENSOR_COLUMNS, PREDICTION_COLUMN, *PIPELINE_COLUMNS, *REFERENCE_COLUMNS]
NUMERIC_COLUMNS = [*SENSOR_COLUMNS, PREDICTION_COLUMN, "rh_reference"]
MAX_ROWS = 5000
EVALUATION_PERIOD_DAYS = 90

LABELS = {
    "co2": "CO₂", "pressure": "Tekanan udara", "air_temperature": "Suhu udara",
    "soil_ec_line1": "Konduktivitas tanah 1", "soil_humidity_line1": "Kelembapan tanah 1", "soil_temp_line1": "Suhu tanah 1",
    "soil_ec_line2": "Konduktivitas tanah 2", "soil_humidity_line2": "Kelembapan tanah 2", "soil_temp_line2": "Suhu tanah 2",
    "soil_ec_line3": "Konduktivitas tanah 3", "soil_humidity_line3": "Kelembapan tanah 3", "soil_temp_line3": "Suhu tanah 3",
    "water_volume_line1": "Volume air 1", "water_volume_line2": "Volume air 2", "water_volume_line3": "Volume air 3",
    "gdd": "Akumulasi panas (GDD)", "standard_day_degree": "Derajat hari standar", "daily_mean_temperature": "Suhu rata-rata harian",
    "daily_max_above_Tbase": "Suhu maks di atas Tbase", "daily_max": "Suhu maks harian", "daily_max_reduction": "Selisih suhu harian",
    "ontario_units": "Satuan panas Ontario",
}


class AuditInputError(ValueError):
    def __init__(self, code: str, message: str, status_code: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rules(path: Path = RULES_PATH) -> dict[str, Any]:
    rules = json.loads(Path(path).read_text(encoding="utf-8"))
    baseline_path = Path(path).parent / rules["d"]["baseline_file"]
    actual = _sha256(baseline_path)
    if actual != rules["d"]["baseline_sha256"]:
        raise RuntimeError("Training drift baseline does not match its pinned SHA256")
    rules["_baseline"] = json.loads(baseline_path.read_text(encoding="utf-8"))
    weights = rules["weights"]
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise RuntimeError("Trust weights must sum to 1")
    return rules


def public_rules(rules: dict[str, Any]) -> dict[str, Any]:
    out = {key: value for key, value in rules.items() if not key.startswith("_")}
    baseline = rules["_baseline"]
    out["baseline_summary"] = {
        "source": baseline.get("source"), "date_min": baseline.get("date_min"), "date_max": baseline.get("date_max"),
        "n_rows": baseline.get("n_rows"), "sha256": rules["d"]["baseline_sha256"],
        "label": "Statistik train.csv asli (Jun–Agu 2023); tetap, tidak dihitung dari data yang dievaluasi.",
    }
    return out


def _fmt(value: float | None, digits: int = 1) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    return f"{value:.{digits}f}".replace(".", ",")


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None or not math.isfinite(value) else round(float(value), digits)


# --------------------------------------------------------------------------- parsing and integrity
def parse_csv(text: str, as_of: str | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not isinstance(text, str) or not text.strip():
        raise AuditInputError("EMPTY_CSV", "CSV kosong.")
    if "\x00" in text:
        raise AuditInputError("INVALID_CSV", "CSV berisi karakter NUL.")
    try:
        raw = pd.read_csv(io.StringIO(text.lstrip("﻿")), dtype=str, keep_default_na=False, on_bad_lines="error")
    except Exception as exc:
        raise AuditInputError("INVALID_CSV", "CSV tidak dapat dibaca dengan aman (jumlah kolom per baris tidak konsisten?).") from exc
    raw.columns = [str(column).strip() for column in raw.columns]
    missing = [column for column in SCHEMA if column not in raw.columns]
    extra = [column for column in raw.columns if column not in SCHEMA]
    if missing:
        raise AuditInputError("MISSING_COLUMNS", "Kolom wajib belum ada: " + ", ".join(missing))
    if extra:
        # Strict schema: nothing outside the evidence columns (e.g. a scenario/label/score column) is accepted.
        raise AuditInputError("UNEXPECTED_COLUMNS", "Kolom tidak dikenal ditolak (dataset hanya boleh berisi bukti observasi): " + ", ".join(extra))
    if len(raw) == 0:
        raise AuditInputError("EMPTY_CSV", "CSV tidak memiliki baris data.")
    if len(raw) > MAX_ROWS:
        raise AuditInputError("TOO_MANY_ROWS", f"Maksimum {MAX_ROWS} baris.", 413)

    times = pd.to_datetime(raw[TIME_COLUMN].str.strip(), format="%Y-%m-%d %H:%M:%S", errors="coerce")
    if times.isna().any():
        row = int(np.flatnonzero(times.isna().to_numpy())[0]) + 2
        raise AuditInputError("INVALID_TIMESTAMP", f"Timestamp baris {row} tidak berformat YYYY-MM-DD HH:MM:SS.")
    if times.duplicated().any():
        row = int(np.flatnonzero(times.duplicated().to_numpy())[0]) + 2
        raise AuditInputError("DUPLICATE_TIMESTAMP", f"Timestamp duplikat pada baris {row} ({raw[TIME_COLUMN].iloc[row - 2]}).")
    aligned = (times.dt.minute.isin([0, 30])) & (times.dt.second == 0)
    if not bool(aligned.all()):
        row = int(np.flatnonzero(~aligned.to_numpy())[0]) + 2
        raise AuditInputError("TIMESTAMP_NOT_ALIGNED", f"Timestamp baris {row} tidak berada pada slot 30 menit.")
    sites = sorted({value.strip() for value in raw[SITE_COLUMN]})
    if sites == [""] or "" in sites:
        raise AuditInputError("MISSING_SITE_ID", "Kolom site_id wajib terisi pada setiap baris.")
    if len(sites) != 1:
        raise AuditInputError("MULTIPLE_SITES", "Satu file hanya boleh berisi satu lokasi (site_id): " + ", ".join(sites))

    chronological = bool(times.is_monotonic_increasing)
    frame = pd.DataFrame({TIME_COLUMN: times})
    unreadable: dict[str, np.ndarray] = {}
    for column in NUMERIC_COLUMNS:
        text_values = raw[column].str.strip()
        numbers = pd.to_numeric(text_values, errors="coerce").astype(float)
        finite = np.isfinite(numbers.to_numpy())
        bad = (text_values != "").to_numpy() & ~finite  # "ERR", "inf", "nan" text: present but unreadable
        frame[column] = np.where(finite, numbers.to_numpy(), np.nan)
        unreadable[column] = bad
    for column in [SITE_COLUMN, *PIPELINE_COLUMNS, "rh_reference_source", "rh_reference_status", "ref_site_id"]:
        frame[column] = raw[column].str.strip()
    for column, mask in unreadable.items():
        frame["_bad_" + column] = mask
    frame = frame.sort_values(TIME_COLUMN).reset_index(drop=True)

    first_day = frame[TIME_COLUMN].min().normalize()
    last_day = frame[TIME_COLUMN].max().normalize()
    period_days = int((last_day - first_day).days) + 1
    if period_days > EVALUATION_PERIOD_DAYS:
        raise AuditInputError("PERIOD_TOO_LONG", f"Periode {period_days} hari melebihi periode evaluasi {EVALUATION_PERIOD_DAYS} hari.")
    meta = {
        "rows": int(len(frame)), "site_id": sites[0], "first_timestamp": frame[TIME_COLUMN].min().strftime("%Y-%m-%d %H:%M:%S"),
        "last_timestamp": frame[TIME_COLUMN].max().strftime("%Y-%m-%d %H:%M:%S"), "period_start": first_day.strftime("%Y-%m-%d"),
        "period_end": last_day.strftime("%Y-%m-%d"), "period_days": period_days, "chronological_in_file": chronological,
        "unreadable_values": int(sum(int(mask.sum()) for mask in unreadable.values())),
    }
    if as_of is not None:
        try:
            cutoff = pd.Timestamp(date.fromisoformat(as_of))
        except (TypeError, ValueError) as exc:
            raise AuditInputError("INVALID_AS_OF", "as_of harus berformat YYYY-MM-DD.") from exc
        if cutoff < first_day or cutoff > last_day:
            raise AuditInputError("INVALID_AS_OF", "as_of berada di luar periode data.")
        # Everything after the cut-off is dropped before any computation: no future information.
        frame = frame.loc[frame[TIME_COLUMN] < cutoff + pd.Timedelta(days=1)].reset_index(drop=True)
        meta["as_of"] = as_of
    else:
        meta["as_of"] = meta["period_end"]
    return frame, meta


# --------------------------------------------------------------------------- daily evaluation
def _slot_index(times: pd.Series) -> np.ndarray:
    return (times.dt.hour * 2 + times.dt.minute // 30).to_numpy()


def _spike_mask(values: np.ndarray, valid: np.ndarray, limit: float) -> np.ndarray:
    """Isolated jumps: a point that differs by > limit from BOTH adjacent valid slots (or the single neighbour at an edge)."""
    spikes = np.zeros(len(values), dtype=bool)
    for i in range(len(values)):
        if not valid[i]:
            continue
        neighbours = [values[j] for j in (i - 1, i + 1) if 0 <= j < len(values) and valid[j]]
        if neighbours and all(abs(values[i] - other) > limit for other in neighbours):
            spikes[i] = True
    return spikes


def _day_grid(day_rows: pd.DataFrame, slots: int) -> pd.DataFrame:
    grid = pd.DataFrame({"slot": np.arange(slots)})
    rows = day_rows.copy()
    rows["slot"] = _slot_index(rows[TIME_COLUMN])
    return grid.merge(rows, on="slot", how="left")


def evaluate_day(day: pd.Timestamp, day_rows: pd.DataFrame, previous_medians: dict[str, float] | None, rules: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float], np.ndarray]:
    slots = rules["slots_per_day"]
    grid = _day_grid(day_rows, slots)
    present = grid[TIME_COLUMN].notna().to_numpy()
    n_present = int(present.sum())
    features = rules["sensor_features"]
    critical = rules["critical_features"]
    ranges = rules["physical_ranges"]
    spike_limits = rules["spike_limits_per_slot"]
    qr = rules["q"]

    # ---- per-feature evidence
    evidence: dict[str, dict[str, Any]] = {}
    clean_values: dict[str, np.ndarray] = {}
    for feature in features:
        values = grid[feature].to_numpy(dtype=float)
        bad = grid["_bad_" + feature].eq(True).to_numpy()
        readable = np.isfinite(values)
        lower, upper = ranges[feature]
        out_of_range = readable & ((values < lower) | (values > upper))
        in_range = readable & ~out_of_range
        spikes = _spike_mask(values, in_range, spike_limits[feature]) if feature in spike_limits else np.zeros(slots, dtype=bool)
        invalid = out_of_range | bad | spikes
        evidence[feature] = {
            "readable": int(readable.sum()) + int(bad.sum()),  # unreadable values are present but invalid
            "missing": int(slots - readable.sum() - bad.sum()),
            "out_of_range": int(out_of_range.sum()), "unreadable": int(bad.sum()), "spikes": int(spikes.sum()),
            "invalid": int(invalid.sum()), "invalid_mask": invalid, "valid_mask": in_range & ~spikes,
            "examples": [float(v) for v in values[out_of_range | spikes][:3]],
        }
        clean_values[feature] = values[in_range & ~spikes]

    # ---- critical gate
    critical_reasons = []
    for feature in critical:
        item = evidence[feature]
        missing_share = item["missing"] / slots
        invalid_share = item["invalid"] / slots
        if missing_share > qr["critical_failure_missing_share"]:
            critical_reasons.append(f"{LABELS[feature]} hilang pada {item['missing']} dari {slots} slot ({_fmt(100 * missing_share, 0)}%)")
        if invalid_share > qr["critical_failure_invalid_share"]:
            critical_reasons.append(f"{LABELS[feature]} tidak valid pada {item['invalid']} dari {slots} slot (di luar rentang fisik/lonjakan/tak terbaca)")
    if n_present < qr["critical_failure_min_slot_share"] * slots:
        critical_reasons.append(f"hanya {n_present} dari {slots} slot waktu tersedia")
    critical_failure = bool(critical_reasons)

    # ---- Q
    q_reasons: list[str] = []
    q_parts = None
    q_score = None
    if n_present >= rules["min_slots_assessable"]:
        slot_share = n_present / slots
        # the weakest critical sensor counts: one dead critical input must not hide behind two healthy ones
        critical_share = float(min(evidence[f]["readable"] / slots for f in critical))
        other = [f for f in features if f not in critical]
        full_credit = qr["other_completeness_full_credit"]
        other_share = float(np.mean([min(1.0, (evidence[f]["readable"] / slots) / full_credit) for f in other]))
        invalid_critical = sum(evidence[f]["invalid"] for f in critical)
        invalid_other = sum(evidence[f]["invalid"] for f in other)
        validity = _clip01(1 - qr["invalid_penalty_critical"] * invalid_critical - qr["invalid_penalty_other"] * invalid_other)
        w = qr["weights"]
        q_parts = {"slot_completeness": slot_share, "critical_completeness": critical_share, "other_completeness": other_share, "validity": validity}
        q_score = 100 * (w["slot_completeness"] * slot_share + w["critical_completeness"] * critical_share
                         + w["other_completeness"] * other_share + w["validity"] * validity)
        if critical_failure and q_score > qr["critical_failure_q_cap"]:
            q_score = float(qr["critical_failure_q_cap"])  # blueprint p.10: critical input failed -> Q 40
            q_reasons.append(f"Q dibatasi {qr['critical_failure_q_cap']} karena input kritis gagal")
        if n_present < slots:
            q_reasons.append(f"{slots - n_present} dari {slots} slot waktu tidak ada di file")
        for feature in critical:
            item = evidence[feature]
            if item["missing"]:
                q_reasons.append(f"{LABELS[feature]} kosong pada {item['missing']} slot (input kritis)")
        if invalid_critical or invalid_other:
            worst = sorted(features, key=lambda f: -evidence[f]["invalid"])
            detail = []
            for feature in worst[:3]:
                item = evidence[feature]
                if not item["invalid"]:
                    continue
                parts = []
                if item["out_of_range"]:
                    parts.append(f"{item['out_of_range']} di luar rentang fisik {ranges[feature][0]}–{ranges[feature][1]}")
                if item["spikes"]:
                    parts.append(f"{item['spikes']} lonjakan antarslot")
                if item["unreadable"]:
                    parts.append(f"{item['unreadable']} tak terbaca")
                example = f" (mis. {_fmt(item['examples'][0])})" if item["examples"] else ""
                detail.append(f"{LABELS[feature]}: " + ", ".join(parts) + example)
            q_reasons.append("Nilai tidak valid — " + "; ".join(detail))
        low_other = [f for f in other if evidence[f]["readable"] / slots < full_credit]
        if low_other:
            q_reasons.append(f"{len(low_other)} sensor non-kritis kosong >{_fmt(100 * (1 - full_credit), 0)}% slot")
        if not q_reasons:
            q_reasons.append("Data lengkap dan seluruh nilai dalam rentang fisik")

    # ---- D (fixed training baseline; values that failed the physical check are excluded)
    dr = rules["d"]
    baseline = rules["_baseline"]["columns"]
    medians: dict[str, float] = {}
    drift_items = []
    for feature in features:
        values = clean_values[feature]
        stats = baseline[feature]
        if len(values) >= dr["min_values_per_feature"]:
            medians[feature] = float(np.median(values))
        if stats.get("daily_increment"):
            ref = stats["daily_increment"]
            if feature not in medians or not previous_medians or feature not in previous_medians:
                continue
            observed = [medians[feature] - previous_medians[feature]]
        else:
            ref = stats
            if len(values) < dr["min_values_per_feature"]:
                continue
            observed = list(values)
        observed_arr = np.asarray(observed, dtype=float)
        z = (float(observed_arr.mean()) - ref["mean"]) / ref["sd"]
        outside = float(np.mean((observed_arr < ref["p1"]) | (observed_arr > ref["p99"])))
        s_z = _clip01((abs(z) - dr["z_free"]) / (dr["z_full"] - dr["z_free"]))
        s_out = _clip01((outside - dr["outside_free"]) / (dr["outside_full"] - dr["outside_free"]))
        drift_items.append({"feature": feature, "label": LABELS[feature], "z": z, "outside_share": outside,
                            "severity": max(s_z, s_out), "cumulative": bool(stats.get("daily_increment"))})
    d_score = None
    d_reasons: list[str] = []
    if len(drift_items) >= dr["min_feature_share"] * len(features):
        d_score = 100 * (1 - float(np.mean([item["severity"] for item in drift_items])))
        shifted = sorted([item for item in drift_items if item["severity"] > 0], key=lambda item: -item["severity"])
        if shifted:
            d_reasons.append("Bergeser dari data latih: " + "; ".join(
                f"{item['label']} (skor geser {_fmt(item['z'], 2)}, {_fmt(100 * item['outside_share'], 0)}% di luar p1–p99)" for item in shifted[:3]))
        else:
            d_reasons.append(f"{len(drift_items)} sensor dalam sebaran data latih (|skor geser| ≤ {_fmt(dr['z_free'])})")
    elif n_present:
        d_reasons.append(f"Hanya {len(drift_items)} sensor yang bisa dibandingkan (< {int(dr['min_feature_share'] * len(features))}); D tidak dinilai")

    # ---- P (technical pipeline evidence, not accuracy)
    pr = rules["p"]
    stage = grid["pipeline_stage"].fillna("").to_numpy(dtype=object)
    status = grid["pipeline_status"].fillna("").to_numpy(dtype=object)
    errors = sorted({str(code) for code in grid["pipeline_error"].fillna("") if str(code)})
    completed = present & (stage == pr["final_stage"]) & (status == "ok")
    p3 = grid[PREDICTION_COLUMN].to_numpy(dtype=float)
    p3_bad = grid["_bad_" + PREDICTION_COLUMN].eq(True).to_numpy()
    p3_valid = np.isfinite(p3) & (p3 >= 0) & (p3 <= 100)
    n_completed = int(completed.sum())
    p_score = None
    p_reasons: list[str] = []
    batch_failed = n_present > 0 and n_completed == 0
    unknown_stages = sorted({str(s) for s in stage[present] if str(s) not in pr["stages"]})
    if n_present:
        completed_share = n_completed / slots
        valid_output_share = float((completed & p3_valid).sum() / n_completed) if n_completed else 0.0
        w = pr["weights"]
        p_score = 100 * (w["completed_share"] * completed_share + w["valid_output_share"] * valid_output_share)
        if errors:
            p_score = max(0.0, p_score - pr["error_code_penalty"])
        if batch_failed:
            reached = pd.Series([s for s in stage[present] if s]).mode()
            p_reasons.append("Batch gagal: tidak ada slot yang mencapai P3" + (f" (berhenti di tahap {reached.iloc[0]})" if len(reached) else ""))
        elif n_completed < slots:
            p_reasons.append(f"{n_completed} dari {slots} slot menyelesaikan seluruh tahap sampai P3")
        if n_completed and valid_output_share < 1:
            p_reasons.append(f"{int(n_completed - (completed & p3_valid).sum())} keluaran P3 kosong/tidak valid pada slot yang dinyatakan selesai")
        if errors:
            p_reasons.append("Kode error tercatat: " + ", ".join(errors))
        if unknown_stages:
            p_reasons.append("Tahap pipeline tidak dikenal: " + ", ".join(unknown_stages))
        if not p_reasons:
            p_reasons.append("Semua slot selesai sampai P3 tanpa error")

    # ---- O (conservative plausibility of P3; plausible by construction ≠ accurate)
    orl = rules["o"]
    candidates = completed & (np.isfinite(p3) | p3_bad)
    o_score = None
    o_reasons: list[str] = []
    if int(candidates.sum()) >= rules["min_slots_assessable"]:
        values = p3[candidates]
        in_range = np.isfinite(values) & (values >= 0) & (values <= 100)
        in_range_share = float(in_range.mean())
        series = values[in_range]
        # consecutive-slot checks only between slots that are adjacent in time
        slots_idx = np.flatnonzero(candidates)[in_range]
        diffs = np.diff(series)
        adjacent = np.diff(slots_idx) == 1
        spikes = int(np.sum((np.abs(diffs) > orl["spike_points_per_slot"]) & adjacent))
        longest = run = 1
        for delta, adj in zip(diffs, adjacent):
            run = run + 1 if (adj and abs(delta) < orl["flatline_tolerance"]) else 1
            longest = max(longest, run)
        amplitude = float(series.max() - series.min()) if len(series) else 0.0
        penalty = min(orl["spike_penalty_cap"], orl["spike_penalty"] * spikes)
        if longest >= orl["flatline_slots"]:
            penalty += orl["flatline_penalty"]
        if amplitude < orl["min_daily_amplitude"]:
            penalty += orl["low_amplitude_penalty"]
        o_score = max(0.0, min(100.0, 100 * in_range_share - penalty))
        if in_range_share < 1:
            o_reasons.append(f"{int(len(values) - in_range.sum())} nilai P3 di luar 0–100% atau tak terbaca")
        if spikes:
            o_reasons.append(f"{spikes} lonjakan P3 > {_fmt(orl['spike_points_per_slot'], 0)} poin antarslot 30 menit")
        if longest >= orl["flatline_slots"]:
            o_reasons.append(f"P3 konstan selama {longest} slot berturut-turut")
        if amplitude < orl["min_daily_amplitude"]:
            o_reasons.append(f"Rentang harian P3 hanya {_fmt(amplitude)} poin")
        if not o_reasons:
            o_reasons.append(f"P3 dalam 0–100%, rentang harian {_fmt(amplitude)} poin, tanpa lonjakan/konstan")
    elif n_present:
        o_reasons.append("Tidak cukup keluaran P3 untuk dinilai")

    # ---- Trust, status, publication
    weights = rules["weights"]
    scores = {"Q": q_score, "D": d_score, "P": p_score, "O": o_score}
    trust = None
    if all(value is not None for value in scores.values()):
        trust = sum(weights[key] * scores[key] for key in weights)
    bands = rules["bands"]
    if n_present == 0 or q_score is None:
        day_status = "tidak_dapat_dinilai"
    elif critical_failure or batch_failed:
        day_status = "kritis"
    elif trust is None:
        day_status = "tidak_dapat_dinilai"
    elif trust >= bands["healthy_min"]:
        day_status = "sehat"
    elif trust >= bands["caution_min"]:
        day_status = "waspada"
    else:
        day_status = "kritis"
    publishable_day = day_status in ("sehat", "waspada") and trust is not None and trust >= rules["publication"]["min_trust"]
    slot_ok = completed & p3_valid
    # A slot is withheld when a critical input at that slot is corrupted (out of physical range, spike,
    # unreadable). A merely missing cell is not: the frozen pipeline handles sparse gaps by design (the official
    # test data has ~10% empty cells); large gaps are caught by the daily critical-input gate above.
    for feature in critical:
        slot_ok &= ~evidence[feature]["invalid_mask"]
    published = slot_ok & publishable_day
    has_p3 = present & np.isfinite(p3)
    withheld = has_p3 & ~published

    issue_rules = rules["issue_thresholds"]
    issues = []
    if critical_failure:
        issues.append("input_kritis")
    if any(evidence[f]["invalid"] for f in features):
        issues.append("anomali_input")
    if n_present < slots or any(evidence[f]["readable"] / slots < qr["other_completeness_full_credit"] for f in features):
        issues.append("data_hilang")
    if d_score is not None and d_score < issue_rules["drift_D_below"]:
        issues.append("drift")
    if batch_failed or errors or (p_score is not None and p_score < issue_rules["pipeline_P_below"]):
        issues.append("pipeline")
    if o_score is not None and o_score < issue_rules["output_O_below"]:
        issues.append("output")
    if int(withheld.sum()):
        issues.append("ditahan")

    summary = {
        "date": day.strftime("%Y-%m-%d"),
        "slots_present": n_present,
        "scores": {key: _round(value, 2) for key, value in scores.items()},
        "trust": _round(trust, 2),
        "status": day_status,
        "critical_input_failure": critical_failure,
        "critical_reasons": critical_reasons,
        "batch_failed": bool(batch_failed),
        "published": bool(publishable_day),
        "published_slots": int(published.sum()),
        "withheld_slots": int(withheld.sum()),
        "issues": issues,
        "reasons": {"Q": q_reasons, "D": d_reasons, "P": p_reasons, "O": o_reasons},
        "q_parts": {key: _round(value, 4) for key, value in (q_parts or {}).items()},
        "drift_top": [{"feature": item["feature"], "label": item["label"], "z": _round(item["z"], 3),
                       "outside_share": _round(item["outside_share"], 3), "severity": _round(item["severity"], 3)}
                      for item in sorted(drift_items, key=lambda item: -item["severity"])[:5]],
        "pipeline": {"completed_slots": n_completed, "errors": errors, "run_ids": sorted({str(r) for r in grid["pipeline_run_id"].fillna("") if str(r)})},
    }
    publish_mask = np.zeros(slots, dtype=bool)
    publish_mask[:] = published
    return summary, medians, publish_mask


# --------------------------------------------------------------------------- audit and decision
def _metrics(errors: np.ndarray) -> dict[str, float | None]:
    if len(errors) == 0:
        return {"n": 0, "rmse": None, "mae": None, "bias": None}
    return {"n": int(len(errors)), "rmse": float(np.sqrt(np.mean(errors ** 2))), "mae": float(np.mean(np.abs(errors))), "bias": float(np.mean(errors))}


def _rounded_metrics(values: dict[str, Any]) -> dict[str, Any]:
    return {key: (_round(value, 4) if isinstance(value, float) else value) for key, value in values.items()}


def run_audit(frame: pd.DataFrame, published: pd.Series, meta: dict[str, Any], rules: dict[str, Any], days: list[dict[str, Any]]) -> dict[str, Any]:
    ar = rules["audit"]
    slots = rules["slots_per_day"]
    period_end = pd.Timestamp(meta["period_end"])
    complete = meta["period_days"] == EVALUATION_PERIOD_DAYS and meta["as_of"] == meta["period_end"]
    window_start = period_end - pd.Timedelta(days=ar["window_days"] - 1)
    base = {
        "label": rules["label"], "thresholds": {"rmse_max": ar["rmse_max"], "abs_bias_max": ar["abs_bias_max"], "min_pairs": ar["min_pairs"],
                                                "window_days": ar["window_days"], "accepted_reference_status": ar["accepted_reference_status"]},
        "definition": "Bias = prediksi P3 − referensi. Hanya P3 yang diterbitkan dan referensi terverifikasi pada timestamp dan site_id yang sama.",
    }
    if not complete:
        base.update({"available": False, "status": "belum_tersedia", "status_label": "Belum tersedia",
                     "reason": f"Audit hanya dihitung setelah periode {EVALUATION_PERIOD_DAYS} hari lengkap (data saat ini: {meta['period_days']} hari, sampai {meta['as_of']})."})
        return base
    in_window = (frame[TIME_COLUMN] >= window_start) & (frame[TIME_COLUMN] < period_end + pd.Timedelta(days=1))
    ref = frame["rh_reference"].to_numpy(dtype=float)
    ref_bad = frame["_bad_rh_reference"].to_numpy(dtype=bool)
    has_ref_value = np.isfinite(ref) | ref_bad
    ref_in_range = np.isfinite(ref) & (ref >= 0) & (ref <= 100)
    verified = frame["rh_reference_status"].isin(ar["accepted_reference_status"]).to_numpy()
    site_ok = (frame["ref_site_id"] == frame[SITE_COLUMN]).to_numpy()
    valid_ref = ref_in_range & verified & site_ok
    pub = published.to_numpy(dtype=bool)
    pairs = valid_ref & pub
    window = in_window.to_numpy()
    window_slots = ar["window_days"] * slots
    p3 = frame[PREDICTION_COLUMN].to_numpy(dtype=float)
    errors_all = p3 - ref

    win_pairs = pairs & window
    metrics = _metrics(errors_all[win_pairs])
    rejected = {
        "missing_value": int((window & ~has_ref_value).sum()),
        "unreadable_or_out_of_range": int((window & has_ref_value & ~ref_in_range).sum()),
        "not_verified": int((window & ref_in_range & ~verified).sum()),
        "site_mismatch": int((window & ref_in_range & verified & ~site_ok).sum()),
        "valid_but_prediction_withheld": int((window & valid_ref & ~pub).sum()),
    }
    n = metrics["n"]
    if int((window & valid_ref).sum()) == 0:
        status, status_label = "belum_memadai", "Audit Belum Memadai"
        reason = "Tidak ada RH referensi terverifikasi di jendela audit."
    elif n < ar["min_pairs"]:
        status, status_label = "belum_memadai", "Audit Belum Memadai"
        reason = f"Hanya {n} pasangan valid, kurang dari minimum {ar['min_pairs']}."
    elif metrics["rmse"] <= ar["rmse_max"] and abs(metrics["bias"]) <= ar["abs_bias_max"]:
        status, status_label = "lolos", "Lolos (demo)"
        reason = f"RMSE {_fmt(metrics['rmse'], 2)} ≤ {_fmt(ar['rmse_max'])} dan |bias| {_fmt(abs(metrics['bias']), 2)} ≤ {_fmt(ar['abs_bias_max'])} pada {n} pasangan."
    else:
        status, status_label = "gagal", "Tidak Lolos Akurasi"
        failed = []
        if metrics["rmse"] > ar["rmse_max"]:
            failed.append(f"RMSE {_fmt(metrics['rmse'], 2)} > {_fmt(ar['rmse_max'])}")
        if abs(metrics["bias"]) > ar["abs_bias_max"]:
            failed.append(f"|bias| {_fmt(abs(metrics['bias']), 2)} > {_fmt(ar['abs_bias_max'])}")
        reason = " dan ".join(failed) + f" pada {n} pasangan."

    times = frame[TIME_COLUMN]
    minutes = (times.dt.hour * 60 + times.dt.minute).to_numpy()
    start_h, start_m = map(int, ar["day_start"].split(":"))
    end_h, end_m = map(int, ar["day_end"].split(":"))
    daytime = (minutes >= start_h * 60 + start_m) & (minutes < end_h * 60 + end_m)
    by_day = []
    for offset in range(ar["window_days"]):
        current = window_start + pd.Timedelta(days=offset)
        mask = (times >= current) & (times < current + pd.Timedelta(days=1))
        mask = mask.to_numpy()
        m = _metrics(errors_all[pairs & mask])
        by_day.append({"date": current.strftime("%Y-%m-%d"), "valid_reference": int((valid_ref & mask).sum()),
                       "published": int((pub & mask).sum()), **_rounded_metrics(m)})
    condition = {
        "siang": _rounded_metrics(_metrics(errors_all[win_pairs & daytime])),
        "malam": _rounded_metrics(_metrics(errors_all[win_pairs & ~daytime])),
    }
    humidity = {
        "kering_ekstrem": _rounded_metrics(_metrics(errors_all[win_pairs & (ref <= ar["humidity_extreme_low"])])),
        "normal": _rounded_metrics(_metrics(errors_all[win_pairs & (ref > ar["humidity_extreme_low"]) & (ref < ar["humidity_extreme_high"])])),
        "lembap_ekstrem": _rounded_metrics(_metrics(errors_all[win_pairs & (ref >= ar["humidity_extreme_high"])])),
    }
    spot = []
    outside = pairs & ~window
    if outside.any():
        dates = times[outside].dt.strftime("%Y-%m-%d")
        for current in sorted(dates.unique()):
            mask = (times.dt.strftime("%Y-%m-%d") == current).to_numpy() & outside
            spot.append({"date": current, **_rounded_metrics(_metrics(errors_all[mask]))})
    base.update({
        "available": True, "status": status, "status_label": status_label, "reason": reason,
        "window": {"start": window_start.strftime("%Y-%m-%d"), "end": meta["period_end"], "slots": window_slots},
        "metrics": _rounded_metrics(metrics),
        "reference_coverage": _round(float((window & valid_ref).sum()) / window_slots, 4),
        "pair_coverage": _round(n / window_slots, 4),
        "withheld_in_window": int((window & np.isfinite(p3) & ~pub).sum()),
        "rejected_reference": rejected,
        "breakdown": {"by_day": by_day, "day_night": condition, "humidity": humidity},
        "spot_checks": spot,
        "spot_checks_note": "Titik cek di luar jendela audit hanya untuk melihat tren; tidak dipakai menentukan status audit.",
    })
    return base


def _mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return float(np.mean(present)) if present else None


def decide(days: list[dict[str, Any]], audit: dict[str, Any], meta: dict[str, Any], rules: dict[str, Any]) -> dict[str, Any]:
    dr = rules["decision"]
    ar = rules["audit"]
    critical_days = [d["date"] for d in days if d["critical_input_failure"]]
    pipeline_days = [d["date"] for d in days if d["batch_failed"]]
    window_days = days[-ar["window_days"]:] if len(days) >= ar["window_days"] else days
    window_dates = {d["date"] for d in window_days}
    critical_in_window = [d for d in critical_days if d in window_dates]
    qw = _mean([d["scores"]["Q"] for d in window_days])
    pw = _mean([d["scores"]["P"] for d in window_days])
    dw = _mean([d["scores"]["D"] for d in window_days])
    tw = _mean([d["trust"] for d in window_days])
    factors = {"mean_Q_window": _round(qw, 2), "mean_P_window": _round(pw, 2), "mean_D_window": _round(dw, 2), "mean_trust_window": _round(tw, 2),
               "critical_input_days": critical_days, "critical_input_days_in_window": critical_in_window, "pipeline_failed_days": pipeline_days,
               "withheld_slots_period": int(sum(d["withheld_slots"] for d in days))}
    reasons: list[str] = []
    steps: list[str] = []
    m = audit.get("metrics") or {}
    if critical_days:
        resolved = not critical_in_window
        reasons.append(f"Input kritis gagal pada {len(critical_days)} hari ({critical_days[0]} s.d. {critical_days[-1]})" + ("; sudah pulih sebelum jendela audit." if resolved else "; masih terjadi di jendela audit."))

    if not audit.get("available"):
        category, title = "periode_belum_lengkap", "Periode Evaluasi Belum Lengkap"
        reasons.insert(0, audit.get("reason", "Audit belum tersedia."))
        steps = ["Lanjutkan monitoring harian sampai periode 90 hari lengkap.", "Keputusan pemakaian model belum dapat diambil tanpa audit akurasi."]
    elif audit["status"] == "belum_memadai" or critical_in_window:
        category, title = "investigasi", "Investigasi Input atau Bukti Audit Tidak Memadai"
        if audit["status"] == "belum_memadai":
            reasons.insert(0, f"Audit belum memadai: {audit['metrics']['n']} pasangan valid dari minimum {ar['min_pairs']} "
                              f"(referensi terverifikasi {_fmt(100 * audit['reference_coverage'], 0)}% slot, {audit['withheld_in_window']} P3 di jendela audit ditahan).")
            if m.get("n"):
                reasons.append(f"Pada {m['n']} pasangan yang tersedia RMSE {_fmt(m['rmse'], 2)} dan bias {_fmt(m['bias'], 2)}, tetapi sampel ini belum cukup untuk menyimpulkan akurasi.")
        if critical_in_window:
            reasons.insert(0, f"Input kritis gagal di jendela audit pada {', '.join(critical_in_window)}; prediksi pada hari itu ditahan.")
        steps = ["Perbaiki dan verifikasi sensor input yang gagal (periksa daya, kabel, kalibrasi) sebelum menilai model.",
                 "Ulangi sampling referensi terverifikasi 7 hari setelah input stabil agar minimal 300 pasangan tersedia.",
                 "Jangan menyimpulkan model perlu dilatih ulang sebelum bukti audit memadai.",
                 "Batasi recovery otomatis sampai audit valid dan lolos."]
    elif audit["status"] == "gagal":
        healthy = (qw is not None and qw >= dr["healthy_input_Q_min"] and pw is not None and pw >= dr["healthy_pipeline_P_min"]
                   and dw is not None and dw >= dr["acceptable_drift_D_min"] and not critical_in_window)
        failed_text = audit["reason"].rstrip(".")
        if healthy:
            category, title = "kandidat_pembaruan", "Kandidat Evaluasi Pembaruan Model"
            reasons.insert(0, f"{failed_text}; bias positif berarti P3 lebih tinggi dari referensi." if (m.get("bias") or 0) > 0 else f"{failed_text}; bias negatif berarti P3 lebih rendah dari referensi.")
            reasons.insert(1, f"Input (Q rata-rata {_fmt(qw)}), pipeline (P {_fmt(pw)}) dan drift (D {_fmt(dw)}) sehat di jendela audit, jadi kesalahan tidak dijelaskan oleh masalah input.")
            spots = audit.get("spot_checks") or []
            if len(spots) >= 2:
                reasons.append(f"Titik cek referensi sebelumnya menunjukkan bias {_fmt(spots[0]['bias'], 2)} ({spots[0]['date']}) → {_fmt(spots[-1]['bias'], 2)} ({spots[-1]['date']}): penurunan ketepatan bertahap.")
            steps = ["Investigasi pola kesalahan (siang/malam, kondisi ekstrem) dan pastikan referensi benar-benar terkalibrasi.",
                     "Kumpulkan data berlabel tepercaya yang cukup, lalu latih kandidat model versi baru dengan validasi temporal.",
                     "Bandingkan kandidat dengan model frozen pada data uji yang sama; terapkan hanya dengan persetujuan.",
                     "Selama evaluasi, model frozen tetap dipakai sebagai pembanding dengan recovery otomatis dibatasi."]
        else:
            category, title = "investigasi", "Investigasi Input atau Bukti Audit Tidak Memadai"
            reasons.insert(0, f"{failed_text}, tetapi input/pipeline/drift di jendela audit tidak sehat (Q {_fmt(qw)}, P {_fmt(pw)}, D {_fmt(dw)}); kesalahan bisa berasal dari input.")
            steps = ["Prioritaskan investigasi input dan pipeline sebelum menyimpulkan model menurun.",
                     "Ulangi audit setelah input stabil.", "Batasi recovery otomatis sampai audit valid dan lolos."]
    else:
        category, title = "pertahankan", "Model Layak Dipertahankan"
        reasons.insert(0, audit["reason"])
        reasons.insert(1, f"Di jendela audit: Q rata-rata {_fmt(qw)}, P {_fmt(pw)}, D {_fmt(dw)}, Trust {_fmt(tw)}.")
        if dw is not None and dw < dr["acceptable_drift_D_min"]:
            reasons.append(f"Catatan: drift input di jendela audit tinggi (D {_fmt(dw)}); pantau lebih ketat.")
        next_audit = (pd.Timestamp(meta["period_end"]) + pd.Timedelta(days=91)).strftime("%Y-%m-%d")
        steps = ["Lanjutkan monitoring harian Q/D/P/O.", f"Audit akurasi berikutnya sekitar {next_audit} (3 bulan), atau lebih cepat bila ada alarm kritis.",
                 "Tidak ada alasan berbasis bukti untuk melatih ulang model."]

    last = days[-1] if days else None
    recovery_reasons = []
    if not audit.get("available"):
        recovery_reasons.append("audit akurasi belum tersedia")
    elif audit["status"] != "lolos":
        recovery_reasons.append(f"audit terakhir berstatus {audit['status_label']}")
    if last and last["critical_input_failure"]:
        recovery_reasons.append(f"input kritis gagal pada {last['date']}")
    if last and (last["trust"] is None or last["trust"] < dr["recovery_trust_min"]):
        recovery_reasons.append(f"Trust hari terakhir {_fmt(last['trust']) if last['trust'] is not None else 'tidak dapat dinilai'} (< {dr['recovery_trust_min']})")
    recovery = {"allowed": not recovery_reasons,
                "label": "Recovery otomatis boleh (estimated)" if not recovery_reasons else "Recovery otomatis dibatasi",
                "reasons": recovery_reasons or ["audit lolos, input kritis sehat, Trust hari terakhir ≥ " + str(dr["recovery_trust_min"])]}
    return {"category": category, "title": title, "reasons": reasons, "next_steps": steps, "factors": factors,
            "retraining_executed": False,
            "retraining_note": "Sistem tidak pernah melatih ulang otomatis. Rekomendasi pembaruan hanya memicu evaluasi model versi baru.",
            "recovery": recovery}


def evaluate(text: str, rules: dict[str, Any], as_of: str | None = None) -> dict[str, Any]:
    frame, meta = parse_csv(text, as_of)
    slots = rules["slots_per_day"]
    start = pd.Timestamp(meta["period_start"])
    end = pd.Timestamp(meta["as_of"])
    days: list[dict[str, Any]] = []
    published = pd.Series(False, index=frame.index)
    previous_medians: dict[str, float] | None = None
    current = start
    while current <= end:
        mask = (frame[TIME_COLUMN] >= current) & (frame[TIME_COLUMN] < current + pd.Timedelta(days=1))
        day_rows = frame.loc[mask]
        summary, medians, publish_mask = evaluate_day(current, day_rows, previous_medians, rules)
        summary["day_index"] = len(days) + 1
        if len(day_rows):
            slot_of_row = _slot_index(day_rows[TIME_COLUMN])
            published.loc[day_rows.index] = publish_mask[slot_of_row]
        summary["reference_slots"] = int(day_rows["rh_reference"].notna().sum())
        days.append(summary)
        previous_medians = medians if medians else None
        current += pd.Timedelta(days=1)

    statuses = [d["status"] for d in days]
    trust_values = [d["trust"] for d in days if d["trust"] is not None]
    summary = {
        "days_evaluated": len(days),
        "mean_trust": _round(float(np.mean(trust_values)), 2) if trust_values else None,
        "status_counts": {key: statuses.count(key) for key in ["sehat", "waspada", "kritis", "tidak_dapat_dinilai"]},
        "days_with_input_issue": sum(1 for d in days if {"input_kritis", "anomali_input"} & set(d["issues"])),
        "critical_input_days": sum(1 for d in days if d["critical_input_failure"]),
        "pipeline_failed_days": sum(1 for d in days if d["batch_failed"]),
        "withheld_slots": int(sum(d["withheld_slots"] for d in days)),
        "withheld_days": sum(1 for d in days if not d["published"] and d["slots_present"]),
        "published_slots": int(sum(d["published_slots"] for d in days)),
        "expected_slots": len(days) * slots,
    }
    audit = run_audit(frame, published, meta, rules, days)
    decision = decide(days, audit, meta, rules)
    return {
        "mode": "synthetic_demo_model_health_audit", "simulation": True,
        "rule_version": rules["version"], "rules_label": rules["label"],
        "disclaimer": [
            "DEMO / DATA SINTETIS: bukan log deployment tiga bulan dan bukan pengukuran lapangan.",
            "Trust Score = kesehatan operasional (Q/D/P/O), bukan probabilitas prediksi benar.",
            "Audit akurasi memakai RH referensi terverifikasi (simulasi); hanya audit yang menilai akurasi.",
            "Model frozen (Stage 1–2, EWM, P1–P3) tidak dijalankan atau diubah oleh pemeriksaan ini; tidak ada retraining otomatis.",
        ],
        "dataset": meta,
        "weights": rules["weights"], "bands": rules["bands"],
        "summary": summary, "days": days, "audit": audit, "decision": decision,
    }
