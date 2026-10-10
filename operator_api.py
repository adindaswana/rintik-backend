"""Read-only operator dashboard views over persisted RINTIK state."""

from __future__ import annotations

import csv
import io
import math
from contextlib import closing
from typing import Any

import pandas as pd

from daily_batch import BatchError, DailyBatchService, parse_iso_date


FEATURE_GROUPS = [
    {
        "id": "atmosphere",
        "label": "Atmosfer",
        "features": [
            ("air_temperature", "Suhu udara", "°C"),
            ("pressure", "Tekanan udara", "hPa"),
            ("co2", "CO₂", "ppm"),
            ("env_battery", "Baterai sensor lingkungan", "%"),
        ],
    },
    {
        "id": "soil_1",
        "label": "Tanah · kanal 1",
        "features": [
            ("soil_ec_line1", "Konduktivitas tanah 1", "µS/cm"),
            ("soil_humidity_line1", "Kelembapan tanah 1", "% tanah"),
            ("soil_temp_line1", "Suhu tanah 1", "°C"),
        ],
    },
    {
        "id": "soil_2",
        "label": "Tanah · kanal 2",
        "features": [
            ("soil_ec_line2", "Konduktivitas tanah 2", "µS/cm"),
            ("soil_humidity_line2", "Kelembapan tanah 2", "% tanah"),
            ("soil_temp_line2", "Suhu tanah 2", "°C"),
        ],
    },
    {
        "id": "soil_3",
        "label": "Tanah · kanal 3",
        "features": [
            ("soil_ec_line3", "Konduktivitas tanah 3", "µS/cm"),
            ("soil_humidity_line3", "Kelembapan tanah 3", "% tanah"),
            ("soil_temp_line3", "Suhu tanah 3", "°C"),
        ],
    },
    {
        "id": "irrigation",
        "label": "Irigasi",
        "features": [
            ("water_volume_line1", "Volume air 1", "m³"),
            ("water_volume_line2", "Volume air 2", "m³"),
            ("water_volume_line3", "Volume air 3", "m³"),
        ],
    },
    {
        "id": "thermal",
        "label": "Indikator termal harian",
        "features": [
            ("gdd", "Growing degree days", "°C·hari"),
            ("standard_day_degree", "Standard day degree", "°C·hari"),
            ("daily_mean_temperature", "Suhu rata-rata harian", "°C"),
            ("daily_max_above_Tbase", "Maksimum di atas Tbase", "°C"),
            ("daily_max", "Suhu maksimum harian", "°C"),
            ("daily_max_reduction", "Reduksi maksimum harian", "°C"),
            ("ontario_units", "Ontario units", "indeks"),
        ],
    },
]

FEATURE_NOTES = {
    "env_battery": "Banyak nilai hilang; tampilkan gap dan jangan diimputasi di UI.",
    "water_volume_line1": "Unit mengikuti laporan; data sumber dapat memuat nilai anomali.",
    "water_volume_line2": "Unit mengikuti laporan; data sumber dapat memuat nilai anomali.",
    "water_volume_line3": "Unit mengikuti laporan; data sumber dapat memuat nilai anomali.",
}


def feature_catalog() -> dict[str, Any]:
    groups = []
    for group in FEATURE_GROUPS:
        features = []
        for key, label, unit in group["features"]:
            features.append(
                {
                    "key": key,
                    "label": label,
                    "unit": unit,
                    "group": group["id"],
                    "note": FEATURE_NOTES.get(key),
                }
            )
        groups.append({"id": group["id"], "label": group["label"], "features": features})
    return {
        "schema_version": "operator-features-v1",
        "source": "official_test_input",
        "simulation": False,
        "excluded_fields": [
            {"key": "env_id", "reason": "identifier_not_physical_signal"},
        ],
        "groups": groups,
    }


def _json_number(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


class OperatorDataService:
    def __init__(self, daily: DailyBatchService):
        self.daily = daily

    def get_series(self, day: str) -> dict[str, Any]:
        parse_iso_date(day)
        with closing(self.daily._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM daily_inputs WHERE date = ?", (day,)
            ).fetchone()
            if not row:
                raise BatchError(
                    "DATE_NOT_INGESTED",
                    "No stored input series exists for this date.",
                    404,
                )
            run = connection.execute(
                "SELECT run_id FROM runs WHERE date = ?", (day,)
            ).fetchone()
        frame = self.daily._frame_from_payload(row["csv_payload"])
        keys = [item[0] for group in FEATURE_GROUPS for item in group["features"]]
        points = []
        for record in frame.itertuples(index=False):
            points.append(
                {
                    "timestamp": record.datetime.strftime("%Y-%m-%d %H:%M:%S"),
                    "values": {key: _json_number(getattr(record, key)) for key in keys},
                }
            )
        return {
            "schema_version": "operator-series-v1",
            "date": day,
            "row_count": len(points),
            "first_timestamp": row["first_timestamp"],
            "last_timestamp": row["last_timestamp"],
            "source": "official_test_input",
            "simulation": False,
            "run_status": "completed" if run else "ingested",
            "run_id": run["run_id"] if run else None,
            "points": points,
        }

    @staticmethod
    def _safe_csv_text(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        if value.startswith(("=", "+", "-", "@", "\t", "\r")):
            return "'" + value
        return value

    def export_day_csv(self, day: str) -> str:
        parse_iso_date(day)
        with closing(self.daily._connect()) as connection:
            run = connection.execute(
                "SELECT * FROM runs WHERE date = ?", (day,)
            ).fetchone()
            if not run:
                raise BatchError(
                    "RUN_NOT_FOUND", "No completed run exists for this date.", 404
                )
            predictions = connection.execute(
                """
                SELECT timestamp, pred_internal, post_1, post_2, post_3, air_humidity
                FROM published_predictions WHERE run_id = ? ORDER BY timestamp
                """,
                (run["run_id"],),
            ).fetchall()
            has_monitoring_tables = connection.execute(
                """
                SELECT COUNT(*) FROM sqlite_master
                WHERE type = 'table' AND name = 'monitoring_runs'
                """
            ).fetchone()[0]
            monitoring = None
            observations: dict[str, Any] = {}
            recovery: dict[str, Any] = {}
            incident = None
            if has_monitoring_tables:
                monitoring = connection.execute(
                    "SELECT * FROM monitoring_runs WHERE date = ?", (day,)
                ).fetchone()
                observations = {
                    row["timestamp"]: row
                    for row in connection.execute(
                        """
                        SELECT timestamp, physical_rh, physical_source
                        FROM demo_sensor_observations WHERE date = ?
                        """,
                        (day,),
                    ).fetchall()
                }
                recovery = {
                    row["timestamp"]: row
                    for row in connection.execute(
                        """
                        SELECT timestamp, recovery_rh, recovery_source
                        FROM recovery_estimates WHERE date = ?
                        """,
                        (day,),
                    ).fetchall()
                }
                incident = connection.execute(
                    "SELECT incident_id, status FROM sensor_incidents WHERE date = ?",
                    (day,),
                ).fetchone()

        columns = [
            "timestamp", "pred_internal", "post_1", "post_2", "post_3",
            "air_humidity", "prediction_source", "simulation", "run_id",
            "model_sha256", "state_sha256", "pipeline_version", "window_mode",
            "window_start_date", "window_end_date", "window_date_count",
            "window_row_count", "simulated_physical_rh", "physical_source",
            "monitoring_verdict", "incident_id", "incident_status", "recovery_rh",
            "recovery_source",
        ]
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for prediction in predictions:
            timestamp = prediction["timestamp"]
            observation = observations.get(timestamp)
            estimate = recovery.get(timestamp)
            record = {
                "timestamp": timestamp,
                "pred_internal": prediction["pred_internal"],
                "post_1": prediction["post_1"],
                "post_2": prediction["post_2"],
                "post_3": prediction["post_3"],
                "air_humidity": prediction["air_humidity"],
                "prediction_source": "real_original_pth",
                "simulation": "false",
                "run_id": run["run_id"],
                "model_sha256": run["model_sha256"],
                "state_sha256": run["state_sha256"],
                "pipeline_version": run["pipeline_version"],
                "window_mode": run["window_mode"],
                "window_start_date": run["window_start_date"],
                "window_end_date": run["window_end_date"],
                "window_date_count": run["window_date_count"],
                "window_row_count": run["window_row_count"],
                "simulated_physical_rh": observation["physical_rh"] if observation else None,
                "physical_source": observation["physical_source"] if observation else None,
                "monitoring_verdict": monitoring["verdict"] if monitoring else None,
                "incident_id": incident["incident_id"] if incident else None,
                "incident_status": incident["status"] if incident else None,
                "recovery_rh": estimate["recovery_rh"] if estimate else None,
                "recovery_source": estimate["recovery_source"] if estimate else None,
            }
            writer.writerow({key: self._safe_csv_text(value) for key, value in record.items()})
        return stream.getvalue()
