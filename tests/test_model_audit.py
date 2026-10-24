"""Model-health / 90-day accuracy audit (demo): scoring, gates, audit boundaries, decisions, API.

The engine evaluates CSV evidence only; none of these tests load the frozen model.
"""
from __future__ import annotations

import io
import math
import sys
import unittest
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import app as api  # noqa: E402
import model_audit as ma  # noqa: E402

FIXTURES = HERE.parent / "fixtures" / "audit_demo"
RULES = ma.load_rules()
WINDOW_START = pd.Timestamp("2026-03-25")


def fixture(name: str = "rintik_demo_001.csv") -> pd.DataFrame:
    return pd.read_csv(FIXTURES / name, dtype=str, keep_default_na=False)


def to_csv(frame: pd.DataFrame) -> str:
    return frame.to_csv(index=False, lineterminator="\n")


def evaluate(frame: pd.DataFrame, as_of: str | None = None) -> dict:
    return ma.evaluate(to_csv(frame), RULES, as_of)


def day_mask(frame: pd.DataFrame, date: str) -> pd.Series:
    return frame["datetime"].str.startswith(date)


def window_mask(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(frame["datetime"]) >= WINDOW_START


def clean_window(frame: pd.DataFrame) -> pd.DataFrame:
    """Every window slot gets an integer verified reference and P3 = reference (perfect, publishable)."""
    w = window_mask(frame)
    slots = np.arange(int(w.sum()))
    ref = (55 + np.round(20 * np.sin(2 * np.pi * slots / 48))).astype(int)
    frame.loc[w, "rh_reference"] = ref.astype(str)
    frame.loc[w, "rintik_p3"] = ref.astype(str)
    frame.loc[w, "rh_reference_status"] = "verified_simulated"
    frame.loc[w, "rh_reference_source"] = "SIMULATED_CALIBRATED_HYGROMETER"
    frame.loc[w, "ref_site_id"] = frame.loc[w, "site_id"]
    return frame


class DatasetHygieneTests(unittest.TestCase):
    def test_fixtures_have_90_days_4320_slots_and_only_evidence_columns(self):
        forbidden = ["scenario", "label", "status_model", "model_status", "retrain", "recommend", "trust", "q_score", "d_score",
                     "p_score", "o_score", "audit_pass", "audit_passed", "decision", "answer"]
        for name in ["rintik_demo_001.csv", "rintik_demo_002.csv", "rintik_demo_003.csv"]:
            frame = fixture(name)
            self.assertEqual(list(frame.columns), ma.SCHEMA, name)
            self.assertEqual(len(frame), 4320, name)
            times = pd.to_datetime(frame["datetime"])
            self.assertTrue(times.is_monotonic_increasing and not times.duplicated().any(), name)
            self.assertEqual(times.dt.date.nunique(), 90, name)
            self.assertTrue((times.diff().dropna() == pd.Timedelta(minutes=30)).all(), name)
            for column in frame.columns:
                self.assertFalse(any(word in column.lower() for word in forbidden), f"{name}: label-like column {column}")
            text = to_csv(frame).lower()
            for word in ["scenario", "retrain", "trust", "keep_model", "investigate"]:
                self.assertNotIn(word, text, f"{name} contains {word!r}")

    def test_reference_is_not_an_input_feature(self):
        # the evidence columns used for Q/D are the sensor columns only; P3/reference never enter drift
        self.assertNotIn("rh_reference", RULES["sensor_features"])
        self.assertNotIn("rintik_p3", RULES["sensor_features"])


class ParsingTests(unittest.TestCase):
    def assertRejected(self, frame, code):
        with self.assertRaises(ma.AuditInputError) as ctx:
            evaluate(frame)
        self.assertEqual(ctx.exception.code, code)

    def test_label_column_is_rejected(self):
        frame = fixture()
        frame["scenario"] = "A"
        self.assertRejected(frame, "UNEXPECTED_COLUMNS")

    def test_missing_column_is_rejected(self):
        self.assertRejected(fixture().drop(columns=["pipeline_stage"]), "MISSING_COLUMNS")

    def test_duplicate_timestamp_is_rejected(self):
        frame = fixture()
        frame.loc[10, "datetime"] = frame.loc[9, "datetime"]
        self.assertRejected(frame, "DUPLICATE_TIMESTAMP")

    def test_bad_and_misaligned_timestamps_are_rejected(self):
        frame = fixture()
        frame.loc[5, "datetime"] = "2026-01-01 02:15:00"
        self.assertRejected(frame, "TIMESTAMP_NOT_ALIGNED")
        frame = fixture()
        frame.loc[5, "datetime"] = "kemarin"
        self.assertRejected(frame, "INVALID_TIMESTAMP")

    def test_multiple_sites_rejected(self):
        frame = fixture()
        frame.loc[100, "site_id"] = "SITE-X"
        self.assertRejected(frame, "MULTIPLE_SITES")

    def test_corrupted_values_count_as_invalid_not_as_numbers(self):
        frame = fixture()
        mask = day_mask(frame, "2026-01-10")
        idx = frame.index[mask][:20]
        frame.loc[idx, "air_temperature"] = "ERR"
        result = evaluate(frame)
        day = result["days"][9]
        self.assertTrue(day["critical_input_failure"])  # 20/48 unreadable > 25%
        self.assertIn("anomali_input", day["issues"])
        self.assertEqual(day["status"], "kritis")
        self.assertEqual(day["published_slots"], 0)


class ScoringTests(unittest.TestCase):
    def test_trust_is_the_blueprint_weighted_sum(self):
        result = evaluate(fixture())
        for day in result["days"]:
            s = day["scores"]
            expected = 0.40 * s["Q"] + 0.30 * s["D"] + 0.20 * s["P"] + 0.10 * s["O"]
            self.assertAlmostEqual(day["trust"], round(expected, 2), delta=0.011)

    def test_critical_input_failure_with_successful_pipeline_is_held(self):
        frame = fixture()
        idx = frame.index[day_mask(frame, "2026-02-03")]
        frame.loc[idx[:30], "air_temperature"] = ""
        day = evaluate(frame)["days"][33]
        self.assertTrue(day["critical_input_failure"])
        self.assertEqual(day["scores"]["P"], 100.0)  # pipeline succeeded technically
        self.assertEqual(day["status"], "kritis")
        self.assertFalse(day["published"])
        self.assertEqual(day["published_slots"], 0)
        self.assertEqual(day["withheld_slots"], 48)

    def test_gate_overrides_a_high_aggregate_score(self):
        frame = fixture()
        idx = frame.index[day_mask(frame, "2026-02-04")]
        frame.loc[idx[:13], "pressure"] = ""  # 27% missing on one critical sensor only
        day = evaluate(frame)["days"][34]
        self.assertTrue(day["critical_input_failure"])
        self.assertGreaterEqual(day["trust"], RULES["publication"]["min_trust"])  # Trust alone would still publish …
        self.assertEqual(day["scores"]["Q"], RULES["q"]["critical_failure_q_cap"])
        self.assertEqual(day["status"], "kritis")  # … but the gate holds the day
        self.assertFalse(day["published"])

    def test_whole_day_missing_is_not_assessable_not_invented(self):
        frame = fixture()
        frame = frame.loc[~day_mask(frame, "2026-01-20")]
        day = evaluate(frame)["days"][19]
        self.assertEqual(day["slots_present"], 0)
        self.assertEqual(day["status"], "tidak_dapat_dinilai")
        self.assertIsNone(day["trust"])
        self.assertTrue(all(value is None for value in day["scores"].values()))

    def test_pipeline_failure_gives_zero_P_and_no_output_score(self):
        frame = fixture()
        idx = frame.index[day_mask(frame, "2026-01-12")]
        frame.loc[idx, ["pipeline_stage", "pipeline_status", "pipeline_error", "rintik_p3"]] = ["stage2", "error", "STAGE2_TIMEOUT", ""]
        day = evaluate(frame)["days"][11]
        self.assertTrue(day["batch_failed"])
        self.assertEqual(day["scores"]["P"], 0.0)
        self.assertIsNone(day["scores"]["O"])
        self.assertIsNone(day["trust"])
        self.assertEqual(day["status"], "kritis")

    def test_isolated_temperature_spike_is_flagged_but_does_not_fail_the_day(self):
        frame = fixture()
        idx = frame.index[day_mask(frame, "2026-01-15")]
        frame.loc[idx[20], "air_temperature"] = "60.6"
        day = evaluate(frame)["days"][14]
        self.assertIn("anomali_input", day["issues"])
        self.assertFalse(day["critical_input_failure"])
        self.assertTrue(day["published"])
        self.assertEqual(day["published_slots"], 47)  # only the corrupted slot is withheld

    def test_output_plausibility_penalises_flatline(self):
        frame = fixture()
        idx = frame.index[day_mask(frame, "2026-01-18")]
        frame.loc[idx, "rintik_p3"] = "62.0"
        day = evaluate(frame)["days"][17]
        self.assertLess(day["scores"]["O"], 80)
        self.assertIn("output", day["issues"])


class AuditTests(unittest.TestCase):
    def test_exact_threshold_rmse_6_and_zero_bias_passes(self):
        frame = clean_window(fixture())
        w = frame.index[window_mask(frame)]
        sign = np.where(np.arange(len(w)) % 2 == 0, 6, -6)
        frame.loc[w, "rintik_p3"] = (frame.loc[w, "rh_reference"].astype(int) + sign).astype(str)
        audit = evaluate(frame)["audit"]
        self.assertEqual(audit["metrics"]["n"], 336)
        self.assertEqual(audit["metrics"]["rmse"], 6.0)
        self.assertEqual(audit["metrics"]["bias"], 0.0)
        self.assertEqual(audit["status"], "lolos")

    def test_just_above_rmse_threshold_fails(self):
        frame = clean_window(fixture())
        w = frame.index[window_mask(frame)]
        sign = np.where(np.arange(len(w)) % 2 == 0, 6.1, -6.1)
        frame.loc[w, "rintik_p3"] = (frame.loc[w, "rh_reference"].astype(int) + sign).round(1).astype(str)
        self.assertEqual(evaluate(frame)["audit"]["status"], "gagal")

    def test_bias_boundary(self):
        frame = clean_window(fixture())
        w = frame.index[window_mask(frame)]
        frame.loc[w, "rintik_p3"] = (frame.loc[w, "rh_reference"].astype(int) + 2).astype(str)
        audit = evaluate(frame)["audit"]
        self.assertEqual(audit["metrics"]["bias"], 2.0)
        self.assertEqual(audit["status"], "lolos")
        frame.loc[w, "rintik_p3"] = (frame.loc[w, "rh_reference"].astype(int) + 2.5).astype(str)
        self.assertEqual(evaluate(frame)["audit"]["status"], "gagal")

    def test_min_pairs_boundary_300_vs_299(self):
        frame = clean_window(fixture())
        w = frame.index[window_mask(frame)]
        frame.loc[w[:36], ["rh_reference", "rh_reference_status", "rh_reference_source", "ref_site_id"]] = ""
        audit = evaluate(frame)["audit"]
        self.assertEqual(audit["metrics"]["n"], 300)
        self.assertEqual(audit["status"], "lolos")
        frame.loc[w[36], ["rh_reference", "rh_reference_status", "rh_reference_source", "ref_site_id"]] = ""
        audit = evaluate(frame)["audit"]
        self.assertEqual(audit["metrics"]["n"], 299)
        self.assertEqual(audit["status"], "belum_memadai")

    def test_unverified_and_wrong_site_references_are_excluded(self):
        frame = clean_window(fixture())
        w = frame.index[window_mask(frame)]
        frame.loc[w[:20], "rh_reference_status"] = "pending"
        frame.loc[w[20:30], "ref_site_id"] = "SITE-OTHER"
        audit = evaluate(frame)["audit"]
        self.assertEqual(audit["metrics"]["n"], 336 - 30)
        self.assertEqual(audit["rejected_reference"]["not_verified"], 20)
        self.assertEqual(audit["rejected_reference"]["site_mismatch"], 10)

    def test_empty_reference_means_insufficient_not_pass_or_fail(self):
        frame = fixture()
        frame[["rh_reference", "rh_reference_status", "rh_reference_source", "ref_site_id"]] = ""
        result = evaluate(frame)
        self.assertEqual(result["audit"]["status"], "belum_memadai")
        self.assertIsNone(result["audit"]["metrics"]["rmse"])
        self.assertEqual(result["decision"]["category"], "investigasi")
        self.assertFalse(result["decision"]["recovery"]["allowed"])

    def test_withheld_predictions_are_never_audited(self):
        frame = clean_window(fixture())
        idx = frame.index[day_mask(frame, "2026-03-28")]
        frame.loc[idx[:30], "co2"] = ""  # critical failure on one window day
        w = frame.index[window_mask(frame)]
        frame.loc[w, "rintik_p3"] = (frame.loc[w, "rh_reference"].astype(int) + 1).astype(str)
        frame.loc[idx, "rintik_p3"] = (frame.loc[idx, "rh_reference"].astype(int) + 40).clip(upper=100).astype(str)
        audit = evaluate(frame)["audit"]
        self.assertEqual(audit["metrics"]["n"], 288)  # the 48 held slots (with huge error) are excluded
        self.assertAlmostEqual(audit["metrics"]["bias"], 1.0)
        self.assertEqual(audit["rejected_reference"]["valid_but_prediction_withheld"], 48)


class DecisionTests(unittest.TestCase):
    def test_three_fixtures_with_one_algorithm(self):
        expected = {"rintik_demo_001.csv": "pertahankan", "rintik_demo_002.csv": "kandidat_pembaruan", "rintik_demo_003.csv": "investigasi"}
        for name, category in expected.items():
            result = ma.evaluate((FIXTURES / name).read_text(encoding="utf-8"), RULES)
            self.assertEqual(result["decision"]["category"], category, name)
            self.assertFalse(result["decision"]["retraining_executed"])

    def test_same_content_any_name_same_decision(self):
        text = (FIXTURES / "rintik_demo_002.csv").read_text(encoding="utf-8")
        self.assertEqual(ma.evaluate(text, RULES)["decision"], ma.evaluate(text.replace("\r\n", "\n"), RULES)["decision"])

    def test_high_trust_but_inaccurate_is_update_candidate(self):
        frame = clean_window(fixture())
        w = frame.index[window_mask(frame)]
        frame.loc[w, "rintik_p3"] = (frame.loc[w, "rh_reference"].astype(int) + 8).clip(upper=100).astype(str)
        result = evaluate(frame)
        self.assertGreaterEqual(result["days"][-1]["trust"], 85)
        self.assertEqual(result["audit"]["status"], "gagal")
        self.assertEqual(result["decision"]["category"], "kandidat_pembaruan")
        self.assertFalse(result["decision"]["recovery"]["allowed"])

    def test_bad_accuracy_with_unhealthy_input_is_not_a_retraining_case(self):
        frame = clean_window(fixture())
        w = frame.index[window_mask(frame)]
        frame.loc[w, "rintik_p3"] = (frame.loc[w, "rh_reference"].astype(int) + 8).clip(upper=100).astype(str)
        # soil sensors drift strongly in the whole window -> D unhealthy
        for k in (1, 2, 3):
            frame.loc[w, f"soil_ec_line{k}"] = (frame.loc[w, f"soil_ec_line{k}"].replace("", np.nan).astype(float) * 3).round(3).astype(str).replace("nan", "")
            frame.loc[w, f"soil_humidity_line{k}"] = (frame.loc[w, f"soil_humidity_line{k}"].replace("", np.nan).astype(float) + 25).round(3).astype(str).replace("nan", "")
        for column in ["co2", "soil_temp_line1", "soil_temp_line2", "soil_temp_line3"]:
            frame.loc[w, column] = (frame.loc[w, column].replace("", np.nan).astype(float) * 1.5).round(3).astype(str).replace("nan", "")
        result = evaluate(frame)
        self.assertEqual(result["audit"]["status"], "gagal")
        self.assertLess(result["decision"]["factors"]["mean_D_window"], RULES["decision"]["acceptable_drift_D_min"])
        self.assertEqual(result["decision"]["category"], "investigasi")

    def test_input_recovered_but_audit_insufficient(self):
        frame = clean_window(fixture())
        for date in ["2026-03-25", "2026-03-26"]:
            idx = frame.index[day_mask(frame, date)]
            frame.loc[idx[:40], "air_temperature"] = ""
        result = evaluate(frame)
        self.assertGreaterEqual(result["days"][-1]["trust"], 85)  # healthy again at the end
        self.assertEqual(result["audit"]["status"], "belum_memadai")  # 240 pairs
        self.assertEqual(result["decision"]["category"], "investigasi")


class TemporalTests(unittest.TestCase):
    def test_daily_scores_do_not_read_future_rows(self):
        text = (FIXTURES / "rintik_demo_003.csv").read_text(encoding="utf-8")
        full = ma.evaluate(text, RULES)
        for k in [1, 21, 41, 56, 89]:
            partial = ma.evaluate(text, RULES, as_of=full["days"][k - 1]["date"])
            self.assertEqual(partial["days"][-1], full["days"][k - 1], k)
            self.assertFalse(partial["audit"]["available"])
            self.assertEqual(partial["decision"]["category"], "periode_belum_lengkap")

    def test_changing_future_rows_never_changes_past_days(self):
        frame = fixture()
        before = evaluate(frame)
        frame.loc[pd.to_datetime(frame["datetime"]) >= pd.Timestamp("2026-03-01"), "air_temperature"] = "70"
        after = evaluate(frame)
        self.assertEqual(before["days"][:59], after["days"][:59])

    def test_incomplete_period_has_no_audit(self):
        frame = fixture()
        frame = frame.loc[pd.to_datetime(frame["datetime"]) < pd.Timestamp("2026-03-01")]
        result = evaluate(frame)
        self.assertFalse(result["audit"]["available"])
        self.assertEqual(result["decision"]["category"], "periode_belum_lengkap")

    def test_drift_baseline_is_fixed_training_statistics(self):
        self.assertEqual(RULES["_baseline"]["source"], "train.csv")
        self.assertEqual(RULES["_baseline"]["date_max"], "2023-08-29 14:30:00")


class ApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://testserver")

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_rules_endpoint_labels_demo_thresholds(self):
        response = await self.client.get("/v1/audit/rules")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["label"], "AMBANG DEMO — BELUM DIKALIBRASI")
        self.assertEqual(body["weights"], {"Q": 0.4, "D": 0.3, "P": 0.2, "O": 0.1})
        self.assertNotIn("_baseline", body)

    async def test_endpoint_evaluates_fixture_and_rejects_labels(self):
        text = (FIXTURES / "rintik_demo_002.csv").read_text(encoding="utf-8")
        response = await self.client.post("/v1/audit/model-health", json={"csv": text})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertTrue(body["simulation"])
        self.assertEqual(len(body["days"]), 90)
        self.assertEqual(body["decision"]["category"], "kandidat_pembaruan")
        labelled = to_csv(fixture().assign(model_status="ok"))
        response = await self.client.post("/v1/audit/model-health", json={"csv": labelled})
        self.assertEqual((response.status_code, response.json()["error"]["code"]), (400, "UNEXPECTED_COLUMNS"))

    async def test_endpoint_rejects_bad_body(self):
        response = await self.client.post("/v1/audit/model-health", json={"rows": []})
        self.assertEqual(response.json()["error"]["code"], "INVALID_BODY")


if __name__ == "__main__":
    unittest.main()
