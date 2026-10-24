"""Independent re-computation of the 90-day model-health audit (QA).

Written from the rules file only, with the Python standard library (csv, math, statistics): no pandas,
no numpy and no import of model_audit's scoring code. It recomputes per-day Q/D/P/O, Trust, status,
published/withheld slots, the audit pairs and RMSE/MAE/bias, and the decision category, then compares
them with the backend (model_audit.evaluate) within explicit tolerances.

Usage: python scripts/verify_audit_independent.py [csv ...] [--report out.json]
Tolerances: daily scores 0.01 (backend rounds to 2 dp); audit metrics 1e-4 (backend rounds to 4 dp).
"""

from __future__ import annotations

import csv
import io
import json
import math
import statistics
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULES = json.loads((ROOT / "audit" / "rules_demo_v1.json").read_text(encoding="utf-8"))
BASE = json.loads((ROOT / "audit" / RULES["d"]["baseline_file"]).read_text(encoding="utf-8"))["columns"]
SLOTS = RULES["slots_per_day"]


def num(text):
    text = (text or "").strip()
    if text == "":
        return "missing"
    try:
        value = float(text)
    except ValueError:
        return "bad"
    return value if math.isfinite(value) else "bad"


def clip01(x):
    return min(1.0, max(0.0, x))


def recompute(text):
    rows = list(csv.DictReader(io.StringIO(text)))
    by_day = {}
    for row in rows:
        stamp = datetime.strptime(row["datetime"], "%Y-%m-%d %H:%M:%S")
        by_day.setdefault(stamp.date(), {})[stamp.hour * 2 + stamp.minute // 30] = row
    first, last = min(by_day), max(by_day)
    features, critical, ranges = RULES["sensor_features"], RULES["critical_features"], RULES["physical_ranges"]
    q, d, p, o = RULES["q"], RULES["d"], RULES["p"], RULES["o"]
    days, published_keys, prev_medians = [], set(), None
    current = first
    while current <= last:
        grid = by_day.get(current, {})
        present = len(grid)
        info = {}
        for f in features:
            vals = [num(grid[s][f]) if s in grid else "missing" for s in range(SLOTS)]
            lo, hi = ranges[f]
            inr = [isinstance(v, float) and lo <= v <= hi for v in vals]
            spike = [False] * SLOTS
            if f in RULES["spike_limits_per_slot"]:
                lim = RULES["spike_limits_per_slot"][f]
                for i in range(SLOTS):
                    if inr[i]:
                        nb = [vals[j] for j in (i - 1, i + 1) if 0 <= j < SLOTS and inr[j]]
                        spike[i] = bool(nb) and all(abs(vals[i] - x) > lim for x in nb)
            bad = [v == "bad" or (isinstance(v, float) and not inr[i]) or spike[i] for i, v in enumerate(vals)]
            missing = sum(v == "missing" for v in vals)
            clean = [v for i, v in enumerate(vals) if inr[i] and not spike[i]]
            info[f] = {"missing": missing, "readable": SLOTS - missing, "invalid": sum(bad), "bad": bad, "clean": clean}
        crit_fail = any(info[f]["missing"] / SLOTS > q["critical_failure_missing_share"] or info[f]["invalid"] / SLOTS > q["critical_failure_invalid_share"] for f in critical)
        crit_fail = crit_fail or present < q["critical_failure_min_slot_share"] * SLOTS
        Q = None
        if present >= RULES["min_slots_assessable"]:
            other = [f for f in features if f not in critical]
            parts = (present / SLOTS,
                     min(info[f]["readable"] / SLOTS for f in critical),
                     sum(min(1.0, info[f]["readable"] / SLOTS / q["other_completeness_full_credit"]) for f in other) / len(other),
                     clip01(1 - q["invalid_penalty_critical"] * sum(info[f]["invalid"] for f in critical) - q["invalid_penalty_other"] * sum(info[f]["invalid"] for f in other)))
            w = q["weights"]
            Q = 100 * (w["slot_completeness"] * parts[0] + w["critical_completeness"] * parts[1] + w["other_completeness"] * parts[2] + w["validity"] * parts[3])
            if crit_fail:
                Q = min(Q, q["critical_failure_q_cap"])
        medians, sev = {}, []
        for f in features:
            clean = info[f]["clean"]
            if len(clean) >= d["min_values_per_feature"]:
                medians[f] = statistics.median(clean)
            stats = BASE[f]
            if stats.get("daily_increment"):
                if f not in medians or not prev_medians or f not in prev_medians:
                    continue
                ref, obs = stats["daily_increment"], [medians[f] - prev_medians[f]]
            else:
                if len(clean) < d["min_values_per_feature"]:
                    continue
                ref, obs = stats, clean
            z = (sum(obs) / len(obs) - ref["mean"]) / ref["sd"]
            out = sum(1 for v in obs if v < ref["p1"] or v > ref["p99"]) / len(obs)
            sev.append(max(clip01((abs(z) - d["z_free"]) / (d["z_full"] - d["z_free"])), clip01((out - d["outside_free"]) / (d["outside_full"] - d["outside_free"]))))
        D = 100 * (1 - sum(sev) / len(sev)) if len(sev) >= d["min_feature_share"] * len(features) else None
        completed = [s in grid and grid[s]["pipeline_stage"].strip() == p["final_stage"] and grid[s]["pipeline_status"].strip() == "ok" for s in range(SLOTS)]
        p3 = [num(grid[s]["rintik_p3"]) if s in grid else "missing" for s in range(SLOTS)]
        p3ok = [isinstance(v, float) and 0 <= v <= 100 for v in p3]
        errors = {grid[s]["pipeline_error"].strip() for s in grid if grid[s]["pipeline_error"].strip()}
        n_done = sum(completed)
        P = None
        if present:
            valid_share = sum(1 for s in range(SLOTS) if completed[s] and p3ok[s]) / n_done if n_done else 0.0
            P = 100 * (p["weights"]["completed_share"] * n_done / SLOTS + p["weights"]["valid_output_share"] * valid_share)
            if errors:
                P = max(0.0, P - p["error_code_penalty"])
        batch_failed = present > 0 and n_done == 0
        cand = [s for s in range(SLOTS) if completed[s] and p3[s] != "missing"]
        O = None
        if len(cand) >= RULES["min_slots_assessable"]:
            good = [s for s in cand if p3ok[s]]
            series = [p3[s] for s in good]
            spikes = sum(1 for a, b in zip(good, good[1:]) if b - a == 1 and abs(p3[b] - p3[a]) > o["spike_points_per_slot"])
            longest = run = 1
            for a, b in zip(good, good[1:]):
                run = run + 1 if (b - a == 1 and abs(p3[b] - p3[a]) < o["flatline_tolerance"]) else 1
                longest = max(longest, run)
            amp = max(series) - min(series) if series else 0.0
            pen = min(o["spike_penalty_cap"], o["spike_penalty"] * spikes) + (o["flatline_penalty"] if longest >= o["flatline_slots"] else 0) + (o["low_amplitude_penalty"] if amp < o["min_daily_amplitude"] else 0)
            O = max(0.0, min(100.0, 100 * len(good) / len(cand) - pen))
        scores = {"Q": Q, "D": D, "P": P, "O": O}
        trust = sum(RULES["weights"][k] * scores[k] for k in scores) if all(v is not None for v in scores.values()) else None
        if present == 0 or Q is None:
            status = "tidak_dapat_dinilai"
        elif crit_fail or batch_failed:
            status = "kritis"
        elif trust is None:
            status = "tidak_dapat_dinilai"
        else:
            status = "sehat" if trust >= RULES["bands"]["healthy_min"] else "waspada" if trust >= RULES["bands"]["caution_min"] else "kritis"
        publish_day = status in ("sehat", "waspada") and trust is not None and trust >= RULES["publication"]["min_trust"]
        pub = [publish_day and completed[s] and p3ok[s] and not any(info[f]["bad"][s] for f in critical) for s in range(SLOTS)]
        for s in range(SLOTS):
            if pub[s]:
                published_keys.add(grid[s]["datetime"])
        withheld = sum(1 for s in range(SLOTS) if s in grid and isinstance(p3[s], float) and not pub[s])
        days.append({"date": current.isoformat(), "scores": scores, "trust": trust, "status": status, "critical": crit_fail,
                     "batch_failed": batch_failed, "published_slots": sum(pub), "withheld_slots": withheld})
        prev_medians = medians or None
        current += timedelta(days=1)

    a = RULES["audit"]
    complete = (last - first).days + 1 == 90
    audit = {"available": complete}
    if complete:
        start = last - timedelta(days=a["window_days"] - 1)
        errs, valid_refs = [], 0
        for row in rows:
            stamp = datetime.strptime(row["datetime"], "%Y-%m-%d %H:%M:%S")
            if stamp.date() < start:
                continue
            ref = num(row["rh_reference"])
            ok = isinstance(ref, float) and 0 <= ref <= 100 and row["rh_reference_status"].strip() in a["accepted_reference_status"] and row["ref_site_id"].strip() == row["site_id"].strip()
            valid_refs += ok
            if ok and row["datetime"] in published_keys:
                errs.append(float(row["rintik_p3"]) - ref)
        n = len(errs)
        rmse = math.sqrt(sum(e * e for e in errs) / n) if n else None
        mae = sum(abs(e) for e in errs) / n if n else None
        bias = sum(errs) / n if n else None
        if valid_refs == 0 or n < a["min_pairs"]:
            status = "belum_memadai"
        elif rmse <= a["rmse_max"] and abs(bias) <= a["abs_bias_max"]:
            status = "lolos"
        else:
            status = "gagal"
        audit.update({"n": n, "rmse": rmse, "mae": mae, "bias": bias, "status": status})
    window = days[-a["window_days"]:]
    crit_window = any(x["critical"] for x in window)

    def mean_of(key):
        vals = [x["scores"][key] for x in window if x["scores"][key] is not None]
        return sum(vals) / len(vals) if vals else None

    dr = RULES["decision"]
    if not audit["available"]:
        category = "periode_belum_lengkap"
    elif audit["status"] == "belum_memadai" or crit_window:
        category = "investigasi"
    elif audit["status"] == "gagal":
        qw, pw, dw = mean_of("Q"), mean_of("P"), mean_of("D")
        healthy = qw is not None and qw >= dr["healthy_input_Q_min"] and pw is not None and pw >= dr["healthy_pipeline_P_min"] and dw is not None and dw >= dr["acceptable_drift_D_min"]
        category = "kandidat_pembaruan" if healthy else "investigasi"
    else:
        category = "pertahankan"
    return {"days": days, "audit": audit, "category": category}


def compare(path: Path) -> dict:
    sys.path.insert(0, str(ROOT))
    import model_audit  # only to obtain the backend result being checked

    text = path.read_text(encoding="utf-8")
    mine = recompute(text)
    theirs = model_audit.evaluate(text, model_audit.load_rules())
    diffs, max_score_diff = [], 0.0
    for a, b in zip(mine["days"], theirs["days"]):
        for key in "QDPO":
            x, y = a["scores"][key], b["scores"][key]
            if (x is None) != (y is None):
                diffs.append(f"{a['date']} {key}: {x} vs {y}")
            elif x is not None:
                max_score_diff = max(max_score_diff, abs(x - y))
        if (a["trust"] is None) != (b["trust"] is None) or (a["trust"] is not None and abs(a["trust"] - b["trust"]) > 0.01):
            diffs.append(f"{a['date']} trust: {a['trust']} vs {b['trust']}")
        for key, other in [("status", "status"), ("critical", "critical_input_failure"), ("batch_failed", "batch_failed"), ("published_slots", "published_slots"), ("withheld_slots", "withheld_slots")]:
            if a[key] != b[other]:
                diffs.append(f"{a['date']} {key}: {a[key]} vs {b[other]}")
    audit_diff = {}
    if mine["audit"]["available"]:
        m = theirs["audit"]["metrics"]
        for key in ["rmse", "mae", "bias"]:
            x, y = mine["audit"][key], m[key]
            audit_diff[key] = None if x is None and y is None else abs(x - y)
            if x is not None and abs(x - y) > 1e-4:
                diffs.append(f"audit {key}: {x} vs {y}")
        if mine["audit"]["n"] != m["n"]:
            diffs.append(f"audit n: {mine['audit']['n']} vs {m['n']}")
        if mine["audit"]["status"] != theirs["audit"]["status"]:
            diffs.append(f"audit status: {mine['audit']['status']} vs {theirs['audit']['status']}")
    if mine["category"] != theirs["decision"]["category"]:
        diffs.append(f"decision: {mine['category']} vs {theirs['decision']['category']}")
    return {"file": path.name, "days": len(mine["days"]), "max_daily_score_abs_diff": round(max_score_diff, 6),
            "audit_abs_diff": audit_diff, "independent": {"audit": mine["audit"], "category": mine["category"]},
            "backend": {"audit_status": theirs["audit"]["status"], "metrics": theirs["audit"].get("metrics"), "category": theirs["decision"]["category"]},
            "mismatches": diffs, "result": "PASS" if not diffs and max_score_diff <= 0.01 else "FAIL"}


def main(argv):
    report_path = None
    if "--report" in argv:
        i = argv.index("--report")
        report_path = Path(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    files = [Path(a) for a in argv] or sorted((ROOT / "fixtures" / "audit_demo").glob("rintik_demo_*.csv"))
    results = [compare(path) for path in files]
    for r in results:
        print(f"{r['result']}  {r['file']}: max |Δ score| {r['max_daily_score_abs_diff']}, audit Δ {r['audit_abs_diff']}, "
              f"category {r['independent']['category']} (backend {r['backend']['category']}), mismatches {len(r['mismatches'])}")
        for line in r["mismatches"][:10]:
            print("   ", line)
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if all(r["result"] == "PASS" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
