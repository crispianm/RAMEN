"""Pre-declared readings (Auditor rulings of 2026-10-14, R1.3): computed verdicts, never written by hand. Pure python, no torch.

clean_noninferiority (tab:clean): one-sided, per dataset, for each subject (RAMEN raw, RAMEN + TAG) separately vs the reference model (released Metric-VDA-S):
    ratio metrics (abs_rel_aligned, abs_rel_unaligned, tae_gtvalid):  subject <= (1 + rel_tol) * reference
    absolute metrics (delta1_aligned):                                subject >= reference - tol
  all conditions on all datasets  -> sentence_preserved; otherwise sentence_differs + the numbers.
  "improves" for a metric only if the subject is better by more than rel_tol (0.01 for delta1) on EVERY dataset. tOF is reported, not part of the sentence.
eps_d_every_dataset (tab:static_diag): "X flickers more than Y" only if X's eps_d exceeds Y's on EVERY dataset shown (by the alignment-invariant diagnostic eps_d), else "mixed";
  every output carries the disclosure that eps_d was chosen after the main results (descriptive, not pre-registered).
"""
from __future__ import annotations

import math


def _num(v):
    return v if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)) else None


def _row_index(result: dict) -> dict:
    return {(r["model"], r["dataset"]): r["metrics"] for r in result["rows"]}


def clean_verdict(rd: dict, result: dict) -> dict:
    idx, ref, tol = _row_index(result), rd["reference_model"], float(rd["rel_tol"])
    absm = rd["abs_metrics"]
    out = {"rule": {"reference": ref, "ratio": f"subject <= {1 + tol:.2f} x reference on {rd['ratio_metrics']}", "absolute": f"subject >= reference - tol on {absm}",
                    "improves": f"better by more than {tol:.0%} (abs tol for absolute metrics) on every dataset", "tof": "reported only"},
           "subjects": {}}
    for s in rd["subjects"]:
        per, fails, improves = {}, [], {m: True for m in list(rd["ratio_metrics"]) + list(absm)}
        for d in rd["datasets"]:
            a, b = idx.get((s, d)), idx.get((ref, d))
            conds = {}
            for m in rd["ratio_metrics"]:
                x, y = (_num(a[m]) if a else None), (_num(b[m]) if b else None)
                ok = x is not None and y is not None and x <= (1 + tol) * y
                conds[m] = {"subject": x, "reference": y, "ratio": (x / y if x is not None and y else None), "ok": bool(ok)}
                improves[m] &= x is not None and y is not None and (y - x) / y > tol
            for m, t in absm.items():
                x, y = (_num(a[m]) if a else None), (_num(b[m]) if b else None)
                ok = x is not None and y is not None and x >= y - float(t)
                conds[m] = {"subject": x, "reference": y, "diff": (x - y if x is not None and y is not None else None), "ok": bool(ok)}
                improves[m] &= x is not None and y is not None and x > y + float(t)
            per[d] = {"conditions": conds, "ok": all(c["ok"] for c in conds.values()),
                      "tof_reported_only": {"subject": _num(a.get("tof")) if a else None, "reference": _num(b.get("tof")) if b else None}}
            fails += [f"{d}/{m} subject={c['subject']} reference={c['reference']}" for m, c in conds.items() if not c["ok"]]
        preserved = all(p["ok"] for p in per.values()) and len(per) == len(rd["datasets"])
        out["subjects"][s] = {"per_dataset": per, "preserved": preserved, "failing": fails, "improves": improves,
                              "sentence": rd["sentence_preserved"] if preserved else rd["sentence_differs"] + " (" + "; ".join(fails) + ")"}
    return out


def eps_d_verdict(rd: dict, item: dict, result: dict) -> dict:
    idx, m, ds = _row_index(result), rd["metric"], list(item["datasets"])
    models = list(dict.fromkeys(r["model"] for r in result["rows"]))
    pairs = []
    for i, x in enumerate(models):
        for y in models[i + 1:]:
            vx = [_num((idx.get((x, d)) or {}).get(m)) for d in ds]
            vy = [_num((idx.get((y, d)) or {}).get(m)) for d in ds]
            if None in vx or None in vy:
                res = "mixed (value missing on a dataset)"
            elif all(a > b for a, b in zip(vx, vy)):
                res = f"{x} flickers more than {y} by the alignment-invariant diagnostic {m}"
            elif all(b > a for a, b in zip(vx, vy)):
                res = f"{y} flickers more than {x} by the alignment-invariant diagnostic {m}"
            else:
                res = "mixed"
            pairs.append({"a": x, "b": y, "datasets": ds, "result": res})
    return {"rule": f"'X flickers more than Y' only if X's {m} exceeds Y's on every dataset shown ({', '.join(ds)}); never the paper's flicker metric; otherwise 'mixed'",
            "pairs": pairs, "disclosure": rd["disclosure"]}


def apply_reading(item: dict, result: dict) -> None:
    """Attach the computed verdict to an aggregated item (result['verdict'], result['disclosure'], row['verdict'])."""
    rd = item.get("reading")
    if not rd:
        return
    if rd["kind"] == "clean_noninferiority":
        v = clean_verdict(rd, result)
        result["verdict"] = v
        for r in result["rows"]:
            sv = v["subjects"].get(r["model"])
            if sv:
                p = sv["per_dataset"].get(r["dataset"])
                r["verdict"] = "within-tolerance" if p and p["ok"] else "differs: " + ", ".join(m for m, c in (p or {"conditions": {}})["conditions"].items() if not c["ok"])
            elif r["model"] == rd["reference_model"]:
                r["verdict"] = "reference"
            else:
                r["verdict"] = "comparator (no verdict)"
    elif rd["kind"] == "eps_d_every_dataset":
        result["verdict"] = eps_d_verdict(rd, item, result)
        result["disclosure"] = rd["disclosure"]
