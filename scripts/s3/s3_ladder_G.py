#!/usr/bin/env python3
"""S3 noise-ladder G, computed OUTSIDE the generator (AUD-FTONLY-S3). Deterministic, CPU only, reads per-clip result JSONs only; writes nothing but its --out dir.

G_s(rung, stage, dataset, metric) = value(numerator model of seed s) / value(denominator model of the SAME seed s), where value = the JSON's `video_macro[metric]`
(= nanmean over clips of per_video[clip][metric]; the script recomputes the nanmean and refuses a JSON where the two differ by more than 1e-9 relative).
The table statistic is the mean over the seed ratios with min/max over seeds (mean of ratios, NOT the ratio of seed means).

usage: python3 -I s3_ladder_G.py --dir DIR [--dir DIR ...] --pairs PAIRS.json --datasets bonn,kitti --rungs llb3x1,llb3x2,llb3x4,llb3x8,llb3x16 --split dev --out OUTDIR [--noise-levels noise_levels.json]
PAIRS.json: {"stages": {"none": "", "agf0.2t0.06": "+agf0.2t0.06"}, "pairs": [{"seed": 46, "num": "<file-name slug of the numerator model>", "den": "<slug of the denominator model>"}, ...]}
Files are found as <DIR>/**/<slug><stage suffix>__<dataset>__<cell>__<split>[__v3].json; exactly one match per (slug, dataset, cell) or the script stops.
Metrics: aligned AbsRel (moving cell, abs_rel_aligned), unaligned AbsRel (moving, abs_rel_unaligned), static flicker (cell st_<rung>, flicker), TAE (moving, tae), TAE GT-valid (moving, tae_gtvalid; text only).
If both slugs of a pair contain `_n<NN>_` the two numbers must be equal (seed matching is enforced by the script)."""
import argparse, csv, hashlib, json, math, re, sys
from pathlib import Path

METRICS = [("aligned AbsRel", "abs_rel_aligned", ""), ("unaligned AbsRel", "abs_rel_unaligned", ""), ("static flicker", "flicker", "st_"), ("TAE", "tae", ""), ("TAE GT-valid", "tae_gtvalid", "")]


def find(dirs, slug, ds, cell, split):
    names = {f"{slug}__{ds}__{cell}__{split}.json", f"{slug}__{ds}__{cell}__{split}__v3.json"}
    hits = sorted({p for d in dirs for p in Path(d).rglob("*.json") if p.name in names})
    if len(hits) != 1:
        raise SystemExit(f"STOP: {len(hits)} files for {slug} {ds} {cell} {split}: {[str(h) for h in hits]}")
    return hits[0]


def value(path, key):
    d = json.load(open(path)); vm = d["video_macro"][key]
    xs = [v[key] for v in d["per_video"].values() if key in v and v[key] is not None]
    xs = [x for x in xs if not math.isnan(x)]
    rec = sum(xs) / len(xs) if xs else float("nan")
    if not (math.isfinite(vm) and math.isfinite(rec)) or abs(vm - rec) > 1e-9 * max(1.0, abs(vm)):
        raise SystemExit(f"STOP: {path.name} {key}: video_macro {vm} vs recomputed nanmean {rec}")
    return float(vm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", required=True); ap.add_argument("--pairs", required=True); ap.add_argument("--datasets", required=True)
    ap.add_argument("--rungs", required=True); ap.add_argument("--split", default="dev"); ap.add_argument("--out", required=True); ap.add_argument("--noise-levels", default=None)
    a = ap.parse_args()
    pj = json.load(open(a.pairs)); stages = pj["stages"]; pairs = sorted(pj["pairs"], key=lambda p: p["seed"])
    for p in pairs:
        sn, sd = re.search(r"_n(\d{2})_", p["num"]), re.search(r"_n(\d{2})_", p["den"])
        if sn and sd and sn.group(1) != sd.group(1):
            raise SystemExit(f"STOP: seed mismatch in pair {p}")
        if sn and int(sn.group(1)) != p["seed"]:
            raise SystemExit(f"STOP: pair seed {p['seed']} != slug seed {sn.group(1)}")
    rungs = a.rungs.split(","); dss = a.datasets.split(",")
    levels = json.load(open(a.noise_levels))["rungs"] if a.noise_levels else {}
    rows = []
    for ds in dss:
        for stg, suf in sorted(stages.items()):
            for label, key, pre in METRICS:
                for r in rungs:
                    for p in pairs:
                        vn = value(find(a.dir, p["num"] + suf, ds, pre + r, a.split), key); vd = value(find(a.dir, p["den"] + suf, ds, pre + r, a.split), key)
                        rows.append((ds, stg, label, r, p["seed"], vn, vd, vn / vd))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "ladder_G_per_seed.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n"); w.writerow(["dataset", "stage", "metric", "rung", "seed", "num_value", "den_value", "G", "G_5dp"])
        for x in rows: w.writerow([*x[:5], repr(x[5]), repr(x[6]), repr(x[7]), f"{x[7]:.5f}"])
    summ = {}
    for x in rows: summ.setdefault(x[:4], []).append(x[7])
    with open(out / "ladder_G_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n"); w.writerow(["dataset", "stage", "metric", "rung", "n_seeds", "G_mean_of_seed_ratios", "G_min", "G_max", "noise_level_dev_median_255"])
        for k in sorted(summ, key=lambda t: (t[0], t[1], [m[0] for m in METRICS].index(t[2]), rungs.index(t[3]))):
            g = summ[k]; lv = levels.get(k[3], {}).get("dev_median", "")
            w.writerow([*k, len(g), repr(sum(g) / len(g)), repr(min(g)), repr(max(g)), lv])
    for f in ("ladder_G_per_seed.csv", "ladder_G_summary.csv"):
        print(hashlib.sha256((out / f).read_bytes()).hexdigest(), f)
    print("rows", len(rows), "cells", len(summ))


if __name__ == "__main__":
    main()
