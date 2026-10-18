#!/usr/bin/env python3
"""Generates MORNING_REPORT_20261018.md tables from the committed score files (L1 8 arms; AV2 Stage 1 verdict + runs). Stdlib only."""
import json, sys, hashlib
from pathlib import Path
H = Path(__file__).resolve().parent
sys.path.insert(0, str(H / "far_diag" / "l1_final"))
import l1_readout as R
SC = H / "far_diag" / "l1_final" / "scores_k2000_sealed"
S, D = {}, {}
for a in R.ARMS:
    S[a], D[a] = R.load(SC, a)
dev = R.dev
out = []
out.append("### L1 (K=2000) eight-arm table: exploratory, exposed DEV, unaudited\n")
nfp = R.nf_rel(dev(R.xv(S['B0'])), dev(R.xv(S['B0b']))); nfd = max(abs(dev(R.xd(D['B0'])) - dev(R.xd(D['B0b']))), 0.01)
d80 = lambda a: dev(R.xd(D[a], "[80,120)"))
nfb = max(abs(d80('B0') - d80('B0b')), 0.01)
out.append(f"Noise floors from |B0-B0b| (dev = |1-x|): NF_p (VKITTI) = {nfp:.5f}; NF_d (DDAD [120,200]) = {nfd:.5f}; NF_b (DDAD [80,120)) = {nfb:.5f}.\n")
gk = [("kitti", "abs_rel_unaligned"), ("nyuv2", "abs_rel_unaligned"), ("scannetpp", "abs_rel_unaligned"), ("kitti", "tae_gtvalid"), ("scannetpp", "tae_gtvalid")]
nfg = {g: R.nf_rel(R.gd(S['B0'], *g), R.gd(S['B0b'], *g)) for g in gk}
hdr = "| arm | verdict | x_V | x_D[120,200] | dev_D[80,120) | DDAD overall AbsRel | " + " | ".join(f"{d}.{k.replace('abs_rel_unaligned','absrel').replace('tae_gtvalid','tae')} (NF {nfg[(d,k)]:.4f})" for d, k in gk) + " |"
out.append(hdr); out.append("|" + "---|" * (hdr.count("|") - 1))
for a in R.ARMS:
    v = R.verdict(S, D, a) if a not in ("B0", "B0b") else dict(label="baseline")
    row = [a, v["label"].split(" (")[0], f"{R.xv(S[a]):.4f}", f"{R.xd(D[a]):.4f}", f"{d80(a):.4f}", f"{D[a]['overall_absrel_0_200']:.4f}"]
    for g in gk:
        val = R.gd(S[a], *g); delta = val - R.gd(S['B0'], *g)
        row.append(f"{val:.4f} ({delta:+.4f})" if a != "B0" else f"{val:.4f}")
    out.append("| " + " | ".join(row) + " |")
out.append("\nGuard cells show the value and (change vs B0). A guard is 'worse than NF' when the change exceeds its NF (KITTI/NYUv2/ScanNet++ abs_rel_unaligned, KITTI/ScanNet++ tae_gtvalid; S1's NYUv2/ScanNet++ guards are reported-only by design).\n")
v = json.load(open(H / "confirm" / "av2" / "stage1_runs_sealed" / ".." / ".." / ".." / ".." / ".." / "ignore_missing.json")) if False else None
V = json.load(open(Path(sys.argv[1])))
out.append("### AV2 Stage 1 (confirmatory strict, TS2+H300 vs TS2, three pairs): verdict " + V["verdict"] + "\n")
out.append("| pair | P_H300 (m) | P_TS2 (m) | d = ln(P_H300/P_TS2) |"); out.append("|---|---|---|---|")
for k, p in V["per_pair"].items():
    out.append(f"| {k} | {p['P_H300']:.1f} | {p['P_TS2']:.1f} | {p['d_ln_PH300_over_PTS2']:.4f} |")
out.append(f"\nd_mean {V['d_mean']:.4f}, d_min {V['d_min']:.4f}, ln 1.5 = {V['ln_margin']:.4f}. Guards (H300 AbsRel / TS2 AbsRel, limit 1.05; px per bin 3.09M / 3.92M / 1.33M / 0.66M / 0.36M, none unevaluable):\n")
bins = list(next(iter(V["per_pair"].values()))["guards"])
out.append("| pair | " + " | ".join(bins) + " |"); out.append("|---|" + "---|" * len(bins))
for k, p in V["per_pair"].items():
    out.append(f"| {k} | " + " | ".join(f"{p['guards'][b]['absrel_ratio_H300_over_TS2']:.4f}{' **FAIL**' if p['guards'][b]['status']=='FAIL' else ''}" for b in bins) + " |")
out.append("\nVerdict string and disclosure (verbatim): " + V["verdict"] + " | " + V["disclosure"] + "\n")
Ref = json.load(open(H / "confirm" / "av2" / "stage1_runs_sealed" / "ref_metric_vda_s.json"))
out.append("Reference row (description only; released Metric-VDA-S at the registered 378 input through the same scorer): P (sky median) %.1f m; %s" % (Ref["sky"]["P_framebalanced_median"], "per-bin AbsRel (pixel-pooled, unaligned): " + ", ".join(f"{b} {Ref['bins'][b]['absrel_unaligned_pooled']:.3f}" for b in bins) + "; median pred:gt: " + ", ".join(f"{b} {Ref['bins'][b]['median_pred_over_gt']:.3f}" for b in bins) + ". (The AV2 portrait camera makes focal length ambiguous: the high AbsRel is equal across arms.)"))
print("\n".join(out))
