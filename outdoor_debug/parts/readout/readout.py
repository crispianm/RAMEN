#!/usr/bin/env python3
"""Pre-declared screen readout (SCREEN_PROPOSAL.md 'Success rule' + Auditor O23): exploratory, unaudited.
PRIMARY quantity x = primary.vkitti_far_80_655.legacy_subsample_median; decision quantity |1-x|; change vs A0 = |1-x_A0| - |1-x_arm| (positive = closer to 1).
NF per metric = max(|A0 - A0b|, 1% relative of A0). BETTER requires primary change > NF_primary AND every guard not worse than A0 by more than its NF.
GUARDS (lower is better): video_macro.abs_rel_unaligned for kitti, nyuv2, scannetpp; video_macro.tae_gtvalid for kitti and scannetpp.
DESCRIPTION ONLY: exact pooled median, per-clip-median mean, ratio of medians, bin AbsRel, by-range AbsRel; if exact and legacy disagree on the SIGN of the change vs A0 for an arm it is stated. Hinge arms: report hinge activity from the training log separately."""
import json, sys
B = "/mnt/DATA/DegradedDepth/claude_outputs/scratch/outdoor_debug_20261016/ARMS/"
ARMS = ["A0", "A0b", "TS2", "TS2_H80", "TS2_H300", "TS2_VK655"]
S = {a: json.load(open(B + a + "/scores.json")) for a in ARMS}
def prim(a, k="legacy_subsample_median"): return S[a]["primary"]["vkitti_far_80_655"][k]
def g(a, ds, k): return S[a]["guards"][ds]["video_macro"][k]
def dev(x): return abs(1 - x)
nf = lambda a0, a0b: max(abs(a0 - a0b), 0.01 * abs(a0))
print("# arm | legacy median x | |1-x| | change vs A0 (+ = closer to 1)")
for a in ARMS: print(f"{a:10s} {prim(a):.5f} {dev(prim(a)):.5f} {dev(prim('A0')) - dev(prim(a)):+.5f}")
NF_P = nf(dev(prim("A0")), dev(prim("A0b"))); print(f"\nNF primary = max(|A0-A0b| in |1-x|, 1% rel) = {NF_P:.5f}  (|A0-A0b| = {abs(dev(prim('A0')) - dev(prim('A0b'))):.5f})")
GUARDS = [("kitti", "abs_rel_unaligned"), ("nyuv2", "abs_rel_unaligned"), ("scannetpp", "abs_rel_unaligned"), ("kitti", "tae_gtvalid"), ("scannetpp", "tae_gtvalid")]
nfg = {(ds, k): nf(g("A0", ds, k), g("A0b", ds, k)) for ds, k in GUARDS}
print("\n# guards: value A0 | A0b | NF")
for (ds, k), v in nfg.items(): print(f"{ds:10s} {k:18s} {g('A0', ds, k):.5f} {g('A0b', ds, k):.5f} NF={v:.5f}")
print("\n# verdicts (exploratory, unaudited)")
for a in ARMS[2:]:
    ch = dev(prim("A0")) - dev(prim(a)); ok_p = ch > NF_P
    worse = [(ds, k, g(a, ds, k) - g("A0", ds, k)) for ds, k in GUARDS if g(a, ds, k) - g("A0", ds, k) > nfg[(ds, k)]]
    ex = S["A0"]["primary"]["vkitti_far_80_655"]["pooled_median"]; exa = prim(a, "pooled_median")
    sign_leg = (dev(prim("A0")) - dev(prim(a)) > 0); sign_ex = (dev(ex) - dev(exa) > 0)
    print(f"{a:10s} primary change {ch:+.5f} vs NF {NF_P:.5f} -> {'beyond NF' if ok_p else 'within/below NF'}; guards worse than NF: {worse if worse else 'none'}; => {'BETTER' if (ok_p and not worse) else 'NOT BETTER'}"
          f" | exact-pooled sign {'agrees' if sign_leg == sign_ex else 'DISAGREES'} with legacy")
print("\n# description only (exact pooled, per-clip-median mean, ratio of medians, bin AbsRel >80)")
for a in ARMS:
    p = S[a]["primary"]["vkitti_far_80_655"]
    print(f"{a:10s} exact {p['pooled_median']:.5f} | perclip {p['per_clip_median_mean']:.5f} | ratio-of-med {p['ratio_of_medians']:.5f} | bin_absrel {p['bin_absrel']:.5f}")
