#!/usr/bin/env python3
"""L1 pre-declared readout (Track 2 light screen; user Add.49/50, Auditor O41). EXPLORATORY, on EXPOSED data (VKITTI DEV = training scenes; DDAD val = 'exposed DEV gate'). Committed BEFORE the first score.
Branch A only (DDAD is a decision leg): `--ddad-gate A` is REQUIRED (any other value is refused: Branch B was not declared).
Inputs: <scores-dir>/<ARM>/scores.json (by-range scorer.py a8b0731e) and <scores-dir>/<ARM>/ddad_scores.json (ddad_scorer.py 80916c4f) for ARM in B0 B0b S1..S6 of ONE stage (--stage 2000|4000).
Output: G0 and, only if G0 passes, per-arm verdicts. Nothing here is a paper claim.
 x_V  = primary.vkitti_far_80_655.legacy_subsample_median  (VKITTI 80-655 m pred:gt; same rule as the Add.O6b screen);  dev(x) = |1 - x|.
 x_D  = DDAD bins['[120,200]'].median_pred_over_gt;  x_D80 = bins['[80,120)']...;  378 input (the registered scorer).
 G0 (validity): for EACH of B0 and B0b:  x_V <= 0.93  AND  x_D <= 0.75.   Both legs, both runs. Fail at --stage 2000 => automatic 4000-step retry of B0/B0b (arms are NOT run at 2000); fail at --stage 4000 => STOP (no arm verdicts).
 NF_p = max(|dev_V(B0) - dev_V(B0b)|, 1% relative of dev_V(B0));  NF_d = max(|dev_D(B0) - dev_D(B0b)|, 0.01 absolute);  guards NF = max(|B0 - B0b|, 1% relative of B0).
 leg V: dev_V(B0) - dev_V(arm) > NF_p.   leg D: dev_D(B0) - dev_D(arm) > NF_d  AND  dev_D80(arm) <= dev_D80(B0) + NF_b  (the [80,120) bin does not grow beyond its own noise floor;  NF_b = max(|dev_D80(B0) - dev_D80(B0b)|, 0.01 absolute)).
 CAUSE-IMPLICATED = leg V and leg D.  FIX-CANDIDATE = CAUSE-IMPLICATED and every GATED guard within NF.  PRIMARY-ONLY = leg V only.  DDAD-ONLY = leg D only.  NOT = neither.
 Integrity (before any verdict): for every scored arm, meta.ckpt.step in scores.json AND ddad_scores.json == --stage, and meta.ckpt.sha256 equal across the two files.  A final readout (more arms than B0,B0b) with a declared arm missing returns rc 5.
 NOTE: NF_p, NF_d, NF_b and the V/D legs are in dev = |1 - x| space (the same space as the legs); the guard NF_g is in the metric's own units (relative 1% of B0, or |B0 - B0b|).
 Guards (lower is better): abs_rel_unaligned for kitti, nyuv2, scannetpp; tae_gtvalid for kitti, scannetpp.  For S1 the nyuv2 and scannetpp guards are REPORTED, not gated (the arm removes those sources by design).
 DESCRIPTION ONLY (never in a verdict): DDAD overall/near-bin AbsRel and the OTHER bin medians (the [80,120) bin median is GATED through NF_b above), VKITTI exact pooled median / per-clip / ratio-of-medians / bin AbsRel."""
import argparse, json, sys
from pathlib import Path
ARMS = ["B0", "B0b", "S1", "S2", "S3", "S4", "S5", "S6"]
GUARDS = [("kitti", "abs_rel_unaligned"), ("nyuv2", "abs_rel_unaligned"), ("scannetpp", "abs_rel_unaligned"), ("kitti", "tae_gtvalid"), ("scannetpp", "tae_gtvalid")]
G0_V, G0_D = 0.93, 0.75
def dev(x): return abs(1.0 - x)
def nf_rel(a, b): return max(abs(a - b), 0.01 * abs(a))
def xv(s): return s["primary"]["vkitti_far_80_655"]["legacy_subsample_median"]
def xd(d, b="[120,200]"): return d["bins"][b]["median_pred_over_gt"]
def gd(s, ds, k): return s["guards"][ds]["video_macro"][k]

def g0(S, D):
    rows = {a: dict(xV=xv(S[a]), xD=xd(D[a]), okV=xv(S[a]) <= G0_V, okD=xd(D[a]) <= G0_D) for a in ("B0", "B0b")}
    return all(r["okV"] and r["okD"] for r in rows.values()), rows

def verdict(S, D, arm):
    """pure function of the 4 score files B0, B0b, arm -> dict (the unit under test; mutants of it are rejected by test_l1_readout.py)."""
    nfp = nf_rel(dev(xv(S["B0"])), dev(xv(S["B0b"]))); nfd = max(abs(dev(xd(D["B0"])) - dev(xd(D["B0b"]))), 0.01)
    nfb = max(abs(dev(xd(D["B0"], "[80,120)")) - dev(xd(D["B0b"], "[80,120)"))), 0.01)   # noise floor of the [80,120) bin (Manager ruling on L1 open item 5)
    chV = dev(xv(S["B0"])) - dev(xv(S[arm])); chD = dev(xd(D["B0"])) - dev(xd(D[arm])); ch80 = dev(xd(D["B0"], "[80,120)")) - dev(xd(D[arm], "[80,120)"))
    legV = chV > nfp; legD = chD > nfd and ch80 >= -nfb - 1e-12   # 1e-12: a growth that is exactly the tolerance (e.g. 0.72 -> 0.71 vs the 0.01 floor) must not flip on float noise
    worse, reported = [], []
    for ds, k in GUARDS:
        nf = nf_rel(gd(S["B0"], ds, k), gd(S["B0b"], ds, k)); w = gd(S[arm], ds, k) - gd(S["B0"], ds, k)
        if w > nf:
            (reported if (arm == "S1" and ds in ("nyuv2", "scannetpp")) else worse).append((ds, k, round(w, 5), round(nf, 5)))
    if legV and legD: label = "FIX-CANDIDATE" if not worse else "CAUSE-IMPLICATED (guard fails: FIX-CANDIDATE not granted)"
    elif legV: label = "PRIMARY-ONLY"
    elif legD: label = "DDAD-ONLY"
    else: label = "NOT"
    if label == "FIX-CANDIDATE": label = "FIX-CANDIDATE (= CAUSE-IMPLICATED + gated guards within NF)"
    return dict(label=label, chV=chV, nfp=nfp, legV=legV, chD=chD, nfd=nfd, legD=legD, ch80=ch80, nfb=nfb, guards_worse=worse, guards_reported_only=reported)

def integrity(S, D, stage):
    """-> list of problems: ckpt step == stage in both files of every scored arm; ckpt sha identical across the two files of an arm."""
    bad = []
    for a in S:
        ms, md = (S[a].get("meta") or {}).get("ckpt") or {}, (D[a].get("meta") or {}).get("ckpt") or {}
        for nm, m in (("scores.json", ms), ("ddad_scores.json", md)):
            if m.get("step") != stage: bad.append(f"{a}: {nm} meta.ckpt.step = {m.get('step')!r} != --stage {stage}")
            if not m.get("sha256"): bad.append(f"{a}: {nm} has no meta.ckpt.sha256")
        if ms.get("sha256") != md.get("sha256"): bad.append(f"{a}: ckpt sha256 differs between scores.json ({str(ms.get('sha256'))[:12]}) and ddad_scores.json ({str(md.get('sha256'))[:12]})")
    return bad

def load(root, arm):
    r = Path(root) / arm
    return json.load(open(r / "scores.json")), json.load(open(r / "ddad_scores.json"))

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--scores-dir", required=True); ap.add_argument("--stage", type=int, required=True, choices=(2000, 4000))
    ap.add_argument("--ddad-gate", required=True); ap.add_argument("--arms", default=",".join(ARMS)); a = ap.parse_args(argv)
    if a.ddad_gate != "A": print(f"REFUSED: --ddad-gate {a.ddad_gate!r}: only Branch A is declared for L1"); return 2
    want = a.arms.split(","); S, D = {}, {}
    for arm in want:
        if not (Path(a.scores_dir) / arm / "scores.json").exists(): print(f"missing {arm}: not scored yet"); continue
        S[arm], D[arm] = load(a.scores_dir, arm)
    bad = integrity(S, D, a.stage)
    if bad:
        print("INTEGRITY FAIL (no verdicts): " + "; ".join(bad)); return 4
    print(f"# L1 readout, stage {a.stage} steps, DDAD gate = A ('exposed DEV gate', exploratory, unaudited)")
    if not all(b in S for b in ("B0", "B0b")): print("G0 needs B0 and B0b: NOT EVALUABLE (no verdicts)"); return 3
    ok, rows = g0(S, D)
    for b, r in rows.items(): print(f"G0 {b}: VKITTI 80-655 m legacy median {r['xV']:.5f} (<= {G0_V}: {'ok' if r['okV'] else 'FAIL'}); DDAD [120,200] median {r['xD']:.5f} (<= {G0_D}: {'ok' if r['okD'] else 'FAIL'})")
    if not ok:
        print("G0: FAIL ->", "AUTOMATIC RETRY: run B0 and B0b at 4000 steps (arms are not run at 2000); no arm verdicts" if a.stage == 2000 else "STOP at 4000 steps: report to the Manager; no arm verdicts")
        return 10 if a.stage == 2000 else 11
    print("G0: PASS (both legs, both baselines)")
    print(f"\n# verdicts vs B0 (in-sweep baseline), NF from |B0 - B0b|")
    for arm in want:
        if arm in ("B0", "B0b") or arm not in S: continue
        v = verdict(S, D, arm)
        print(f"{arm:4s} {v['label']:62s} | V: change {v['chV']:+.5f} vs NF_p {v['nfp']:.5f} ({'leg' if v['legV'] else 'no'}) | D[120,200]: change {v['chD']:+.5f} vs NF_d {v['nfd']:.5f}, [80,120) change {v['ch80']:+.5f} ({'leg' if v['legD'] else 'no'}) | gated guards worse than NF: {v['guards_worse'] or 'none'}"
              + (f" | reported-only guards worse: {v['guards_reported_only']}" if v["guards_reported_only"] else ""))
    print("\n# description only (exposed; never in a verdict)")
    for arm in want:
        if arm not in S: continue
        p = S[arm]["primary"]["vkitti_far_80_655"]; d = D[arm]
        print(f"{arm:4s} VKITTI exact {p['pooled_median']:.4f} perclip {p['per_clip_median_mean']:.4f} ratio-of-med {p['ratio_of_medians']:.4f} bin_absrel {p['bin_absrel']:.4f} | DDAD overall AbsRel {d['overall_absrel_0_200']:.4f}, "
              + ", ".join(f"{b} {d['bins'][b]['median_pred_over_gt']:.3f}/{d['bins'][b]['absrel_unaligned_pooled']:.3f}" for b in d["bins"]) + "  (median pred:gt / AbsRel)")
    miss = [x for x in want if x not in S]
    if miss: print(f"INCOMPLETE: declared arm(s) {miss} not scored: rc 5 (this is not a complete readout)"); return 5
    return 0

if __name__ == "__main__":
    sys.exit(main())
