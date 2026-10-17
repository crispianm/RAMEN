#!/usr/bin/env python3
"""O28 output checks for a ddad_scorer.py result. PASS/FAIL names only (no metric values) so sealed arm outputs stay sealed.
usage: check_ddad_output.py ddad_scores.json --expect-ckpt-sha SHA --reference ref_ddad_scores.json [--log ddad_scorer.log] [--release]"""
import argparse, json, math, sys
SCORER = "80916c4f3b2ab0b8999b07e142847f8df157db5da756192f02f6df5dea1ed09f"; BJ = "5d3d048ce7b26fb53e1e6a5aadf8265cb92c19dafc7340bc2e04cc6bcdf6e7ef"
FRAMES = "b8d9664fb0c28df0fab61ad71383d48660c5444c4538a8f471fd4bbc8f0e5098"; PAIRS = "962d499612c40bca7d314f8ff16ce1d6c7b03962ae78938f472a101766150245"
OURS_LABEL = "exploratory DEV, report-only (DDAD val LR_DEV; not a held-out test, never enters the decision)"
REL_LABEL = "released Metric-VDA-S at the registered 378 input (not its default 518), fp16, exploratory DEV, report-only"
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("scores"); ap.add_argument("--expect-ckpt-sha", required=True); ap.add_argument("--reference", required=True)
    ap.add_argument("--log"); ap.add_argument("--release", action="store_true"); a = ap.parse_args()
    d = json.load(open(a.scores)); ref = json.load(open(a.reference)); m = d.get("meta", {}); res = []
    def chk(n, ok): res.append((n, bool(ok)))
    chk("status FULL", m.get("status") == "FULL"); chk("pins_asserted true", m.get("pins_asserted") is True)
    chk("scorer sha 80916c4f", m.get("ddad_scorer_sha256") == SCORER); chk("bench_joint sha 5d3d048c", m.get("bench_joint_sha256") == BJ)
    p = m.get("pins", {}); chk("frame list sha b8d9664f", p.get("frames_sha256") == FRAMES); chk("pairs sha 962d4996", p.get("pairs_sha256") == PAIRS)
    chk("precision fp16 and short_side 378", m.get("precision") == "fp16" and m.get("short_side") == 378)
    chk("GH200", "GH200" in str(m.get("gpu"))); chk("no limit_scenes", m.get("limit_scenes") is None)
    chk("395 frames / 50 scenes", d.get("n_scored_frames") == 395 and d.get("n_scenes") == 50)
    chk("0 non-finite pred px", d.get("n_nonfinite_pred_px") == 0)
    bs = d.get("bins", {}); chk("bin px sum == overall n", sum(b["n_px"] for b in bs.values()) == d.get("overall_n_px"))
    chk("same GT px/frames/scenes per bin as the reference", {k: (b["n_px"], b["n_frames"], b["n_scenes"]) for k, b in bs.items()} == {k: (b["n_px"], b["n_frames"], b["n_scenes"]) for k, b in ref["bins"].items()} and d.get("overall_n_px") == ref.get("overall_n_px"))
    chk("all metrics finite", all(math.isfinite(b[k]) for b in bs.values() for k in ("absrel_unaligned_pooled", "median_pred_over_gt")) and math.isfinite(d.get("overall_absrel_0_200", float("nan"))))
    if a.release:
        rw = m.get("release_weights", {}); chk("release weights sha 3c28432b", rw.get("sha256") == "3c28432b4e1f0d7bb31cad5151b6313b49457db5aa58d82e85bfb0f8b1311b33"); chk("release label exact (meta+top)", m.get("label") == REL_LABEL == d.get("label"))
    else:
        chk("ckpt sha == independent sha", (m.get("ckpt") or {}).get("sha256") == a.expect_ckpt_sha); chk("ckpt weights_key trainable_ema", (m.get("ckpt") or {}).get("weights_key") == "trainable_ema"); chk("label exact", d.get("label") == OURS_LABEL or m.get("label") == OURS_LABEL)
    if a.log:
        t = open(a.log, errors="replace").read(); chk("shape log: first clip HxW identical", "same HxW, identity resample" in t)
    for n, ok in res: print(("PASS " if ok else "FAIL ") + n)
    nf = sum(1 for _, ok in res if not ok); print(f"RESULT: {'ALL PASS' if nf == 0 else str(nf) + ' FAIL'} ({len(res)} checks)"); sys.exit(1 if nf else 0)
main()
