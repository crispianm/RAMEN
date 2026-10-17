#!/usr/bin/env python3
"""A4 output checks for the released Metric-VDA-S DDAD row at 518 (REGISTRATION.md A4). PASS/FAIL names only (no metric values).
usage: check_a4_518.py ddad_scores.json --log ddad_scorer.log --scorer-sha SHA --ref378 <the A3 release ddad_scores.json>   (exit 1 on any FAIL)"""
import argparse, json, math, re, sys
BJ = "5d3d048ce7b26fb53e1e6a5aadf8265cb92c19dafc7340bc2e04cc6bcdf6e7ef"; W = "3c28432b4e1f0d7bb31cad5151b6313b49457db5aa58d82e85bfb0f8b1311b33"
FRAMES = "b8d9664fb0c28df0fab61ad71383d48660c5444c4538a8f471fd4bbc8f0e5098"; PAIRS = "962d499612c40bca7d314f8ff16ce1d6c7b03962ae78938f472a101766150245"
L518 = "released Metric-VDA-S at its default 518 input (not the registered 378), fp16, exploratory DEV, report-only"
SHAPE = re.compile(r"first clip: model input \((\d+), 3, 518, 826\) -> depth \(\1, 518, 826\) \(same HxW, identity resample\)")
SAME = ("precision", "max_len", "resolution_rule", "alignment", "bins_m", "bench_joint_sha256", "pins", "release_weights", "spec", "status", "limit_scenes", "pins_asserted", "per_scene_scored")


def checks(d, log_text, scorer_sha, ref):
    m, r, res = d.get("meta", {}), ref.get("meta", {}), []
    c = lambda n, ok: res.append((n, bool(ok)))
    c("status FULL, pins_asserted", m.get("status") == "FULL" and m.get("pins_asserted") is True and m.get("limit_scenes") is None)
    c("short_side == 518 (meta)", m.get("short_side") == 518); c("label exact (meta+top)", m.get("label") == L518 == d.get("label"))
    c("scorer sha == A4 pin", m.get("ddad_scorer_sha256") == scorer_sha); c("bench_joint sha pin", m.get("bench_joint_sha256") == BJ)
    p = m.get("pins", {}); c("frame list + pairs sha", p.get("frames_sha256") == FRAMES and p.get("pairs_sha256") == PAIRS)
    c("release weights sha", (m.get("release_weights") or {}).get("sha256") == W); c("GH200, fp16", "GH200" in str(m.get("gpu")) and m.get("precision") == "fp16")
    c("395 frames / 50 scenes, 0 non-finite", d.get("n_scored_frames") == 395 and d.get("n_scenes") == 50 and d.get("n_nonfinite_pred_px") == 0)
    c("every other meta field == the 378 release row", all(m.get(k) == r.get(k) for k in SAME))
    bs, rb = d.get("bins", {}), ref.get("bins", {})
    c("same GT px/frames/scenes per bin as the 378 row", {k: (b["n_px"], b["n_frames"], b["n_scenes"]) for k, b in bs.items()} == {k: (b["n_px"], b["n_frames"], b["n_scenes"]) for k, b in rb.items()} and d.get("overall_n_px") == ref.get("overall_n_px"))
    c("all metrics finite", all(math.isfinite(b[k]) for b in bs.values() for k in ("absrel_unaligned_pooled", "median_pred_over_gt")) and math.isfinite(d.get("overall_absrel_0_200", float("nan"))))
    c("shape log: first clip (N,3,518,826) -> depth (N,518,826)", SHAPE.search(log_text or "") is not None)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("scores"); ap.add_argument("--log", required=True); ap.add_argument("--scorer-sha", required=True); ap.add_argument("--ref378", required=True)
    a = ap.parse_args(argv)
    res = checks(json.load(open(a.scores)), open(a.log, errors="replace").read(), a.scorer_sha, json.load(open(a.ref378)))
    for n, ok in res: print(("PASS " if ok else "FAIL ") + n)
    nf = sum(1 for _, ok in res if not ok); print(f"RESULT: {'ALL PASS' if nf == 0 else str(nf) + ' FAIL'} ({len(res)} checks)"); return 1 if nf else 0


if __name__ == "__main__":
    sys.exit(main())
