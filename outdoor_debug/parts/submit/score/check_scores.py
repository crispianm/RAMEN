#!/usr/bin/env python3
"""Validate a by-range scorer scores.json (outdoor-debug screen). Exit 0 = all PASS, 1 = any FAIL.
Usage: python -I check_scores.py scores.json [--expect-step N] [--expect-ckpt-sha SHA] [--expect-host-contains GH200]
Checks: pins_asserted true; scorer sha; bench_joint sha (+ equals its pin); clip-list pins; ckpt step/sha/weights_key=trainable_ema; host type string;
nyuv2_frames 160; precision fp16; no limit_clips/max_frames; all four datasets present with the pinned clip counts; TAE (finite tae_gtvalid and tae) for kitti and scannetpp;
primary VKITTI far statistic finite."""
import argparse, json, math, sys

SCORER_SHA = "a8b0731e7270b911f92d56eade55b413338b12fd4373aac2b93ae1a7da8171df"
BJ_SHA = "5d3d048ce7b26fb53e1e6a5aadf8265cb92c19dafc7340bc2e04cc6bcdf6e7ef"
COMBINED = "aca96bbd894855cc99b8d97dbdf6e957d1c1f4efb0d49c77ff6108d8dccd7de6"
DSETS = {"kitti": 7, "nyuv2": 4, "scannetpp": 31, "vkitti": 10}
TAE_DS = ("kitti", "scannetpp")
R3 = {"kitti": "e187a28b87d43f3b3e6f687a0acc944f216fa754b4512fc373d616c5bece0a3a", "nyuv2": "49adc06914d2bfcdec0d602a37099f0289e498780302df39f512a010c6e58797",
      "scannetpp": "010a310b828dedeaac77700463d243644d83b35a8146d75fa88da79862bb37c6", "vkitti": "2f57ab4872f2ff8e82e86c30f89fa21de5db3351d457df7dc9bb4e7389f15073"}
PRIMARY_MASK = "80.0<=gt<655.0, finite; sentinel 655.35 excluded on GT"

def fin(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)

def get(d, *path):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d

def run(d, a):
    out = []
    def chk(name, ok, detail=""):
        out.append((name, bool(ok), detail))
    m = d.get("meta", {}) if isinstance(d, dict) else {}
    chk("pins_asserted is true", m.get("pins_asserted") is True, f"got {m.get('pins_asserted')!r}")
    chk("scorer sha256", m.get("scorer_sha256") == SCORER_SHA, f"got {m.get('scorer_sha256')}")
    chk("bench_joint sha256", m.get("bench_joint_sha256") == BJ_SHA and m.get("bench_joint_pinned") == BJ_SHA, f"got {m.get('bench_joint_sha256')} pinned {m.get('bench_joint_pinned')}")
    chk("combined clip sha", m.get("combined_clip_sha256") == COMBINED == m.get("combined_pinned"), f"got {m.get('combined_clip_sha256')}")
    ck = m.get("ckpt", {}) or {}
    chk("ckpt weights_key trainable_ema", ck.get("weights_key") == "trainable_ema", f"got {ck.get('weights_key')!r}")
    chk("--expect-step / --expect-ckpt-sha / --expect-host-contains were all passed (strict mode, reviewer finding 3/4)", a.expect_step is not None and bool(a.expect_ckpt_sha) and bool(a.expect_host_contains), "an expect flag is missing")
    if a.expect_step is not None:
        chk(f"ckpt step == {a.expect_step}", ck.get("step") == a.expect_step, f"got {ck.get('step')!r}")
    else:
        chk("ckpt step present", isinstance(ck.get("step"), int), f"got {ck.get('step')!r}")
    if a.expect_ckpt_sha:
        chk("ckpt sha256", ck.get("sha256") == a.expect_ckpt_sha, f"got {ck.get('sha256')}")
    else:
        s = ck.get("sha256")
        chk("ckpt sha256 recorded (64 hex)", isinstance(s, str) and len(s) == 64, f"got {s!r}")
    hs = f"{m.get('host')} | {m.get('gpu')}"
    if a.expect_host_contains:
        chk(f"gpu string contains {a.expect_host_contains!r} (gpu field only: the hostname can contain it spuriously)", a.expect_host_contains in str(m.get("gpu")), f"got gpu {m.get('gpu')!r}")
    else:
        chk("host type string recorded", bool(m.get("host")) and bool(m.get("gpu")), f"got {hs}")
    chk("nyuv2_frames == 160", m.get("nyuv2_frames") == 160, f"got {m.get('nyuv2_frames')!r}")
    chk("precision fp16", m.get("precision") == "fp16", f"got {m.get('precision')!r}")
    chk("no limit_clips/max_frames (keys present and null)", "limit_clips" in m and "max_frames" in m and m.get("limit_clips") is None and m.get("max_frames") is None, f"got {m.get('limit_clips')!r}/{m.get('max_frames')!r}")
    chk("meta.datasets has all four", all(k in (m.get("datasets") or []) for k in DSETS), f"got {m.get('datasets')}")
    g = d.get("guards", {}) if isinstance(d, dict) else {}
    cl = m.get("clip_lists", {}) or {}
    for ds in DSETS:
        c = cl.get(ds) or {}
        chk(f"{ds} clip-key sha256 == R3 pin", c.get("keys_sha256") == R3[ds] == c.get("pinned"), f"got {c.get('keys_sha256')}")
        chk(f"{ds} per_video keys == clip_lists keys", sorted((get(g, ds, "per_video") or {}).keys()) == sorted(c.get("keys") or ["<none>"]), "per_video keys differ from clip list")
    for ds, n in DSETS.items():
        gd = g.get(ds)
        chk(f"{ds} present with {n} clips", isinstance(gd, dict) and gd.get("n_clips") == n and len(gd.get("per_video", {})) == n,
            f"n_clips={get(g, ds, 'n_clips')!r} per_video={len(get(g, ds, 'per_video') or {})}")
        chk(f"{ds} abs_rel_unaligned finite", fin(get(g, ds, "video_macro", "abs_rel_unaligned")), f"got {get(g, ds, 'video_macro', 'abs_rel_unaligned')!r}")
    for ds in TAE_DS:
        for k in ("tae_gtvalid", "tae"):
            v = get(g, ds, "video_macro", k)
            chk(f"{ds} {k} present+finite", fin(v), f"got {v!r}")
    for ds in ("kitti", "scannetpp"):
        pv = get(g, ds, "per_video") or {}
        bad = [k for k, r in pv.items() if not fin((r or {}).get("tae_gtvalid"))]
        chk(f"{ds} every clip has finite tae_gtvalid", not bad and bool(pv), f"bad clips {bad}")
    v = get(d, "primary", "vkitti_far_80_655", "legacy_subsample_median")
    chk("primary vkitti far legacy_subsample_median finite and in (0,2)", fin(v) and 0.0 < v < 2.0, "missing, non-finite or out of range (value withheld: sealed)")
    pr = get(d, "primary", "vkitti_far_80_655") or {}
    chk("primary legacy_subsample_n == 1000000", pr.get("legacy_subsample_n") == 1000000, f"got {pr.get('legacy_subsample_n')!r}")
    chk("primary mask string == [80,655) GT-side, sentinel excluded", pr.get("mask") == PRIMARY_MASK, f"got {pr.get('mask')!r}")
    chk("primary n (exact far px) >= 1e6 and pooled_median finite", isinstance(pr.get("n"), int) and pr.get("n") >= 1000000 and fin(pr.get("pooled_median")), f"n={pr.get('n')!r}")
    return out

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("scores")
    ap.add_argument("--expect-step", type=int, default=None)
    ap.add_argument("--expect-ckpt-sha", default=None)
    ap.add_argument("--expect-host-contains", default=None)
    a = ap.parse_args(argv)
    d = json.load(open(a.scores))
    res = run(d, a)
    for n, ok, det in res:
        print(("PASS " if ok else "FAIL ") + n + ("" if ok else f"   [{det}]"))
    nf = sum(1 for _, ok, _ in res if not ok)
    print(f"RESULT: {'ALL PASS' if nf == 0 else str(nf) + ' FAIL'} ({len(res)} checks) {a.scores}")
    return 0 if nf == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
