#!/usr/bin/env python3
"""N7 dry run on the SIX REAL step-14000 Stage 1 checkpoints (before av2_stage1_ckpts.txt is filled). Reads each checkpoint ONLY for step + its own stored cfg (no model forward, no AV2 data):
for each (pair, arm): S.assert_ckpt_facts (step == 14000, pair known) and S.assert_ckpt_cfg (hinge keys / seed / resume_reseed / resume_from / max_steps / ts2 / zero_disparity facts) must PASS on the real file;
per pair: the H300 and TS2 stored cfgs differ ONLY in the allowed keys (S.ALLOWED_PAIR_DIFF); the six cfg hashes are printed; negative controls: the same real cfgs checked under the OTHER arm label or another pair label must be REFUSED.
usage: dryrun_stage1_real_ckpts.py CKPT_ROOT   (CKPT_ROOT/S1_<nXX>_<ARM>/step14000.pt)"""
import copy, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
import av2_scorer as S

root = Path(sys.argv[1]); PAIR = {"n46": "s50", "n47": "s51", "n48": "s52"}; ok = True
cfgs = {}
def refused(fn):
    try: fn(); return False
    except AssertionError: return True
for n, pair in PAIR.items():
    for arm in ("TS2", "H300"):
        p = root / f"S1_{n}_{arm}" / "step14000.pt"
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        ck = torch.load(p, map_location="cpu", weights_only=False)
        step, cfg = ck.get("step"), ck.get("cfg"); del ck
        try:
            S.assert_ckpt_facts({"step": step}, pair); S.assert_ckpt_cfg(cfg, pair, arm); res = "PASS"
        except AssertionError as e:
            res = "FAIL: " + str(e)[:160]; ok = False
        other = "TS2" if arm == "H300" else "H300"
        neg_arm = refused(lambda: S.assert_ckpt_cfg(cfg, pair, other))
        neg_pair = refused(lambda: S.assert_ckpt_cfg(cfg, {"s50": "s51", "s51": "s52", "s52": "s50"}[pair], arm))
        ok &= neg_arm and neg_pair
        cfgs[(pair, arm)] = cfg
        print(f"{pair} {arm}: sha256 {sha} step {step} cfg_sha256 {S.cfg_sha256(cfg)[:16]} | real-file facts {res} | negative controls: other-arm label refused={neg_arm}, other-pair label refused={neg_pair}")
for pair in PAIR.values():
    def norm(c):
        c = json.loads(json.dumps(c))
        if c["loss"].get("sky_hinge_max_depth") is None: c["loss"].pop("sky_hinge_max_depth", None)
        return c
    d = S.cfg_diff(norm(cfgs[(pair, "H300")]), norm(cfgs[(pair, "TS2")]))
    good = d <= S.ALLOWED_PAIR_DIFF
    ok &= good
    print(f"{pair}: H300 vs TS2 stored-cfg differing keys = {sorted(d)} -> {'within the allowed set' if good else 'OUTSIDE the allowed set ' + str(sorted(d - S.ALLOWED_PAIR_DIFF))}")
print("N7 DRY RUN:", "ALL PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
