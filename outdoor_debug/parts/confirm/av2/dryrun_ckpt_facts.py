#!/usr/bin/env python3
"""Dry run of the scorer's checkpoint-fact assertions (N1) on the REAL local start checkpoints c3_n46/47/48 step12000 (arm checkpoints do not exist yet).
(a) the real start checkpoints: assert_ckpt_facts must REFUSE (step 12000 != 14000) and assert_ckpt_cfg must refuse (no resume_from, seed 46/47/48, no hinge min depth for H300);
(b) each real start cfg + the declared Stage 1 overrides of (pair, arm) (seed, resume_reseed, resume_from, max_steps 14000, hinge, TS2 labels): assert_ckpt_cfg must PASS for all six;
(c) the declared stage1/configs yaml of all six must pass;
(d) check_pair_cfgs (D2/D3: stored cfg vs arm label, pair differs only in the hinge keys + checkpoint dir) on run-meta-shaped dicts built from (b) and from (c), plus the real start cfgs' loss.sky_vda_zero_disparity / sky_hinge_max_depth.
Lines (b), (c), (d) are CIRCULAR: the derived cfgs are built from the scorer's own constants / the declared yamls, so they only show that the checker is self-consistent with the declared design; only (a) and the
'real start cfg facts' line say something about real files. The arm checkpoints (step 14000) do not exist yet: THIS DRY RUN MUST BE REPEATED ON THE SIX REAL step-14000 CHECKPOINTS BEFORE THE REGISTRY IS FILLED (RUNBOOK_STAGE1.md).
usage: dryrun_ckpt_facts.py [CKPT_ROOT]"""
import copy, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import av2_scorer as S

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/DATA/DegradedDepth/checkpoints_v3/eval_20261012_c3rep/RAMEN_train_dn01_c3rep_20261009/runs")
PAIR_OF = {"46": "s50", "47": "s51", "48": "s52"}


def refusal(fn):
    try:
        fn(); return "PASSED"
    except AssertionError as e:
        return "REFUSED: " + str(e)[:150]


def derive(real_cfg, pair, arm):
    import yaml  # noqa: F401
    c = copy.deepcopy(real_cfg); n = {v: k for k, v in PAIR_OF.items()}[pair]
    c["data"]["seed"] = S.SEED_OF[pair]; c["training"]["resume_reseed"] = S.SEED_OF[pair]
    c["training"]["resume_from"] = S.START_FMT.format(n=n); c["training"]["max_steps"] = 14000
    for s in c["data"]["sources"]:
        if s.get("type") == "tartanair":
            s["ts2_sky"] = True
    c["training"]["checkpoint_dir"] = f"runs/confirm1_S1_n{n}_{arm}"
    c["loss"]["sky_hinge_weight"] = S.HINGE_W if arm == "H300" else 0.0
    if arm == "H300":
        c["loss"]["sky_hinge_min_depth"] = S.HINGE_MIN
    return c


def run_meta(cfg, pair, arm):
    """A run-json-shaped dict as the scorer would write it for a checkpoint with this cfg."""
    c = json.loads(S.cfg_canonical(cfg))
    return dict(meta=dict(arm=arm, seed_label=pair, cfg=c, cfg_sha256=S.cfg_sha256(c), **S.assert_ckpt_cfg(c, pair, arm)))


def main():
    import torch, yaml
    for n, pair in PAIR_OF.items():
        p = ROOT / f"dn01_c3rep_20261009_c3_n{n}" / "step12000.pt"
        ck = torch.load(p, map_location="cpu", weights_only=False); cfg = ck["cfg"]
        print(f"== real start checkpoint c3_n{n} step12000 ({p.name}) sha256 {S.A.sha256_file(p)[:16]}.. pair {pair}")
        print("  facts: step", ck["step"], "| cfg: data.seed", cfg["data"].get("seed"), "resume_from", cfg["training"].get("resume_from"), "hinge w/min", cfg["loss"].get("sky_hinge_weight"), cfg["loss"].get("sky_hinge_min_depth"))
        print("  assert_ckpt_facts(step):", refusal(lambda: S.assert_ckpt_facts(dict(step=ck["step"]), pair)))
        for arm in ("H300", "TS2"):
            print(f"  assert_ckpt_cfg(real start cfg, {pair}, {arm}):", refusal(lambda: S.assert_ckpt_cfg(cfg, pair, arm)))
        for arm in ("H300", "TS2"):
            d = derive(cfg, pair, arm)
            print(f"  (b, CIRCULAR) derived arm cfg ({pair}, {arm}) from the REAL start cfg + declared overrides: assert_ckpt_facts(14000) + assert_ckpt_cfg:",
                  refusal(lambda: (S.assert_ckpt_facts(dict(step=14000), pair), S.assert_ckpt_cfg(d, pair, arm))))
            other = "TS2" if arm == "H300" else "H300"
            print(f"     same cfg scored as --arm {other}:", refusal(lambda: S.assert_ckpt_cfg(d, pair, other)))
        print("  real start cfg facts: loss.sky_vda_zero_disparity", cfg["loss"].get("sky_vda_zero_disparity"), "| loss.sky_hinge_max_depth", cfg["loss"].get("sky_hinge_max_depth"), "| cfg sha256", S.cfg_sha256(cfg)[:16] + "..")
        print("  (d, CIRCULAR) check_pair_cfgs(derived H300, derived TS2 from the REAL start cfg):", refusal(lambda: print("      differing paths:", sorted(S.check_pair_cfgs(run_meta(derive(cfg, pair, "H300"), pair, "H300"), run_meta(derive(cfg, pair, "TS2"), pair, "TS2"))))))
        ys = {a: yaml.safe_load(open(HERE.parent / "stage1" / "configs" / f"arm_S1_n{n}_{a}.yaml")) for a in ("H300", "TS2")}
        print("  (d, CIRCULAR) check_pair_cfgs(declared yamls):", refusal(lambda: print("      differing paths:", sorted(S.check_pair_cfgs(run_meta(ys["H300"], pair, "H300"), run_meta(ys["TS2"], pair, "TS2"))))))
        for arm in ("H300", "TS2"):
            y = yaml.safe_load(open(HERE.parent / "stage1" / "configs" / f"arm_S1_n{n}_{arm}.yaml"))
            print(f"  (c, CIRCULAR) declared stage1 yaml arm_S1_n{n}_{arm}: assert_ckpt_cfg:", refusal(lambda: S.assert_ckpt_cfg(y, pair, arm)))


if __name__ == "__main__":
    main()
