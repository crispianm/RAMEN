"""a1: forced-fail of arms.py diff logic. Copy of arms.py with REPO/OUT redirected; perturbations injected into the copy only."""
import re, sys, io, contextlib, runpy
from pathlib import Path
REPO=Path("/home/wg19671/.claude/jobs/a0d391cb/tmp/od"); AC=REPO/"outdoor_debug/parts/arm_checks"
src=(AC/"arms.py").read_text()
scratch=Path("/tmp/rv_a1"); scratch.mkdir(exist_ok=True)
def run(label, patch=None, post_build=None):
    s=src.replace('REPO = Path(__file__).resolve().parents[3]', f'REPO = Path("{REPO}")').replace('OUT = Path(__file__).resolve().parent', f'OUT = Path("{scratch}")')
    if patch: s=patch(s)
    ns={"__name__":"__main__","__file__":str(scratch/"arms.py")}
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf): exec(compile(s,"arms_copy","exec"),ns)
    o=buf.getvalue()
    summ=[l for l in o.splitlines() if "DEFECT" in l or l.startswith("ALL EXPECTED")]
    print(f"--- {label}\n"+"\n".join(summ) if summ else f"--- {label}\n(no summary lines)")
run("baseline (unperturbed)")
# 1. undeclared key perturbation: TS2 arm also changes data.max_depth? (non-declared key: loss.silog_weight-like). Inject into TS2 overrides a non-declared key
run("PERT1 TS2 gets undeclared key data.clip_stride=9", lambda s: s.replace('"TS2": TS2,','"TS2": TS2+[("data.clip_stride", 9)],'))
run("PERT2 H80 gets undeclared key loss.log_terms_every=11 (also LR-key check?)", lambda s: s.replace('("loss.sky_hinge_min_depth", 80.0)]','("loss.sky_hinge_min_depth", 80.0),("loss.sky_loss_weight",5)]'))
# 2. declared key wrong value
run("PERT3 H300 declared key min_depth wrong value 30.0 (should be 300)", lambda s: s.replace('("loss.sky_hinge_min_depth", 300.0)','("loss.sky_hinge_min_depth", 30.0)'))
run("PERT4 H80 weight 0.1 -> 0.0 (hinge off but 'declared')", lambda s: s.replace('("loss.sky_hinge_weight", 0.1), ("loss.sky_hinge_min_depth", 80.0)','("loss.sky_hinge_weight", 0.0), ("loss.sky_hinge_min_depth", 80.0)'))
run("PERT5 VK655 bound 656 -> 700", lambda s: s.replace('("src:vkitti.vkitti_max_depth", 656)','("src:vkitti.vkitti_max_depth", 700)'))
run("PERT6 TS2 flag ts2_sky True->False", lambda s: s.replace('("src:tartanair.ts2_sky", True)','("src:tartanair.ts2_sky", False)'))
run("PERT7 LR key differs in one arm (H300 lr_decay_steps)", lambda s: s.replace('("loss.sky_hinge_min_depth", 300.0)]','("loss.sky_hinge_min_depth", 300.0),("training.lr_decay_steps", 99999)]'))
run("PERT8 A0b seed 49->50", lambda s: s.replace('("data.seed", 49)','("data.seed", 50)'))
