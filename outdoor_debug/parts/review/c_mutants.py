"""c: mutation testing of scorer.py pure functions against tests/test_outdoor_scorer.py (unchanged). Each mutant = textual edit of a COPY of scorer.py; the real test file is run against it (CPU)."""
import subprocess, shutil, os, sys
from pathlib import Path
REPO=Path("/home/wg19671/.claude/jobs/a0d391cb/tmp/od"); SC=REPO/"outdoor_debug/parts/scorer"
W=Path("/tmp/rv_c/x/y/sc"); 
M=[
 ("far_mask lo inclusive -> exclusive (gt >= lo -> gt > lo)", "(gt >= lo)", "(gt > lo)"),
 ("far_mask sentinel exclusive -> inclusive (gt < sentinel -> <=)", "(gt < sentinel)", "(gt <= sentinel)"),
 ("SENTINEL_M 655 -> 655.4 (admits 655.35 sentinel)", "FAR_LO, SENTINEL_M = 80.0, 655.0", "FAR_LO, SENTINEL_M = 80.0, 655.4"),
 ("SENTINEL_M 655 -> 654 ", "FAR_LO, SENTINEL_M = 80.0, 655.0", "FAR_LO, SENTINEL_M = 80.0, 654.0"),
 ("FAR_LO 80 -> 79 (def-time default arg)", "FAR_LO, SENTINEL_M = 80.0, 655.0", "FAR_LO, SENTINEL_M = 79.0, 655.0".replace("79.0, 655.0","655.0").replace("FAR_LO, SENTINEL_M = 79.0","FAR_LO, SENTINEL_M = 79.0")),
 ("far_mask drops isfinite", "return np.isfinite(gt) & (gt >= lo)", "return (gt >= lo)"),
 ("range_masks lower edge (gt > lo) -> (gt >= lo)", "(gt > max(lo, 0.0))", "(gt >= max(lo, 0.0))"),
 ("range_masks upper edge (gt <= hi) -> (gt < hi)", "(gt <= hi)", "(gt < hi)"),
 ("RANGES 10-30 -> 11-30 edge move", '"10-30": (10.0, 30.0)', '"10-30": (11.0, 30.0)'),
 ("RANGES >80 lo 80 -> 81", '">80": (80.0, np.inf)', '">80": (81.0, np.inf)'),
 ("range_masks min_depth 0.1 -> 0.0", "min_depth: float = 0.1) -> tuple", "min_depth: float = 0.0) -> tuple"),
 ("range_masks sky not excluded", "(~sky if sky is not None else True)", "True"),
 ("absrel_sums divides by pred", "np.abs(p - g) / g).sum()", "np.abs(p - g) / p).sum()"),
 ("absrel_sums returns mean not sum (n,s) wrong", "float((np.abs(p - g) / g).sum())", "float((np.abs(p - g) / g).mean())"),
 ("ratio_stats median -> mean", "m = float(np.median(ratios))", "m = float(np.mean(ratios))"),
 ("sky_levels frac_gt80 threshold 80 -> 90", "(allv > 80).mean()", "(allv > 90).mean()"),
 ("sky_levels frame-balanced cap 3000 -> 30", "v.size <= 3000 else rng.choice(v, 3000", "v.size <= 30 else rng.choice(v, 30"),
 ("main(): vkitti sky def gtf >= sky_depth -> > (not unit-testable pure fn)", 'sky = (gtf >= proto["sky_depth_m"])', 'sky = (gtf > proto["sky_depth_m"])'),
 ("main(): legacy subsample 100000 -> 1 (far)", "[:100000].numpy()", "[:1].numpy()"),
 ("main(): near bin gtf < 30 -> < 31", "(gtf < 30)", "(gtf < 31)"),
 ("main(): far bin uses range_masks instead of far_mask (80<gt<=inf vs 80<=gt<655)", "fm = far_mask(gtf)", "fm = range_masks(gtf, sky)[0]['>80']"),
 ("main(): primary success quantity abs(1-median) -> median", "abs_dev=abs(1.0 - m)", "abs_dev=m"),
 ("main(): pooled absrel accumulates into range bins wrongly (acc_rng[r][0] += n_)->(+= 0)", "acc_rng[r][0] += n_;", "acc_rng[r][0] += 0;"),
]
def run(name, a, b):
    if W.exists(): shutil.rmtree(W)
    (W/"tests").mkdir(parents=True)
    t=(SC/"scorer.py").read_text()
    if a is not None:
        assert t.count(a)==1, f"pattern count {t.count(a)}: {name}"
        t=t.replace(a,b)
    (W/"scorer.py").write_text(t); shutil.copy(SC/"tests/test_outdoor_scorer.py", W/"tests/")
    env={**os.environ,"CUDA_VISIBLE_DEVICES":"","PYTHONPATH":str(REPO),"PYTHONDONTWRITEBYTECODE":"1"}
    r=subprocess.run([os.path.expanduser("~/.venvs/ramen/bin/python"),"-m","pytest","-q","-x","-p","no:cacheprovider",str(W/"tests/test_outdoor_scorer.py")],capture_output=True,text=True,env=env,cwd="/tmp/rv_c")
    last=[l for l in r.stdout.splitlines() if l.startswith(("FAILED","ERROR")) or " passed" in l or " failed" in l]
    return r.returncode, "; ".join(last)[:160]
rc,o=run("unmutated",None,None); print(f"[BASELINE unmutated] rc={rc} {o}")
for name,a,b in M:
    if "FAR_LO 80 -> 79" in name: a,b=("FAR_LO, SENTINEL_M = 80.0, 655.0","FAR_LO, SENTINEL_M = 79.0, 655.0")
    rc,o=run(name,a,b); print(f"[{'CAUGHT' if rc!=0 else 'SURVIVED'}] {name} :: {o}")
