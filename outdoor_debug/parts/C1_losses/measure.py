"""Measurement script (CPU). Writes loss_diagnostics_C1.csv and prints the tables quoted in FINDINGS.md."""
import csv, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from c1_common import *

dep, m, sky, lab = outdoor_scene()
batch = make_batch(dep, m, sky)
gt = dep.clone()
scen = {
  "perfect(sky=655)": gt.clone(),
  "scale x1.5": gt * 1.5,
  "scale x0.5": gt * 0.5,
  "sky_pred=40m (valid perfect)": torch.where(sky > 0.5, torch.full_like(gt, 40.0), gt),
  "head saturates at 100m": gt.clamp(max=100.0),
  "far valid x0.5, rest perfect": torch.where(torch.tensor(lab)[None, None, None].expand_as(gt) == 4, gt * 0.5, gt),
}
rows = []
def run(cname, c):
    for sn, pr in scen.items():
        pr = pr.clone()
        Lv, gv = loss_and_grad(pr, batch, dict(c, loss=dict(c["loss"], metric_silog_weight=0.0)))
        Ls, gs = silog_only(pr, batch, c)
        for term, g, L in (("VDA", gv, Lv), ("SiLog", gs, Ls)):
            for (n, cnt, ga, gl, gsum) in per_bin(g, gt, lab, pr):
                rows.append([cname, sn, term, L, n, cnt, ga, gl, gsum])
effective = cfg()
nosky = cfg(sky_vda_zero_disparity=False)
run("effective(C3/FT: sky_vda_zero_disparity=true, silog w=1 lam=.5)", effective)
run("sky_vda_zero_disparity=false", nosky)
with open(os.path.join(os.path.dirname(__file__), "loss_diagnostics_C1.csv"), "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["config", "scenario", "term", "loss_value", "bin", "n_px", "mean_abs_grad_per_m", "mean_abs_grad_wrt_log_pred", "sum_abs_grad"]); wr.writerows(rows)
cur = None
for r in rows:
    key = (r[0], r[1], r[2])
    if key != cur:
        print(f"\n[{r[0][:30]}] {r[1]} | {r[2]} loss={r[3]:.5f}"); cur = key
    print(f"   {r[4]:22s} n={r[5]:5d} mean|g|/px={r[6]:.3e} mean|g|*pred={r[7]:.3e} sum|g|={r[8]:.3e}")
