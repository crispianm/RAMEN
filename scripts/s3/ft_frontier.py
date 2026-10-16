"""FT version of frontier.py (Part F of TAG_diagnostics_predeclared_20261007.md, old sha 1bb6ff22...): same indices, re-pinned to FT n46 as the denominator. Reads only ft_cells.csv (from ft_filter_cells.py); python -I. Usage: ft_frontier.py <ft_cells.csv> <ft_cells.csv sha256> <out dir>"""
import csv, math, hashlib, sys
from pathlib import Path
P = Path(sys.argv[1])
OUT = Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)
assert hashlib.sha256(P.read_bytes()).hexdigest() == sys.argv[2], "CSV sha mismatch"
R = {r["row"]: r for r in csv.DictReader(open(P))}
REF = "ft"
ROWS = ["ft+TAG", "ft+kal0.05", "ft+oneeuro", "ft+ema0.7", "ft+flowema0.7", "ft+tmed5"]
FL = ["bvi_rlv|fnoise|f_noise", "kitti|st_llvar|flicker", "bonn|st_llvar|flicker", "nyuv2|st_llvar|flicker", "scannet|st_llvar|flicker"]
AC = [f"{d}|llvar|abs_rel_aligned" for d in ("kitti", "bonn", "scannet", "sintel", "nyuv2")] + ["nuscenes_night|real|abs_rel_aligned"]
NIGHT = "nuscenes_night|real|abs_rel_aligned"
TAE = ["kitti|llvar|tae", "bonn|llvar|tae", "scannet|llvar|tae", "sintel|llvar|tae", "nuscenes_night|real|tae"]
def f(x):
    try: return float(x)
    except (TypeError, ValueError): return None
def gm(row, cols, drop_kitti=False):
    v = []
    for c in cols:
        if drop_kitti and c.startswith("kitti"): continue
        a, b = f(R[row].get(c)), f(R[REF].get(c))
        if a and b: v.append(a / b)
    return (math.exp(sum(map(math.log, v)) / len(v)), len(v), v) if v else (None, 0, [])
def table(drop_kitti):
    T = {}
    for r in ROWS:
        fl, nf, _ = gm(r, FL, drop_kitti); ac, na, accv = gm(r, AC, drop_kitti)
        nt, _, _ = gm(r, [NIGHT]); ta, nt_, _ = gm(r, TAE, drop_kitti)
        cells = [(c, f(R[r].get(c)) / f(R[REF].get(c))) for c in AC if not (drop_kitti and c.startswith("kitti")) and f(R[r].get(c)) and f(R[REF].get(c))]
        T[r] = dict(FLICKER=fl, nF=nf, ACCURACY=ac, nA=na, NIGHT=nt, TAE=ta, nT=nt_, worst=max(cells, key=lambda x: x[1]) if cells else None, n_gt101=sum(1 for _, v in cells if v > 1.01))
    return T
def dom(T, ycol):
    res = {}
    for a in ROWS:
        res[a] = [b for b in ROWS if b != a and T[b]["FLICKER"] <= T[a]["FLICKER"] and T[b][ycol] <= T[a][ycol] and (T[b]["FLICKER"] < T[a]["FLICKER"] or T[b][ycol] < T[a][ycol])]
    return res
lines = []
for tag, dk in (("WITH KITTI", False), ("WITHOUT KITTI", True)):
    T = table(dk); lines.append(f"## {tag}")
    lines.append("row | FLICKER (n) | ACCURACY (n) | NIGHT | TAE (n) | worst accuracy cell | #acc cells >1.01")
    for r in ROWS:
        t = T[r]; w = t["worst"]
        lines.append(f"{r} | {t['FLICKER']:.3f} ({t['nF']}) | {t['ACCURACY']:.3f} ({t['nA']}) | {t['NIGHT']:.3f} | {t['TAE']:.3f} ({t['nT']}) | {w[0]}={w[1]:.3f} | {t['n_gt101']}")
    for ycol in ("ACCURACY", "NIGHT"):
        d = dom(T, ycol); lines.append(f"F2 Pareto (FLICKER,{ycol}): " + "; ".join(f"{r}: {'DOMINATED by ' + ','.join(x) if x else 'ON FRONTIER'}" for r, x in d.items()))
    neutral = [r for r in ROWS if T[r]["ACCURACY"] <= 1.005 and T[r]["NIGHT"] <= 1.005]
    best = min(neutral, key=lambda r: T[r]["FLICKER"]) if neutral else None
    lines.append(f"F3 accuracy-neutral (ACCURACY<=1.005 and NIGHT<=1.005): {neutral}; lowest FLICKER among them: {best}\n")
    if not dk:
        with open(OUT / "frontier.csv", "w") as fh:
            w = csv.writer(fh); w.writerow(["row", "FLICKER", "nF", "ACCURACY", "nA", "NIGHT", "TAE", "nT", "n_acc_cells_gt1.01"])
            for r in ROWS: t = T[r]; w.writerow([r, t["FLICKER"], t["nF"], t["ACCURACY"], t["nA"], t["NIGHT"], t["TAE"], t["nT"], t["n_gt101"]])
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        for a, y in zip(ax, ("ACCURACY", "NIGHT")):
            for r in ROWS:
                a.scatter(T[r]["FLICKER"], T[r][y], s=60, marker="*" if r.endswith("TAG") else "o", color="C3" if r.endswith("TAG") else "C0")
                a.annotate(r.replace("ft+", ""), (T[r]["FLICKER"], T[r][y]), textcoords="offset points", xytext=(5, 4), fontsize=9)
            a.axhline(1.0, color="grey", lw=.8, ls="--"); a.axhline(1.005, color="grey", lw=.5, ls=":")
            a.set_xlabel("flicker index (ratio vs Metric-VDA-S FT n46, lower better)"); a.set_ylabel(f"{y.lower()} ratio (lower better)")
        ax[0].set_title("Accuracy index vs flicker index"); ax[1].set_title("Night aligned AbsRel vs flicker index")
        fig.suptitle("Post-hoc filters on Metric-VDA-S FT n46: one setting each, a point comparison (not a matched frontier)", fontsize=10)
        fig.tight_layout(); fig.savefig(OUT / "frontier.png", dpi=150)
(OUT / "frontier_report.txt").write_text("\n".join(lines)); print("\n".join(lines))
