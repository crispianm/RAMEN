"""FT version of sr_view.py's cell table for the supp_filters point comparison (open item 6, package_s3). python -I, stdlib only.
Usage: python -I ft_filter_cells.py <F_FILTERS model dir> <out dir>
Reads ONLY the 7 released variants of FT n46 from the S3 F_FILTERS job (one named file per variant x cell; any missing / duplicate / non-finite value aborts).
Writes ft_cells.csv (same column ids as WP31 sr_cells.csv: '<ds>|<cell>|<metric>', rows = variants) and ft_cells_provenance.csv (sha256 of every file used)."""
import csv, hashlib, json, math, sys
from pathlib import Path
D = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
PFX = "metric_vda_s_ft:_projects_b5dh_RAMEN_train_dn01_c3rep_20261009_runs_dn01_c3rep_20261009_ft_n46_step12000"
VARIANTS = [("ft", ""), ("ft+TAG", "+agf0.2t0.06"), ("ft+kal0.05", "+kal0.05"), ("ft+oneeuro", "+oneeuro1.0b1.0"), ("ft+ema0.7", "+ema0.7"), ("ft+flowema0.7", "+flowema0.7"), ("ft+tmed5", "+tmed5")]
# (column id, file dir, dataset in the file name, cell in the file name, field path in the json). Column ids / fields are the PROTOCOL_TABLE counted cells used by frontier.py
# (dn01_v3_table1.v3_cells('scannetpp','llvar','st_llvar')): scannet -> scannetpp, sintel -> sintel_rgbfix, aligned AbsRel / TAE / flicker = video_macro.*, F_noise = macro.f_noise.
CELLS = []
for d, fd in (("kitti", "kitti"), ("bonn", "bonn"), ("scannet", "scannetpp"), ("sintel", "sintel_rgbfix"), ("nyuv2", "nyuv2")):
    CELLS.append((f"{d}|llvar|abs_rel_aligned", "bench", fd, "llvar", "video_macro.abs_rel_aligned"))
for d, fd in (("kitti", "kitti"), ("bonn", "bonn"), ("scannet", "scannetpp"), ("sintel", "sintel_rgbfix")):
    CELLS.append((f"{d}|llvar|tae", "bench", fd, "llvar", "video_macro.tae"))
for d, fd in (("kitti", "kitti"), ("bonn", "bonn"), ("nyuv2", "nyuv2"), ("scannet", "scannetpp")):
    CELLS.append((f"{d}|st_llvar|flicker", "bench", fd, "st_llvar", "video_macro.flicker"))
CELLS += [("nuscenes_night|real|abs_rel_aligned", "heldout", "nuscenes_night_test", "real", "video_macro.abs_rel_aligned"),
          ("nuscenes_night|real|tae", "heldout", "nuscenes_night_test", "real", "video_macro.tae"),
          ("bvi_rlv|fnoise|f_noise", "fnoise", "bvi_rlv_test", "fnoise", "macro.f_noise")]
def fname(var, sub, ds, cell):
    return f"{PFX}{var}__{ds}__{cell}__test.json" if sub == "heldout" else (f"{PFX}{var}__{ds}__fnoise.json" if sub == "fnoise" else f"{PFX}{var}__{ds}__{cell}__all__v3.json")
def get(obj, path):
    for k in path.split("."):
        if not isinstance(obj, dict) or k not in obj: raise SystemExit(f"STOP: field {path} missing")
        obj = obj[k]
    if isinstance(obj, bool) or not isinstance(obj, (int, float)) or not math.isfinite(obj): raise SystemExit(f"STOP: field {path} not a finite number")
    return float(obj)
rows, prov = {}, []
for name, var in VARIANTS:
    rows[name] = {}
    for cid, sub, ds, cell, field in CELLS:
        f = D / sub / fname(var, sub, ds, cell)
        if not f.is_file(): raise SystemExit(f"STOP: missing {f.name}")
        b = f.read_bytes(); rows[name][cid] = get(json.loads(b), field); prov.append((name, cid, sub + "/" + f.name, hashlib.sha256(b).hexdigest()))
with open(OUT / "ft_cells.csv", "w", newline="") as fh:
    w = csv.writer(fh, lineterminator="\n"); w.writerow(["row"] + [c[0] for c in CELLS])
    for name, _ in VARIANTS: w.writerow([name] + [f"{rows[name][c[0]]:.5g}" for c in CELLS])        # same %.5g formatting as sr_view.py
with open(OUT / "ft_cells_provenance.csv", "w", newline="") as fh:
    w = csv.writer(fh, lineterminator="\n"); w.writerow(["row", "column", "file", "sha256"]); w.writerows(prov)
print(f"rows {len(VARIANTS)} columns {len(CELLS)} files {len(prov)}")
