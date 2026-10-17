"""Check 1b: dataset CLASS instantiated per source for each arm, through the REAL train._build_multisource (-> train._make_source -> _make_source_inner)
with a recorder wrapped around train._make_source. Roots remapped /projects/b5dh/data/train -> /mnt/DATA/DegradedDepth/data/TRAIN (local); the
ts2_seg_root -> local TA_seg. Sources whose local tree fails to build are reported (class then derived from a per-source try with the SAME real
builder, see per-source fallback). Also records the post-construction flag state of each instance."""
import sys, copy, json, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).parent))
import yaml, train
from arms import build
LOCAL = "/mnt/DATA/DegradedDepth/data/TRAIN"
SEGL = "/mnt/DATA/DegradedDepth/claude_outputs/scratch/outdoor_debug_20261016/TA_seg"

def remap(cfg):
    c = copy.deepcopy(cfg)
    for s in c["data"]["sources"] + c["data"].get("val_sources", []):
        if "root" in s: s["root"] = s["root"].replace("/projects/b5dh/data/train", LOCAL)
        if "gt_root" in s: s["gt_root"] = s["gt_root"].replace("/projects/b5dh/data/train", LOCAL)
        if "extra_root" in s: s["extra_root"] = s["extra_root"].replace("/projects/b5dh/data/train", LOCAL)
        if "ts2_seg_root" in s: s["ts2_seg_root"] = SEGL
    return c

def state(ds):
    return {k: ((type(getattr(ds, k, None)).__name__) if k == "ts2" else getattr(ds, k, None)) for k in
            ("ts2", "vkitti_max_depth", "sky_as_valid_depth", "silog_max_depth", "metric_supervision", "sparse_gt", "max_depth")}

if __name__ == "__main__":
    base, arms = build()
    real = train._make_source
    res = {}
    for name, cfg in arms.items():
        cfg = remap(cfg); rec = []
        def wrapped(spec, split, common, _rec=rec):
            t0 = time.time()
            try:
                ds = real(spec, split, common)
            except Exception as e:
                _rec.append(dict(split=split, type=spec["type"], root=spec.get("root"), error=f"{type(e).__name__}: {str(e)[:200]}")); raise
            _rec.append(dict(split=split, type=spec["type"], root=spec.get("root"), cls=type(ds).__name__, mro=[c.__name__ for c in type(ds).__mro__[:4]], n=len(ds), state=state(ds), sec=round(time.time()-t0,1)))
            return ds
        train._make_source = wrapped
        try:
            train._build_multisource(cfg, "train", 1, 0); err = None
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:300]}"
        res[name] = dict(records=rec, build_error=err)
        print(name, "build_error:", err)
        for r in rec: print("  ", json.dumps(r))
    json.dump(res, open(Path(__file__).parent / "classes.json", "w"), indent=1)
