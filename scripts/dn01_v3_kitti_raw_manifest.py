"""Protocol v3 WP13: write provenance/dn01_v3/kitti_raw_train_drives.json from the listing of data_depth_annotated.zip (train/ = the 138 depth-annotated train drives, val/ = the 13 VDA val drives).
Run once: ``python scripts/dn01_v3_kitti_raw_manifest.py [zip]``; the loader (data/kitti_raw_train.py) asserts the result (138 drives, val pinned, disjoint)."""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

ZIP = "/mnt/DATA/DegradedDepth/data/TEST/kitti_depth_completion_gt/data_depth_annotated.zip"
OUT = Path(__file__).resolve().parents[1] / "provenance" / "dn01_v3" / "kitti_raw_train_drives.json"


def main(zpath: str = ZIP) -> None:
    names = zipfile.ZipFile(zpath).namelist()
    split = {s: sorted({n.split("/")[1] for n in names if n.startswith(s + "/") and n.count("/") >= 2}) for s in ("train", "val")}
    rec = {"source": "data_depth_annotated.zip listing", "zip_size": Path(zpath).stat().st_size, "n_train_drives": len(split["train"]), "train_drives": split["train"], "vda_val_drives": split["val"],
           "train_gt_image_02_files": sum(1 for n in names if n.startswith("train/") and "/image_02/" in n and n.endswith(".png")), "val_gt_image_02_files": sum(1 for n in names if n.startswith("val/") and "/image_02/" in n and n.endswith(".png"))}
    assert not set(rec["train_drives"]) & set(rec["vda_val_drives"]) and rec["n_train_drives"] == 138 and len(rec["vda_val_drives"]) == 13
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1) + "\n")
    print(OUT, hashlib.sha256(OUT.read_bytes()).hexdigest()[:8], rec["n_train_drives"], rec["train_gt_image_02_files"])


if __name__ == "__main__":
    main(*sys.argv[1:])
