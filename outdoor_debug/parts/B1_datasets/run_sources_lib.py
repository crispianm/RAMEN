"""Decode real train frames per source with the REAL dataset classes (degrade off); raw-decoded stats + processed stats + panels."""
import sys; sys.path.insert(0, "/home/wg19671/.claude/jobs/a0d391cb/tmp/od/outdoor_debug/parts/B1_datasets")
from common import *
import torch, json, time
from PIL import Image
from data.vda_semantics_dataset import (VDASemanticsTartanAirDataset, VDASemanticsVKITTIDataset, VDASemanticsPointOdysseyDataset, VDASemanticsIRSDataset)
from data.video_clip_dataset import _decode_depth
from data.vkitti_dataset import _decode_vkitti_depth
from data.pointodyssey_dataset import _PO_DEPTH_SCALE
from data.kitti_raw_train import KITTIRawTrainDataset, read_kitti_gt
from data.spring_dataset import SpringDataset
import yaml
random.seed(0); np.random.seed(0)
common = dict(num_frames=32, image_size=(518, 518), split="train", clip_stride=32, frame_stride=1, seed=45, degrade=False)

def make(name):
    if name == "tartanair": return VDASemanticsTartanAirDataset(root=f"{T}/tartanair", val_fraction=0.1, max_depth=10000.0, **common)
    if name == "vkitti": 
        ds = VDASemanticsVKITTIDataset(root=f"{T}/vkitti", variants=["clone","fog","morning","overcast","rain","sunset","15-deg-left","15-deg-right","30-deg-left","30-deg-right"], cameras=["Camera_0","Camera_1"], no_holdout=True, max_depth=80.0, **common)
        ds.vkitti_max_depth = 655.0; return ds
    if name == "pointodyssey": return VDASemanticsPointOdysseyDataset(root=f"{T}/point_odyssey/train", keep_list=REPO+"/provenance/dn01_vdamix/po_filter_v1.json", keep_sha256="ff4851c6ad1d8bb0754ef9ef30aeb1c83562624041074acd41e41b5e9eedb34c", max_depth=999.0, **common)
    if name == "kitti_raw": return KITTIRawTrainDataset(root=f"{T}/kitti_raw", gt_root=f"{T}/kitti_raw_gt", allow_missing=False, max_depth=80.0, **common)
    if name == "irs": return VDASemanticsIRSDataset(root=f"{T}/irs", val_fraction=0.0, max_depth=80.0, **common)
    if name == "spring": return SpringDataset(root=f"{T}/spring", val_fraction=0.03, max_depth=80.0, **common)

def raw_depth(name, ds, clip, i):
    files = ds.clips[clip]
    dep = files[1][i]
    if name == "tartanair": return _decode_depth(str(dep))
    if name == "vkitti": return _decode_vkitti_depth(str(dep))
    if name == "pointodyssey": return cv2.imread(str(dep), cv2.IMREAD_UNCHANGED).astype(np.float32) * _PO_DEPTH_SCALE
    if name == "kitti_raw": return read_kitti_gt(dep)
    if name == "irs": 
        from data.irs_dataset import _read_exr_depth; return _read_exr_depth(dep)
    if name == "spring": return ds.read_depth(dep)

