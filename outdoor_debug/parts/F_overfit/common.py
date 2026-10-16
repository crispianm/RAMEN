"""Shared helpers for F_overfit: load the effective FT n46 config with local paths, build real dataset sources, collate like the trainer."""
import sys, json, pathlib, copy
import numpy as np, torch
REPO = pathlib.Path('/home/wg19671/.claude/jobs/a0d391cb/tmp/od')
sys.path.insert(0, str(REPO))
import train
T = pathlib.Path('/mnt/DATA/DegradedDepth/data/TRAIN')
SCR = pathlib.Path('/mnt/DATA/DegradedDepth/claude_outputs/scratch/outdoor_debug_20261016/F_overfit')
SMALL = REPO / 'outdoor_debug/parts/F_overfit'
VDA_REPO = '/mnt/DATA/DegradedDepth/competition/Video-Depth-Anything'
CK = '/mnt/DATA/DegradedDepth/RAMEN/checkpoints/'

def load_cfg():
    cfg = train.load_config(REPO / 'configs/experiments/dn01_c3rep_20261009_ft_n46.yaml')
    m = cfg['model']
    m['vda_repo'] = VDA_REPO
    m['vda_checkpoint_path'] = CK + 'metric_video_depth_anything_vits.pth'
    m['da_v2_checkpoint_path'] = CK + 'depth_anything_v2_vits.pth'
    m['device'] = 'cuda'
    return cfg

COMMON = dict(num_frames=32, image_size=(518, 518), clip_stride=32, frame_stride=1, max_depth=80.0, seed=45,
              degrade=True, degrade_mode=['awgn_blur'], degrade_severity=0.0, max_clips_per_video=None, resize_mode='vda')

def make_source(kind):
    if kind == 'vkitti':
        spec = dict(type='vkitti', root=str(T / 'vkitti'), variants=['clone'], cameras=['Camera_0'], no_holdout=True, vkitti_max_depth=655,
                    metric_supervision=True, sparse_gt=False, resize_mode='vda')
    else:
        spec = dict(type='tartanair', root=str(T / 'tartanair'), metric_supervision=True, sparse_gt=False, silog_max_depth=655.0, max_depth=10000.0, resize_mode='vda')
    return train._make_source(spec, 'train', COMMON)

def collate(items):
    out = {}
    for k in items[0]:
        v = [it[k] for it in items]
        out[k] = torch.stack(v) if torch.is_tensor(v[0]) else v
    return out
