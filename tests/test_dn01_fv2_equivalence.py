"""DN-01 final-v2: LINEAGE of the reused CONTROL seed-42 arm (Auditor required item 1). CONTROL seed 42 = the v2 ST-U (Isambard job 6892107) was
trained at commit 2ce123e; the final-v2 seeds 43 / 44 are trained at the LAUNCH commit, and train.py / data/ / models/ differ between the two
(+351 / -18). The reuse is only honest if the launch tree's non-regime training path is EQUIVALENT to 2ce123e's: the WHOLE tree of 2ce123e (git
archive) and the working tree each run the REAL train.train() on the CONTROL recipe (the depth-only degraded fine-tune, no static regime) on a fixed
tiny CPU run (same seed, data, config) in a subprocess, and the trained weights must be IDENTICAL tensor by tensor. Skipped when the local weights /
data / git object are absent."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
CONTROL_COMMIT = "2ce123e"
R = "/mnt/DATA/DegradedDepth/RAMEN"
LOCAL_DATA = {"vkitti": "/mnt/DATA/DegradedDepth/data/TRAIN/vkitti", "irs": "/mnt/DATA/DegradedDepth/data/TRAIN/irs"}
LOCAL_MODEL = {"checkpoint_path": f"{R}/checkpoints/depth-anything-v3-small.safetensors",
               "vda_repo": "/mnt/DATA/DegradedDepth/competition/Video-Depth-Anything",
               "da_v2_checkpoint_path": f"{R}/checkpoints/depth_anything_v2_vits.pth",
               "vda_checkpoint_path": f"{R}/checkpoints/metric_video_depth_anything_vits.pth"}
INIT = (f"{R}/results/dn01_st_recal/recal_full/temp2_distilled.pt", "96d35f08ecd5c988387e502bbd0ad85875539cafd9e329448ad5c76804916c4b")

RUNNER = '''
import json, sys
tree, cfg_json = sys.argv[1], sys.argv[2]
sys.path.insert(0, tree)
import torch
torch.set_num_threads(2)
import train
train.train(json.load(open(cfg_json)))
'''


def _have():
    need = [*LOCAL_DATA.values(), *LOCAL_MODEL.values(), INIT[0]]
    git = subprocess.run(["git", "cat-file", "-e", f"{CONTROL_COMMIT}:train.py"], cwd=ROOT, capture_output=True).returncode == 0
    return git and all(Path(p).exists() for p in need)


needs_local = pytest.mark.skipif(not _have(), reason="needs the local VKITTI / IRS data, the weights, the recal init and the 2ce123e git object")


def _control_cfg(out):
    import train
    cfg = train.load_config(ROOT / "configs/experiments/dn01_v2_st_u_full.yaml")   # the CONTROL recipe: no static regime, no L_rg
    assert float(cfg["data"].get("static_prob", 0.0)) == 0.0 and float(cfg["loss"].get("rg_weight", 0.0)) == 0.0
    for s in cfg["data"]["sources"] + cfg["data"].get("val_sources", []):
        s["root"] = LOCAL_DATA[s["type"]]
    assert hashlib.sha256(Path(INIT[0]).read_bytes()).hexdigest() == INIT[1]
    cfg["model"].update({**{k: v for k, v in LOCAL_MODEL.items() if k in cfg["model"]}, "st_temp2_init": INIT[0], "device": "cpu"})
    cfg["data"].update(image_size=[126, 126], num_frames=8, clip_stride=8, num_workers=0, val_clips_per_source=2)
    cfg["training"].update(max_steps=2, grad_accum=2, checkpoint_at_steps=[2], checkpoint_dir=str(out), val_every_steps=100000, val_max_batches=2)
    return cfg


def _train_in(tree, cfg, tmp_path, name):
    (tmp_path / "runner.py").write_text(RUNNER)
    cfg_path = tmp_path / f"{name}.json"
    cfg_path.write_text(json.dumps(cfg))
    p = subprocess.run([sys.executable, str(tmp_path / "runner.py"), str(tree), str(cfg_path)], cwd=tree, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-1500:]
    return torch.load(Path(cfg["training"]["checkpoint_dir"]) / "step2.pt", map_location="cpu", weights_only=False)["trainable"]


@needs_local
def test_with_no_regime_the_launch_tree_trains_the_control_identically_to_2ce123e(tmp_path):
    old_tree = tmp_path / "tree_2ce123e"
    old_tree.mkdir()
    arch = subprocess.run(["git", "archive", CONTROL_COMMIT], cwd=ROOT, capture_output=True, check=True)
    subprocess.run(["tar", "-x", "-C", str(old_tree)], input=arch.stdout, check=True)
    assert (old_tree / "train.py").read_text() != (ROOT / "train.py").read_text()      # the trees really differ (the reason for this test)
    sd_old = _train_in(old_tree, _control_cfg(tmp_path / "out_old"), tmp_path, "old")
    sd_new = _train_in(ROOT, _control_cfg(tmp_path / "out_new"), tmp_path, "new")
    assert sd_old.keys() == sd_new.keys() and len(sd_old) > 100
    diff = [k for k in sd_old if not torch.equal(sd_old[k], sd_new[k])]
    assert not diff, f"{len(diff)} tensors differ between the whole tree @{CONTROL_COMMIT} and the working tree, e.g. {diff[:3]}"
    assert any(k.startswith("motion_modules") for k in sd_old)                      # the run trained the motion modules (not a vacuous equality)
