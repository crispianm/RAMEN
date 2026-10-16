"""DN-01 Phase D: metric VDA-S -- config carve-out, the scale-pinning SiLog term, and the
eval wrapper reproducing the trained metric model's forward() (no depth<->disparity flip)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

import train
from train import _assert_vda_uses_vda_loss_mode, compute_loss, load_config

ROOT = Path(__file__).resolve().parents[1]
VDA = Path("/mnt/DATA/DegradedDepth/competition/Video-Depth-Anything")


def test_carve_out_is_metric_only():
    custom = {"loss_mode": "custom", "sky_zero_disparity": False}
    _assert_vda_uses_vda_loss_mode({"model": {"arch": "vda", "metric": True}, "loss": custom})
    for model in ({"arch": "vda"}, {"arch": "vda", "metric": False}):   # relative still hard-fails
        with pytest.raises(ValueError, match="loss_mode"):
            _assert_vda_uses_vda_loss_mode({"model": model, "loss": custom})
    with pytest.raises(ValueError, match="sky_zero_disparity"):   # the sky rule is not carved out
        _assert_vda_uses_vda_loss_mode({"model": {"arch": "vda", "metric": True},
                                        "loss": {"loss_mode": "custom", "sky_zero_disparity": True}})


class _ZeroVDALoss(torch.nn.Module):
    def __init__(self, **_):
        super().__init__()

    def forward(self, prediction, target, mask):
        return {"total_loss": prediction.sum() * 0.0}


@pytest.fixture
def zero_vda_loss(monkeypatch):
    monkeypatch.setattr(train, "_get_vda_video_depth_loss", lambda repo, lc: _ZeroVDALoss())


def _metric_cfg(w=1.0):
    return {"model": {"arch": "vda", "metric": True},
            "loss": {"loss_mode": "vda", "sky_zero_disparity": False, "metric_silog_weight": w}}


def test_metric_silog_pins_absolute_scale_on_native_depth(zero_vda_loss):
    depth = torch.rand(1, 1, 2, 8, 8) * 20 + 1.0
    batch = {"depth": depth, "mask": torch.ones_like(depth)}
    assert compute_loss(depth.clone(), batch, _metric_cfg()).item() == pytest.approx(0.0, abs=1e-5)
    # 2x too far everywhere: DA-V2's SiLog(lambd=0.5) = sqrt(0.5) * ln 2, i.e. NOT scale-invariant
    got = compute_loss(2 * depth, batch, _metric_cfg()).item()
    assert got == pytest.approx(math.sqrt(0.5) * math.log(2), rel=1e-4)
    # disparity-native output must never reach a depth-space log loss
    with pytest.raises(ValueError, match="depth-native"):
        compute_loss(depth, batch, _metric_cfg(), pred_is_disparity=True)


def test_metric_silog_off_by_default(zero_vda_loss):
    depth = torch.rand(1, 1, 2, 8, 8) + 1.0
    batch = {"depth": depth, "mask": torch.ones_like(depth)}
    assert compute_loss(2 * depth, batch, _metric_cfg(0.0)).item() == 0.0


def test_dn01_d0_config_resolves_to_the_metric_recipe():
    c = load_config(ROOT / "configs/experiments/dn01_d0_metric_clean.yaml")
    assert c["model"]["metric"] is True and c["loss"]["loss_mode"] == "vda"
    assert c["loss"]["metric_silog_weight"] > 0
    assert [s["type"] for s in c["data"]["sources"]] == ["vkitti", "irs"]
    assert all(s.get("vkitti_max_depth") == 80 for s in c["data"]["sources"] + c["data"]["val_sources"]
               if s["type"] == "vkitti")
    assert {19500, 20000} <= set(c["training"]["checkpoint_at_steps"])
    assert c["training"]["max_steps"] == 20000 and c["training"]["learning_rate"] == 1e-4


@pytest.mark.skipif(not VDA.is_dir(), reason="vendored Video-Depth-Anything checkout not present")
def test_eval_wrapper_equals_forward_for_metric_model(tmp_path):
    """scripts/val01_eval_vda.py rebuilds a train.py checkpoint and runs VideoDepthAnything
    on ImageNet-normalised frames; that must equal VDANet.forward() on [0,1] frames for the
    METRIC model -- same depth, no inversion (a wrapper once inverted metric depth)."""
    from models.vda_net import VDANet
    sys.path.insert(0, str(ROOT / "scripts"))
    import val01_eval_vda as ev

    torch.manual_seed(0)
    net = VDANet(repo=str(VDA), encoder="vits", metric=True, init="random").eval()
    with torch.no_grad():
        for p in net.core.head.parameters():   # a "trained" head, motion modules included
            p.add_(0.02 * torch.randn_like(p))
    dav2 = {k: v for k, v in net.core.state_dict().items() if k.startswith("pretrained.")}
    dav2.update({"depth_head." + k: v for k, v in net.core.head.state_dict().items()
                 if "motion_modules" not in k})
    torch.save(dav2, tmp_path / "dav2.pth")
    torch.save({"trainable_ema": {"head." + k: v for k, v in net.core.head.state_dict().items()}},
               tmp_path / "step.pt")
    core = ev.load_model_for_inference(str(VDA), str(tmp_path / "step.pt"), "vits", True,
                                       ramen_checkpoint=True,
                                       da_v2_checkpoint_path=str(tmp_path / "dav2.pth")).eval()
    assert core.metric is True and net.predicts_disparity is False
    rgb = torch.rand(1, 3, 2, 70, 70)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
    with torch.no_grad():
        ours = net(rgb)[:, 0]
        theirs = core((rgb.permute(0, 2, 1, 3, 4) - mean) / std)
    assert ours.abs().max() > 0
    torch.testing.assert_close(theirs, ours)


def _vkitti_root(tmp_path):
    import cv2
    import numpy as np
    rgb = tmp_path / "Scene01" / "clone" / "frames" / "rgb" / "Camera_0"
    dep = tmp_path / "Scene01" / "clone" / "frames" / "depth" / "Camera_0"
    rgb.mkdir(parents=True)
    dep.mkdir(parents=True)
    d = np.full((540, 960), 5000, np.uint16)   # 50 m (cm units)
    d[:, 480:] = 10000                          # 100 m: real geometry past 80 m
    d[:100] = 65535                             # sky sentinel
    for i in range(3):
        cv2.imwrite(str(rgb / f"rgb_{i:05d}.jpg"), np.full((540, 960, 3), 128, np.uint8))
        cv2.imwrite(str(dep / f"depth_{i:05d}.png"), d)
    return str(tmp_path)


def test_vkitti_max_depth_default_unchanged_and_cap_masks_far(tmp_path):
    from data.vda_semantics_dataset import VDASemanticsVKITTIDataset
    root = _vkitti_root(tmp_path)
    common = dict(num_frames=2, image_size=(518, 518), clip_stride=1, frame_stride=1, max_depth=80.0,
                  seed=42, degrade=False, degrade_mode="full", degrade_severity=None,
                  max_clips_per_video=None, resize_mode="vda")
    spec = {"type": "vkitti", "root": root, "variants": ["clone"], "val_fraction": 1.0}
    base = train._make_source(spec, "val", common)
    capped = train._make_source({**spec, "vkitti_max_depth": 80}, "val", common)
    assert base.vkitti_max_depth is None and capped.vkitti_max_depth == 80.0
    a, b = base[0], capped[0]
    for k in ("frames", "depth", "sky"):   # only the mask may change
        assert torch.equal(a[k], b[k])
    far = (a["depth"] > 99) & (a["depth"] < 101)          # the 100 m block interior, not sky
    near = (a["depth"] > 49) & (a["depth"] < 51)
    assert far.any() and near.any()
    assert (a["mask"][far] > 0).all()        # default: far geometry stays supervised (relative runs)
    assert (b["mask"][far] == 0).all()       # cap: d >= 80 m dropped
    assert torch.equal(a["mask"][near], b["mask"][near])   # below the cap: identical
    with pytest.raises(ValueError, match="resize_mode: vda"):
        train._make_source({**spec, "vkitti_max_depth": 80}, "val", {**common, "resize_mode": None})
