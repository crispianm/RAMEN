"""DIST-01 arm (c): the unlabeled-real-video source and the no-GT loss path.

Covers (a) the dataset's key/shape contract with zero depth/mask under both loaders,
(b) compute_loss / the loss_mode=vda path returning a finite zero on an all-zero mask,
(c) load_config refusing video_only in val_sources, and (d) mixing a synthetic source and
a video_only source through build_dataloader's uniform sampler.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

import train
from data.video_only_dataset import VideoOnlyDataset
from data.vda_semantics_dataset import (
    VDASemanticsDegradedVideoOnlyDataset,
    VDASemanticsVideoOnlyDataset,
)
from train import _make_source, compute_loss, load_config

SIZE = 70          # small square target: 518 crops are far too slow for a CPU unit test
COMMON = dict(
    num_frames=2, image_size=(SIZE, SIZE), clip_stride=1, frame_stride=1, max_depth=80.0,
    seed=42, degrade=False, degrade_mode="full", degrade_severity=None,
    max_clips_per_video=None, resize_mode="vda",
)


def _video_root(tmp_path, names=("clip_a",), num_images=3, w=96, h=64):
    for name in names:
        d = tmp_path / name
        d.mkdir(parents=True)
        for i in range(num_images):
            arr = np.zeros((h, w, 3), np.uint8)
            arr[..., 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
            arr[..., 1] = (i * 37) % 256
            Image.fromarray(arr).save(d / f"{i:06d}.png")
    return tmp_path


def _tartanair_root(tmp_path, trajs=("P000",), num_images=3, w=96, h=64):
    import cv2
    for traj in trajs:
        img_dir = tmp_path / "Env" / "Data_easy" / traj / "image_lcam_front"
        dep_dir = tmp_path / "Env" / "Data_easy" / traj / "depth_lcam_front"
        img_dir.mkdir(parents=True)
        dep_dir.mkdir(parents=True)
        for i in range(num_images):
            Image.new("RGB", (w, h), color=(i * 10 % 256, 0, 0)).save(img_dir / f"{i:06d}_lcam_front.png")
            depth = np.full((h, w), 5.0, dtype=np.float32)
            cv2.imwrite(str(dep_dir / f"{i:06d}_lcam_front.png"),
                        depth.reshape(h, w, 1).view(np.uint8).reshape(h, w, 4))
    return tmp_path


def _assert_contract(item, num_frames=2, size=SIZE, degraded=False):
    assert item["frames"].shape == (3, num_frames, size, size)
    for k in ("depth", "mask", "sky"):
        assert item[k].shape == (1, num_frames, size, size)
        assert float(item[k].abs().sum()) == 0.0, f"{k} must be all zeros (no GT)"
    expected = {"frames", "depth", "mask", "sky", "video_id", "start_frame"}
    if degraded:
        expected |= {"degraded_frames", "degrade_params"}
    assert set(item) == expected


# --------------------------------------------------------------------------- (a) dataset

def test_vda_video_only_yields_zero_depth_and_mask(tmp_path):
    root = _video_root(tmp_path)
    ds = VDASemanticsVideoOnlyDataset(rgb_root=str(root), split="val", val_fraction=1.0,
                                      **{k: v for k, v in COMMON.items() if k != "resize_mode"})
    assert len(ds) > 0
    _assert_contract(ds[0])


def test_legacy_video_only_yields_zero_depth_and_mask(tmp_path):
    root = _video_root(tmp_path)
    ds = VideoOnlyDataset(rgb_root=str(root), split="val", val_fraction=1.0,
                          **{k: v for k, v in COMMON.items() if k != "resize_mode"})
    _assert_contract(ds[0])


def test_video_only_degraded_frames_differ_from_clean(tmp_path):
    root = _video_root(tmp_path)
    src = _make_source({"type": "video_only", "rgb_root": str(root), "val_fraction": 1.0},
                       "val", {**COMMON, "degrade": True})
    assert isinstance(src, VDASemanticsDegradedVideoOnlyDataset)
    item = src[0]
    _assert_contract(item, degraded=True)
    assert item["degraded_frames"].shape == item["frames"].shape
    assert not torch.equal(item["degraded_frames"], item["frames"])


def test_video_only_reads_no_depth_files(tmp_path):
    """The source must not require (or open) a parallel depth tree -- that is the whole
    point of the type; pseudo_gt refuses to enumerate a sequence without one."""
    root = _video_root(tmp_path)
    ds = VideoOnlyDataset(rgb_root=str(root), split="val", val_fraction=1.0,
                          **{k: v for k, v in COMMON.items() if k != "resize_mode"})
    assert all(clip[1] is None for clip in ds.clips)


def test_video_only_respects_num_frames_and_frame_stride(tmp_path):
    root = _video_root(tmp_path, num_images=10)
    common = {**COMMON, "num_frames": 3, "frame_stride": 2, "clip_stride": 4}
    ds = VideoOnlyDataset(rgb_root=str(root), split="val", val_fraction=1.0,
                          **{k: v for k, v in common.items() if k != "resize_mode"})
    assert [c[3] for c in ds.clips] == [0, 4]          # span 5, stride 4, 10 frames
    assert ds[0]["frames"].shape[1] == 3


def test_video_only_respects_max_clips_per_video(tmp_path):
    root = _video_root(tmp_path, num_images=10)
    common = {**COMMON, "max_clips_per_video": 3}
    ds = VideoOnlyDataset(rgb_root=str(root), split="val", val_fraction=1.0,
                          **{k: v for k, v in common.items() if k != "resize_mode"})
    assert len(ds) == 3


# ----------------------------------------------------------------- (b) no-GT loss safety

def _empty_gt_batch(B=1, T=2, H=8, W=8):
    z = torch.zeros(B, 1, T, H, W)
    return {"depth": z, "mask": z.clone(), "sky": z.clone()}


def test_compute_loss_is_finite_zero_on_an_all_zero_mask():
    pred = torch.rand(1, 1, 2, 8, 8, requires_grad=True) + 0.5
    loss = compute_loss(pred, _empty_gt_batch(), {"loss": {}}, pred_is_disparity=True)
    assert torch.isfinite(loss) and float(loss.detach()) == 0.0
    assert loss.requires_grad, "must keep a grad_fn: the trainer backwards through it"
    loss.backward()   # must not raise -- the batch is NOT skipped


def test_compute_loss_vda_mode_is_finite_zero_on_an_all_zero_mask(monkeypatch):
    """The vendored VDA loss is not present on this workstation, so the guard has to sit at
    OUR call site: with an empty mask the VDA loss must never be called at all."""
    def _boom(*a, **k):
        raise AssertionError("VDA's VideoDepthLoss must not be reached for a no-GT clip")

    monkeypatch.setattr(train, "_get_vda_video_depth_loss", _boom)
    pred = torch.rand(1, 1, 2, 8, 8, requires_grad=True) + 0.5
    loss = compute_loss(pred, _empty_gt_batch(), {"loss": {"loss_mode": "vda"}},
                        pred_is_disparity=True)
    assert torch.isfinite(loss) and float(loss.detach()) == 0.0
    loss.backward()


def test_compute_loss_vda_mode_still_calls_the_vda_loss_when_gt_exists(monkeypatch):
    called = {}

    def _fake(repo, lc):
        def _loss(prediction, target, mask):
            called["yes"] = True
            return {"total_loss": prediction.new_tensor(1.0)}
        return _loss

    monkeypatch.setattr(train, "_get_vda_video_depth_loss", _fake)
    batch = _empty_gt_batch()
    batch["depth"] = torch.full_like(batch["depth"], 5.0)
    batch["mask"] = torch.ones_like(batch["mask"])
    loss = compute_loss(torch.rand(1, 1, 2, 8, 8) + 0.5, batch, {"loss": {"loss_mode": "vda"}},
                        pred_is_disparity=True)
    assert called.get("yes") and float(loss) == 1.0


# --------------------------------------------------------------- (c) val_sources rejection

def _write_cfg(tmp_path, cfg, name="c.yaml"):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_load_config_rejects_video_only_in_val_sources(tmp_path):
    cfg = {"data": {"sources": [{"type": "video_only", "rgb_root": "/x"}],
                    "val_sources": [{"type": "tartanair", "root": "/y"},
                                    {"type": "video_only", "rgb_root": "/x"}]}}
    with pytest.raises(ValueError, match="video_only"):
        load_config(_write_cfg(tmp_path, cfg))


def test_load_config_accepts_video_only_in_train_sources(tmp_path):
    cfg = {"data": {"sources": [{"type": "video_only", "rgb_root": "/x"}],
                    "val_sources": [{"type": "tartanair", "root": "/y"}]}}
    assert load_config(_write_cfg(tmp_path, cfg))["data"]["sources"][0]["type"] == "video_only"


def test_load_config_rejects_video_only_inherited_into_val_sources(tmp_path):
    parent = _write_cfg(tmp_path, {"data": {"val_sources": [{"type": "video_only", "rgb_root": "/x"}]}},
                        "parent.yaml")
    child = _write_cfg(tmp_path, {"extends": parent.name, "data": {"sources": []}}, "child.yaml")
    with pytest.raises(ValueError, match="video_only"):
        load_config(child)


# ------------------------------------------------- (d) mixing GT and video_only sources

def test_mixed_synthetic_and_video_only_sources_share_one_uniform_sampler(tmp_path):
    # two sequences per source: the deterministic split always holds one out for val
    ta_root = _tartanair_root(tmp_path / "ta", trajs=("P000", "P001"))
    vo_root = _video_root(tmp_path / "vo", names=("clip_a", "clip_b"))
    cfg = {
        "data": {
            "sources": [
                {"type": "tartanair", "root": str(ta_root), "val_fraction": 0.0},
                {"type": "video_only", "rgb_root": str(vo_root), "val_fraction": 0.0},
            ],
            "val_sources": [{"type": "tartanair", "root": str(ta_root), "val_fraction": 1.0}],
            "num_frames": 2, "image_size": [SIZE, SIZE], "clip_stride": 1, "frame_stride": 1,
            "resize_mode": "vda", "degrade": True, "val_clips_per_source": 1, "num_workers": 0,
        },
        "training": {"batch_size": 2},
    }
    loader, _ = train.build_dataloader(cfg, "train")
    assert loader.gt_clips > 0 and loader.video_only_clips > 0
    # uniform sampler: each SOURCE carries equal total mass (1/len per clip within a source)
    w = loader.sampler.weights
    assert pytest.approx(float(w.sum()), rel=1e-6) == 2.0

    batch = next(iter(loader))
    for key in ("frames", "degraded_frames", "depth", "mask", "sky", "degrade_params"):
        assert key in batch, f"mixed batch lost {key} -- the two sources' key sets must match"
    assert batch["frames"].shape == (2, 3, 2, SIZE, SIZE)
    # the GT source still supervises normally; the loss is finite either way
    loss = compute_loss(torch.rand(2, 1, 2, SIZE, SIZE) + 0.5, batch, {"loss": {}},
                        pred_is_disparity=True)
    assert torch.isfinite(loss)
