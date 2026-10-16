"""loss.loss_mode='vda' (VAL-01) reuses VDA's own loss/loss.py::VideoDepthLoss directly
instead of train.py's hand-rolled SSI/TGM. These tests use a fully scripted fake
loss.loss module (same pattern as tests/test_vda_net.py's vda_repo fixture, hermetic --
no dependency on the real vendored checkout) to verify OUR plumbing: raw disparity handed
straight through with no depth round-trip or pre-normalisation, sky pixels excluded (not
force-included), and the exact hyperparameters VDA's own composition actually uses
(trim=0.0, not TrimmedProcrustesLoss's own unused 0.2 default; temp_grad_scales left to
the class's own default) are what gets constructed."""

from __future__ import annotations

import sys
import types

import pytest
import torch
from torch import nn

from train import compute_loss, _assert_vda_uses_vda_loss_mode


class _RecordingVideoDepthLoss(nn.Module):
    """Records every call's arguments and returns a fixed loss so tests can assert on
    exactly what train.py handed it, not on VDA's own internal math."""
    last_call = None

    def __init__(self, alpha, scales, trim, stable_scale):
        super().__init__()
        self.init_args = dict(alpha=alpha, scales=scales, trim=trim, stable_scale=stable_scale)

    def forward(self, prediction, target, mask):
        _RecordingVideoDepthLoss.last_call = dict(
            prediction=prediction.detach().clone(),
            target=target.detach().clone(),
            mask=mask.detach().clone(),
        )
        return {"total_loss": prediction.sum() * 0.0 + 1.23, "spatial_loss": None, "stable_loss": None}


@pytest.fixture
def vda_loss_repo(monkeypatch):
    import train as train_module
    # Clear BOTH module-level caches -- _VDA_LOSS_MODULE_CACHE is keyed only by repo path
    # and independent of _VDA_VIDEO_DEPTH_LOSS_CACHE; any other test file reusing this
    # same literal repo path would otherwise reuse this cached loss.loss module object
    # (found the hard way via tests/test_loss_mode_vda_arch_agnostic.py).
    train_module._VDA_VIDEO_DEPTH_LOSS_CACHE.clear()
    train_module._VDA_LOSS_MODULE_CACHE.clear()
    _RecordingVideoDepthLoss.last_call = None
    pkg = types.ModuleType("loss")
    pkg.__path__ = []
    mod = types.ModuleType("loss.loss")
    mod.VideoDepthLoss = _RecordingVideoDepthLoss
    monkeypatch.setitem(sys.modules, "loss", pkg)
    monkeypatch.setitem(sys.modules, "loss.loss", mod)
    return "/fake/vda/repo"


def _cfg(vda_loss_repo, **loss_overrides):
    loss = {"loss_mode": "vda", "sky_zero_disparity": False, **loss_overrides}
    return {"model": {"arch": "vda", "vda_repo": vda_loss_repo}, "loss": loss}


def test_compute_loss_vda_mode_uses_the_composed_loss_defaults(vda_loss_repo):
    pred = torch.rand(1, 1, 2, 4, 4) * 0.5 + 0.1   # disparity, positive
    depth = torch.rand(1, 1, 2, 4, 4) * 5.0 + 1.0
    mask = torch.ones(1, 1, 2, 4, 4)
    compute_loss(pred, {"depth": depth, "mask": mask}, _cfg(vda_loss_repo), pred_is_disparity=True)
    # constructed with VDA's own composition defaults, not TrimmedProcrustesLoss's
    # standalone (and here unused) trim=0.2
    from train import _VDA_VIDEO_DEPTH_LOSS_CACHE
    (init_args,) = [v.init_args for v in _VDA_VIDEO_DEPTH_LOSS_CACHE.values()]
    assert init_args == {"alpha": 0.5, "scales": 4, "trim": 0.0, "stable_scale": 10.0}


def test_compute_loss_vda_mode_hands_raw_disparity_unconverted(vda_loss_repo):
    """pred must reach VideoDepthLoss as raw disparity -- no depth round-trip, no
    pre-normalisation (VideoDepthLoss's own TrimmedProcrustesLoss normalises internally)."""
    pred = torch.tensor([[[[[0.2, 0.3], [0.4, 0.5]], [[0.6, 0.7], [0.8, 0.9]]]]])  # (1,1,2,2,2)
    depth = torch.full((1, 1, 2, 2, 2), 4.0)
    mask = torch.ones(1, 1, 2, 2, 2)
    compute_loss(pred, {"depth": depth, "mask": mask}, _cfg(vda_loss_repo), pred_is_disparity=True)
    got = _RecordingVideoDepthLoss.last_call["prediction"]
    assert got.shape == (1, 2, 2, 2)   # (B,T,H,W), reshaped from (B,1,T,H,W)
    assert torch.equal(got, pred.reshape(1, 2, 2, 2))


def test_compute_loss_vda_mode_target_is_raw_inverse_depth(vda_loss_repo):
    pred = torch.full((1, 1, 1, 2, 2), 0.5)
    depth = torch.full((1, 1, 1, 2, 2), 4.0)
    mask = torch.ones(1, 1, 1, 2, 2)
    compute_loss(pred, {"depth": depth, "mask": mask}, _cfg(vda_loss_repo), pred_is_disparity=True)
    target = _RecordingVideoDepthLoss.last_call["target"]
    assert torch.allclose(target, torch.full((1, 1, 2, 2), 0.25))   # 1/4.0, no normalisation


def test_compute_loss_vda_mode_excludes_sky_rather_than_zero_disparity(vda_loss_repo):
    """VDA's own loss has no sky concept -- sky pixels must stay masked out, never
    force-included the way RAMEN's sky_zero_disparity convention does for other archs."""
    pred = torch.full((1, 1, 1, 2, 2), 0.5)
    depth = torch.full((1, 1, 1, 2, 2), 4.0)
    mask = torch.tensor([[[[[1.0, 0.0], [1.0, 1.0]]]]])   # top-right pixel invalid (e.g. sky)
    batch = {"depth": depth, "mask": mask, "sky": torch.tensor([[[[[0.0, 1.0], [0.0, 0.0]]]]])}
    compute_loss(pred, batch, _cfg(vda_loss_repo), pred_is_disparity=True)
    m = _RecordingVideoDepthLoss.last_call["mask"]
    assert not bool(m[0, 0, 0, 1])   # sky pixel stays excluded, not force-included


def test_compute_loss_vda_mode_converts_depth_native_pred_once_before_disparity(vda_loss_repo):
    """Matrix item 2 (Opus, 2026-08-28): loss_mode='vda' is no longer disparity-only --
    D3Net is genuinely depth-native (verified, see models/d3_net.py), so
    pred_is_disparity=False must do exactly ONE depth->disparity conversion (not raise),
    using the SAME clamp floor (1e-3) as train.py's own non-vda compute_loss path, not a
    redundant round-trip."""
    pred = torch.full((1, 1, 1, 2, 2), 4.0)   # depth-native pred
    depth = torch.full((1, 1, 1, 2, 2), 4.0)
    mask = torch.ones(1, 1, 1, 2, 2)
    compute_loss(pred, {"depth": depth, "mask": mask}, _cfg(vda_loss_repo), pred_is_disparity=False)
    got = _RecordingVideoDepthLoss.last_call["prediction"]
    assert torch.allclose(got, torch.full((1, 1, 2, 2), 0.25))   # 1/4.0, exactly one conversion


def test_compute_loss_vda_mode_depth_native_matches_disparity_native_when_equivalent(vda_loss_repo):
    """A depth-native pred of value 1/d and a disparity-native pred of value d must
    produce the identical disparity handed to VideoDepthLoss -- the two code paths (skip
    conversion vs convert once) must agree away from the clamp floor."""
    depth_native_pred = torch.full((1, 1, 1, 2, 2), 4.0)      # depth-native: value IS depth
    disparity_native_pred = torch.full((1, 1, 1, 2, 2), 0.25)  # disparity-native: value IS disparity
    gt_depth = torch.full((1, 1, 1, 2, 2), 4.0)
    mask = torch.ones(1, 1, 1, 2, 2)
    compute_loss(depth_native_pred, {"depth": gt_depth, "mask": mask}, _cfg(vda_loss_repo),
                pred_is_disparity=False)
    from_depth = _RecordingVideoDepthLoss.last_call["prediction"].clone()
    compute_loss(disparity_native_pred, {"depth": gt_depth, "mask": mask}, _cfg(vda_loss_repo),
                pred_is_disparity=True)
    from_disparity = _RecordingVideoDepthLoss.last_call["prediction"].clone()
    torch.testing.assert_close(from_depth, from_disparity)


def test_compute_loss_vda_mode_nan_depth_is_excluded_not_propagated(vda_loss_repo):
    """A NaN depth pixel fails depth > 0 (NaN comparisons are always False), so it's
    excluded from the mask before ever reaching 1/depth -- gd stays finite by construction.
    This documents why the module's own finite-target assertion normally can't fire
    through this path, rather than silently having no coverage for it."""
    pred = torch.full((1, 1, 1, 2, 2), 0.5)
    depth = torch.tensor([[[[[float("nan"), 4.0], [4.0, 4.0]]]]])
    mask = torch.ones(1, 1, 1, 2, 2)
    compute_loss(pred, {"depth": depth, "mask": mask}, _cfg(vda_loss_repo), pred_is_disparity=True)
    target = _RecordingVideoDepthLoss.last_call["target"]
    passed_mask = _RecordingVideoDepthLoss.last_call["mask"]
    assert torch.isfinite(target).all()
    assert not bool(passed_mask[0, 0, 0, 0])   # the NaN-depth pixel is excluded


def test_load_config_requires_loss_mode_vda_for_arch_vda():
    with pytest.raises(ValueError, match="loss_mode"):
        _assert_vda_uses_vda_loss_mode({"model": {"arch": "vda"}, "loss": {}})


def test_load_config_rejects_sky_zero_disparity_for_arch_vda():
    with pytest.raises(ValueError, match="sky_zero_disparity"):
        _assert_vda_uses_vda_loss_mode(
            {"model": {"arch": "vda"}, "loss": {"loss_mode": "vda", "sky_zero_disparity": True}})


def test_load_config_accepts_correct_vda_loss_config():
    _assert_vda_uses_vda_loss_mode(
        {"model": {"arch": "vda"}, "loss": {"loss_mode": "vda", "sky_zero_disparity": False}})   # must not raise


def test_load_config_skips_the_check_for_non_vda_arch():
    _assert_vda_uses_vda_loss_mode({"model": {"arch": "d3"}, "loss": {}})   # must not raise
