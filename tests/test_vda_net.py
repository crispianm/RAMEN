"""models/vda_net.py wraps the vendored video_depth_anything.video_depth.VideoDepthAnything
directly rather than reimplementing it -- these tests use a fully scripted fake module (same
pattern as tests/test_ovda_streaming.py's ovda_repo fixture) so the suite stays hermetic and
does not depend on the real vendored checkout or any downloaded checkpoint being present."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn

from models import build_model
from models.vda_net import VDANet

_NUM_MOTION = 3
_REPO_ROOT = Path(__file__).resolve().parents[1]


class _FakePretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.patch_size = 14

    def get_intermediate_layers(self, x, idx, return_class_token=True):
        raise NotImplementedError("not exercised -- forward() is faked at the VideoDepthAnything level")


class _FakeHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial = nn.Linear(4, 4)                         # stand-in for the 64 DA-V2-shared keys
        self.motion_modules = nn.ModuleList(nn.Linear(4, 4) for _ in range(_NUM_MOTION))  # zero-init temporal


class _FakeVideoDepthAnything(nn.Module):
    def __init__(self, encoder="vits", features=64, out_channels=None, use_bn=False,
                 use_clstoken=False, num_frames=32, pe="ape", metric=False):
        super().__init__()
        self.pretrained = _FakePretrained()
        self.head = _FakeHead()
        self.metric = metric

    def forward(self, x):   # x: (B,T,C,H,W)
        B, T, C, H, W = x.shape
        scale = self.head.spatial.weight.sum() + self.pretrained.proj.weight.sum()
        return torch.ones(B, T, H, W) * scale + 1.0   # +1 keeps it away from 0 (disparity clamp)


@pytest.fixture
def vda_repo(tmp_path, monkeypatch):
    pkg = types.ModuleType("video_depth_anything")
    pkg.__path__ = []
    mod = types.ModuleType("video_depth_anything.video_depth")
    mod.VideoDepthAnything = _FakeVideoDepthAnything
    monkeypatch.setitem(sys.modules, "video_depth_anything", pkg)
    monkeypatch.setitem(sys.modules, "video_depth_anything.video_depth", mod)
    # A real, existing directory -- TOOL-11-style fallback (VDANet.__init__ now checks
    # Path(repo).is_dir()) must not divert an existing, valid configured repo.
    real_repo = tmp_path / "real_vda_repo"
    real_repo.mkdir()
    return str(real_repo)


def _da_v2_state_dict(pretrained_val=1.0, head_val=2.0):
    """A DA-V2-shaped state dict: pretrained.* keys match VDA's own directly; depth_head.*
    stands in for DA-V2's head prefix that must be remapped to VDA's head.* on load."""
    sd = {}
    ref = _FakeVideoDepthAnything()
    for k, v in ref.pretrained.state_dict().items():
        sd[f"pretrained.{k}"] = torch.full_like(v, pretrained_val)
    for k, v in ref.head.state_dict().items():
        if "motion_modules" in k:
            continue   # DA-V2 has no temporal layers -- not present in a real DA-V2 checkpoint
        sd[f"depth_head.{k}"] = torch.full_like(v, head_val)
    return sd


def test_wrapper_builds_with_random_init(vda_repo):
    model = VDANet(repo=vda_repo, init="random")
    assert isinstance(model.core, _FakeVideoDepthAnything)


def test_encoder_frozen_head_trainable(vda_repo):
    model = VDANet(repo=vda_repo, init="random")
    model.assert_backbone_frozen()   # must not raise
    assert not any(p.requires_grad for p in model.core.pretrained.parameters())
    assert all(p.requires_grad for p in model.core.head.parameters())
    trainable_ids = {id(p) for p in model.trainable_parameters()}
    head_ids = {id(p) for p in model.core.head.parameters()}
    assert trainable_ids == head_ids


def test_trainable_modules_returns_the_whole_head(vda_repo):
    model = VDANet(repo=vda_repo, init="random")
    mods = model.trainable_modules()
    assert set(mods) == {"head"}
    assert mods["head"] is model.core.head


def test_unfrozen_backbone_fails_the_assertion(vda_repo):
    model = VDANet(repo=vda_repo, init="random")
    next(model.core.pretrained.parameters()).requires_grad_(True)
    with pytest.raises(AssertionError, match="frozen"):
        model.assert_backbone_frozen()


def test_forward_relative_returns_native_disparity_unconverted(vda_repo):
    """VDA's relative-mode output IS disparity natively -- forward() must pass it through
    unchanged, not convert it to depth the way OVDANet does for its own disparity-native
    model (train.py's compute_loss/evaluate read self.predicts_disparity to know which
    space they're getting, instead of every arch being forced into one convention)."""
    model = VDANet(repo=vda_repo, init="random", metric=False)
    assert model.predicts_disparity is True
    rgb = torch.rand(2, 3, 4, 28, 28)   # (B,C,T,H,W), multiples of 14
    with torch.no_grad():
        out = model(rgb)
    assert out.shape == (2, 1, 4, 28, 28)
    x = (rgb.permute(0, 2, 1, 3, 4) - model._mean) / model._std
    with torch.no_grad():
        disp = model.core(x).unsqueeze(1)
    assert torch.equal(out, disp)


def test_forward_metric_returns_depth_without_inversion(vda_repo):
    model = VDANet(repo=vda_repo, init="random", metric=True)
    assert model.predicts_disparity is False
    rgb = torch.rand(1, 3, 2, 28, 28)
    with torch.no_grad():
        out = model(rgb)
    x = (rgb.permute(0, 2, 1, 3, 4) - model._mean) / model._std
    with torch.no_grad():
        depth = model.core(x).unsqueeze(1)
    assert torch.equal(out, depth)


def test_metric_predicts_disparity_false_agrees_with_the_recorded_empirical_units_check(vda_repo):
    """predicts_disparity=False for metric mode used to be an unverified assumption
    (carried over from the relative variant's own documented behaviour); Fable/Opus's
    2026-08-27 ruling required an empirical check before trusting it further --
    scripts/val01_metric_units_check.py, run for real on crispwork against the released
    Metric-Video-Depth-Anything-Small checkpoint (see models/vda_net.py's class docstring
    for the full pre-registration + measured numbers). This test doesn't re-run that real,
    GPU-and-checkpoint-dependent check (this suite is hermetic and has no CUDA-compatible
    GPU) -- it cross-checks the cheap, checkable half: the committed artifact's own
    conclusion must still say "depth_like", AND this wrapper's own predicts_disparity flag
    must agree with it, so a future re-run that flips the empirical finding without anyone
    updating this class fails loudly instead of silently."""
    artifact_path = _REPO_ROOT / "provenance" / "inventory" / "val01_metric_units_check_kitti_20260827.json"
    artifact = json.loads(artifact_path.read_text())
    assert artifact["verdict"]["conclusion"] == "depth_like"
    model = VDANet(repo=vda_repo, init="random", metric=True)
    # depth_like -> forward() must NOT be treated as disparity -> predicts_disparity is False
    assert model.predicts_disparity is False


def test_forward_rejects_non_multiple_of_14_dims(vda_repo):
    model = VDANet(repo=vda_repo, init="random")
    rgb = torch.rand(1, 3, 2, 30, 30)
    with pytest.raises(AssertionError, match="multiple"):
        model(rgb)


def test_da_v2_init_loads_encoder_and_spatial_head_leaves_motion_modules_untouched(vda_repo, tmp_path):
    sd = _da_v2_state_dict(pretrained_val=3.0, head_val=5.0)
    ckpt_path = tmp_path / "da_v2_vits.pth"
    torch.save(sd, ckpt_path)

    # Same seed for both constructions: the fake's motion_modules are randomly initialised,
    # so "unchanged by DA-V2 init" can only be checked against a construction with identical
    # random state, not a fresh instance of its own.
    torch.manual_seed(0)
    ref = _FakeVideoDepthAnything()
    motion_before = {k: v.clone() for k, v in ref.head.state_dict().items() if "motion_modules" in k}

    torch.manual_seed(0)
    model = VDANet(repo=vda_repo, init="da_v2", da_v2_checkpoint_path=str(ckpt_path))

    assert torch.equal(model.core.pretrained.proj.weight, torch.full_like(model.core.pretrained.proj.weight, 3.0))
    assert torch.equal(model.core.head.spatial.weight, torch.full_like(model.core.head.spatial.weight, 5.0))
    for k, v in model.core.head.state_dict().items():
        if "motion_modules" in k:
            assert torch.equal(v, motion_before[k]), f"{k} should stay at its zero-init value, not be overwritten"


def test_nonexistent_da_v2_checkpoint_path_falls_back_to_local_default(vda_repo, tmp_path, monkeypatch, capsys):
    """Same TOOL-11-shaped defect as vda_repo's own fallback test: a checkpoint's saved cfg
    bakes in the training machine's da_v2_checkpoint_path, which does not exist on crispwork."""
    import models.vda_net as vda_net_module
    sd = _da_v2_state_dict(pretrained_val=7.0, head_val=9.0)
    real_ckpt = tmp_path / "da_v2_vits.pth"
    torch.save(sd, real_ckpt)
    monkeypatch.setattr(vda_net_module, "_DEFAULT_DA_V2_CHECKPOINT", str(real_ckpt))

    stale_isambard_path = "/projects/b5dh/DaBiT/weights/depth_anything_v2_vits.pth"
    model = VDANet(repo=vda_repo, init="da_v2", da_v2_checkpoint_path=stale_isambard_path)

    assert torch.equal(model.core.pretrained.proj.weight, torch.full_like(model.core.pretrained.proj.weight, 7.0))
    assert "does not exist on this host" in capsys.readouterr().err


def test_da_v2_init_rejects_an_unexpected_extra_key(vda_repo, tmp_path):
    sd = _da_v2_state_dict()
    sd["depth_head.nonexistent_layer.weight"] = torch.zeros(1)
    ckpt_path = tmp_path / "bad.pth"
    torch.save(sd, ckpt_path)
    with pytest.raises(AssertionError, match="unexpected"):
        VDANet(repo=vda_repo, init="da_v2", da_v2_checkpoint_path=str(ckpt_path))


def test_da_v2_init_rejects_a_missing_required_key(vda_repo, tmp_path):
    sd = _da_v2_state_dict()
    sd.pop("pretrained.proj.weight")
    ckpt_path = tmp_path / "incomplete.pth"
    torch.save(sd, ckpt_path)
    with pytest.raises(AssertionError, match="diverge"):
        VDANet(repo=vda_repo, init="da_v2", da_v2_checkpoint_path=str(ckpt_path))


def test_released_init_strict_loads_the_full_state_dict(vda_repo, tmp_path):
    ref = _FakeVideoDepthAnything()
    for p in ref.parameters():
        nn.init.constant_(p, 7.0)
    ckpt_path = tmp_path / "released.pth"
    torch.save(ref.state_dict(), ckpt_path)

    model = VDANet(repo=vda_repo, init="released", checkpoint_path=str(ckpt_path))
    assert all(torch.equal(p, torch.full_like(p, 7.0)) for p in model.core.parameters())


def test_unknown_encoder_rejected(vda_repo):
    with pytest.raises(ValueError, match="encoder"):
        VDANet(repo=vda_repo, encoder="vitxl", init="random")


def test_unknown_init_rejected(vda_repo):
    with pytest.raises(ValueError, match="init"):
        VDANet(repo=vda_repo, init="not_a_real_mode")


def test_train_mode_keeps_backbone_in_eval(vda_repo):
    model = VDANet(repo=vda_repo, init="random")
    model.train()
    assert model.training
    assert not model.core.pretrained.training


def test_patch_size_read_from_the_encoder_not_hardcoded(vda_repo):
    """infer.py's StreamingDepth.step reads self.m.PATCH_SIZE to pad non-patch-multiple
    frames (TOOL-08); VDANet lacked this attribute entirely until it was actually run
    through that path for the first time tonight (AttributeError). Must come from the
    encoder itself, not a literal, so a different encoder size can't silently mismatch."""
    model = VDANet(repo=vda_repo, init="random")
    assert model.PATCH_SIZE == model.core.pretrained.patch_size == 14


def test_build_model_dispatches_arch_vda(vda_repo):
    cfg = {"model": {"arch": "vda", "vda_repo": vda_repo, "vda_init": "random", "metric": True}}
    model = build_model(cfg)
    assert isinstance(model, VDANet)
    assert model.metric is True


def test_nonexistent_configured_repo_falls_back_to_local_default(vda_repo, monkeypatch, capsys):
    """TOOL-14 root cause: matrix_vda_degraded's own checkpoint cfg bakes in Isambard's
    training-machine vda_repo path, which does not exist on crispwork -- same bug class
    TOOL-11 fixed for OVDA. VDANet must fall back rather than fail deep inside sys.path."""
    import models.vda_net as vda_net_module
    monkeypatch.setattr(vda_net_module, "_DEFAULT_VDA_REPO", vda_repo)
    stale_isambard_path = "/projects/b5dh/repos/Video-Depth-Anything"
    model = VDANet(repo=stale_isambard_path, init="random")
    assert isinstance(model.core, _FakeVideoDepthAnything)
    assert vda_repo in sys.path
    assert stale_isambard_path not in sys.path
    assert "does not exist on this host" in capsys.readouterr().err


def test_existing_configured_repo_is_used_as_is_with_no_warning(vda_repo, capsys):
    VDANet(repo=vda_repo, init="random")
    assert capsys.readouterr().err == ""
