from __future__ import annotations

import sys
import types

import pytest
import torch
from torch import nn

sys.path.insert(0, "scripts")
import val01_eval_vda as ev  # noqa: E402

_NUM_MOTION = 3


class _FakePretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.patch_size = 14


class _FakeHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial = nn.Linear(4, 4)
        self.motion_modules = nn.ModuleList(nn.Linear(4, 4) for _ in range(_NUM_MOTION))


class _FakeVideoDepthAnything(nn.Module):
    def __init__(self, encoder="vits", features=64, out_channels=None, use_bn=False,
                use_clstoken=False, num_frames=32, pe="ape", metric=False):
        super().__init__()
        self.pretrained = _FakePretrained()
        self.head = _FakeHead()
        self.metric = metric

    def forward(self, x):
        return torch.ones(x.shape[0], x.shape[1], x.shape[3], x.shape[4])


@pytest.fixture
def fake_vda_repo_for_reconstruction(monkeypatch):
    """Combines test_vda_net.py's real-nn.Module video_depth_anything fake (needed to
    actually verify weight reconstruction) with empty eval/metric module stand-ins
    (needed only so _import_vda_modules doesn't crash -- load_model_for_inference's
    ramen_checkpoint path never calls into either)."""
    pkg = types.ModuleType("video_depth_anything")
    pkg.__path__ = []
    mod = types.ModuleType("video_depth_anything.video_depth")
    mod.VideoDepthAnything = _FakeVideoDepthAnything
    monkeypatch.setitem(sys.modules, "video_depth_anything", pkg)
    monkeypatch.setitem(sys.modules, "video_depth_anything.video_depth", mod)
    monkeypatch.setitem(sys.modules, "eval", types.ModuleType("eval"))
    monkeypatch.setitem(sys.modules, "metric", types.ModuleType("metric"))
    return "/fake/vda/repo"


def _da_v2_state_dict(pretrained_val=1.0, head_val=2.0):
    sd = {}
    ref = _FakeVideoDepthAnything()
    for k, v in ref.pretrained.state_dict().items():
        sd[f"pretrained.{k}"] = torch.full_like(v, pretrained_val)
    for k, v in ref.head.state_dict().items():
        if "motion_modules" in k:
            continue
        sd[f"depth_head.{k}"] = torch.full_like(v, head_val)
    return sd


def test_ramen_checkpoint_reconstruction_overlays_trainable_ema_onto_head(fake_vda_repo_for_reconstruction, tmp_path):
    """Reproduces the real failure found running this for real on job 6155634's
    step2000.pt (2026-08-27): loading a train.py-saved checkpoint directly with
    model.load_state_dict(checkpoint, strict=True) fails because the checkpoint's
    top-level keys are {cfg, epoch, step, ..., trainable, trainable_ema}, not a flat
    state dict -- the actual weights are nested inside trainable_ema, head.*-prefixed."""
    vda_repo = fake_vda_repo_for_reconstruction
    da_v2_path = tmp_path / "da_v2_vits.pth"
    torch.save(_da_v2_state_dict(pretrained_val=1.0, head_val=2.0), da_v2_path)

    # Build a reference model with known, distinguishable head weights to save as the
    # checkpoint's trainable_ema -- verifies the OVERLAY actually happened, not just that
    # loading didn't crash.
    from models.vda_net import VDANet
    trained = VDANet(repo=vda_repo, init="da_v2", da_v2_checkpoint_path=str(da_v2_path))
    with torch.no_grad():
        trained.core.head.spatial.weight.fill_(9.0)
    ema_target_sd = {f"head.{k}": v for k, v in trained.core.head.state_dict().items()}
    ckpt_path = tmp_path / "step2000.pt"
    torch.save({"cfg": {}, "epoch": 0, "step": 2000, "micro": 32000, "best": 0.1,
               "metrics": {}, "optimizer": {}, "ema": {}, "rng": {},
               "trainable_ema": ema_target_sd}, ckpt_path)

    model = ev.load_model_for_inference(vda_repo, str(ckpt_path), "vits", metric=False,
                                        ramen_checkpoint=True, da_v2_checkpoint_path=str(da_v2_path))
    assert torch.equal(model.head.spatial.weight, torch.full_like(model.head.spatial.weight, 9.0))
    # pretrained (frozen backbone) still comes from the DA-V2 init, unaffected by the checkpoint
    assert torch.equal(model.pretrained.proj.weight, torch.full_like(model.pretrained.proj.weight, 1.0))


def test_ramen_checkpoint_requires_da_v2_checkpoint_path(fake_vda_repo_for_reconstruction, tmp_path):
    ckpt_path = tmp_path / "step2000.pt"
    torch.save({"trainable_ema": {}}, ckpt_path)
    with pytest.raises(ValueError, match="da_v2_checkpoint_path"):
        ev.load_model_for_inference(fake_vda_repo_for_reconstruction, str(ckpt_path), "vits",
                                    metric=False, ramen_checkpoint=True)


def test_ramen_checkpoint_rejects_a_checkpoint_with_neither_trainable_key(fake_vda_repo_for_reconstruction, tmp_path):
    da_v2_path = tmp_path / "da_v2_vits.pth"
    torch.save(_da_v2_state_dict(), da_v2_path)
    ckpt_path = tmp_path / "bad.pt"
    torch.save({"cfg": {}, "model": {}}, ckpt_path)   # unfrozen-backbone shape, not yet supported
    with pytest.raises(ValueError, match="trainable"):
        ev.load_model_for_inference(fake_vda_repo_for_reconstruction, str(ckpt_path), "vits",
                                    metric=False, ramen_checkpoint=True,
                                    da_v2_checkpoint_path=str(da_v2_path))


def test_non_ramen_checkpoint_path_unchanged(fake_vda_repo_for_reconstruction, tmp_path):
    """The default (ramen_checkpoint=False) path must still work exactly as before --
    strict=True load of a flat, standalone state dict."""
    model_for_shape = _FakeVideoDepthAnything()
    full_sd = model_for_shape.state_dict()
    ckpt_path = tmp_path / "released.pth"
    torch.save(full_sd, ckpt_path)
    model = ev.load_model_for_inference(fake_vda_repo_for_reconstruction, str(ckpt_path), "vits", metric=False)
    assert isinstance(model, _FakeVideoDepthAnything)
