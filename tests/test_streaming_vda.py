"""TOOL-16: infer.py's StreamingVDA drives an in-repo VDANet checkpoint through the
vendored video_depth_anything.video_depth_stream.VideoDepthAnything's real streaming
API. Uses a fully scripted fake module (same pattern as tests/test_vda_net.py) so the
suite stays hermetic -- no real vendored checkout or checkpoint required.

The fake streaming/whole-clip classes share an IDENTICAL pretrained/head structure
(same as tests/test_vda_net.py's own fakes) so a strict state_dict() transplant
between them is a meaningful test of the real code path, not something a shape
mismatch would let pass vacuously. The final test also exercises evaluation.model_registry.build_vda directly (its own,
separate code path -- loads a plain VideoDepthAnything state_dict, no VDANet
involved) against the same fakes, as the drift guard RAMEN Opus asked for between
StreamingVDA's and build_vda's duplicated reset()/predict_frame logic.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import torch
from torch import nn

from evaluation.model_registry import build_vda
from infer import StreamingVDA
from models.vda_net import VDANet

_NUM_MOTION = 3


class _FakePretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.patch_size = 14

    def get_intermediate_layers(self, x, idx, return_class_token=True):
        raise NotImplementedError("not exercised -- inference is faked at the whole-class level")


class _FakeHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial = nn.Linear(4, 4)
        self.motion_modules = nn.ModuleList(nn.Linear(4, 4) for _ in range(_NUM_MOTION))


class _FakeWholeClipCore(nn.Module):
    """Stands in for video_depth_anything.video_depth.VideoDepthAnything (VDANet.core)."""

    def __init__(self, encoder="vits", features=64, out_channels=None, use_bn=False,
                 use_clstoken=False, num_frames=32, pe="ape", metric=False):
        super().__init__()
        self.pretrained = _FakePretrained()
        self.head = _FakeHead()
        self.metric = metric

    def forward(self, x):
        B, T, C, H, W = x.shape
        scale = self.head.spatial.weight.sum() + self.pretrained.proj.weight.sum()
        return torch.ones(B, T, H, W) * scale + 1.0


class _FakeStreamingCore(nn.Module):
    """Stands in for video_depth_anything.video_depth_stream.VideoDepthAnything --
    identical pretrained/head structure to _FakeWholeClipCore (real code builds both
    from the same constructor args), plus the real class's four streaming-state
    attributes and its infer_video_depth_one single-frame API."""

    def __init__(self, encoder="vits", features=64, out_channels=None, use_bn=False,
                 use_clstoken=False, num_frames=32, pe="ape"):
        super().__init__()
        self.pretrained = _FakePretrained()
        self.head = _FakeHead()
        self.transform = None
        self.frame_id_list = []
        self.frame_cache_list = []
        self.id = -1

    def infer_video_depth_one(self, frame, device="cpu", fp32=False):
        assert frame.ndim == 3 and frame.shape[2] == 3, f"expected HWC, got {frame.shape}"
        assert frame.max() > 1.5, "input must be scaled to ~[0,255] HWC, not left as [0,1]"
        self.id += 1
        scale = (self.head.spatial.weight.sum() + self.pretrained.proj.weight.sum()).item()
        # Deterministic function of the loaded weights, self.id (so a test can tell apart
        # "fresh state" from "state carried over from a previous step() call") AND the
        # frame's own content (so a test can catch a wrong/missing CHW->HWC or [0,1]->
        # [0,255] conversion -- a content-blind fake would let that class of bug pass).
        content = float(frame.mean())
        return np.full(frame.shape[:2], abs(scale) + 1.0 + 0.1 * self.id + 0.001 * content,
                       dtype=np.float32)


@pytest.fixture
def vda_modules(tmp_path, monkeypatch):
    pkg = types.ModuleType("video_depth_anything")
    pkg.__path__ = []
    whole_clip_mod = types.ModuleType("video_depth_anything.video_depth")
    whole_clip_mod.VideoDepthAnything = _FakeWholeClipCore
    stream_mod = types.ModuleType("video_depth_anything.video_depth_stream")
    stream_mod.VideoDepthAnything = _FakeStreamingCore
    monkeypatch.setitem(sys.modules, "video_depth_anything", pkg)
    monkeypatch.setitem(sys.modules, "video_depth_anything.video_depth", whole_clip_mod)
    monkeypatch.setitem(sys.modules, "video_depth_anything.video_depth_stream", stream_mod)
    real_repo = tmp_path / "real_vda_repo"
    real_repo.mkdir()
    return str(real_repo)


def test_weights_transplant_strictly_from_vda_net_core(vda_modules):
    vda_net = VDANet(repo=vda_modules, init="random")
    # Give the trained checkpoint's weights a distinctive, non-default value so a
    # transplant failure (e.g. silently keeping the streaming core's own freshly
    # random-initialised weights instead) would change the predicted output.
    with torch.no_grad():
        for p in vda_net.core.parameters():
            p.fill_(0.25)

    runner = StreamingVDA(vda_net, device="cpu", dtype=None)
    assert torch.equal(
        runner.core.head.spatial.weight, vda_net.core.head.spatial.weight,
    )
    assert torch.equal(
        runner.core.pretrained.proj.weight, vda_net.core.pretrained.proj.weight,
    )


def test_step_converts_chw_01_to_hwc_0_255_before_calling_vendored_api(vda_modules):
    vda_net = VDANet(repo=vda_modules, init="random")
    runner = StreamingVDA(vda_net, device="cpu", dtype=None)
    frame_chw = torch.rand(3, 8, 6)   # this repo's shared manifest convention: CHW, [0,1]
    depth = runner.step(frame_chw)
    assert depth.shape == (8, 6)                 # (H, W), matching frame_chw's own H,W
    assert torch.all(depth > 0)                  # 1/raw_disparity of a positive fake output


def test_step_persists_state_across_calls_until_reset(vda_modules):
    vda_net = VDANet(repo=vda_modules, init="random")
    runner = StreamingVDA(vda_net, device="cpu", dtype=None)
    frame_chw = torch.rand(3, 8, 6)

    first = runner.step(frame_chw)
    second = runner.step(frame_chw)
    assert not torch.equal(first, second), (
        "_FakeStreamingCore.id increments each call -- two step()s on the same input "
        "must differ, or StreamingVDA is resetting the vendored class's own state "
        "between calls instead of letting it persist as a real stream would"
    )

    runner.reset()
    assert runner.core.id == -1
    assert runner.core.transform is None
    assert runner.core.frame_id_list == []
    assert runner.core.frame_cache_list == []
    after_reset = runner.step(frame_chw)
    assert torch.equal(after_reset, first), (
        "reset() must restore the exact state infer_video_depth_one starts a fresh "
        "video from -- the first step() after reset must reproduce the very first "
        "step()'s own output"
    )


def test_rejects_a_non_chw_frame(vda_modules):
    vda_net = VDANet(repo=vda_modules, init="random")
    runner = StreamingVDA(vda_net, device="cpu", dtype=None)
    with pytest.raises(ValueError, match="C,H,W"):
        runner.step(torch.rand(8, 6))


def test_agrees_with_build_vda_given_the_same_image_in_each_ones_native_convention(
    vda_modules, monkeypatch, tmp_path,
):
    """The drift guard RAMEN Opus asked for (2026-08-30): StreamingVDA and
    evaluation.model_registry.build_vda are two separate, deliberately duplicated copies
    of the same reset()/predict_frame logic (scripts/eval_streaming_npz.py is frozen and
    cannot be refactored to share code with StreamingVDA). Feeding the SAME underlying
    image to each in ITS OWN native convention -- CHW/[0,1] to StreamingVDA, HWC/[0,255]
    to build_vda -- and requiring identical output pins the equivalence explicitly,
    rather than leaving the duplication free to drift unnoticed."""
    vda_net = VDANet(repo=vda_modules, init="random")
    with torch.no_grad():
        for p in vda_net.core.parameters():
            p.uniform_(-1.0, 1.0)   # distinctive, non-constant weights the transplant must carry over

    checkpoint_path = tmp_path / "vda_checkpoint.pth"
    torch.save(vda_net.core.state_dict(), checkpoint_path)
    monkeypatch.setenv("VDA_ROOT", vda_modules)
    predict_frame, _reset, _digest = build_vda(str(checkpoint_path), device="cpu", dtype_str="fp32", window=32)

    runner = StreamingVDA(vda_net, device="cpu", dtype=None)

    frame_chw = torch.rand(3, 8, 6)                       # this repo's shared manifest convention
    frame_hwc_255 = frame_chw.permute(1, 2, 0) * 255.0    # the SAME image, build_vda's own convention

    from_wrapper = runner.step(frame_chw)
    from_build_vda = predict_frame(frame_hwc_255)

    assert torch.allclose(from_wrapper, from_build_vda.float(), atol=1e-5)


@pytest.mark.parametrize("encoder,metric", [("vitl", False), ("vitl", True), ("vits", True)])
def test_agrees_with_streaming_vda_teacher_on_output_semantics(vda_modules, monkeypatch, tmp_path,
                                                               encoder, metric):
    """2026-09-19: the train-path rebuild scored via ``--model ramen`` did not reproduce
    TEACH-01's vanilla teacher scores at step 0 -- StreamingVDA inverted the METRIC
    branch's depth-like output as if it were disparity, and left released relative
    VDA-L's ~450-median disparity un-rescaled (most pixels under the harness min_depth).
    The same weights must give the same depth through both wrappers."""
    from evaluation.teacher_models import StreamingVDATeacher

    vda_net = VDANet(repo=vda_modules, encoder=encoder, metric=metric, init="random")
    checkpoint_path = tmp_path / "vda_checkpoint.pth"
    torch.save(vda_net.core.state_dict(), checkpoint_path)
    monkeypatch.setenv("VDA_ROOT", vda_modules)
    teacher = StreamingVDATeacher(str(checkpoint_path), encoder, metric, device="cpu", dtype=None)
    runner = StreamingVDA(vda_net, device="cpu", dtype=None)

    frame_chw = torch.rand(3, 8, 6)
    assert torch.allclose(runner.step(frame_chw), teacher.step(frame_chw), atol=1e-6)
