"""TOOL-08 / docs/DEGRADED_EVAL_PROTOCOL.md §3: RAMEN and OVDA pad native,
non-patch-multiple frames (KITTI-13drive) before their patch embeddings see them,
and crop the decoded depth back to native resolution before returning. The single
most important property (RAMEN Opus's review): every number produced before this
existed used a patch-size-divisible resolution, and must be provably unaffected --
so the no-op case is tested first and explicitly, not just the padded case."""
from __future__ import annotations

import torch

from infer import StreamingDepth, StreamingOVDA, _pad_to_patch_multiple


def test_pad_to_patch_multiple_is_a_true_noop_for_divisible_input():
    x = torch.randn(1, 3, 1, 28, 42)  # both dims already multiples of 14
    out, native_h, native_w = _pad_to_patch_multiple(x, 14)
    assert out is x  # the padded branch was never entered, not just numerically equal
    assert (native_h, native_w) == (28, 42)


def test_pad_to_patch_multiple_pads_bottom_right_with_replicated_edges():
    x = torch.arange(5 * 6, dtype=torch.float32).reshape(1, 1, 5, 6)
    out, native_h, native_w = _pad_to_patch_multiple(x, 4)
    assert (native_h, native_w) == (5, 6)
    assert out.shape[-2:] == (8, 8)  # next multiple of 4 in each dim
    torch.testing.assert_close(out[..., :5, :6], x)  # native content unchanged, top-left anchored
    torch.testing.assert_close(out[..., 5:, :6], x[..., 4:5, :].expand(-1, -1, 3, -1))  # bottom edge replicated
    torch.testing.assert_close(out[..., :5, 6:], x[..., :, 5:6].expand(-1, -1, -1, 2))  # right edge replicated


class _RecordingD3Net:
    """Records the (H, W) every stage actually receives, so a bug that pads before
    encode but reads native H/W into apply_temporal/decode_modules (using the wrong
    grid for the tokens actually produced) would surface as a recorded mismatch."""

    def __init__(self, patch_size):
        self.PATCH_SIZE = patch_size
        self.channel = None
        self.head = None
        self.seen_hw = []

    def condition(self, frames):
        return None, None

    def encode(self, frames, cond=None):
        return [(frames.mean(dim=(1, 3, 4), keepdim=False)[:, :, None, None], None)]

    def apply_temporal(self, feats, height, width, cond=None):
        self.seen_hw.append(("apply_temporal", height, width))
        token, cam = feats[0]
        return [(token, cam)]

    def decode_modules(self, temporal, channel, head, feats, height, width):
        self.seen_hw.append(("decode_modules", height, width))
        value = feats[0][0][..., 0, 0]
        return value[:, None, :, None, None].expand(-1, 1, -1, height, width)


def test_streaming_depth_divisible_input_is_untouched():
    model = _RecordingD3Net(patch_size=14)
    runner = StreamingDepth(model, window=1, device="cpu", dtype=None)
    frame = torch.ones(3, 28, 42)  # already multiples of 14
    out = runner.step(frame)
    assert out.shape == (28, 42)
    assert model.seen_hw == [("apply_temporal", 28, 42), ("decode_modules", 28, 42)]


def test_streaming_depth_pads_non_multiple_input_and_crops_back_to_native():
    model = _RecordingD3Net(patch_size=4)
    runner = StreamingDepth(model, window=1, device="cpu", dtype=None)
    frame = torch.ones(3, 5, 6)  # not multiples of 4 -- pads to (8, 8)
    out = runner.step(frame)
    assert out.shape == (5, 6)  # caller only ever sees native shape
    # Internal stages ran on the PADDED grid, not the native one -- the tokens encode()
    # actually produced correspond to (8, 8), so apply_temporal/decode_modules using a
    # different (H, W) would reconstruct the wrong spatial layout.
    assert model.seen_hw == [("apply_temporal", 8, 8), ("decode_modules", 8, 8)]


class _RecordingOVDA:
    """Records the (H, W) init_stream_state and step_stream each receive -- TOOL-10's
    mismatch was exactly two call sites disagreeing about which grid a padded/native
    frame maps to, so this is checked directly rather than trusted from the pad logic
    alone."""

    def __init__(self):
        self._mean = torch.zeros(1, 3, 1, 1, 1)
        self._std = torch.ones(1, 3, 1, 1, 1)
        self.init_hw = None
        self.step_hw = None

    def init_stream_state(self, H, W, device):
        self.init_hw = (H, W)
        return {"H": H, "W": W}

    def step_stream(self, x, state):
        self.step_hw = (x.shape[-2], x.shape[-1])
        assert (x.shape[-2], x.shape[-1]) == (state["H"], state["W"]), (
            "step_stream saw a different (H, W) than init_stream_state cached -- "
            "exactly TOOL-10's setup_cache-vs-step_stream mismatch, reached from "
            "infer.py's own wiring rather than OVDA's vendored formula"
        )
        return torch.ones(1, 1, x.shape[-2], x.shape[-1])


def test_streaming_ovda_pads_before_computing_hw_so_cache_and_step_agree():
    model = _RecordingOVDA()
    runner = StreamingOVDA(model, device="cpu", dtype=None)
    frame = torch.ones(3, 5, 6)  # not multiples of 14 -- pads to (14, 14)
    out = runner.step(frame)
    assert out.shape == (5, 6)  # caller only ever sees native shape
    assert model.init_hw == (14, 14)
    assert model.step_hw == (14, 14)


def test_streaming_ovda_divisible_input_is_untouched():
    model = _RecordingOVDA()
    runner = StreamingOVDA(model, device="cpu", dtype=None)
    frame = torch.ones(3, 14, 28)  # already multiples of 14
    out = runner.step(frame)
    assert out.shape == (14, 28)
    assert model.init_hw == (14, 28)
