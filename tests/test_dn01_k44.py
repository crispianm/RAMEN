"""DN-01 K44 proxy options (both opt-in, default paths unchanged): the display-space (gain-weighted) restoration L1 and
the final 8-bit quantisation of the synthetic lowlight input."""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from data.elvis_noise import ELVIS_PHI_PATHS
from data.video_clip_dataset import _degrade_clip
from train import enhancement_error

needs_bank = pytest.mark.skipif(not any(os.path.isfile(p) for p in ELVIS_PHI_PATHS), reason="needs the ELVIS bank")


def test_gain_weighted_l1_is_display_space_l1():
    g = torch.Generator().manual_seed(0)
    recon, clean = torch.rand(3, 3, 2, 8, 8, generator=g), torch.rand(3, 3, 2, 8, 8, generator=g)
    params = torch.tensor([[0.5, 1.0, 2.0 ** -2, 0.0, 0.0],                   # darkened lowlight clip: g = 4
                           [0.5, 1.0, 1.0, 0.1, 1.0],                          # awgn_blur: g = 1
                           [-1.0, -1.0, -1.0, -1.0, -1.0]])                    # paired real clip: g = 1
    batch = {"degrade_params": params}
    plain = enhancement_error(recon, clean, batch, {})
    assert plain == (recon - clean).abs().mean()                              # default unchanged
    w = torch.tensor([4.0, 1.0, 1.0]).view(3, 1, 1, 1, 1)
    got = enhancement_error(recon, clean, batch, {"enh_gain_weight": True})
    assert torch.allclose(got, (w * recon - w * clean).abs().mean())


@needs_bank
def test_quantized_lowlight_input_is_on_the_8bit_grid_and_default_is_not():
    clip = torch.rand(2, 3, 16, 16, generator=torch.Generator().manual_seed(1)) * 0.8 + 0.1
    ll = {"ev": [-2.0, -1.0], "kappa": 0.186}
    a, pa = _degrade_clip(clip, np.random.RandomState(3), "lowlight", severity=0.5, blur_severity=0.3, lowlight=ll)
    b, pb = _degrade_clip(clip, np.random.RandomState(3), "lowlight", severity=0.5, blur_severity=0.3,
                          lowlight=dict(ll, quantize=True))
    assert np.array_equal(pa, pb)                                                   # same RNG stream
    assert torch.equal(b, torch.floor(a * 255.0) / 255.0)                            # the floor of the float input
    assert torch.equal(b * 255.0, torch.round(b * 255.0)) and not torch.equal(a * 255.0, torch.round(a * 255.0))
