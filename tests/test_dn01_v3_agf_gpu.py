"""Protocol v3 WP24: the device implementation of the agf stage (evaluation/depth_probe.depth_agf_torch) vs the numpy REFERENCE (depth_agf). Stated tolerance: max |diff| < 1e-4 m on metric depth (float32 output)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import depth_probe as dp  # noqa: E402

TOL_M = 1e-4


def _clip(seed=0, n=32, h=40, w=56, noise=0.06):
    rng = np.random.RandomState(seed)
    base = rng.uniform(1.0, 80.0, (h, w))
    t = np.arange(n)[:, None, None]
    d = base * (1 + 0.2 * np.sin(t / 5.0)) * (1 + noise * rng.randn(n, h, w))        # flicker + a slow real change
    d[10] *= 1.8                                                                       # a step (the gain must follow)
    return np.clip(d, 0.1, 400.0).astype(np.float32)


@pytest.mark.parametrize("amin,tau", [(0.2, 0.06), (0.2, 0.03), (0.3, 0.12), (0.5, 0.25)])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gpu_agf_matches_the_numpy_reference_within_tolerance(amin, tau, seed):
    d = _clip(seed)
    ref = dp.depth_agf(d, amin, tau)
    got = dp.depth_agf_torch(d, amin, tau)                                            # numpy in -> numpy out (CUDA if present, else CPU torch)
    assert got.dtype == ref.dtype == np.float32 and got.shape == ref.shape
    assert float(np.abs(got - ref).max()) < TOL_M, float(np.abs(got - ref).max())
    assert float(np.abs(got - ref).max() / np.abs(ref).max()) < 1e-6                   # and relative


def test_amin_one_is_the_exact_identity_like_the_reference():
    d = _clip(3)
    assert np.array_equal(dp.depth_agf_torch(d, 1.0, 0.06), d) and np.array_equal(dp.depth_agf(d, 1.0, 0.06), d)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_a_cuda_tensor_stays_on_the_device_and_matches():
    d = _clip(4)
    t = torch.from_numpy(d).cuda()
    out = dp.depth_agf_torch(t, 0.2, 0.06)
    assert out.is_cuda and out.dtype == torch.float32 and out.shape == t.shape
    assert float(np.abs(out.cpu().numpy() - dp.depth_agf(d, 0.2, 0.06)).max()) < TOL_M


def test_the_backend_flag_routes_make_post_and_the_default_is_the_reference(monkeypatch):
    d = _clip(5)
    fn = dp.make_post("agf0.2t0.06")
    assert dp.AGF_BACKEND == "cpu" and np.array_equal(fn(d), dp.depth_agf(d, 0.2, 0.06))             # default = reference, bit for bit
    monkeypatch.setattr(dp, "AGF_BACKEND", "gpu")
    got = fn(d)
    assert float(np.abs(got - dp.depth_agf(d, 0.2, 0.06)).max()) < TOL_M and not np.array_equal(got, d)
    assert isinstance(fn(torch.from_numpy(d)), torch.Tensor)                                          # a tensor input always takes the device path


def test_scoring_metrics_are_unchanged_by_the_backend(monkeypatch):
    """The scoring decision inputs (AbsRel / TAE-style temporal stats on the filtered depth) agree to far better than any reported digit."""
    d = _clip(6)
    gt = d.mean(0, keepdims=True).repeat(len(d), 0)
    mets = lambda x: (float(np.mean(np.abs(x - gt) / gt)), float(np.mean(np.abs(np.diff(np.log(x), axis=0)))))
    ref = mets(dp.depth_agf(d, 0.2, 0.06))
    got = mets(dp.depth_agf_torch(d, 0.2, 0.06))
    assert all(abs(a - b) < 1e-7 for a, b in zip(ref, got)), (ref, got)
