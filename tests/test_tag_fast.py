"""Unit equivalence of repro/tag_fast.py vs the numpy reference and depth_agf_torch on synthetic inputs (no model). EXPLORATORY, UNAUDITED."""
import numpy as np
import pytest
import torch

import subprocess
from pathlib import Path

import evaluation.depth_probe as dp
from repro import tag_fast_patch as tfp
from repro.tag_fast import Agf3Stream, agf_v2, agf_v3

# Reference = the scorer's own float64 numpy depth_agf, imported from evaluation/depth_probe.py (NOT copied); the file must be byte-identical to scorer commit 368b5c53.
SCORER_COMMIT = "368b5c53b37ebd22f5d44ce0c9d514122efc88da"
SCORER_DEPTH_PROBE_BLOB = "4b2269d10a35f2da9071b65592dab9816862a667"  # git blob of evaluation/depth_probe.py at SCORER_COMMIT (pinned: the public repo has no history)
depth_agf, depth_agf_torch = tfp.original("depth_agf"), tfp.original("depth_agf_torch")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
PARAMS = [(0.2, 0.06), (0.5, 0.12), (0.05, 0.02)]


def _inputs():
    g = np.random.default_rng(0)
    base = 0.3 + 9 * g.random((1, 64, 80))
    rnd = (base * (1 + 0.05 * g.standard_normal((32, 64, 80)))).astype(np.float32)
    step = rnd.copy(); step[16:, :32] *= 1.5
    spike = rnd.copy(); spike[10, 5:20, 5:20] *= 8; spike[11] = rnd[11]
    near0 = rnd.copy(); near0[:, :8, :8] = g.random((32, 8, 8)).astype(np.float32) * 1e-6; near0[5, 8:16, :8] = 0.0
    long = (base * (1 + 0.05 * g.standard_normal((600, 64, 80)))).astype(np.float32)
    return {"random": rnd, "step": step, "spike": spike, "near_zero": near0, "long_T600": long, "T1": rnd[:1], "T2": rnd[:2]}


@pytest.mark.parametrize("name", list(_inputs()))
@pytest.mark.parametrize("amin,tau", PARAMS)
def test_equivalence(name, amin, tau):
    x = _inputs()[name]
    ref = depth_agf(x, amin, tau).astype(np.float64)
    xt = torch.from_numpy(x).cuda()
    v1 = depth_agf_torch(xt, amin, tau)
    outs = {"v1": v1.double().cpu().numpy(), "v2": agf_v2(xt, amin, tau).double().cpu().numpy(), "v3": agf_v3(xt, amin, tau).double().cpu().numpy()}
    for n, o in outs.items():
        assert np.isfinite(o).all(), n
        err = np.abs(o - ref)
        rel = err / np.maximum(np.abs(ref), 1e-6)
        if name != "near_zero":
            assert err.max() < 1e-3, (n, err.max())          # metres; typically ~1e-5 for fp32 on 0.3-10 m
            assert rel.max() < 1e-4, (n, rel.max())
        else:
            assert err.max() < 1e-3, (n, err.max())
        print(f"{name} amin={amin} tau={tau} {n}: max_abs={err.max():.3e} max_rel={rel.max():.3e} mean_abs={err.mean():.3e}")


@pytest.mark.parametrize("name", ["random", "spike", "near_zero", "long_T600"])
@pytest.mark.parametrize("amin,tau", PARAMS)
def test_stream_bit_identical(name, amin, tau):
    xt = torch.from_numpy(_inputs()[name]).cuda()
    win = agf_v3(xt, amin, tau)
    st = Agf3Stream(xt[0], amin, tau)
    out = [xt[0].float()] + [st.step(xt[t]) for t in range(1, xt.shape[0])]
    assert torch.equal(torch.stack(out), win)


def test_amin_one_identity():
    x = _inputs()["random"]; xt = torch.from_numpy(x).cuda()
    assert np.array_equal(depth_agf(x, 1.0, 0.06), x)
    assert torch.equal(agf_v3(xt, 1.0, 0.06), xt) and torch.equal(agf_v2(xt, 1.0, 0.06), xt)
    st = Agf3Stream(xt[0], 1.0, 0.06)
    assert torch.equal(st.step(xt[1]), xt[1])


def test_fp16_input():
    x = _inputs()["random"]; xt = torch.from_numpy(x).cuda()
    h = xt.half()
    ref = depth_agf(h.float().cpu().numpy(), 0.2, 0.06).astype(np.float64)    # the filter applied to the fp16-quantised depth
    o = agf_v3(h, 0.2, 0.06)
    assert o.dtype == torch.float32
    assert np.abs(o.double().cpu().numpy() - ref).max() < 1e-3


def _dist(o, ref):
    r = torch.from_numpy(ref).cuda().double(); o = o.double()
    rel = ((o - r).abs() / r.abs().clamp_min(1e-6)).flatten()
    q = rel[::max(1, rel.numel() // 4_000_000)].cpu().numpy()
    return float(rel.max()), float(np.quantile(q, 0.999)), float(np.median(q))


def test_reference_is_scorer_368b5c53():
    f = Path(dp.__file__)
    blob = SCORER_DEPTH_PROBE_BLOB
    assert subprocess.check_output(["git", "hash-object", str(f)], text=True).strip() == blob, "depth_probe.py differs from the scorer at 368b5c53"
    assert depth_agf.__module__ == "evaluation.depth_probe" and not tfp._state["patched"]


def test_eps_guard_and_655m_cap():
    g = np.random.default_rng(3)
    near0 = np.concatenate([(g.random((16, 32, 40)) * 2e-6).astype(np.float32), np.zeros((16, 32, 40), np.float32)])
    far = (655.0 * (1 + 0.02 * g.standard_normal((32, 32, 40)))).astype(np.float32)
    for x in (near0, far):
        for amin, tau in PARAMS[:2]:
            ref = depth_agf(x, amin, tau); xt = torch.from_numpy(x).cuda()
            for o in (agf_v2(xt, amin, tau), agf_v3(xt, amin, tau)):
                mx, p, md = _dist(o, ref)
                assert mx <= 1e-5, (mx, p, md)


def test_state_carries_across_stitched_windows():
    """Stitched sequence emitted as 32 + 22 + 22 + ... frames (window 32, stride 22): one carried stream over the chunks == whole-sequence V3 bitwise, and == the numpy reference to <=1e-5."""
    x = _inputs()["long_T600"][:32 + 22 * 9]
    xt = torch.from_numpy(x).cuda()
    for amin, tau in PARAMS[:2]:
        whole = agf_v3(xt, amin, tau)
        st = Agf3Stream(xt[0], amin, tau)
        parts = [torch.cat([xt[:1].float(), st.chunk(xt[1:32])])]
        for a in range(32, xt.shape[0], 22):
            parts.append(st.chunk(xt[a:a + 22]))
        assert torch.equal(torch.cat(parts), whole)
        assert _dist(whole, depth_agf(x, amin, tau))[0] <= 1e-5
        # a per-window RESET is a different filter (class A): it must differ
        reset = torch.cat([agf_v3(xt[a:a + 32], amin, tau) for a in range(0, 32, 32)] + [agf_v3(xt[a:a + 22], amin, tau) for a in range(32, xt.shape[0], 22)])
        assert not torch.equal(reset, whole)


def test_nan_inf_behaviour_matches_reference():
    x = _inputs()["random"][:12].copy(); x[5, 3, 3] = np.nan; x[7, 10, 10] = np.inf; x[2, 20, 20] = -np.inf
    with np.errstate(all="ignore"):
        ref = depth_agf(x, 0.2, 0.06)
    xt = torch.from_numpy(x).cuda()
    for o in (depth_agf_torch(xt, 0.2, 0.06), agf_v2(xt, 0.2, 0.06), agf_v3(xt, 0.2, 0.06)):
        o = o.cpu().numpy()
        assert np.array_equal(np.isnan(o), np.isnan(ref)) and np.array_equal(np.isinf(o), np.isinf(ref))


@pytest.mark.parametrize("amin,tau", PARAMS[:2])
def test_T1000_full_518(amin, tau):
    g = np.random.default_rng(5)
    x = (7.0 * (1 + 0.03 * g.standard_normal((1000, 518, 518)))).astype(np.float32); x[500:, :259] *= 1.4
    ref = depth_agf(x, amin, tau); xt = torch.from_numpy(x).cuda()
    for o in (agf_v2(xt, amin, tau), agf_v3(xt, amin, tau)):
        mx, p, md = _dist(o, ref); print(f"T1000 518x518 {amin}/{tau}: max {mx:.2e} p99.9 {p:.2e} median {md:.2e}")
        assert mx <= 1e-5


def test_patch_routes_scorer_to_v3_and_restores():
    x = _inputs()["step"][:20]; xt = torch.from_numpy(x).cuda()
    orig_name, orig_fn, orig_np = dp.AGF_BACKEND, dp.depth_agf_torch, dp.depth_agf
    tok = dp.make_post("agf0.2t0.06")
    base = tok(x)                                            # unpatched: the numpy reference path
    tfp.patch()
    try:
        assert dp.AGF_BACKEND == "fast_triton_v3" == tfp.BACKEND_NAME
        assert dp.depth_agf_torch is tfp._v3 and dp.depth_agf is tfp._v3
        post = dp.make_post("agf0.2t0.06")
        on_np, on_t = post(x), post(xt)
        assert isinstance(on_np, np.ndarray) and on_np.dtype == x.dtype and torch.is_tensor(on_t)
        assert torch.equal(on_t, agf_v3(xt, 0.2, 0.06))
        assert np.array_equal(on_np, agf_v3(xt, 0.2, 0.06).cpu().numpy())
        assert np.abs(on_np.astype(np.float64) - base).max() < 1e-5 * np.abs(base).max()
        assert np.array_equal(dp.make_post("agf1t0.06")(x), x)       # amin=1 identity
    finally:
        tfp.unpatch()
    assert (dp.AGF_BACKEND, dp.depth_agf_torch, dp.depth_agf) == (orig_name, orig_fn, orig_np)
    assert np.array_equal(dp.make_post("agf0.2t0.06")(x), base)       # numpy reference path again, bit for bit
