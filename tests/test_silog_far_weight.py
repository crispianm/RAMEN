"""L1 arm S6: far-weighted SiLog (loss.metric_silog_far_weight / loss.metric_silog_far_from), opt-in, default OFF.
Covers: OFF is bit-identical (key absent == far_weight 1.0 == the original expression, loss AND gradient, torch.equal); the weighted formula against an independent float64 reference;
invariants (all-near, all-far, w=1 reduce to the original); gradcheck; far weight raises the far pull; config validation (refusal paths); the log term; and FORCED-FAIL controls:
every property check is also run against deliberately wrong implementations (mutants) and MUST raise AssertionError, so a vacuous check cannot pass."""
from __future__ import annotations

import math

import pytest
import torch

import train
from train import check_metricfix, compute_loss, silog_far_weight_cfg, silog_far_weighted


class _ZeroVDALoss(torch.nn.Module):
    def __init__(self, **_):
        super().__init__()

    def forward(self, prediction, target, mask):
        return {"total_loss": prediction.sum() * 0.0}


@pytest.fixture
def zero_vda_loss(monkeypatch):
    monkeypatch.setattr(train, "_get_vda_video_depth_loss", lambda repo, lc: _ZeroVDALoss())


def _cfg(**loss):
    return {"model": {"arch": "vda", "metric": True},
            "loss": {"loss_mode": "vda", "sky_zero_disparity": False, "metric_silog_weight": 1.0, **loss}}


def _orig(diff_log, lambd):   # the ORIGINAL (pre-S6) expression, verbatim
    return torch.sqrt((diff_log.pow(2).mean() - lambd * diff_log.mean().pow(2)).clamp_min(1e-12))


def _ref64(d, g, lambd, fw, ff):   # independent reference: python floats, explicit loops, no torch ops
    d = [float(x) for x in d]; g = [float(x) for x in g]
    w = [fw if gi >= ff else 1.0 for gi in g]
    sw = sum(w); m1 = sum(wi * di for wi, di in zip(w, d)) / sw; m2 = sum(wi * di * di for wi, di in zip(w, d)) / sw
    return math.sqrt(max(m2 - lambd * m1 * m1, 1e-12))


def _data(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    gt = torch.cat([torch.rand(n // 2, generator=g) * 40 + 1, torch.rand(n // 2, generator=g) * 400 + 50]).double()   # near 1-41 m, far 50-450 m
    gt[0] = 50.0   # a pixel EXACTLY at the boundary: must count as far (>=)
    d = (torch.randn(n, generator=g) * 0.3 + 0.1).double()
    return d, gt


def check_impl(fn):
    """Property checks of a silog_far_weighted-like callable fn(diff_log, gt, lambd, fw, ff). Raises AssertionError on any violation."""
    d, gt = _data()
    for lambd in (0.5, 0.15):
        for fw, ff in ((4.0, 50.0), (2.5, 80.0), (4.0, 1e9), (4.0, 0.0)):
            got = float(fn(d, gt, lambd, fw, ff)); want = _ref64(d, gt, lambd, fw, ff)
            assert abs(got - want) < 1e-9, f"formula: got {got} want {want} (lambd {lambd}, fw {fw}, ff {ff})"
    # reduces to the original: w == 1 (fw 1), no far pixel (ff huge), every pixel far (ff 0)
    for fw, ff in ((1.0, 50.0), (4.0, 1e9), (4.0, 0.0)):
        assert abs(float(fn(d, gt, 0.5, fw, ff)) - float(_orig(d, 0.5))) < 1e-9, f"must equal the original for fw {fw}, ff {ff}"
    # boundary pixel: lowering ff just below / raising just above the exact boundary value 50.0 must change the result (>= semantics)
    a = float(fn(d, gt, 0.5, 4.0, 50.0)); b = float(fn(d, gt, 0.5, 4.0, 50.0 + 1e-6))
    assert abs(a - b) > 1e-6, "boundary pixel at exactly far_from must be treated as far"
    # far weight raises the far pull: share of |grad| on far pixels grows with fw
    def share(fw):
        dd = d.clone().requires_grad_(True); fn(dd, gt, 0.5, fw, 50.0).backward(); gr = dd.grad.abs()
        return float(gr[gt >= 50.0].sum() / gr.sum())
    assert share(4.0) > share(1.0) + 1e-3, "far weight must raise the far share of the gradient"
    assert share(1.0) < share(4.0) < share(16.0), "far gradient share must increase monotonically with the weight"
    # gradcheck (float64)
    dd = d.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(lambda x: fn(x, gt, 0.5, 4.0, 50.0), (dd,), eps=1e-6, atol=1e-5)


def test_good_implementation_passes_all_properties():
    check_impl(silog_far_weighted)


def _mut_weight_on_near(d, gt, lambd, fw, ff):
    w = 1.0 + (fw - 1.0) * (gt < ff).to(d.dtype); sw = w.sum()
    return torch.sqrt(((w * d.pow(2)).sum() / sw - lambd * ((w * d).sum() / sw).pow(2)).clamp_min(1e-12))


def _mut_norm_by_n(d, gt, lambd, fw, ff):   # forgets to normalise by sum(w): weights inflate the loss
    w = 1.0 + (fw - 1.0) * (gt >= ff).to(d.dtype); n = d.numel()
    return torch.sqrt(((w * d.pow(2)).sum() / n - lambd * ((w * d).sum() / n).pow(2)).clamp_min(1e-12))


def _mut_strict_greater(d, gt, lambd, fw, ff):   # '>' instead of '>='
    w = 1.0 + (fw - 1.0) * (gt > ff).to(d.dtype); sw = w.sum()
    return torch.sqrt(((w * d.pow(2)).sum() / sw - lambd * ((w * d).sum() / sw).pow(2)).clamp_min(1e-12))


def _mut_unweighted_mean_term(d, gt, lambd, fw, ff):   # weights only the second moment, the squared-mean term stays unweighted
    w = 1.0 + (fw - 1.0) * (gt >= ff).to(d.dtype); sw = w.sum()
    return torch.sqrt(((w * d.pow(2)).sum() / sw - lambd * d.mean().pow(2)).clamp_min(1e-12))


def _mut_ignores_weight(d, gt, lambd, fw, ff):
    return _orig(d, lambd)


def _mut_weight_squared(d, gt, lambd, fw, ff):   # w^2 instead of w
    w = (1.0 + (fw - 1.0) * (gt >= ff).to(d.dtype)) ** 2; sw = w.sum()
    return torch.sqrt(((w * d.pow(2)).sum() / sw - lambd * ((w * d).sum() / sw).pow(2)).clamp_min(1e-12))


def _mut_detached(d, gt, lambd, fw, ff):   # value right, no gradient
    with torch.no_grad():
        v = silog_far_weighted(d, gt, lambd, fw, ff)
    return v + 0.0 * d.sum()


@pytest.mark.parametrize("mut", [_mut_weight_on_near, _mut_norm_by_n, _mut_strict_greater, _mut_unweighted_mean_term, _mut_ignores_weight, _mut_weight_squared, _mut_detached])
def test_forced_fail_mutants_are_rejected(mut):
    with pytest.raises((AssertionError, RuntimeError)):   # RuntimeError: gradcheck raises GradcheckError (a RuntimeError subclass) on a wrong gradient
        check_impl(mut)


# ---- OFF == byte-identical through the real compute_loss -----------------------------------------------------------------------------------------
def _batch(seed=1, sparse_mask=True):
    g = torch.Generator().manual_seed(seed)
    depth = torch.cat([torch.rand(1, 1, 2, 8, 4, generator=g) * 40 + 1, torch.rand(1, 1, 2, 8, 4, generator=g) * 300 + 60], dim=-1)   # left near, right far
    mask = torch.ones_like(depth)
    if sparse_mask:
        mask[..., 0, 0] = 0.0
    pred = depth * torch.exp(torch.randn(depth.shape, generator=g) * 0.3 + 0.05)
    return pred, {"depth": depth, "mask": mask}


def _loss_and_grad(cfg):
    pred, batch = _batch(); pred = pred.clone().requires_grad_(True)
    loss = compute_loss(pred, batch, cfg); loss.backward()
    return loss.detach(), pred.grad.detach().clone()


def test_off_is_bit_identical(zero_vda_loss):
    l0, g0 = _loss_and_grad(_cfg())                                         # keys absent
    l1, g1 = _loss_and_grad(_cfg(metric_silog_far_weight=1.0))              # far_weight exactly 1.0 (+ no far_from)
    l2, g2 = _loss_and_grad(_cfg(metric_silog_far_weight=1.0, metric_silog_far_from=50.0))
    for l, g in ((l1, g1), (l2, g2)):
        assert torch.equal(l0, l) and torch.equal(g0, g), "OFF must be bit-identical (loss and gradient)"
    pred, batch = _batch(); m = batch["mask"] > 0.5
    dl = torch.log(batch["depth"][m].float()) - torch.log(pred[m].float().clamp_min(1e-3))
    assert torch.equal(l0, _orig(dl, 0.5)), "OFF must equal the original expression evaluated independently"


def test_on_matches_reference_through_compute_loss(zero_vda_loss):
    l_on, g_on = _loss_and_grad(_cfg(metric_silog_far_weight=4.0, metric_silog_far_from=50.0))
    l_off, g_off = _loss_and_grad(_cfg())
    pred, batch = _batch(); m = batch["mask"] > 0.5
    dl = (torch.log(batch["depth"][m].float()) - torch.log(pred[m].float().clamp_min(1e-3)))
    want = _ref64(dl.double(), batch["depth"][m].double(), 0.5, 4.0, 50.0)
    assert abs(float(l_on) - want) < 1e-5, f"ON loss {float(l_on)} vs reference {want}"
    assert abs(float(l_on) - float(l_off)) > 1e-4, "ON must change the loss on a batch with both near and far pixels"
    assert not torch.equal(g_on, g_off)
    far = batch["depth"] >= 50.0
    sh = lambda g: float(g[far & (batch["mask"] > 0.5)].abs().sum() / g.abs().sum())
    assert sh(g_on) > sh(g_off), "far pixels must receive a larger share of the gradient when ON"


def test_masked_pixels_do_not_count(zero_vda_loss):
    pred, batch = _batch(); pred2 = pred.clone(); pred2[..., 0, 0] = 1e3   # the masked-out pixel (mask 0) changes wildly
    cfg = _cfg(metric_silog_far_weight=4.0, metric_silog_far_from=50.0)
    assert torch.equal(compute_loss(pred, batch, cfg), compute_loss(pred2, batch, cfg))


def test_forced_fail_wrong_integration_is_detected(zero_vda_loss, monkeypatch):
    """The end-to-end ON test must FAIL if the loss silently ignored the keys (forced-fail control of the previous test)."""
    monkeypatch.setattr(train, "silog_far_weighted", lambda d, g, lambd, fw, ff: _orig(d, lambd))
    with pytest.raises(AssertionError):
        test_on_matches_reference_through_compute_loss(None)


def test_log_term_only_when_on(zero_vda_loss):
    for cfg, present in ((_cfg(log_terms=True), False), (_cfg(log_terms=True, metric_silog_far_weight=1.0), False),
                         (_cfg(log_terms=True, metric_silog_far_weight=4.0, metric_silog_far_from=50.0), True)):
        train._TERM_ACC.clear() if hasattr(train, "_TERM_ACC") else None
        pred, batch = _batch(); compute_loss(pred, batch, cfg)
        acc = getattr(train, "_TERM_ACC", {})
        assert ("silog_far_px_frac" in acc) == present, f"log term presence {('silog_far_px_frac' in acc)} want {present}"
        if present:
            frac = (batch["depth"][batch["mask"] > 0.5] >= 50.0).float().mean()
            s = acc["silog_far_px_frac"]; v = s[0] / s[1] if isinstance(s, (list, tuple)) else s
            assert abs(float(v) - float(frac)) < 1e-6


def test_log_term_uses_ge_at_the_boundary(zero_vda_loss):
    """second review (S6-code mutant): the logged far-pixel fraction counts GT >= far_from (a pixel exactly AT far_from is far, as in the loss weight).  Pixels set to exactly 50.0 make `>` differ from `>=`."""
    cfg = _cfg(log_terms=True, metric_silog_far_weight=4.0, metric_silog_far_from=50.0)
    pred, batch = _batch(); d = batch["depth"].clone(); d[..., 0, :2] = 50.0; batch["depth"] = d     # a few pixels at exactly 50.0
    m = batch["mask"] > 0.5
    ge, gt = (d[m] >= 50.0).float().mean(), (d[m] > 50.0).float().mean()
    assert float(ge) > float(gt), "vacuous: the batch must contain pixels exactly at far_from"
    train._TERM_ACC.clear() if hasattr(train, "_TERM_ACC") else None
    compute_loss(pred, batch, cfg); s = getattr(train, "_TERM_ACC", {})["silog_far_px_frac"]; v = s[0] / s[1] if isinstance(s, (list, tuple)) else s
    assert abs(float(v) - float(ge)) < 1e-6 and abs(float(v) - float(gt)) > 1e-6


def test_config_validation_refusals():
    ok = _cfg(metric_silog_far_weight=4.0, metric_silog_far_from=50.0); check_metricfix(ok)
    check_metricfix(_cfg()); check_metricfix(_cfg(metric_silog_far_weight=1.0))
    bad = [dict(metric_silog_far_from=50.0),                                       # alone: no-op the author should not believe in
           dict(metric_silog_far_weight=4.0),                                      # far_weight != 1 without far_from: no hidden default
           dict(metric_silog_far_weight=0.0, metric_silog_far_from=50.0),
           dict(metric_silog_far_weight=-2.0, metric_silog_far_from=50.0),
           dict(metric_silog_far_weight=True, metric_silog_far_from=50.0),
           dict(metric_silog_far_weight=4.0, metric_silog_far_from=0.0),
           dict(metric_silog_far_weight=4.0, metric_silog_far_from="50")]
    for b in bad:
        with pytest.raises(ValueError):
            check_metricfix(_cfg(**b))
    with pytest.raises(ValueError, match="metric_silog_weight"):
        check_metricfix(_cfg(metric_silog_weight=0.0, metric_silog_far_weight=4.0, metric_silog_far_from=50.0))
    with pytest.raises(ValueError):
        silog_far_weight_cfg({"metric_silog_far_weight": 4.0})
    assert silog_far_weight_cfg({}) is None and silog_far_weight_cfg({"metric_silog_far_weight": 1.0}) is None
    assert silog_far_weight_cfg({"metric_silog_far_weight": 4.0, "metric_silog_far_from": 50.0}) == (4.0, 50.0)
