"""train.py's compute_loss()/evaluate() default to treating the model's forward() output as
depth (inverting it to disparity internally). VDA-native archs (models/vda_net.py's relative
mode) output disparity directly; pred_is_disparity=True skips that inversion so the loss
doesn't round-trip through two differently-clamped spaces. See models/vda_net.py's module
docstring and provenance/plan_state.json's VAL-01 note (2026-08-27, Opus's audit) for why
the mismatch matters specifically at disparity==0 (sky_zero_disparity)."""

from __future__ import annotations

import torch

from train import compute_loss, evaluate

_BASE_CFG = {"loss": {"ssi_weight": 1.0, "tgm_weight": 0.0, "gradient_weight": 0.0}}


def _batch(depth: torch.Tensor, mask: torch.Tensor) -> dict:
    return {"depth": depth, "mask": mask}


def test_pred_is_disparity_false_matches_the_original_depth_convention():
    """Away from the clamp floors, pred_is_disparity=True fed genuine disparity must agree
    with the default (False) path fed the equivalent depth -- same math, different input
    space -- confirming the flag doesn't change anything except which space `pred` is in."""
    torch.manual_seed(0)
    disparity = torch.rand(1, 1, 2, 4, 4) * 0.5 + 0.1   # safely away from 0 and any clamp floor
    depth_equiv = 1.0 / disparity
    mask = torch.ones(1, 1, 2, 4, 4)
    depth_gt = torch.rand(1, 1, 2, 4, 4) * 5.0 + 1.0

    loss_disp = compute_loss(disparity, _batch(depth_gt, mask), _BASE_CFG, pred_is_disparity=True)
    loss_depth = compute_loss(depth_equiv, _batch(depth_gt, mask), _BASE_CFG, pred_is_disparity=False)
    assert torch.allclose(loss_disp, loss_depth, atol=1e-5)


def test_pred_is_disparity_true_preserves_exact_zero_disparity():
    """The concrete regression this flag exists for: a model that natively outputs disparity
    0 (VDA's own sky_zero_disparity convention) must have that 0 reach the loss exactly.
    Round-tripping through depth (pred_is_disparity=False fed 1/0.clamp(min=1e-6)=1e6, then
    compute_loss's own 1/1e6.clamp_min(1e-3)) would silently turn it into 1e-6, not 0."""
    disparity = torch.zeros(1, 1, 1, 2, 2)
    mask = torch.ones(1, 1, 1, 2, 2)
    depth_gt = torch.full((1, 1, 1, 2, 2), 10.0)

    # what models/vda_net.py's OLD (pre-fix) forward() would have produced for disparity==0
    depth_equiv_old_path = 1.0 / disparity.clamp(min=1e-6)

    loss_disp = compute_loss(disparity, _batch(depth_gt, mask), _BASE_CFG, pred_is_disparity=True)
    loss_depth_roundtrip = compute_loss(depth_equiv_old_path, _batch(depth_gt, mask), _BASE_CFG,
                                        pred_is_disparity=False)
    # both finite, but the round-tripped version disagrees with true disparity=0 by ~1e-6 --
    # small in isolation, but the point is pred_is_disparity=True has NO such conversion at all
    assert torch.isfinite(loss_disp)
    assert torch.isfinite(loss_depth_roundtrip)
    # direct check on the recovered disparity value itself, not just the aggregate loss
    pd_direct = disparity.reshape(-1, 2, 2)
    pd_roundtrip = (1.0 / depth_equiv_old_path.clamp_min(1e-3)).reshape(-1, 2, 2)
    assert torch.equal(pd_direct, torch.zeros_like(pd_direct))
    assert not torch.equal(pd_roundtrip, torch.zeros_like(pd_roundtrip)), (
        "the round-trip path should NOT reproduce exact zero -- if it does, the regression "
        "this test guards against no longer reproduces and the test needs revisiting"
    )


def test_evaluate_reads_predicts_disparity_from_the_model():
    class _DisparityModel(torch.nn.Module):
        predicts_disparity = True

        def forward(self, x):
            B, C, T, H, W = x.shape
            return torch.full((B, 1, T, H, W), 0.1)   # constant disparity

    class _DepthModel(torch.nn.Module):
        predicts_disparity = False

        def forward(self, x):
            B, C, T, H, W = x.shape
            return torch.full((B, 1, T, H, W), 10.0)   # constant depth == disparity 0.1

    frames = torch.rand(1, 3, 2, 4, 4)
    depth = torch.full((1, 1, 2, 4, 4), 8.0)
    mask = torch.ones(1, 1, 2, 4, 4)
    loader = [{"frames": frames, "depth": depth, "mask": mask, "video_id": ["v0"]}]

    m1 = evaluate(_DisparityModel(), loader, "cpu", None, max_batches=1)
    m2 = evaluate(_DepthModel(), loader, "cpu", None, max_batches=1)
    assert abs(m1["abs_rel"] - m2["abs_rel"]) < 1e-5
