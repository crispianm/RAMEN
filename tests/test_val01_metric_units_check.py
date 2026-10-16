from __future__ import annotations

import sys

import torch

sys.path.insert(0, "scripts")
import val01_metric_units_check as uc  # noqa: E402


def test_pearson_corr_perfect_positive():
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    b = torch.tensor([2.0, 4.0, 6.0, 8.0])
    assert abs(uc.pearson_corr(a, b) - 1.0) < 1e-9


def test_pearson_corr_perfect_negative():
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    b = torch.tensor([8.0, 6.0, 4.0, 2.0])
    assert abs(uc.pearson_corr(a, b) - (-1.0)) < 1e-9


def test_interpret_depth_like_when_positive_depth_corr_and_low_inverse_corr():
    result = uc.interpret({"correlation_with_gt_depth": 0.9, "correlation_with_gt_inverse_depth": 0.05})
    assert result["conclusion"] == "depth_like"
    assert result["matches_if_depth_like"] is True
    assert result["matches_if_disparity_like"] is False


def test_interpret_disparity_like_when_positive_inverse_corr_and_low_depth_corr():
    result = uc.interpret({"correlation_with_gt_depth": 0.02, "correlation_with_gt_inverse_depth": 0.85})
    assert result["conclusion"] == "disparity_like"
    assert result["matches_if_disparity_like"] is True
    assert result["matches_if_depth_like"] is False


def test_interpret_inconclusive_when_both_or_neither_correlate():
    result = uc.interpret({"correlation_with_gt_depth": 0.1, "correlation_with_gt_inverse_depth": 0.1})
    assert result["conclusion"] == "inconclusive"


def test_pre_registration_is_written_before_any_measurement_field_exists():
    """The pre-registration constant is a module-level literal, defined and fully
    populated before run_units_check (the only function that touches real data) exists in
    the file -- this is a structural, not a runtime, guarantee, but the test pins the
    constant's shape so it can't be silently narrowed after the fact."""
    assert set(uc.PRE_REGISTRATION) == {
        "hypothesis_under_test", "if_depth_like", "if_disparity_like", "context_only_not_a_units_measurement",
    }
    assert "scale,shift=1.0,0.0" in uc.PRE_REGISTRATION["context_only_not_a_units_measurement"]
