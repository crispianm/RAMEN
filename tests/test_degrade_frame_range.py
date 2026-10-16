import numpy as np
import pytest
import torch

from data.video_clip_dataset import (degrade_frame_range, endpoint_degradation_seed,
                                     stratified_blur_assignment)


def _clean_frames(n, seed):
    rng = np.random.RandomState(seed)
    return torch.from_numpy(rng.rand(n, 3, 16, 12).astype(np.float32))


def _degrade_in_ranges(clean, seed, severity, blurred, ranges):
    """Call degrade_frame_range once per (lo, hi) in ranges, sharing one rng
    across all calls (as RecipeManifest's regenerate_range contract requires),
    and concatenate the results (frames and per-call profiles both)."""
    rng = np.random.RandomState(seed)
    parts, profiles = [], []
    for lo, hi in ranges:
        out, params = degrade_frame_range(clean[lo:hi], rng, severity, blurred)
        parts.append(out)
        profiles.append(params)
    return torch.cat(parts, dim=0), profiles


def test_endpoint_degradation_seed_matches_the_preregistered_formula():
    import hashlib
    got = endpoint_degradation_seed("bonn", "rgbd_bonn_balloon", "full", 1.0)
    key = "bonn|rgbd_bonn_balloon|full|1.0"
    expected = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    assert got == expected


def test_endpoint_degradation_seed_is_sensitive_to_every_component():
    base = endpoint_degradation_seed("bonn", "seq0", "full", 1.0)
    assert endpoint_degradation_seed("vkitti", "seq0", "full", 1.0) != base
    assert endpoint_degradation_seed("bonn", "seq1", "full", 1.0) != base
    assert endpoint_degradation_seed("bonn", "seq0", "full", 0.5) != base


def test_degrade_frame_range_is_independent_of_chunk_boundaries():
    """The property this whole function exists for: whatever ranges the caller
    splits a video into, sharing one rng across the calls, the output for a
    given video must be identical -- required for RecipeManifest's read-side
    chunk_size to be free to differ from whatever the builder used."""
    clean = _clean_frames(11, seed=1)
    seed = endpoint_degradation_seed("bonn", "test_video", "full", 1.0)

    whole, _ = _degrade_in_ranges(clean, seed, 1.0, True, [(0, 11)])
    in_threes, _ = _degrade_in_ranges(clean, seed, 1.0, True, [(0, 3), (3, 6), (6, 9), (9, 11)])
    uneven, _ = _degrade_in_ranges(clean, seed, 1.0, True, [(0, 1), (1, 7), (7, 11)])

    torch.testing.assert_close(whole, in_threes)
    torch.testing.assert_close(whole, uneven)


def test_degrade_frame_range_profile_is_constant_across_every_call_for_a_video():
    """docs/DEGRADED_EVAL_PROTOCOL.md §1's one-profile-per-video invariant, now
    enforced by degrade_frame_range instead of _degrade_clip (§4's per-frame-
    derivation amendment): severity, gamma, gain and blur_sigma must be IDENTICAL
    across every range of a given video, since they are fixed inputs to every call,
    not redrawn -- only the per-pixel noise realisation varies frame to frame."""
    clean = _clean_frames(9, seed=6)
    seed = endpoint_degradation_seed("bonn", "profile_check_video", "full", 1.0)
    _, profiles = _degrade_in_ranges(clean, seed, 1.0, True, [(0, 2), (2, 5), (5, 9)])
    for p in profiles[1:]:
        np.testing.assert_array_equal(p, profiles[0])


def test_degrade_frame_range_unblurred_is_also_chunk_independent():
    clean = _clean_frames(7, seed=2)
    seed = endpoint_degradation_seed("bonn", "unblurred_video", "full", 0.6)
    whole, _ = _degrade_in_ranges(clean, seed, 0.6, False, [(0, 7)])
    chunked, _ = _degrade_in_ranges(clean, seed, 0.6, False, [(0, 2), (2, 5), (5, 7)])
    torch.testing.assert_close(whole, chunked)


def test_degrade_frame_range_blurred_vs_unblurred_actually_differ():
    clean = _clean_frames(5, seed=3)
    seed_a = endpoint_degradation_seed("bonn", "v_blur_check", "full", 1.0)
    blurred, _ = degrade_frame_range(clean, np.random.RandomState(seed_a), 1.0, True)
    unblurred, _ = degrade_frame_range(clean, np.random.RandomState(seed_a), 1.0, False)
    assert not torch.allclose(blurred, unblurred)


def test_degrade_frame_range_severity_changes_output():
    clean = _clean_frames(4, seed=4)
    seed_a = endpoint_degradation_seed("bonn", "v_severity_check", "full", 1.0)
    mild, _ = degrade_frame_range(clean, np.random.RandomState(seed_a), 0.2, False)
    severe, _ = degrade_frame_range(clean, np.random.RandomState(seed_a), 1.8, False)
    assert not torch.allclose(mild, severe)


def test_degrade_frame_range_output_stays_in_unit_range():
    clean = _clean_frames(6, seed=5)
    seed_a = endpoint_degradation_seed("bonn", "v_range_check", "full", 2.0)
    out, _ = degrade_frame_range(clean, np.random.RandomState(seed_a), 2.0, True)
    assert out.min() >= 0.0
    assert out.max() <= 1.0
    assert out.shape == clean.shape


@pytest.mark.parametrize("n, expected_k", [(5, 4), (25, 18), (13, 9)])
def test_stratified_blur_assignment_matches_preregistered_counts(n, expected_k):
    """docs/DEGRADED_EVAL_PROTOCOL.md §1's literal counts: 4/5 (VKITTI-5scene),
    18/25 (Bonn primary), 9/13 (KITTI 13-drive) -- the first two are exact ties
    broken by rounding up, the third has no tie."""
    video_ids = [f"seq{i:02d}" for i in range(n)]
    assignment = stratified_blur_assignment(video_ids)
    assert sum(assignment.values()) == expected_k


def test_stratified_blur_assignment_is_first_k_by_sorted_id():
    video_ids = ["z", "b", "m", "a", "y"]
    assignment = stratified_blur_assignment(video_ids)  # n=5 -> k=4
    ordered = sorted(video_ids)
    blurred = {vid for vid, is_blurred in assignment.items() if is_blurred}
    assert blurred == set(ordered[:4])


def test_degrade_frame_range_has_no_mode_argument_and_cannot_reach_elvis():
    """docs/DEGRADED_EVAL_PROTOCOL.md §9's freeze block claims elvis_noise.py's
    hash is not load-bearing for the endpoint protocol, because no builder can
    reach _degrade_clip's mode="elvis" branch. That claim needs a check, not
    just a paragraph (RAMEN Opus's review): degrade_frame_range has NO `mode`
    parameter at all -- it always implements _degrade_clip's mode="full"
    formulas inline, with no branch to any other mode -- so this asserts the
    function's actual signature, which is the real mechanism providing the
    guarantee. If a future edit ever adds a `mode` parameter, this test forces
    a decision about whether "elvis" must be rejected explicitly."""
    import inspect

    params = inspect.signature(degrade_frame_range).parameters
    assert "mode" not in params, (
        "degrade_frame_range gained a mode parameter -- docs/DEGRADED_EVAL_PROTOCOL.md "
        "§9's elvis-unreachability claim needs re-checking, not just this test updating.")


@pytest.mark.parametrize("builder_module", [
    "scripts.build_bonn_recipe_manifest",
    "scripts.build_vkitti_recipe_manifest",
    "scripts.build_kitti_recipe_manifest",
])
def test_endpoint_builders_never_import_elvis_or_degrade_clip(builder_module):
    """The other half of the same freeze-block claim: no endpoint builder imports
    elvis_noise or _degrade_clip directly (they all use degrade_frame_range,
    which cannot reach either). Checks the module's actual namespace after
    import, not a source-text grep -- a docstring mentioning "_degrade_clip"
    in prose (to explain what degrade_frame_range deliberately does NOT do)
    must not fail this; an actual `from ... import _degrade_clip` must."""
    import importlib

    module = importlib.import_module(builder_module)
    assert not hasattr(module, "_degrade_clip")
    assert not hasattr(module, "sample_profile")
    assert not hasattr(module, "degrade_clip_elvis")
