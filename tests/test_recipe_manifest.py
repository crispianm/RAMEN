import hashlib

import numpy as np
import pytest
import torch

from data.video_clip_dataset import _degrade_clip
from evaluation.alignment_free import (FrameManifest, RecipeManifest, load_frame_source,
                                       update_pixel_digest, write_frame_manifest, write_recipe_manifest)


def _synthetic_video(video_id: str, n_frames: int, seed: int):
    rng = np.random.RandomState(seed)
    frames = [rng.rand(3, 4, 4).astype(np.float32) for _ in range(n_frames)]
    gts = [rng.rand(4, 4).astype(np.float32) + 0.1 for _ in range(n_frames)]
    masks = [np.ones((4, 4), dtype=bool) for _ in range(n_frames)]
    return frames, gts, masks


def _digest_video(frames, gts, masks) -> str:
    """Uses the same update_pixel_digest helper RecipeManifest._verify calls, not a
    reimplementation -- a divergent test digest would make every test in this file
    prove nothing about whether the two halves (builder and reader) actually agree."""
    d = hashlib.sha256()
    for f, g, m in zip(frames, gts, masks):
        update_pixel_digest(d, f, g, m)
    return d.hexdigest()


def _range_callback(videos):
    """A regenerate_range callback (video_id, start, end) -> rows over an in-memory
    dict of (frames, gts, masks) tuples, tracking every call it receives so tests
    can assert on chunk boundaries."""
    calls = []

    def regenerate_range(vid, start, end):
        calls.append((vid, start, end))
        frames, gts, masks = videos[vid]
        return list(zip(frames[start:end], gts[start:end], masks[start:end]))

    return regenerate_range, calls


def test_recipe_manifest_matches_materialized_frame_manifest(tmp_path):
    """The core proof of equivalence: regenerating via RecipeManifest must produce
    byte-identical arrays to a materialized FrameManifest built from the same data."""
    videos = {"v0": _synthetic_video("v0", 3, seed=1), "v1": _synthetic_video("v1", 2, seed=2)}

    pixel_rows = []
    for vid, (frames, gts, masks) in videos.items():
        for t, (f, g, m) in enumerate(zip(frames, gts, masks)):
            pixel_rows.append((vid, float(t), f, g, m))
    pixel_path = tmp_path / "pixels.npz"
    write_frame_manifest(pixel_path, pixel_rows)

    digests = {vid: _digest_video(*data) for vid, data in videos.items()}
    recipe_path = tmp_path / "recipe.npz"
    video_ids, frame_indices = zip(*[(vid, float(t)) for vid, data in videos.items() for t in range(len(data[0]))])
    write_recipe_manifest(recipe_path, video_ids, frame_indices, root=str(tmp_path), digests=digests)

    regenerate_range, _ = _range_callback(videos)
    fm = FrameManifest(pixel_path)
    rm = RecipeManifest(recipe_path, regenerate_range, chunk_size=2)
    assert len(fm) == len(rm)
    assert fm.video_id == rm.video_id
    for row in range(len(fm)):
        np.testing.assert_array_equal(fm.frame(row), rm.frame(row))
        np.testing.assert_array_equal(fm.gt(row), rm.gt(row))
        np.testing.assert_array_equal(fm.mask(row), rm.mask(row))


def test_recipe_manifest_never_holds_more_than_one_chunk(tmp_path):
    """The regression this whole rewrite exists to prevent: a long video must be
    served in bounded chunks, not materialised whole. Uses a 20-frame video with
    chunk_size=3 and checks every regenerate_range call requested <= chunk_size
    rows, including during the verify pass."""
    videos = {"long": _synthetic_video("long", 20, seed=9)}
    digests = {"long": _digest_video(*videos["long"])}
    recipe_path = tmp_path / "recipe.npz"
    write_recipe_manifest(recipe_path, ["long"] * 20, list(range(20)), root=str(tmp_path), digests=digests)

    regenerate_range, calls = _range_callback(videos)
    rm = RecipeManifest(recipe_path, regenerate_range, chunk_size=3)
    for row in range(20):
        rm.frame(row)
    assert calls, "expected at least one regenerate_range call"
    assert all(end - start <= 3 for _, start, end in calls)
    # verify pass (7 chunks of <=3 over 20 frames) + serve pass chunks, never one
    # single (0, 20) call anywhere.
    assert (0, 20) not in [(s, e) for _, s, e in calls]


def test_recipe_manifest_matches_with_degraded_and_forced_blur(tmp_path):
    """Same proof, but through the degradation path with a forced blur assignment --
    the case the stratified-blur pre-registration (§1) actually needs verified."""
    clean_frames, gts, masks = _synthetic_video("v0", 4, seed=3)
    clip = torch.from_numpy(np.stack(clean_frames, axis=0))  # (T, 3, H, W) -- _degrade_clip's own
    # documented shape (its docstring, and the blur loop's `for t in range(out.shape[0])`, both
    # assume T is axis 0); an earlier version of this test built (3, T, H, W) instead, which still
    # passed because both sides of the comparison used the same wrong shape, self-consistently --
    # it proved nothing about the shape the real pipeline actually produces.

    def make_degraded(force_blur):
        rng = np.random.RandomState(42)
        degraded, _ = _degrade_clip(clip, rng, mode="full", severity=1.0, force_blur=force_blur)
        return [degraded[t].numpy() for t in range(degraded.shape[0])]

    degraded_frames = make_degraded(force_blur=True)
    digest = _digest_video(degraded_frames, gts, masks)

    recipe_path = tmp_path / "recipe_degraded.npz"
    write_recipe_manifest(recipe_path, ["v0"] * 4, [0.0, 1.0, 2.0, 3.0], root=str(tmp_path),
                          digests={"v0": digest})

    def regenerate_range(vid, start, end):
        assert vid == "v0"
        frames = make_degraded(force_blur=True)
        return list(zip(frames[start:end], gts[start:end], masks[start:end]))

    rm = RecipeManifest(recipe_path, regenerate_range, chunk_size=4)
    for row in range(4):
        np.testing.assert_array_equal(rm.frame(row), degraded_frames[row])

    # A different force_blur must NOT reproduce the same digest -- proves the digest
    # is actually sensitive to the degradation outcome, not a no-op check.
    other_frames = make_degraded(force_blur=False)
    assert _digest_video(other_frames, gts, masks) != digest


def test_recipe_manifest_digest_mismatch_hard_stops_and_names_the_video(tmp_path):
    frames, gts, masks = _synthetic_video("v0", 2, seed=5)
    recipe_path = tmp_path / "recipe.npz"
    write_recipe_manifest(recipe_path, ["v0", "v0"], [0.0, 1.0], root=str(tmp_path),
                          digests={"v0": "0" * 64})  # deliberately wrong

    regenerate_range, _ = _range_callback({"v0": (frames, gts, masks)})
    rm = RecipeManifest(recipe_path, regenerate_range)
    with pytest.raises(ValueError, match=r"DIGEST MISMATCH.*'v0'"):
        rm.frame(0)


def test_recipe_manifest_callback_wrong_row_count_raises_naming_video(tmp_path):
    frames, gts, masks = _synthetic_video("v0", 4, seed=6)
    digest = _digest_video(frames, gts, masks)
    recipe_path = tmp_path / "recipe.npz"
    write_recipe_manifest(recipe_path, ["v0"] * 4, list(range(4)), root=str(tmp_path), digests={"v0": digest})

    def bad_regenerate_range(vid, start, end):
        return list(zip(frames[start:end], gts[start:end], masks[start:end]))[:-1]  # drops one row

    rm = RecipeManifest(recipe_path, bad_regenerate_range, chunk_size=4)
    with pytest.raises(ValueError, match=r"returned 3 rows, expected 4.*'v0'|'v0'.*returned 3 rows"):
        rm.frame(0)


def test_recipe_manifest_revisiting_an_evicted_video_raises(tmp_path):
    videos = {"v0": _synthetic_video("v0", 2, seed=1), "v1": _synthetic_video("v1", 2, seed=2)}
    digests = {vid: _digest_video(*data) for vid, data in videos.items()}
    recipe_path = tmp_path / "recipe.npz"
    write_recipe_manifest(recipe_path, ["v0", "v0", "v1", "v1"], [0, 1, 0, 1], root=str(tmp_path), digests=digests)

    regenerate_range, _ = _range_callback(videos)
    rm = RecipeManifest(recipe_path, regenerate_range)
    rm.frame(0)  # verifies+serves v0
    rm.frame(2)  # moves on to v1, evicting v0
    with pytest.raises(ValueError, match="already consumed and evicted"):
        rm.frame(0)  # v0 again -- must raise, not silently re-verify


def test_write_recipe_manifest_rejects_video_without_a_digest(tmp_path):
    with pytest.raises(ValueError, match="no digest recorded"):
        write_recipe_manifest(tmp_path / "recipe.npz", ["v0", "v1"], [0.0, 0.0], root=str(tmp_path),
                              digests={"v0": "a" * 64})  # v1 missing


def test_load_frame_source_dispatches_by_content_not_a_flag(tmp_path):
    write_frame_manifest(tmp_path / "pixels.npz", [("v0", 0.0, np.zeros((3, 2, 2), np.float32),
                                                    np.ones((2, 2), np.float32), np.ones((2, 2), bool))])
    assert isinstance(load_frame_source(tmp_path / "pixels.npz"), FrameManifest)

    write_recipe_manifest(tmp_path / "recipe.npz", ["v0"], [0.0], root=str(tmp_path), digests={"v0": "a" * 64})
    src = load_frame_source(tmp_path / "recipe.npz", regenerate_range=lambda vid, s, e: [])
    assert isinstance(src, RecipeManifest)

    with pytest.raises(ValueError, match="regenerate_range must be supplied"):
        load_frame_source(tmp_path / "recipe.npz")


def test_force_blur_none_preserves_existing_unforced_behaviour():
    """Additive-only check: force_blur=None (the default) must reproduce byte-identical
    output to the pre-existing call signature, for the same seed."""
    clean = torch.rand(2, 3, 5, 5)  # (T, C, H, W), matching _degrade_clip's documented shape
    out_a, params_a = _degrade_clip(clean.clone(), np.random.RandomState(7), mode="full", severity=0.5)
    out_b, params_b = _degrade_clip(clean.clone(), np.random.RandomState(7), mode="full", severity=0.5,
                                    force_blur=None)
    torch.testing.assert_close(out_a, out_b)
    np.testing.assert_array_equal(params_a, params_b)


def test_force_blur_overrides_outcome_without_changing_rng_consumption():
    """The forced-True and forced-False branches must differ only in blur_sigma/output,
    not in the OTHER drawn params (s/gamma/gain/sigma_read) -- proving the RNG stream
    position used by the rest of the call is unaffected by force_blur."""
    clean = torch.rand(2, 3, 5, 5)  # (T, C, H, W), matching _degrade_clip's documented shape
    _, params_true = _degrade_clip(clean.clone(), np.random.RandomState(11), mode="full",
                                   severity=0.9, force_blur=True)
    _, params_false = _degrade_clip(clean.clone(), np.random.RandomState(11), mode="full",
                                    severity=0.9, force_blur=False)
    s_true, gamma_true, gain_true, sigma_r_true, blur_true = params_true
    s_false, gamma_false, gain_false, sigma_r_false, blur_false = params_false
    assert s_true == pytest.approx(s_false)
    assert gamma_true == pytest.approx(gamma_false)
    assert gain_true == pytest.approx(gain_false)
    assert sigma_r_true == pytest.approx(sigma_r_false)
    assert blur_true > 0.0
    assert blur_false == 0.0
