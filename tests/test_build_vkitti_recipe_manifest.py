import numpy as np
import pytest
from PIL import Image

from evaluation.alignment_free import RecipeManifest
from scripts.build_vkitti_recipe_manifest import (
    SKY_DEPTH_M, build_manifest, build_regenerate_range, scene_frame_pairs,
)


def _write_scene(root, scene, n_rgb, n_depth, depth_cm_fn, variant="clone", camera="Camera_0",
                 h=40, w=60):
    """depth_cm_fn(frame_index) -> (h, w) uint16 array of centimetre depth values."""
    rgb_dir = root / scene / variant / "frames" / "rgb" / camera
    dep_dir = root / scene / variant / "frames" / "depth" / camera
    rgb_dir.mkdir(parents=True)
    dep_dir.mkdir(parents=True)
    for i in range(n_rgb):
        Image.fromarray((np.random.rand(h, w, 3) * 255).astype(np.uint8)).save(
            rgb_dir / f"rgb_{i:05d}.jpg")
    for i in range(n_depth):
        Image.fromarray(depth_cm_fn(i).astype(np.uint16)).save(dep_dir / f"depth_{i:05d}.png")


def _uniform_depth_cm(value_m):
    def fn(i):
        return np.full((40, 60), int(value_m * 100), dtype=np.uint16)
    return fn


def test_scene_frame_pairs_positional_no_timestamp_association(tmp_path):
    """VKITTI needs no timestamp association -- rgb[i] pairs with depth[i] by
    sorted positional index, unlike Bonn's TUM timestamps."""
    _write_scene(tmp_path, "Scene01", n_rgb=5, n_depth=5, depth_cm_fn=_uniform_depth_cm(10.0))
    pairs = scene_frame_pairs(tmp_path, "Scene01")
    assert len(pairs) == 5
    for rgb_path, dep_path in pairs:
        assert rgb_path.name.startswith("rgb_")
        assert dep_path.name.startswith("depth_")


def test_scene_frame_pairs_truncates_to_the_shorter_count(tmp_path):
    """The positional equivalent of Bonn's file-existence hard-stop: frames
    without both an rgb and depth file present are dropped by truncation."""
    _write_scene(tmp_path, "Scene02", n_rgb=7, n_depth=4, depth_cm_fn=_uniform_depth_cm(10.0))
    pairs = scene_frame_pairs(tmp_path, "Scene02")
    assert len(pairs) == 4


def test_mask_excludes_non_positive_depth(tmp_path):
    _write_scene(tmp_path, "SceneZero", n_rgb=1, n_depth=1, depth_cm_fn=_uniform_depth_cm(0.0))
    regenerate_range = build_regenerate_range("clean", root=tmp_path)
    (_, gt, mask), = regenerate_range("SceneZero", 0, 1)
    assert not mask.any()


def test_mask_keeps_real_geometry_beyond_max_depth_cap(tmp_path):
    """The load-bearing convention (RAMEN Opus's review, verified against
    data/vkitti_dataset.py directly): the mask excludes ONLY the sky sentinel
    and non-positive depth. Real geometry beyond the 80m max_depth cap (e.g.
    100m, still well under the 655m sky sentinel) stays VALID in the mask with
    its true depth -- the 80m cap is applied at scoring time, not in the mask.
    A mask that excluded >80m pixels would silently change every VKITTI number
    without looking like an error."""
    _write_scene(tmp_path, "SceneFar", n_rgb=1, n_depth=1, depth_cm_fn=_uniform_depth_cm(100.0))
    regenerate_range = build_regenerate_range("clean", root=tmp_path)
    (_, gt, mask), = regenerate_range("SceneFar", 0, 1)
    assert mask.all()
    np.testing.assert_allclose(gt, 100.0, atol=1e-3)


def test_mask_excludes_the_sky_sentinel_not_admits_it_as_valid_geometry(tmp_path):
    """65535cm (~655.35m) is VKITTI's sky/no-return sentinel -- must be masked
    out, not admitted as a real 655m-distant surface."""
    _write_scene(tmp_path, "SceneSky", n_rgb=1, n_depth=1, depth_cm_fn=_uniform_depth_cm(655.35))
    regenerate_range = build_regenerate_range("clean", root=tmp_path)
    (_, gt, mask), = regenerate_range("SceneSky", 0, 1)
    assert not mask.any()
    assert gt.max() <= SKY_DEPTH_M  # clipped, never exceeds the sentinel


def test_build_regenerate_range_clean_is_chunk_independent(tmp_path):
    _write_scene(tmp_path, "Scene01", n_rgb=6, n_depth=6, depth_cm_fn=_uniform_depth_cm(10.0))
    regen_a = build_regenerate_range("clean", root=tmp_path)
    whole = regen_a("Scene01", 0, 6)
    regen_b = build_regenerate_range("clean", root=tmp_path)
    chunked = regen_b("Scene01", 0, 2) + regen_b("Scene01", 2, 6)
    for (f1, g1, m1), (f2, g2, m2) in zip(whole, chunked):
        np.testing.assert_array_equal(f1, f2)
        np.testing.assert_array_equal(g1, g2)
        np.testing.assert_array_equal(m1, m2)


def test_build_regenerate_range_degraded_gt_and_mask_stay_clean(tmp_path):
    """docs/DEGRADED_EVAL_PROTOCOL.md §3's hard precondition, checked for the
    property the recipe design actually guarantees (RAMEN Opus's review):
    GT/mask byte-identical between clean and degraded, frame differs."""
    _write_scene(tmp_path, "Scene02", n_rgb=4, n_depth=4, depth_cm_fn=_uniform_depth_cm(20.0))
    clean_rows = build_regenerate_range("clean", root=tmp_path)("Scene02", 0, 4)
    degraded_rows = build_regenerate_range("degraded", root=tmp_path, severity=1.0)("Scene02", 0, 4)
    for (f1, g1, m1), (f2, g2, m2) in zip(clean_rows, degraded_rows):
        np.testing.assert_array_equal(g1, g2)
        np.testing.assert_array_equal(m1, m2)
        assert not np.array_equal(f1, f2)


def test_build_regenerate_range_degraded_third_pass_from_zero_raises(tmp_path):
    _write_scene(tmp_path, "Scene18", n_rgb=3, n_depth=3, depth_cm_fn=_uniform_depth_cm(10.0))
    regen = build_regenerate_range("degraded", root=tmp_path, severity=1.0)
    regen("Scene18", 0, 3)
    regen("Scene18", 0, 3)
    with pytest.raises(ValueError, match="only ever makes 2 passes"):
        regen("Scene18", 0, 3)


def test_build_manifest_round_trips_through_recipe_manifest(tmp_path):
    _write_scene(tmp_path, "Scene20", n_rgb=5, n_depth=5, depth_cm_fn=_uniform_depth_cm(10.0))
    out_path = tmp_path / "manifest.npz"
    stats = build_manifest("degraded", out_path, ["Scene20"], root=tmp_path, severity=1.0)
    assert stats["frame_counts"]["Scene20"] == 5
    assert stats["builder_commit"]

    regenerate_range = build_regenerate_range("degraded", root=tmp_path, severity=1.0)
    rm = RecipeManifest(out_path, regenerate_range, chunk_size=2)
    assert len(rm) == 5
    for row in range(5):
        rm.frame(row)


def test_build_manifest_digest_mismatch_if_severity_differs_at_read_time(tmp_path):
    _write_scene(tmp_path, "Scene01", n_rgb=3, n_depth=3, depth_cm_fn=_uniform_depth_cm(10.0))
    out_path = tmp_path / "manifest6.npz"
    build_manifest("degraded", out_path, ["Scene01"], root=tmp_path, severity=1.0)

    wrong_severity_regen = build_regenerate_range("degraded", root=tmp_path, severity=0.5)
    rm = RecipeManifest(out_path, wrong_severity_regen)
    with pytest.raises(ValueError, match="DIGEST MISMATCH"):
        rm.frame(0)
