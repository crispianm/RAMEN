import numpy as np
import pytest
from PIL import Image

from evaluation.alignment_free import RecipeManifest
from scripts.build_kitti_recipe_manifest import (
    build_manifest, build_regenerate_range, drive_frame_pairs,
)


def _write_drive(root, drive, filenames, depth_cm_fn, h=40, w=64):
    """depth_cm_fn(filename) -> (h, w) uint16 array of raw depth-completion values
    (metres = value / 256.0). filenames is the exact rgb filename list; depth files
    are written for all but any names passed via `missing`."""
    rgb_dir = root / drive / "rgb"
    depth_dir = root / drive / "depth"
    rgb_dir.mkdir(parents=True)
    depth_dir.mkdir(parents=True)
    for fn in filenames:
        Image.fromarray((np.random.rand(h, w, 3) * 255).astype(np.uint8)).save(rgb_dir / fn)
    return rgb_dir, depth_dir


def _uniform_depth_raw(value_m):
    def fn(shape):
        return np.full(shape, int(value_m * 256), dtype=np.uint16)
    return fn


def test_drive_frame_pairs_matches_by_filename(tmp_path):
    filenames = ["0000000005.png", "0000000006.png", "0000000007.png"]
    rgb_dir, depth_dir = _write_drive(tmp_path, "driveA", filenames, None)
    for fn in filenames:
        Image.fromarray(np.full((40, 64), 2560, dtype=np.uint16)).save(depth_dir / fn)
    pairs = drive_frame_pairs(tmp_path, "driveA")
    assert len(pairs) == 3
    for rgb_path, depth_path in pairs:
        assert rgb_path.name == depth_path.name


def test_drive_frame_pairs_raises_on_missing_depth(tmp_path):
    """Unlike Bonn (silently drops) and VKITTI (truncates), a missing depth file
    for an existing rgb frame is a hard failure -- matching
    build_kitti_streaming_manifest.py exactly, since KITTI's depth-completion GT
    is expected to always exist for an extracted frame."""
    filenames = ["0000000005.png", "0000000006.png"]
    rgb_dir, depth_dir = _write_drive(tmp_path, "driveB", filenames, None)
    Image.fromarray(np.full((40, 64), 2560, dtype=np.uint16)).save(depth_dir / filenames[0])
    # filenames[1]'s depth is deliberately never written
    with pytest.raises(ValueError, match="missing depth GT"):
        drive_frame_pairs(tmp_path, "driveB")


def test_mask_bakes_in_the_max_depth_cap(tmp_path):
    """Deliberately DIFFERENT from VKITTI: the 80m cap is baked into the mask
    itself here, matching build_kitti_streaming_manifest.py's own
    (depth > MIN_DEPTH) & (depth < MAX_DEPTH) convention -- geometry beyond 80m
    is INVALID in this mask, not kept valid the way VKITTI keeps it."""
    filenames = ["0000000000.png"]
    rgb_dir, depth_dir = _write_drive(tmp_path, "driveC", filenames, None)
    Image.fromarray(_uniform_depth_raw(100.0)((40, 64))).save(depth_dir / filenames[0])  # 100m > 80m cap
    regenerate_range = build_regenerate_range("clean", root=tmp_path)
    (_, gt, mask), = regenerate_range("driveC", 0, 1)
    assert not mask.any()  # excluded by the cap, unlike VKITTI's equivalent case
    np.testing.assert_allclose(gt, 100.0, atol=1e-2)  # gt itself is NOT clipped, unlike VKITTI


def test_mask_excludes_non_positive_and_keeps_in_range_depth(tmp_path):
    filenames = ["0000000000.png", "0000000001.png"]
    rgb_dir, depth_dir = _write_drive(tmp_path, "driveD", filenames, None)
    Image.fromarray(_uniform_depth_raw(0.0)((40, 64))).save(depth_dir / filenames[0])
    Image.fromarray(_uniform_depth_raw(20.0)((40, 64))).save(depth_dir / filenames[1])
    regenerate_range = build_regenerate_range("clean", root=tmp_path)
    rows = regenerate_range("driveD", 0, 2)
    assert not rows[0][2].any()
    assert rows[1][2].all()
    np.testing.assert_allclose(rows[1][1], 20.0, atol=1e-2)


def test_no_resize_frames_stay_at_native_resolution(tmp_path):
    filenames = ["0000000000.png"]
    _write_drive(tmp_path, "driveE", filenames, None, h=37, w=53)
    depth_dir = tmp_path / "driveE" / "depth"
    Image.fromarray(_uniform_depth_raw(10.0)((37, 53))).save(depth_dir / filenames[0])
    regenerate_range = build_regenerate_range("clean", root=tmp_path)
    (frame, gt, mask), = regenerate_range("driveE", 0, 1)
    assert frame.shape == (3, 37, 53)
    assert gt.shape == (37, 53)
    assert mask.shape == (37, 53)


def test_build_regenerate_range_clean_is_chunk_independent(tmp_path):
    filenames = [f"{i:010d}.png" for i in range(6)]
    rgb_dir, depth_dir = _write_drive(tmp_path, "driveF", filenames, None)
    for fn in filenames:
        Image.fromarray(_uniform_depth_raw(15.0)((40, 64))).save(depth_dir / fn)
    regen_a = build_regenerate_range("clean", root=tmp_path)
    whole = regen_a("driveF", 0, 6)
    regen_b = build_regenerate_range("clean", root=tmp_path)
    chunked = regen_b("driveF", 0, 2) + regen_b("driveF", 2, 6)
    for (f1, g1, m1), (f2, g2, m2) in zip(whole, chunked):
        np.testing.assert_array_equal(f1, f2)
        np.testing.assert_array_equal(g1, g2)
        np.testing.assert_array_equal(m1, m2)


def test_build_regenerate_range_degraded_gt_and_mask_stay_clean(tmp_path):
    """docs/DEGRADED_EVAL_PROTOCOL.md §3's hard precondition, checked for the
    property the recipe design actually guarantees (RAMEN Opus's review):
    GT/mask byte-identical between clean and degraded, frame differs."""
    filenames = [f"{i:010d}.png" for i in range(4)]
    rgb_dir, depth_dir = _write_drive(tmp_path, "driveG", filenames, None)
    for fn in filenames:
        Image.fromarray(_uniform_depth_raw(15.0)((40, 64))).save(depth_dir / fn)
    clean_rows = build_regenerate_range("clean", root=tmp_path)("driveG", 0, 4)
    degraded_rows = build_regenerate_range("degraded", root=tmp_path, severity=1.0)("driveG", 0, 4)
    for (f1, g1, m1), (f2, g2, m2) in zip(clean_rows, degraded_rows):
        np.testing.assert_array_equal(g1, g2)
        np.testing.assert_array_equal(m1, m2)
        assert not np.array_equal(f1, f2)


def test_build_regenerate_range_degraded_third_pass_from_zero_raises(tmp_path):
    filenames = [f"{i:010d}.png" for i in range(3)]
    rgb_dir, depth_dir = _write_drive(tmp_path, "driveH", filenames, None)
    for fn in filenames:
        Image.fromarray(_uniform_depth_raw(15.0)((40, 64))).save(depth_dir / fn)
    regen = build_regenerate_range("degraded", root=tmp_path, severity=1.0)
    regen("driveH", 0, 3)
    regen("driveH", 0, 3)
    with pytest.raises(ValueError, match="only ever makes 2 passes"):
        regen("driveH", 0, 3)


def test_build_manifest_round_trips_through_recipe_manifest(tmp_path):
    filenames = [f"{i:010d}.png" for i in range(5)]
    rgb_dir, depth_dir = _write_drive(tmp_path, "driveI", filenames, None)
    for fn in filenames:
        Image.fromarray(_uniform_depth_raw(15.0)((40, 64))).save(depth_dir / fn)
    out_path = tmp_path / "manifest.npz"
    stats = build_manifest("degraded", out_path, ["driveI"], root=tmp_path, severity=1.0)
    assert stats["frame_counts"]["driveI"] == 5
    assert stats["builder_commit"]

    regenerate_range = build_regenerate_range("degraded", root=tmp_path, severity=1.0)
    rm = RecipeManifest(out_path, regenerate_range, chunk_size=2)
    assert len(rm) == 5
    for row in range(5):
        rm.frame(row)


def test_build_manifest_digest_mismatch_if_severity_differs_at_read_time(tmp_path):
    filenames = [f"{i:010d}.png" for i in range(3)]
    rgb_dir, depth_dir = _write_drive(tmp_path, "driveJ", filenames, None)
    for fn in filenames:
        Image.fromarray(_uniform_depth_raw(15.0)((40, 64))).save(depth_dir / fn)
    out_path = tmp_path / "manifest6.npz"
    build_manifest("degraded", out_path, ["driveJ"], root=tmp_path, severity=1.0)

    wrong_severity_regen = build_regenerate_range("degraded", root=tmp_path, severity=0.5)
    rm = RecipeManifest(out_path, wrong_severity_regen)
    with pytest.raises(ValueError, match="DIGEST MISMATCH"):
        rm.frame(0)


def test_build_regenerate_range_degraded_handles_ragged_frame_shapes_within_a_drive(tmp_path):
    """The reason build_kitti_streaming_manifest.py crops depth to each frame's
    OWN (h, w) rather than a drive-wide constant: real KITTI frames are not
    guaranteed uniform shape frame-to-frame (RAMEN Opus's review). A prior draft
    of this builder batched a whole chunk into one torch.stack before degrading
    it, which would raise on exactly this input. Two frames of genuinely
    different native resolution in the same drive must degrade without error."""
    rgb_dir = tmp_path / "driveRagged" / "rgb"
    depth_dir = tmp_path / "driveRagged" / "depth"
    rgb_dir.mkdir(parents=True)
    depth_dir.mkdir(parents=True)
    shapes = [(40, 64), (37, 59)]  # deliberately different h, w
    for i, (h, w) in enumerate(shapes):
        fn = f"{i:010d}.png"
        Image.fromarray((np.random.rand(h, w, 3) * 255).astype(np.uint8)).save(rgb_dir / fn)
        Image.fromarray(_uniform_depth_raw(15.0)((h, w))).save(depth_dir / fn)

    regenerate_range = build_regenerate_range("degraded", root=tmp_path, severity=1.0)
    rows = regenerate_range("driveRagged", 0, 2)
    assert rows[0][0].shape == (3, 40, 64)
    assert rows[1][0].shape == (3, 37, 59)
