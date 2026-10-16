import numpy as np
import pytest
from PIL import Image

from evaluation.alignment_free import RecipeManifest
from scripts.build_kitti_val_recipe_manifest import (
    _video_id, build_manifest, build_regenerate_range, resolved_pairs,
)


def _image_name(drive, frame, cam="02"):
    return f"{drive}_image_{frame:010d}_image_{cam}.png"


def _depth_name(drive, frame, cam="02"):
    return f"{drive}_groundtruth_depth_{frame:010d}_image_{cam}.png"


def _write_val_selection(root, n, depth_value_m=15.0, extra_files=0, drive="2011_09_26_drive_0002_sync"):
    """Filenames match the REAL val_selection_cropped naming pattern exactly
    (image/..._image_<frame>_image_0N.png vs
    groundtruth_depth/..._groundtruth_depth_<frame>_image_0N.png) -- both
    so these tests exercise positional pairing the way real data requires,
    and so the correspondence check (drive/frame/camera extracted from both
    names) has something real to check against."""
    img_dir = root / "image"
    depth_dir = root / "groundtruth_depth"
    img_dir.mkdir(parents=True)
    depth_dir.mkdir(parents=True)
    for i in range(n + extra_files):
        Image.fromarray((np.random.rand(20, 24, 3) * 255).astype(np.uint8)).save(
            img_dir / _image_name(drive, i))
        Image.fromarray(np.full((20, 24), int(depth_value_m * 256), dtype=np.uint16)).save(
            depth_dir / _depth_name(drive, i))


def _first_video_id(root):
    pairs = resolved_pairs(root, n=20)
    return _video_id(pairs[0][0])


def test_resolved_pairs_is_first_n_by_independently_sorted_globs(tmp_path):
    _write_val_selection(tmp_path, n=20, extra_files=5)
    pairs = resolved_pairs(tmp_path, n=20)
    assert len(pairs) == 20
    img_names = [p[0] for p in pairs]
    depth_names = [p[1] for p in pairs]
    assert img_names == sorted(img_names)
    assert depth_names == sorted(depth_names)
    # Positional correspondence: the i-th image pairs with the i-th depth file
    # by SORT ORDER, not by any shared substring in their names.
    assert img_names[0] == _image_name("2011_09_26_drive_0002_sync", 0)
    assert depth_names[0] == _depth_name("2011_09_26_drive_0002_sync", 0)


def test_resolved_pairs_raises_on_correspondence_mismatch(tmp_path):
    """The load-bearing check (RAMEN Opus's review): positional pairing is
    exactly Bonn's original defect, so it is checked here rather than
    assumed. Two drives, sorted so their image/depth files interleave out of
    correspondence -- e.g. drive A's frame 1 image lands next to drive B's
    frame 0 depth once both globs are sorted independently -- must raise
    naming both filenames, not silently pair the wrong frames."""
    img_dir = tmp_path / "image"
    depth_dir = tmp_path / "groundtruth_depth"
    img_dir.mkdir(parents=True)
    depth_dir.mkdir(parents=True)
    # Image side: drive A frames 0-1. Depth side: drive B frames 0-1 -- same
    # frame numbers, different drive, so sorted-glob position 0 still pairs
    # frame 0 with frame 0, but the DRIVE does not correspond.
    for i in range(2):
        Image.fromarray((np.random.rand(20, 24, 3) * 255).astype(np.uint8)).save(
            img_dir / _image_name("driveA", i))
        Image.fromarray(np.full((20, 24), 15 * 256, dtype=np.uint16)).save(
            depth_dir / _depth_name("driveB", i))
    with pytest.raises(ValueError, match="does not correspond"):
        resolved_pairs(tmp_path, n=2)


def test_identity_raises_on_unrecognised_filename(tmp_path):
    img_dir = tmp_path / "image"
    depth_dir = tmp_path / "groundtruth_depth"
    img_dir.mkdir(parents=True)
    depth_dir.mkdir(parents=True)
    Image.fromarray((np.random.rand(20, 24, 3) * 255).astype(np.uint8)).save(
        img_dir / "not_the_right_pattern.png")
    Image.fromarray(np.full((20, 24), 15 * 256, dtype=np.uint16)).save(
        depth_dir / _depth_name("2011_09_26_drive_0002_sync", 0))
    with pytest.raises(ValueError, match="does not match the expected"):
        resolved_pairs(tmp_path, n=1)


def test_resolved_pairs_raises_if_not_enough_files(tmp_path):
    _write_val_selection(tmp_path, n=5)
    with pytest.raises(ValueError, match="need 20"):
        resolved_pairs(tmp_path, n=20)


def test_mask_has_no_upper_cap(tmp_path):
    """The fourth mask convention: depth > 1e-3 with NO upper bound at all --
    unlike KITTI-13drive, which bakes the 80m cap into its mask."""
    # 150m: far beyond any max_depth (80m) but within uint16's representable range at
    # the /256 encoding (max ~256m) -- 500m would silently wrap around uint16, which
    # is a fixture bug, not something this test is about.
    _write_val_selection(tmp_path, n=20, depth_value_m=150.0)
    vid = _first_video_id(tmp_path)
    regenerate_range = build_regenerate_range("clean", root=tmp_path)
    (_, gt, mask), = regenerate_range(vid, 0, 1)
    assert mask.all()  # 150m stays valid -- no cap in this mask at all
    np.testing.assert_allclose(gt, 150.0, atol=1)


def test_mask_excludes_non_positive_depth(tmp_path):
    _write_val_selection(tmp_path, n=20, depth_value_m=0.0)
    vid = _first_video_id(tmp_path)
    regenerate_range = build_regenerate_range("clean", root=tmp_path)
    (_, gt, mask), = regenerate_range(vid, 0, 1)
    assert not mask.any()


def test_regenerate_range_rejects_any_range_other_than_0_1(tmp_path):
    _write_val_selection(tmp_path, n=20)
    vid = _first_video_id(tmp_path)
    regenerate_range = build_regenerate_range("clean", root=tmp_path)
    with pytest.raises(ValueError, match="exactly one frame"):
        regenerate_range(vid, 0, 2)


def test_build_regenerate_range_degraded_gt_and_mask_stay_clean(tmp_path):
    """docs/DEGRADED_EVAL_PROTOCOL.md §3's hard precondition, checked for the
    property the recipe design actually guarantees (RAMEN Opus's review):
    GT/mask byte-identical between clean and degraded, frame differs."""
    _write_val_selection(tmp_path, n=20, depth_value_m=15.0)
    vid = _first_video_id(tmp_path)
    clean_rows = build_regenerate_range("clean", root=tmp_path)(vid, 0, 1)
    degraded_rows = build_regenerate_range("degraded", root=tmp_path, severity=1.0)(vid, 0, 1)
    np.testing.assert_array_equal(clean_rows[0][1], degraded_rows[0][1])
    np.testing.assert_array_equal(clean_rows[0][2], degraded_rows[0][2])
    assert not np.array_equal(clean_rows[0][0], degraded_rows[0][0])


def test_build_regenerate_range_degraded_third_pass_raises(tmp_path):
    _write_val_selection(tmp_path, n=20)
    vid = _first_video_id(tmp_path)
    regen = build_regenerate_range("degraded", root=tmp_path, severity=1.0)
    regen(vid, 0, 1)
    regen(vid, 0, 1)
    with pytest.raises(ValueError, match="only ever makes 2 passes"):
        regen(vid, 0, 1)


def test_build_manifest_round_trips_through_recipe_manifest(tmp_path):
    _write_val_selection(tmp_path, n=20)
    out_path = tmp_path / "manifest.npz"
    stats = build_manifest("degraded", out_path, root=tmp_path, severity=1.0)
    assert stats["total_frames"] == 20

    regenerate_range = build_regenerate_range("degraded", root=tmp_path, severity=1.0)
    rm = RecipeManifest(out_path, regenerate_range)
    assert len(rm) == 20
    for row in range(20):
        rm.frame(row)


def test_manifest_records_the_resolved_source_filenames(tmp_path):
    """Population identity is pinned by WHICH files these are, not by the
    sorted-glob rule that selected them (RAMEN Opus's review) -- both the
    image and depth lists, since they're paired positionally, not by name."""
    _write_val_selection(tmp_path, n=20, extra_files=3)
    out_path = tmp_path / "manifest.npz"
    build_manifest("clean", out_path, root=tmp_path)
    data = np.load(out_path, allow_pickle=False)
    assert "source_image_filenames" in data.files
    assert "source_depth_filenames" in data.files
    pairs = resolved_pairs(tmp_path, n=20)
    assert list(data["source_image_filenames"]) == [p[0] for p in pairs]
    assert list(data["source_depth_filenames"]) == [p[1] for p in pairs]


def test_stratified_blur_is_14_of_20(tmp_path):
    _write_val_selection(tmp_path, n=20)
    from data.video_clip_dataset import stratified_blur_assignment
    pairs = resolved_pairs(tmp_path, n=20)
    video_ids = [_video_id(img_fn) for img_fn, _ in pairs]
    assignment = stratified_blur_assignment(video_ids)
    assert sum(assignment.values()) == 14


def test_build_manifest_digest_mismatch_if_severity_differs_at_read_time(tmp_path):
    _write_val_selection(tmp_path, n=20)
    out_path = tmp_path / "manifest.npz"
    build_manifest("degraded", out_path, root=tmp_path, severity=1.0)

    wrong_severity_regen = build_regenerate_range("degraded", root=tmp_path, severity=0.5)
    rm = RecipeManifest(out_path, wrong_severity_regen)
    with pytest.raises(ValueError, match="DIGEST MISMATCH"):
        rm.frame(0)
