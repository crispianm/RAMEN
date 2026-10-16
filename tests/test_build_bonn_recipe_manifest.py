import numpy as np
import pytest
from PIL import Image

from evaluation.alignment_free import RecipeManifest
from scripts.build_bonn_recipe_manifest import (
    associate_sequence, build_manifest, build_regenerate_range,
)


def _write_tum_sequence(seq_dir, n_rgb, n_depth, offsets, missing_depth_indices=()):
    """Build a tiny synthetic TUM-style sequence: rgb.txt/depth.txt plus real
    (small) PNG files, with rgb timestamps at 0,1,2,... and depth timestamps
    offset by `offsets[i]` from the matching rgb index (so association is
    checkable), and some depth files deliberately absent from disk."""
    seq_dir.mkdir(parents=True)
    rgb_lines, depth_lines = [], []
    for i in range(n_rgb):
        ts = float(i)
        fn = f"rgb/{i}.png"
        (seq_dir / "rgb").mkdir(exist_ok=True)
        Image.fromarray((np.random.rand(4, 4, 3) * 255).astype(np.uint8)).save(seq_dir / fn)
        rgb_lines.append(f"{ts:.4f} {fn}")
    (seq_dir / "depth").mkdir(exist_ok=True)
    for i in range(n_depth):
        ts = float(i) + (offsets[i] if i < len(offsets) else 0.0)
        fn = f"depth/{i}.png"
        depth_lines.append(f"{ts:.4f} {fn}")
        if i in missing_depth_indices:
            continue
        Image.fromarray((np.random.rand(4, 4) * 5000).astype(np.uint16)).save(seq_dir / fn)
    (seq_dir / "rgb.txt").write_text("\n".join(rgb_lines) + "\n")
    (seq_dir / "depth.txt").write_text("\n".join(depth_lines) + "\n")


def test_associate_sequence_pairs_within_tolerance(tmp_path):
    seq_dir = tmp_path / "seq0"
    _write_tum_sequence(seq_dir, n_rgb=5, n_depth=5, offsets=[0.001] * 5)
    matched = associate_sequence(seq_dir)
    assert len(matched) == 5
    for ts, rf, df in matched:
        assert rf.startswith("rgb/")
        assert df.startswith("depth/")


def test_associate_sequence_drops_pairs_outside_tolerance(tmp_path):
    seq_dir = tmp_path / "seq1"
    # offset 0.05 > MAX_OFFSET (0.02) for every frame -- nothing should match
    _write_tum_sequence(seq_dir, n_rgb=4, n_depth=4, offsets=[0.05] * 4)
    matched = associate_sequence(seq_dir)
    assert len(matched) == 0


def test_associate_sequence_file_existence_hard_stop(tmp_path):
    """A depth.txt entry within tolerance but whose file doesn't exist on disk must
    never be matched -- the hard-stop that runs before the tolerance check."""
    seq_dir = tmp_path / "seq2"
    _write_tum_sequence(seq_dir, n_rgb=5, n_depth=5, offsets=[0.001] * 5,
                        missing_depth_indices={2})
    matched = associate_sequence(seq_dir)
    assert len(matched) == 4
    matched_depth_files = {df for _, _, df in matched}
    assert "depth/2.png" not in matched_depth_files


def test_build_regenerate_range_clean_decodes_correctly(tmp_path):
    seq_dir = tmp_path / "bonn_seq"
    _write_tum_sequence(seq_dir, n_rgb=6, n_depth=6, offsets=[0.001] * 6)
    regenerate_range = build_regenerate_range("clean", root=tmp_path)
    rows = regenerate_range("bonn_seq", 0, 6)
    assert len(rows) == 6
    for frame, gt, mask in rows:
        assert frame.shape == (3, 518, 518)
        assert gt.shape == (518, 518)
        assert mask.shape == (518, 518)
        assert frame.min() >= 0.0 and frame.max() <= 1.0


def test_build_regenerate_range_clean_is_chunk_independent(tmp_path):
    seq_dir = tmp_path / "bonn_seq2"
    _write_tum_sequence(seq_dir, n_rgb=7, n_depth=7, offsets=[0.001] * 7)
    regen_a = build_regenerate_range("clean", root=tmp_path)
    whole = regen_a("bonn_seq2", 0, 7)
    regen_b = build_regenerate_range("clean", root=tmp_path)
    chunked = regen_b("bonn_seq2", 0, 3) + regen_b("bonn_seq2", 3, 7)
    for (f1, g1, m1), (f2, g2, m2) in zip(whole, chunked):
        np.testing.assert_array_equal(f1, f2)
        np.testing.assert_array_equal(g1, g2)
        np.testing.assert_array_equal(m1, m2)


def test_build_regenerate_range_degraded_out_of_order_raises(tmp_path):
    seq_dir = tmp_path / "bonn_seq3"
    _write_tum_sequence(seq_dir, n_rgb=5, n_depth=5, offsets=[0.001] * 5)
    regen = build_regenerate_range("degraded", root=tmp_path, severity=1.0)
    regen("bonn_seq3", 0, 2)
    with pytest.raises(ValueError, match="strictly increasing"):
        regen("bonn_seq3", 3, 5)  # skips [2,3) -- out of order


def test_build_regenerate_range_degraded_third_pass_from_zero_raises(tmp_path):
    """RecipeManifest only ever makes two passes over a video (verify, then serve),
    both restarting at local frame 0 -- a third restart means something other than
    RecipeManifest's own contract is driving this callback, and must raise rather
    than be silently accepted as 'just another pass'."""
    seq_dir = tmp_path / "bonn_seq3b"
    _write_tum_sequence(seq_dir, n_rgb=3, n_depth=3, offsets=[0.001] * 3)
    regen = build_regenerate_range("degraded", root=tmp_path, severity=1.0)
    regen("bonn_seq3b", 0, 3)  # pass 1 (verify)
    regen("bonn_seq3b", 0, 3)  # pass 2 (serve)
    with pytest.raises(ValueError, match="only ever makes 2 passes"):
        regen("bonn_seq3b", 0, 3)  # pass 3 -- must raise


def test_build_regenerate_range_degraded_gt_and_mask_stay_clean(tmp_path):
    """docs/DEGRADED_EVAL_PROTOCOL.md §3's hard precondition (RAMEN Opus's
    review flagged the recipe design silently dropped the mechanical build-time
    check §3 originally specified against pixel-materialised manifests -- this
    proves the property the recipe builders actually guarantee instead: GT
    depth and mask are byte-identical between clean and degraded regeneration
    for the same video, and the frame differs -- verifying degradation reaches
    the pixels, not just that gt/mask stayed clean)."""
    seq_dir = tmp_path / "bonn_seq4"
    _write_tum_sequence(seq_dir, n_rgb=4, n_depth=4, offsets=[0.001] * 4)
    clean_rows = build_regenerate_range("clean", root=tmp_path)("bonn_seq4", 0, 4)
    degraded_rows = build_regenerate_range("degraded", root=tmp_path, severity=1.0)("bonn_seq4", 0, 4)
    for (f1, g1, m1), (f2, g2, m2) in zip(clean_rows, degraded_rows):
        np.testing.assert_array_equal(g1, g2)
        np.testing.assert_array_equal(m1, m2)
        assert not np.array_equal(f1, f2)


def test_build_manifest_round_trips_through_recipe_manifest(tmp_path):
    seq_dir = tmp_path / "bonn_seq5"
    _write_tum_sequence(seq_dir, n_rgb=5, n_depth=5, offsets=[0.001] * 5)
    out_path = tmp_path / "manifest.npz"
    stats = build_manifest("degraded", out_path, ["bonn_seq5"], root=tmp_path, severity=1.0)
    assert stats["frame_counts"]["bonn_seq5"] == 5
    assert stats["builder_commit"]

    regenerate_range = build_regenerate_range("degraded", root=tmp_path, severity=1.0)
    rm = RecipeManifest(out_path, regenerate_range, chunk_size=2)
    assert len(rm) == 5
    for row in range(5):
        rm.frame(row)  # must not raise (digest verifies on first access to the video)


def test_build_manifest_digest_mismatch_if_severity_differs_at_read_time(tmp_path):
    seq_dir = tmp_path / "bonn_seq6"
    _write_tum_sequence(seq_dir, n_rgb=3, n_depth=3, offsets=[0.001] * 3)
    out_path = tmp_path / "manifest6.npz"
    build_manifest("degraded", out_path, ["bonn_seq6"], root=tmp_path, severity=1.0)

    wrong_severity_regen = build_regenerate_range("degraded", root=tmp_path, severity=0.5)
    rm = RecipeManifest(out_path, wrong_severity_regen)
    with pytest.raises(ValueError, match="DIGEST MISMATCH"):
        rm.frame(0)
