"""Protocol v3 WP12: the Spring train loader (data/spring_dataset.py): disparity (2x grid, float16) -> metric depth with the per-frame fx and the 0.065 m baseline."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
h5py = pytest.importorskip("h5py")

import train  # noqa: E402
from data.spring_dataset import BASELINE_M, SpringDataset  # noqa: E402

REAL = Path("/mnt/DATA/DegradedDepth/data/TRAIN/spring")
W, H = 64, 36                                                    # frame grid; the disparity file is 2x: (2H, 2W)


def _tree(tmp, seqs=("0001", "0002"), n=12, fx0=100.0):
    rng = np.random.RandomState(0)
    for k, s in enumerate(seqs):
        base = tmp / "extracted" / "spring" / "train" / s
        for d in ("frame_left", "disp1_left", "cam_data"):
            (base / d).mkdir(parents=True)
        fxs = fx0 + 10 * k + np.arange(n)                        # fx varies per frame
        np.savetxt(base / "cam_data" / "intrinsics.txt", np.stack([fxs, fxs, np.full(n, W / 2), np.full(n, H / 2)], 1))
        for i in range(1, n + 1):
            cv2.imwrite(str(base / "frame_left" / f"frame_left_{i:04d}.png"), rng.randint(0, 255, (H, W, 3), dtype=np.uint8))
            disp = rng.uniform(0.5, 8.0, (2 * H, 2 * W)).astype(np.float16)
            disp[0, 0], disp[2, 2], disp[4, 4] = np.nan, 0.0, np.inf
            with h5py.File(base / "disp1_left" / f"disp1_left_{i:04d}.dsp5", "w") as f:
                f["disparity"] = disp
    return str(tmp)


def _ds(tmp, **kw):
    args = dict(root=kw.pop("root", None) or _tree(tmp), num_frames=4, clip_stride=4, image_size=(56, 56), split="train", degrade=False, seed=0)
    args.update(kw)
    return SpringDataset(**args)


def test_read_depth_is_fx_times_baseline_over_the_frame_grid_disparity(tmp_path):
    ds = _ds(tmp_path)
    base = tmp_path / "extracted" / "spring" / "train" / "0002"
    for frame in (1, 7):
        p = base / "disp1_left" / f"disp1_left_{frame:04d}.dsp5"
        d = ds.read_depth(p)
        with h5py.File(p) as f:
            disp = np.asarray(f["disparity"])[::2, ::2].astype(np.float32)
        fx = 110.0 + (frame - 1)                                           # sequence 0002 starts at fx 110, +1 per frame
        ok = np.isfinite(disp) & (disp > 1e-3)
        assert d.shape == (H, W) and d.dtype == np.float32
        assert np.allclose(d[ok], fx * BASELINE_M / disp[ok], rtol=1e-5)
        assert bool((d[~ok] == 0).all()) and (~ok).any()                    # nan / 0 / inf disparity -> invalid (0)


def test_baseline_is_65_mm():
    assert BASELINE_M == 0.065


def test_clips_shapes_cap_and_sequence_ids(tmp_path):
    ds = _ds(tmp_path, max_depth=80.0)
    assert ds.sequences == ["0001", "0002"] and len(ds) == 2 * 3          # 12 frames -> 3 clips of 4
    x = ds[0]
    assert x["frames"].shape == (3, 4, 56, 56) and x["depth"].shape == (1, 4, 56, 56) and x["video_id"] in ds.sequences
    m = x["mask"] > 0.5
    assert float(m.float().mean()) > 0.9 and float(x["depth"][m].max()) < 80.0


def test_far_pixels_are_masked_by_the_per_source_cap(tmp_path):
    root = _tree(tmp_path)
    near, far = _ds(tmp_path, root=root, max_depth=80.0), _ds(tmp_path, root=root, max_depth=1.0)
    assert float(far[0]["mask"].mean()) < float(near[0]["mask"].mean()) and float(far[0]["depth"].max()) < 1.0


def test_incomplete_sequences_fail_loudly(tmp_path):
    root = _tree(tmp_path)
    (tmp_path / "extracted" / "spring" / "train" / "0001" / "disp1_left" / "disp1_left_0003.dsp5").unlink()
    with pytest.raises(FileNotFoundError, match="incomplete"):
        SpringDataset(root=root, num_frames=4, image_size=(56, 56), degrade=False)
    with pytest.raises(FileNotFoundError, match="not found"):
        SpringDataset(root=str(tmp_path / "nowhere"), degrade=False)


def test_val_needs_a_fraction_and_holds_out_sequences(tmp_path):
    root = _tree(tmp_path)
    kw = dict(root=root, num_frames=4, image_size=(56, 56), degrade=False, seed=3)
    with pytest.raises(ValueError, match="val_fraction"):
        SpringDataset(split="val", **kw)
    tr, va = SpringDataset(split="train", val_fraction=0.5, **kw), SpringDataset(split="val", val_fraction=0.5, **kw)
    assert set(tr.sequences).isdisjoint(va.sequences)


def test_make_source_wires_spring(tmp_path):
    common = dict(num_frames=4, image_size=(56, 56), clip_stride=4, degrade=False, seed=1, max_depth=80.0, resize_mode="vda")
    ds = train._make_source({"type": "spring", "root": _tree(tmp_path), "max_depth": 60.0, "sparse_gt": False}, "train", common)
    assert ds.max_depth == 60.0 and float(ds[0]["sparse"]) == 0.0


@pytest.mark.skipif(not (REAL / "extracted/spring/train/0001/disp1_left").is_dir() or not (REAL / "extracted/spring/train/0047/disp1_left").is_dir(), reason="Spring not (fully) extracted here")
def test_real_spring_depths_are_plausible_metric():
    ds = SpringDataset(root=str(REAL), sequences=["0001"], num_frames=4, clip_stride=100, image_size=(518, 518), degrade=False, seed=0)
    x = ds[0]
    m = x["mask"] > 0.5
    assert x["frames"].shape == (3, 4, 518, 518) and float(m.float().mean()) > 0.5
    med = float(x["depth"][m].median())
    assert 1.0 < med < 80.0, med                                      # a rendered outdoor scene: metres, not pixels


def test_both_layouts_resolve_and_load_identically(tmp_path, capsys):
    """Local layout <root>/extracted/spring/train and Isambard layout <root>/spring/train (no `extracted`): same sequences, same clips; the chosen dir is logged; `extracted` wins if both exist."""
    import shutil
    from data.spring_dataset import resolve_spring_train
    loc = Path(_tree(tmp_path / "loc"))
    isa = tmp_path / "isa"
    (isa / "spring").mkdir(parents=True)
    shutil.copytree(loc / "extracted" / "spring" / "train", isa / "spring" / "train")
    a, b = _ds(None, root=str(loc)), _ds(None, root=str(isa))
    assert a.sequences == b.sequences and len(a) == len(b) > 0
    out = capsys.readouterr().out
    assert f"[spring] train dir = {loc}/extracted/spring/train" in out and f"[spring] train dir = {isa}/spring/train" in out
    both = tmp_path / "both"
    shutil.copytree(loc, both)
    shutil.copytree(isa / "spring", both / "spring")
    assert resolve_spring_train(str(both)) == both / "extracted" / "spring" / "train"
    with pytest.raises(FileNotFoundError, match="not found"):
        resolve_spring_train(str(tmp_path / "none"))
