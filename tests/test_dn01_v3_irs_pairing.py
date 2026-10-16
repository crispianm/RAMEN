"""Protocol v3 WP25 irsfix: IRSDataset pairs l_/d_ by FRAME NUMBER and builds clips only over consecutive runs."""
import logging
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data import irs_dataset as M  # noqa: E402


def _scene(tmp, name, l_frames, d_frames, fam="Home"):
    sc = tmp / fam / name
    sc.mkdir(parents=True)
    for f in l_frames:
        (sc / f"l_{f:05d}.png").write_bytes(b"x")
    for f in d_frames:
        (sc / f"d_{f:05d}.exr").write_bytes(b"x")
    return sc


def _ds(tmp, **kw):
    a = dict(root=str(tmp), num_frames=3, clip_stride=1, image_size=(8, 8), degrade=False, val_fraction=0.0, seed=1)
    a.update(kw)
    return M.IRSDataset(**a)


def _frames(path_list, idx):
    return int(path_list[idx].stem.split("_")[1])


def _clip_frames(ds, clip, stride=1):
    imgs, deps, vid, start = clip
    return [(_frames(imgs, start + t * stride), _frames(deps, start + t * stride)) for t in range(ds.num_frames)]


def test_different_missing_frames_pair_by_number_and_no_gap(tmp_path):
    _scene(tmp_path, "s", [k for k in range(1, 21) if k not in (5, 6)], [k for k in range(1, 21) if k not in (12,)])
    ds = _ds(tmp_path)
    assert len(ds) > 0
    for c in ds.clips:
        fr = _clip_frames(ds, c)
        assert all(l == d for l, d in fr)
        ks = [l for l, _ in fr]
        assert ks == list(range(ks[0], ks[0] + 3))            # no gap inside a clip
    st = next(iter(ds.pairing_stats.values()))
    assert st["paired"] == 17 and st["dropped_unpaired"] == 3
    assert {k for c in ds.clips for k, _ in _clip_frames(ds, c)}.isdisjoint({5, 6, 12})


def test_dotfiles_and_partials_are_ignored(tmp_path):
    sc = _scene(tmp_path, "s", range(1, 9), range(1, 9))
    (sc / ".n_00481.exr.Y61Bsm").write_bytes(b"x")
    (sc / ".l_00002.png.abc").write_bytes(b"x")
    (sc / ".d_00003.exr").write_bytes(b"x")
    (sc / "l_00009.png.part").write_bytes(b"x")
    ds = _ds(tmp_path)
    assert next(iter(ds.pairing_stats.values()))["paired"] == 8 and next(iter(ds.pairing_stats.values()))["dropped_unpaired"] == 0


def test_royalroom_like_gaps_make_no_gap_spanning_clip(tmp_path, caplog):
    miss = set(range(803, 805)) | set(range(960, 971))
    frames = [k for k in range(1, 978) if k not in miss]
    _scene(tmp_path, "RoyalRoom_T1", frames, frames)
    with caplog.at_level(logging.WARNING, logger=M.logger.name):
        ds = _ds(tmp_path, num_frames=8, clip_stride=8)
    assert len(ds) > 0 and any("RoyalRoom_T1" in r.getMessage() for r in caplog.records)
    for c in ds.clips:
        ks = [l for l, d in _clip_frames(ds, c) if l == d]
        assert len(ks) == 8 and ks == list(range(ks[0], ks[0] + 8))
    st = next(iter(ds.pairing_stats.values()))
    assert st["paired"] == len(frames) and st["runs"] == 3
    assert len({c[2] for c in ds.clips}) == 2 and st["dropped_short_run"] == 7   # per-run ids; the 971..977 tail (7 < span 8) is dropped


def test_frame_stride_clip_never_spans_a_gap(tmp_path):
    frames = [k for k in range(1, 30) if k != 15]
    _scene(tmp_path, "s", frames, frames)
    ds = _ds(tmp_path, num_frames=3, frame_stride=2)
    for c in ds.clips:
        ks = [l for l, _ in _clip_frames(ds, c, stride=2)]
        assert not (ks[0] < 15 < ks[-1]) and ks[-1] - ks[0] == 4
    st = next(iter(ds.pairing_stats.values()))
    assert st["dropped_short_run"] == 0


def test_short_run_is_dropped_and_counted(tmp_path):
    frames = list(range(1, 4)) + list(range(10, 20))        # run of 3 (span 5 below) + run of 10
    _scene(tmp_path, "s", frames, frames)
    ds = _ds(tmp_path, num_frames=5)
    st = next(iter(ds.pairing_stats.values()))
    assert st["dropped_short_run"] == 3 and st["runs"] == 2 and st["clips"] == len(ds) == 6


def test_complete_scene_unchanged_vs_old_logic(tmp_path):
    sc = _scene(tmp_path, "s", range(1, 31), range(1, 31))
    ds = _ds(tmp_path, num_frames=4, clip_stride=3, frame_stride=2, max_clips_per_video=5)
    imgs, deps = sorted(sc.glob("l_*.png")), sorted(sc.glob("d_*.exr"))
    span = 7
    starts = list(range(0, 30 - span + 1, 3))
    step = len(starts) / 5
    starts = [starts[int(i * step)] for i in range(5)]
    assert [(c[0], c[1], c[2], c[3]) for c in ds.clips] == [(imgs, deps, "Home/s", s) for s in starts]
