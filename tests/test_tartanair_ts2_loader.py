"""Opt-in TS2 TartanAir sky/mask loader (data/tartanair_ts2.py, wired via VDASemanticsTartanAirDataset.ts2 and the
train.py source keys ts2_sky / ts2_seg_root / ts2_env_table).  Flag off must equal the released loader exactly."""
from __future__ import annotations

import importlib.util
import json
import os
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from data.tartanair_ts2 import DEFAULT_TABLE, TS2Labeller, load_env_table, table_sha256, ts2_sky_valid
from data.vda_semantics_dataset import VDASemanticsDegradedTartanAirDataset, VDASemanticsTartanAirDataset
from train import _make_source

REPO = Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("ts_v2_draft", REPO / "outdoor_debug" / "parts" / "TA_sky" / "ts_v2_draft.py")
draft = importlib.util.module_from_spec(_s); _s.loader.exec_module(draft)

W = H = 120
NB = 4                      # 4 column blocks of 30 px
SKY = 7
INF = np.inf
COMMON = dict(num_frames=2, image_size=(518, 518), clip_stride=1, frame_stride=1, max_depth=80.0, seed=42)


def _table(tmp_path, **envs):
    t = {"schema": "ts2_env_table/1", "n_envs": len(envs), "envs": envs}
    p = tmp_path / "table.json"; p.write_text(json.dumps(t)); return p


def _flags(sky_ids=(SKY,), far_dome=False, seg_sky_ok=True, sat_nonsky_is_sky=False, fallback_sat_is_sky=False):
    return dict(sky_ids=list(sky_ids), far_dome=far_dome, seg_sky_ok=seg_sky_ok, sat_nonsky_is_sky=sat_nonsky_is_sky,
                fallback_sat_is_sky=fallback_sat_is_sky)


def _write_depth(path, depth):
    h, w = depth.shape
    cv2.imwrite(str(path), np.ascontiguousarray(depth.astype(np.float32)).reshape(h, w, 1).view(np.uint8).reshape(h, w, 4))


def _blocks(vals):
    """(H, W) array whose 4 column blocks take vals."""
    return np.repeat(np.array(vals, np.float32)[None, :], H, 0).repeat(W // NB, 1)


def _build(tmp_path, env, depth_vals, seg_vals, n_frames=3, with_seg=True, seg_shape=None):
    """TRAIN-style tree + seg zip (seg_root/env/Data_easy/seg_lcam_front.zip). Returns (data_root, seg_root)."""
    root, seg_root = tmp_path / "train", tmp_path / "seg"
    img_dir = root / env / "Data_easy" / "P000" / "image_lcam_front"
    dep_dir = root / env / "Data_easy" / "P000" / "depth_lcam_front"
    img_dir.mkdir(parents=True); dep_dir.mkdir(parents=True)
    (seg_root / env / "Data_easy").mkdir(parents=True)
    d = _blocks(depth_vals); s = np.repeat(np.array(seg_vals, np.uint8)[None, :], H, 0).repeat(W // NB, 1)
    if seg_shape is not None:
        s = s[:seg_shape[0], :seg_shape[1]]
    with zipfile.ZipFile(seg_root / env / "Data_easy" / "seg_lcam_front.zip", "w") as z:
        for i in range(n_frames):
            Image.new("RGB", (W, H), (i * 10, 0, 0)).save(img_dir / f"{i:06d}_lcam_front.png")
            _write_depth(dep_dir / f"{i:06d}_lcam_front_depth.png", d)
            if with_seg:
                ok, buf = cv2.imencode(".png", s); assert ok
                z.writestr(f"{env}/Data_easy/P000/seg_lcam_front/{i:06d}_lcam_front_seg.png", buf.tobytes())
    return root, seg_root


def _ds(root, ts2=None, cls=VDASemanticsTartanAirDataset):
    ds = cls(root=str(root), split="val", val_fraction=1.0, **COMMON)
    if ts2 is not None:
        ds.ts2 = ts2
    return ds


def _at(sample, key, block):          # value at the centre of a column block (518 px wide)
    return float(sample[key][0, 0, 259, int((block + 0.5) * 518 / NB)])


# ---- flag off == released --------------------------------------------------------------------------------------------
def test_flag_off_is_the_released_rule(tmp_path):
    root, _ = _build(tmp_path, "Env", [20.0, 8000.0, INF, 65504.0], [3, SKY, SKY, 3])
    ds = _ds(root)
    assert ds.ts2 is None
    out = ds[0]
    # released: valid = finite & 0<d<max_depth(80); sky = ~finite | d>1e4 (so the 8000 m dome is neither sky nor valid)
    assert [_at(out, "mask", b) for b in range(4)] == [1.0, 0.0, 0.0, 0.0]
    assert [_at(out, "sky", b) for b in range(4)] == [0.0, 0.0, 1.0, 1.0]


def test_flag_off_matches_a_reference_implementation_of_the_released_code(tmp_path):
    """Re-derive the released masks from the raw depth with the released lines and compare exactly."""
    from data.video_clip_dataset import _decode_depth
    from data.vda_resize import vda_resize_depth_like
    root, _ = _build(tmp_path, "Env", [20.0, 8000.0, INF, 65504.0], [3, SKY, SKY, 3])
    ds = _ds(root)
    out = ds[0]
    dep = sorted((root / "Env" / "Data_easy" / "P000" / "depth_lcam_front").glob("*.png"))[0]
    d_raw = _decode_depth(str(dep))
    valid = np.isfinite(d_raw) & (d_raw > 0) & (d_raw < 80.0)
    sky = ~np.isfinite(d_raw) | (d_raw > 1.0e4)
    for key, a in (("mask", valid), ("sky", sky)):
        r = vda_resize_depth_like(a.astype(np.float32), 518, 518)
        assert torch.equal(out[key][0, 0], torch.from_numpy(r.copy()))


# ---- flag on: the rule, through the dataset --------------------------------------------------------------------------
@pytest.mark.parametrize("flags,depth,seg,sky,mask", [
    # far-dome, seg-sky at 8000 m (below 9000): uncertain -> sky 0 mask 0
    (_flags(far_dome=True), 8000.0, SKY, 0.0, 0.0),
    # far-dome, seg-sky at 9500 m (>= 9000, < 1e4): sky 1
    (_flags(far_dome=True), 9500.0, SKY, 1.0, 0.0),
    # near-dome (seg_sky_ok True): seg-sky at 8000 m is sky
    (_flags(far_dome=False), 8000.0, SKY, 1.0, 0.0),
    # saturated seg-sky -> sky
    (_flags(far_dome=True), 65504.0, SKY, 1.0, 0.0),
    (_flags(far_dome=False), INF, SKY, 1.0, 0.0),
    # saturated non-sky -> uncertain; sat_nonsky_is_sky -> sky
    (_flags(), 65504.0, 3, 0.0, 0.0),
    (_flags(sat_nonsky_is_sky=True), 65504.0, 3, 1.0, 0.0),
    # near-dome with seg_sky_ok False -> everything seg-sky is uncertain (never sky)
    (_flags(far_dome=False, seg_sky_ok=False), 8000.0, SKY, 0.0, 0.0),
    (_flags(far_dome=False, seg_sky_ok=False), INF, SKY, 0.0, 0.0),
    # non-sky finite valid depth stays valid
    (_flags(), 20.0, 3, 0.0, 1.0),
    # id-0 guard: id 0 is sky only where saturated; finite id 0 uncertain
    (_flags(sky_ids=(0,)), 65504.0, 0, 1.0, 0.0),
    (_flags(sky_ids=(0,)), 20.0, 0, 0.0, 0.0),
    # unlabelled env: fallback_sat_is_sky flagged -> saturated is sky; unflagged -> uncertain
    (_flags(sky_ids=(), fallback_sat_is_sky=True), 65504.0, 3, 1.0, 0.0),
    (_flags(sky_ids=(), fallback_sat_is_sky=False), 65504.0, 3, 0.0, 0.0),
])
def test_ts2_rule_through_the_dataset(tmp_path, flags, depth, seg, sky, mask):
    root, seg_root = _build(tmp_path, "Env", [depth] * 4, [seg] * 4)
    lab = TS2Labeller(seg_root, _table(tmp_path, Env=flags), max_depth=80.0)
    out = _ds(root, lab)[0]
    assert {_at(out, "sky", b) for b in range(4)} == {sky}
    assert {_at(out, "mask", b) for b in range(4)} == {mask}
    # depth tensor is NOT touched by the flag: same as the released loader
    base = _ds(root)[0]
    assert torch.equal(out["depth"], base["depth"]) and torch.equal(out["frames"], base["frames"])


def test_masks_only_binary_after_the_released_resize_and_class_ids_never_interpolated(tmp_path):
    """Seg with a class boundary mid-image. At raw resolution the TS2 masks are exactly {0,1}; the dataset resizes
    them with the released nearest-neighbour float-mask resize, so they stay {0,1} (no blended class-id artefacts)."""
    root, seg_root = _build(tmp_path, "Env", [20.0, 65504.0, 65504.0, 20.0], [3, SKY, 200, 9])
    lab = TS2Labeller(seg_root, _table(tmp_path, Env=_flags()), max_depth=80.0)
    d = _blocks([20.0, 65504.0, 65504.0, 20.0])
    seg = lab.read_seg(root / "Env" / "Data_easy" / "P000" / "depth_lcam_front" / "000000_lcam_front_depth.png")
    assert seg.dtype == np.uint8 and set(np.unique(seg).tolist()) == {3, SKY, 200, 9}
    sky, mask, unc = lab.label(root / "Env" / "Data_easy" / "P000" / "depth_lcam_front" / "000000_lcam_front_depth.png", d)
    assert sky.dtype == mask.dtype == unc.dtype == bool
    assert (sky[:, 30:60]).all() and not sky[:, 60:].any() and unc[:, 60:90].all() and mask[:, :30].all() and mask[:, 90:].all()
    out = _ds(root, lab)[0]
    for key in ("mask", "sky"):
        assert set(torch.unique(out[key]).tolist()) <= {0.0, 1.0}


def test_degraded_subclass_inherits_ts2(tmp_path):
    root, seg_root = _build(tmp_path, "Env", [8000.0] * 4, [SKY] * 4)
    lab = TS2Labeller(seg_root, _table(tmp_path, Env=_flags(far_dome=True)), max_depth=80.0)
    assert issubclass(VDASemanticsDegradedTartanAirDataset, VDASemanticsTartanAirDataset)
    ds = VDASemanticsDegradedTartanAirDataset(root=str(root), split="val", val_fraction=1.0, degrade=False, degrade_mode="full",
                                              degrade_severity=None, **COMMON)
    ds.ts2 = lab
    out = ds[0]
    assert _at(out, "sky", 0) == 0.0 and _at(out, "mask", 0) == 0.0


# ---- failures --------------------------------------------------------------------------------------------------------
def test_missing_env_in_table_raises(tmp_path):
    root, seg_root = _build(tmp_path, "Env", [20.0] * 4, [3] * 4)
    lab = TS2Labeller(seg_root, _table(tmp_path, Other=_flags()), max_depth=80.0)
    with pytest.raises(KeyError, match="not in the env table"):
        _ds(root, lab)[0]


def test_missing_seg_zip_or_frame_raises(tmp_path):
    root, seg_root = _build(tmp_path, "Env", [20.0] * 4, [3] * 4, with_seg=False)
    lab = TS2Labeller(seg_root, _table(tmp_path, Env=_flags()), max_depth=80.0)
    with pytest.raises(FileNotFoundError, match="seg frame missing"):
        _ds(root, lab)[0]
    os.remove(seg_root / "Env" / "Data_easy" / "seg_lcam_front.zip")
    lab2 = TS2Labeller(seg_root, _table(tmp_path, Env=_flags()), max_depth=80.0)
    with pytest.raises(FileNotFoundError, match="seg zip missing"):
        _ds(root, lab2)[0]
    with pytest.raises(FileNotFoundError):
        TS2Labeller(tmp_path / "nope", _table(tmp_path, Env=_flags()))


def test_shape_mismatch_asserts(tmp_path):
    root, seg_root = _build(tmp_path, "Env", [20.0] * 4, [3] * 4, seg_shape=(H - 4, W))
    lab = TS2Labeller(seg_root, _table(tmp_path, Env=_flags()), max_depth=80.0)
    with pytest.raises(AssertionError, match="seg shape"):
        _ds(root, lab)[0]
    with pytest.raises(AssertionError, match="seg shape"):
        ts2_sky_valid(np.zeros((4, 4), np.float32), np.zeros((4, 5), np.uint8), [1], False)


def test_zip_handles_are_cached_per_pid(tmp_path):
    root, seg_root = _build(tmp_path, "Env", [20.0] * 4, [3] * 4)
    lab = TS2Labeller(seg_root, _table(tmp_path, Env=_flags()), max_depth=80.0)
    ds = _ds(root, lab); ds[0]; ds[1] if len(ds) > 1 else None
    assert len(lab._zips) == 1 and next(iter(lab._zips))[0] == os.getpid()
    import pickle
    assert pickle.loads(pickle.dumps(lab))._zips == {}      # workers re-open their own handles


# ---- train.py wiring -------------------------------------------------------------------------------------------------
TCOMMON = dict(num_frames=2, image_size=(518, 518), clip_stride=1, frame_stride=1, max_depth=80.0, seed=42, degrade=False,
               degrade_mode="full", degrade_severity=None, max_clips_per_video=None, resize_mode="vda")


def test_train_source_keys(tmp_path):
    root, seg_root = _build(tmp_path, "Env", [20.0] * 4, [3] * 4)
    base = {"type": "tartanair", "root": str(root), "val_fraction": 1.0}
    assert _make_source(base, "val", TCOMMON).ts2 is None                       # absent = unchanged
    tbl = _table(tmp_path, Env=_flags())
    ds = _make_source({**base, "ts2_sky": True, "ts2_seg_root": str(seg_root), "ts2_env_table": str(tbl)}, "val", TCOMMON)
    assert isinstance(ds.ts2, TS2Labeller) and ds.ts2.max_depth == 80.0 and ds[0]["mask"].shape[0] == 1
    dd = _make_source({**base, "ts2_sky": True, "ts2_seg_root": str(seg_root)}, "val", {**TCOMMON, "degrade": True})
    assert isinstance(dd, VDASemanticsDegradedTartanAirDataset) and dd.ts2 is not None   # default table
    with pytest.raises(ValueError, match="ts2_seg_root"):
        _make_source({**base, "ts2_sky": True}, "val", TCOMMON)
    with pytest.raises(ValueError, match="without ts2_sky"):
        _make_source({**base, "ts2_seg_root": str(seg_root)}, "val", TCOMMON)
    with pytest.raises(ValueError, match="resize_mode"):
        _make_source({**base, "ts2_sky": True, "ts2_seg_root": str(seg_root)}, "val", {**TCOMMON, "resize_mode": None})


# ---- frozen table ----------------------------------------------------------------------------------------------------
def test_frozen_table_matches_generator_and_spec_counts():
    spec = importlib.util.spec_from_file_location("mk", REPO / "outdoor_debug/parts/TA_sky/make_ts2_env_table.py")
    mk = importlib.util.module_from_spec(spec); spec.loader.exec_module(mk)
    t = mk.build()
    assert json.loads(Path(DEFAULT_TABLE).read_text()) == t        # table on disk == regenerated from the pinned sources
    envs = load_env_table()
    assert len(envs) == 74 and len(table_sha256()) == 64
    assert sum(v["far_dome"] for v in envs.values()) == 43
    assert sorted(k for k, v in envs.items() if not v["seg_sky_ok"]) == sorted(
        ["RetroOffice", "AbandonedCable", "CarWelding", "Hospital", "OldBrickHouseDay", "CountryHouse", "SoulCity"])
    assert sorted(k for k, v in envs.items() if v["sat_nonsky_is_sky"]) == ["ArchVizTinyHouseNight", "SoulCity"]
    assert sum(v["fallback_sat_is_sky"] for v in envs.values()) == 8


def test_port_equals_draft_on_random_arrays():
    rng = np.random.RandomState(0)
    d = rng.choice(np.array([0.0, 5.0, 100.0, 8999.0, 9000.0, 9500.0, 9999.9, 10000.0, 65504.0, np.inf, np.nan], np.float32), (64, 64))
    s = rng.randint(0, 4, (64, 64)).astype(np.uint8)
    for ids in ((), (1,), (0,), (0, 2)):
        for far in (False, True):
            for ok in (True, False):
                for sn in (False, True):
                    for fb in (False, True):
                        a = ts2_sky_valid(d, s, ids, far, ok, sn, 1e4, fb)
                        b = draft.ts_v2_guarded(d, s, ids, far, ok, sn, 1e4, fb)
                        assert all(np.array_equal(x, y) for x, y in zip(a, b))


# ---- real data (skips when absent) -----------------------------------------------------------------------------------
TRAIN = Path(os.environ.get("TARTANAIR_TRAIN_ROOT", "/mnt/DATA/DegradedDepth/data/TRAIN/tartanair"))
SEGR = Path(os.environ.get("TARTANAIR_SEG_ROOT", "/mnt/DATA/DegradedDepth/claude_outputs/scratch/outdoor_debug_20261016/TA_seg"))


@pytest.mark.skipif(not (TRAIN.is_dir() and SEGR.is_dir()), reason="local TartanAir TRAIN tree or seg zips absent")
@pytest.mark.parametrize("env", ["Downtown", "AmericanDiner", "HongKong", "AbandonedFactory"])
def test_real_frame_equals_ts_v2_guarded(env):
    from data.video_clip_dataset import _decode_depth
    if not (TRAIN / env).is_dir() or not (SEGR / env / "Data_easy" / "seg_lcam_front.zip").is_file():
        pytest.skip(f"{env} not local")
    traj = sorted((TRAIN / env / "Data_easy").glob("P*/depth_lcam_front"))[0]
    files = sorted(traj.glob("*.png")); f = files[len(files) // 2]
    lab = TS2Labeller(SEGR, None, max_depth=1.0e4)
    d = _decode_depth(str(f))
    seg = lab.read_seg(f)
    assert seg.shape == d.shape and seg.dtype == np.uint8
    fl = load_env_table()[env]
    sky, mask, unc = lab.label(f, d)
    ref = draft.ts_v2_guarded(d, seg, fl["sky_ids"], fl["far_dome"], fl["seg_sky_ok"], fl["sat_nonsky_is_sky"], 1e4, fl["fallback_sat_is_sky"])
    assert np.array_equal(sky, ref[0]) and np.array_equal(mask, ref[1]) and np.array_equal(unc, ref[2])
    assert not (sky & mask).any() and not (unc & mask).any()
