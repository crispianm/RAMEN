"""ScanNet++ train depth decode: uint16 mm (98,304 B) and float32 m (196,608 B) payloads -> (192,256) float32 metres, 0 = invalid.
11 subset140 scenes store float32 metres raw-deflate (protocol v3 gates at b0b2c06 died on them). Real-frame tests skip when the data is absent."""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pytest

from data.scannetpp_dataset import _index_depth_bin
from data.scannetpp_train import decode_depth_frame

ROOT = Path("/mnt/DATA/DegradedDepth/data/TRAIN/scannetpp_train_full/data")


def _deflate(raw: bytes) -> bytes:
    c = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    return c.compress(raw) + c.flush()


def test_uint16_mm_to_metres_zero_invalid():
    a = np.full((192, 256), 1500, np.uint16)
    a[:5] = 0
    d = decode_depth_frame(_deflate(a.tobytes()))
    assert d.shape == (192, 256) and d.dtype == np.float32
    assert np.allclose(d[10:], 1.5) and (d[:5] == 0).all()


def test_float32_metres_kept_invalid_zeroed():
    a = np.full((192, 256), 1.5, np.float32)
    a[0], a[1], a[2], a[3] = 0.0, np.nan, np.inf, -1.0
    d = decode_depth_frame(_deflate(a.tobytes()))
    assert d.shape == (192, 256) and d.dtype == np.float32
    assert np.allclose(d[10:], 1.5) and (d[:4] == 0).all()


def test_other_payload_size_raises():
    with pytest.raises(ValueError, match="decodes to 1000 bytes"):
        decode_depth_frame(_deflate(b"\x00" * 1000))


def _real_frame(scene: str, k: int = 100) -> np.ndarray:
    p = ROOT / scene / "iphone" / "depth.bin"
    if not p.exists():
        pytest.skip(f"{p} absent")
    off, n = _index_depth_bin(p)[k]
    with open(p, "rb") as f:
        f.seek(off)
        return decode_depth_frame(f.read(n))


def test_real_float32_scene_is_metres():
    d = _real_frame("27dc178a3d")                    # float32 scene
    v = d[d > 0]
    assert d.shape == (192, 256) and 0.3 < np.median(v) < 10 and v.max() < 20


def test_real_uint16_scene_is_metres():
    d = _real_frame("00777c41d4")                    # uint16 scene
    v = d[d > 0]
    assert d.shape == (192, 256) and 0.3 < np.median(v) < 10 and v.max() < 20


def test_build_multisource_clips_match_independent_decode(tmp_path):
    """F4: a clip from a float32 scene and from a uint16 scene, drawn through the REAL train._build_multisource, carry exactly the independently decoded depth values."""
    import copy, os
    for s in ("27dc178a3d", "00777c41d4"):
        if not (ROOT / s / "iphone" / "depth.bin").exists():
            pytest.skip("local ScanNet++ scenes absent")
    repo = Path(__file__).resolve().parents[1]
    (tmp_path / "data").mkdir(); (tmp_path / "splits").mkdir()
    for s in ("27dc178a3d", "00777c41d4"):
        (tmp_path / "data" / s).symlink_to(ROOT / s)
    (tmp_path / "splits" / "sem_test.txt").write_text("27dc178a3d\n00777c41d4\n")
    (tmp_path / "splits" / "nvs_sem_val.txt").write_text("")
    (tmp_path / "splits" / "nvs_test_iphone.txt").write_text("")
    os.chdir(repo)
    import train
    cfg = copy.deepcopy(train.load_config(repo / "configs" / "experiments" / "dn01_v3_syn_A.yaml"))
    src = dict(type="scannetpp_train", root=str(tmp_path), splits=["sem_test"], frame_stride=6, max_depth=10.0, resize_mode="vda", metric_supervision=True, sparse_gt=False, sampling_mass=1)
    cfg["data"].update(sources=[src], val_sources=[src], num_workers=0, clip_stride=32, num_frames=4, degrade=False, sampling="equal")
    loader, _ = train._build_multisource(cfg, "train", 1, 0)
    ds = loader.dataset
    ds = getattr(ds, "datasets", [ds])[0]
    for scene, kind in (("27dc178a3d", np.float32), ("00777c41d4", np.uint16)):
        idx = next(i for i, c in enumerate(ds.clips) if c[2] == scene)
        _a, _b, _s, start = ds.clips[idx]
        out = ds[idx]
        p = ROOT / scene / "iphone" / "depth.bin"
        sp = _index_depth_bin(p)
        with open(p, "rb") as f:
            for t in range(4):
                off, n = sp[start + t * 6]
                f.seek(off)
                raw = zlib.decompress(f.read(n), wbits=-zlib.MAX_WBITS) if kind is np.float32 else None
                if kind is np.float32:
                    ind = np.frombuffer(raw, np.float32)
                else:
                    f.seek(off); b = f.read(n)
                    import lz4.block
                    ind = np.frombuffer(lz4.block.decompress(b, uncompressed_size=98304), np.uint16).astype(np.float32) / 1000.0
                got = out["depth"][0, t].numpy(); g = got[got > 0]
                valid_ind = ind[np.isfinite(ind) & (ind > 0) & (ind < 10.0)]
                assert g.size > 0 and np.isin(np.round(g, 4), np.round(np.unique(valid_ind), 4)).all(), (scene, t)
                assert abs(np.median(g) - np.median(valid_ind)) < 0.25 * np.median(valid_ind), (scene, t)
