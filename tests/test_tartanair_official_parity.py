"""Parity of our TartanAir decoding with the OFFICIAL tartanairpy reader (tartanair==1.4.0).

* depth: data.video_clip_dataset._decode_depth  vs  TartanAirImageReader.read_depth (exact float32 equality)
* rgb:   PIL ``Image.open().convert("RGB")`` (what VideoDepthClipDataset does)  vs  TartanAirImageReader.read_rgb (exact)

Reference: outdoor_debug/parts/TA_sky/parity_reference.json holds sha256 of the official reader's outputs
(made in an isolated venv by outdoor_debug/parts/TA_sky/make_parity_reference.py), so the test runs without the
official package.  If the official package IS importable the test additionally compares against it directly.
Skips when the local TartanAir TRAIN tree or the reference file is absent.  TRAIN data only.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(os.environ.get("TARTANAIR_TRAIN_ROOT", "/mnt/DATA/DegradedDepth/data/TRAIN/tartanair"))
REF = Path(__file__).resolve().parents[1] / "outdoor_debug" / "parts" / "TA_sky" / "parity_reference.json"

pytestmark = pytest.mark.skipif(not (ROOT.is_dir() and REF.is_file()), reason="local TartanAir TRAIN tree or reference json absent")


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _entries():
    if not REF.is_file():
        return []
    return json.loads(REF.read_text())["entries"]


def _official():
    try:
        from tartanair.reader import TartanAirImageReader
        return TartanAirImageReader()
    except Exception:  # package (or its heavy deps) absent
        return None


@pytest.mark.parametrize("e", _entries(), ids=lambda e: f"{e['env']}/{e['traj']}/{e['depth'][:6]}")
def test_depth_and_rgb_parity(e):
    from data.video_clip_dataset import _decode_depth
    base = ROOT / e["env"] / "Data_easy" / e["traj"]
    dp, ip = base / "depth_lcam_front" / e["depth"], base / "image_lcam_front" / e["image"]
    if not (dp.is_file() and ip.is_file()):
        pytest.skip("frame not present locally")
    d = _decode_depth(str(dp))
    assert d.dtype == np.float32 and list(d.shape) == e["depth_shape"]
    assert _sha(d) == e["depth_sha256"], "depth differs from official read_depth"
    rgb = np.asarray(Image.open(ip).convert("RGB"))
    assert list(rgb.shape) == e["rgb_shape"]
    assert _sha(rgb) == e["rgb_sha256"], "rgb differs from official read_rgb"
    off = _official()
    if off is not None:
        np.testing.assert_array_equal(d, off.read_depth(str(dp)))
        np.testing.assert_array_equal(rgb, off.read_rgb(str(ip)))


def test_reference_covers_key_envs_and_sky_values():
    ents = _entries()
    envs = {e["env"] for e in ents}
    assert {"Downtown", "GreatMarsh", "HongKong"} <= envs
    assert any(e["n_65504"] > 0 for e in ents), "reference should include a stored-65504 frame"
