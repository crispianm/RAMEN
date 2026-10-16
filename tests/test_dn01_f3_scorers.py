"""DN-01 F3: the `dvp` token (and dvp+agf...) through ALL THREE scorer entries, called the way the scoring jobs call them (F3 smoke job 6986546 failed in the nuScenes-night
scorer: its ``score()`` is ``@torch.no_grad`` and DVP's test-time training had no graph). The Auditor's re-audit bar: (1) bench_joint ``cmd_run``, scripts/dn01_heldout_eval
``score`` / ``score_static`` (with their no_grad decorators) and scripts/dn01_bvi_fnoise ``run`` (low AND normal clip) each run dvp / dvp+agf; (2) the grad-mode switch is
CONFINED to ``f3_baselines.dvp_filter`` and every OTHER variant's output is byte-identical with and without the DVP variants in the grid. A tiny fake depth model and a tiny
DVP UNet / VGG keep it on CPU; the DVP training loop, the memo and the record are the real code."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

import bench_joint as bj
from evaluation import f3_baselines as f3

ROOT = Path(__file__).resolve().parents[1]
BANK = any(Path(p).is_file() for p in __import__("data.elvis_noise", fromlist=["x"]).ELVIS_PHI_PATHS)
BASEGRID = ["none", "ema0.7", "agf0.2t0.06", "agf0.2t0.03"]
FULLGRID = BASEGRID + ["f3blind"]
DVP_VARS = ("dvp", "dvp+agf0.2t0.06", "dvp+agf0.2t0.03")
REC = {"dvp_out_of_range_px", "dvp_floored_px", "dvp_px", "dvp_seconds"}


class _TinyUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, init_features=32):
        super().__init__()
        self.a = nn.Conv2d(in_channels, 4, 3, padding=1)
        self.b = nn.Conv2d(4, out_channels, 3, padding=1)

    def forward(self, x):
        return self.b(torch.relu(self.a(x)))


class _TinyVGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Conv2d(3, 2, 3, padding=1)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return [self.c(x)] * 5


def _fake_depth(dev):
    """A metric 'depth model' (window 1): depth = 2 + 6 * luma of its input frames (metres)."""
    return dict(fwd=lambda x: (2.0 + 6.0 * x.mean(1), None), metric=True, window=1, weights={})


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setitem(bj.DEPTH_MODELS, "f3fake", _fake_depth)
    monkeypatch.setattr(f3, "_unet_cls", lambda repo: _TinyUNet)
    monkeypatch.setitem(f3._VGG, "cpu", _TinyVGG())
    monkeypatch.setattr(f3, "DVP_EPOCHS", 3)
    monkeypatch.setattr(f3.dvp_filter, "__defaults__", (3, None))
    f3.MEMO.key = f3.MEMO.out = None
    f3.MEMO.stats = {}
    return tmp_path


def _clip(n=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    clean = torch.rand(n, 3, 28, 42, generator=g) * 0.6 + 0.2
    gt = (2.0 + 6.0 * torch.rand(n, 30, 40, generator=g)).double().numpy()
    return clean, gt


def _non_dvp_equal(a: dict, b: dict) -> None:
    for v in BASEGRID:
        assert json.dumps(a[v], sort_keys=True) == json.dumps(b[v], sort_keys=True), v


def _check_dvp(res: dict, suffixes=("",)) -> None:
    for v in DVP_VARS:
        assert {k + s for k in REC for s in suffixes} <= set(res[v]), v
    for v in BASEGRID:
        assert not any(k.startswith("dvp_") for k in res[v]), v


# ----------------------------------------------------------------------------- (1)+(2) scripts/dn01_heldout_eval.score / score_static (@torch.no_grad)

def test_heldout_score_and_score_static_run_dvp_under_their_no_grad_decorators(env):
    from scripts import dn01_heldout_eval as H
    p = bj.build_pipeline("f3fake", "cpu")
    clean, gt = _clip()
    x = (clean * 0.3).clamp(0, 1)
    proto = bj.DATASETS["kitti"]
    grad_before = torch.is_grad_enabled()
    a = H.score(p, x, clean, gt, proto, "real", 0.0, 0.0, None, BASEGRID)
    b = H.score(p, x, clean, gt, proto, "real", 0.0, 0.0, None, bj._grid_variants(argparse.Namespace(grid=FULLGRID, model="f3fake")))
    assert torch.is_grad_enabled() == grad_before
    _non_dvp_equal(a, b)
    _check_dvp(b)
    assert b["dvp"]["abs_rel_aligned"] != b["none"]["abs_rel_aligned"]                       # DVP actually changed the depth
    if BANK:
        s1 = H.score_static(p, clean, gt, proto, "kitti", "c1", "st_llb3", "cpu", BASEGRID)
        s2 = H.score_static(p, clean, gt, proto, "kitti", "c1", "st_llb3", "cpu", list(b))
        _non_dvp_equal(s1, s2)
        _check_dvp(s2)


# ----------------------------------------------------------------------------- bench_joint cmd_run (apply_variant outside run_pipeline's no_grad)

@pytest.mark.skipif(not BANK, reason="ELVIS profile bank (low-light cells) not on this host")
def test_bench_cmd_run_dvp_and_other_variants_byte_identical(env, monkeypatch):
    clean, gt = _clip()
    man = env / "manifest.json"
    man.write_text("{}")
    monkeypatch.setitem(bj.PATHS, "kitti_manifest", str(man))
    monkeypatch.setattr(bj, "load_clips", lambda ds, split, limit=None: [("clipA", [{}] * 8)])
    monkeypatch.setattr(bj, "load_frames", lambda ds, frames: (clean.clone(), (30, 40)))
    monkeypatch.setattr(bj, "load_gt", lambda ds, frames: gt.copy())
    import evaluation.sky_metrics as sm       # F3 x sky-fix merge: cmd_run now reads KITTI sky masks from image files; this test feeds fake {} frames, so stub the mask (the test is about DVP byte-identity)
    monkeypatch.setattr(sm, "kitti_sky_masks", lambda root, frames, crop, g: np.zeros(g.shape, dtype=bool))

    def run(grid, out):
        bj.cmd_run(argparse.Namespace(model="f3fake", out=str(out), dataset=["kitti"], cells=["llb3", "st_llb3"], split="dev", limit_clips=None, device="cpu",
                                      temporal=False, grid=grid, diag=False, allow_test=False))
    run(BASEGRID, env / "a")
    run(FULLGRID, env / "b")
    for c in ("llb3", "st_llb3"):
        for v in BASEGRID:
            name = bj.art_path(".", 'f3fake' if v == 'none' else 'f3fake+' + v, "kitti", c, "dev").name      # v3: the __v3 suffix
            assert (env / "a" / name).read_bytes() == (env / "b" / name).read_bytes(), name      # (2) byte-identical files
        for v in DVP_VARS:
            art = json.loads(bj.art_path(env / "b", f"f3fake+{v}", "kitti", c, "dev").read_text())
            assert REC <= set(art["per_video"]["clipA"]) and art["depth_post"] == v.split("+")


# ----------------------------------------------------------------------------- scripts/dn01_bvi_fnoise.run: DVP on the low AND the normal clip

def test_bvi_fnoise_run_dvp_on_low_and_normal_with_limit(env, monkeypatch):
    from scripts import dn01_bvi_fnoise as B
    reg = env / "registration.json"
    reg.write_text(json.dumps({"sequences": {"S1__10": {"included": True}, "S2__10": {"included": True}}, "excluded": []}))
    clean, _ = _clip(n=12)
    monkeypatch.setattr(B, "checked_files", lambda root, key, rec: {"low": "low", "normal": "normal"})
    monkeypatch.setattr(B, "load", lambda files: (clean * (0.25 if files == "low" else 1.0)).clone())

    def run(grid, out, limit):
        B.run(argparse.Namespace(model="f3fake", root="unused", registration=str(reg), device="cpu", relative=False, grid=grid, out=str(out), limit_clips=limit))
    run(BASEGRID, env / "a", 1)
    run(FULLGRID, env / "b", 1)
    for v in BASEGRID:
        name = f"{bj._slug('f3fake' if v == 'none' else 'f3fake+' + v)}__bvi_rlv_test__fnoise.json"
        assert (env / "a" / name).read_bytes() == (env / "b" / name).read_bytes(), name
    art = {v: json.loads((env / "b" / f"f3fake{'' if v == 'none' else '+' + v}__bvi_rlv_test__fnoise.json").read_text()) for v in FULLGRID[:-1] + list(DVP_VARS)}
    assert all(list(a["per_sequence"]) == ["S1__10"] and a["limit_clips"] == 1 for a in art.values())          # --limit-clips 1: one sequence
    _check_dvp({v: a["per_sequence"]["S1__10"] for v, a in art.items()}, suffixes=("_low", "_normal"))       # DVP ran on BOTH clips
    seq = art["dvp"]["per_sequence"]["S1__10"]
    assert seq["dvp_px_low"] == seq["dvp_px_normal"] == 12 * 28 * 42
    run(BASEGRID, env / "c", None)
    assert list(json.loads((env / "c" / "f3fake__bvi_rlv_test__fnoise.json").read_text())["per_sequence"]) == ["S1__10", "S2__10"]
    assert "limit_clips" not in json.loads((env / "c" / "f3fake__bvi_rlv_test__fnoise.json").read_text())   # default: unchanged artifact


# ----------------------------------------------------------------------------- (2) the grad-mode switch is confined to dvp_filter

def test_grad_mode_switch_is_confined_to_dvp_filter():
    src = (ROOT / "evaluation/f3_baselines.py").read_text()
    code = [ln.strip() for ln in src.splitlines() if ("enable_grad" in ln or "inference_mode" in ln or "set_grad_enabled" in ln) and ln.strip().startswith(("with ", "torch."))]
    assert code == ["with torch.inference_mode(False), torch.enable_grad():"]                    # one grad-mode switch in the whole module ...
    body = src.split("def dvp_filter(")[1].split("\ndef ")[0]
    assert "    with torch.inference_mode(False), torch.enable_grad():" in body                    # ... and it is inside dvp_filter
    for f in ("bench_joint.py", "scripts/dn01_heldout_eval.py", "scripts/dn01_bvi_fnoise.py", "evaluation/depth_probe.py"):
        t = (ROOT / f).read_text()
        assert "enable_grad" not in t and "inference_mode(False)" not in t, f
