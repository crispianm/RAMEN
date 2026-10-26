"""RAMEN release loader: a composed ckpt (rgb_restorer) must NOT load silently through load_d3_checkpoint / ours:; load_ramen round-trips v2 and v3 packs (CPU, tiny synthetic nets built on the headc-b a1h arm; skipped without the VDA-S weights)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train  # noqa: E402

ARM = "dn01_v3_headc_b_a1h"
LOCAL = {"vda_repo": "/mnt/DATA/DegradedDepth/competition/Video-Depth-Anything",
         "vda_checkpoint_path": "/mnt/DATA/DegradedDepth/RAMEN/checkpoints/metric_video_depth_anything_vits.pth",
         "da_v2_checkpoint_path": "/mnt/DATA/DegradedDepth/RAMEN/checkpoints/depth_anything_v2_vits.pth", "st_temp2_init": None}
needs_all = pytest.mark.skipif(not Path(LOCAL["vda_checkpoint_path"]).is_file() or not Path(LOCAL["vda_repo"]).is_dir(),
                               reason="Metric-VDA-S weights / VDA repo not on this host")


def _cfg(name, **model):
    cfg = train.load_config(ROOT / "configs" / "experiments" / f"{name}.yaml")
    cfg["model"].update(**LOCAL, **model)
    return cfg


def _mk(**model):
    from models import build_model
    torch.manual_seed(0)
    return build_model(_cfg(ARM, **model)).eval()


def _jitter(n, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in n.parameters():
            if p.requires_grad:
                p.add_(torch.randn(p.shape, generator=g) * 0.01)
        n.gate.g.fill_(0.9); n.gate2.g.fill_(0.3)


def _pack(wide_depth="wide4x", wide_rgb="wide2x"):
    """depth-width plain net (its trainable state = trainable_ema) + a rgb-width plain net (its restorer = rgb_restorer)."""
    dn = _mk(st_stage2_width=wide_depth); _jitter(dn, 1)
    rn = _mk(st_stage2_width=wide_rgb); _jitter(rn, 2)
    sd = {k: v.clone() for k, v in torch.nn.ModuleDict(dn.trainable_modules()).state_dict().items()}
    rr = {k: v.clone() for k, v in torch.nn.ModuleDict({"temp2": rn.temp2, "block2": rn.block2, "gate2": rn.gate2}).state_dict().items()}
    return {"trainable_ema": sd, "rgb_restorer": rr}


def _ck(rgb_width, depth_width="wide4x", own_width="wide2x"):
    pk = _pack(depth_width, own_width)
    m = {"st_stage2_width": depth_width, "st_decoupled_restorer": True}
    if rgb_width:
        m["st_rgb_stage2_width"] = rgb_width
    cfg = _cfg(ARM, **m)
    return {"cfg": cfg, **pk, "step": 1}


def _save(tmp_path, ck, name):
    p = tmp_path / name
    torch.save(ck, p)
    return str(p)


@needs_all
def test_load_d3_checkpoint_raises_on_composed_forced_fail(tmp_path):
    from models import build_model, load_d3_checkpoint
    ck = _ck("wide2x")
    net = build_model(ck["cfg"])
    with pytest.raises(ValueError, match="composed checkpoint"):
        load_d3_checkpoint(net, ck)                                  # forced fail: silent drop is the trap
    with pytest.raises(ValueError, match="composed checkpoint"):
        load_d3_checkpoint(net, _save(tmp_path, ck, "c.pt"))


@needs_all
def test_depth_only_passes_and_plain_ckpt_unchanged(tmp_path):
    from models import build_model, load_d3_checkpoint
    ck = _ck("wide2x")
    plain = {k: v for k, v in ck.items() if k != "rgb_restorer"}
    net = build_model({**ck["cfg"], "model": {**ck["cfg"]["model"], "st_decoupled_restorer": False, "st_rgb_stage2_width": None, "st_stage2_width": "wide4x"}})
    load_d3_checkpoint(net, dict(plain, rgb_restorer=ck["rgb_restorer"]), depth_only=True)    # explicit opt-in: ok
    load_d3_checkpoint(net, plain)                                                              # no display head: ok


@needs_all
def test_bench_joint_ours_raises_on_composed(tmp_path):
    import bench_joint
    with pytest.raises(ValueError, match="l15d:"):
        bench_joint.load_ours(_save(tmp_path, _ck("wide2x"), "c.pt"), "cpu")


@needs_all
@pytest.mark.parametrize("rgb_width", [None, "wide2x"], ids=["v2", "v3"])
def test_load_ramen_roundtrip(tmp_path, rgb_width):
    from models import load_ramen
    ck = _ck(rgb_width, own_width="wide2x") if rgb_width else _ck(None, depth_width="wide2x", own_width="wide2x")
    net = load_ramen(_save(tmp_path, ck, "r.pt"))
    assert not net.training and net.decoupled_restorer
    for k, v in ck["rgb_restorer"].items():                          # display head landed in the own restorer
        mod, rest = k.split(".", 1)
        assert torch.equal(getattr(net, mod).state_dict()[rest], v)


@needs_all
def test_load_ramen_rejects_plain_ckpt_forced_fail(tmp_path):
    from models import load_ramen
    ck = _ck("wide2x"); ck.pop("rgb_restorer")
    with pytest.raises(ValueError, match="not a composed"):
        load_ramen(_save(tmp_path, ck, "p.pt"))


@needs_all
def test_streaming_load_ignores_training_host_temp2_init_path(tmp_path):
    """infer.load (streaming --selftest path) must not try to read cfg model.st_temp2_init (a training-host path)."""
    import infer
    n = _mk(st_stage2_width="wide2x"); _jitter(n, 1)
    cfg = _cfg(ARM, st_stage2_width="wide2x"); cfg["model"]["st_temp2_init"] = "/projects/nonexistent/temp2_distilled.pt"
    p = tmp_path / "s.pt"
    torch.save({"cfg": cfg, "trainable_ema": torch.nn.ModuleDict(n.trainable_modules()).state_dict(), "step": 1}, p)
    model, res = infer.load(str(p), "cpu")
    assert res == 518 or res > 0 and not model.training
    cfg2 = {**cfg, "model": {**cfg["model"], "st_temp2_init": "/projects/nonexistent/temp2_distilled.pt"}}
    from models import build_model
    with pytest.raises(Exception):                       # forced-fail control: WITHOUT the override the build reads the missing path
        build_model(cfg2)
