"""scripts/ramen_l_preflight.py --resume: positive control and one forced fail per refusal class (CPU; synthetic last.pt, fake weights file)."""
from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import ramen_l_preflight as P  # noqa: E402
import train  # noqa: E402

CFG = ROOT / "configs/experiments"
OUT = "runs/_t_ramen_l_resume"


@pytest.fixture()
def env(tmp_path):
    w = tmp_path / "w.pth"; w.write_bytes(b"fake weights" * 10)
    sha = hashlib.sha256(w.read_bytes()).hexdigest()
    full, res = CFG / "_t_full.yaml", CFG / "_t_full_resume.yaml"
    full.write_text(f"extends: ramen_l_c3_n46.yaml\nmodel:\n  vda_checkpoint_sha256: {sha}\ntraining:\n  checkpoint_dir: {OUT}\n")
    res.write_text(f"extends: _t_full.yaml\ntraining:\n  resume_from: {OUT}/last.pt\n")
    (ROOT / OUT).mkdir(parents=True, exist_ok=True)
    yield dict(w=w, sha=sha, size=w.stat().st_size, full=full, res=res)
    for f in (full, res): f.unlink()
    shutil.rmtree(ROOT / OUT, ignore_errors=True)


def make_ckpt(env, step=11000, drop=(), cfg_patch=None, shape=(256, 1024, 1, 1)):
    cfg = train.load_config(env["full"])
    if cfg_patch: cfg = cfg_patch(cfg)
    ck = {"cfg": cfg, "step": step, "micro": step * 4, "epoch": 0, "best": 1.0, "optimizer": {"state": {}, "param_groups": []}, "rng": {"x": 1}, "ema": {},
          "trainable": {"depth_head.projects.0.weight": torch.zeros(*shape)}, "trainable_ema": {"depth_head.projects.0.weight": torch.zeros(*shape)}}
    for k in drop: ck.pop(k)
    torch.save(ck, ROOT / OUT / "last.pt")


def run(env, config=None, *extra):
    return P.main(["--config", str(config or env["res"]), "--resume", "--no-git", "--no-paths", "--out-dir", OUT, "--weights", str(env["w"]), "--weights-sha", env["sha"],
                   "--weights-size", str(env["size"]), *extra])


def test_positive(env, capsys):
    make_ckpt(env); assert run(env) == 0
    o = capsys.readouterr().out; assert "PREFLIGHT OK" in o and "resume=True resume_step=11000" in o


@pytest.mark.parametrize("step,ok", [(500, True), (11500, True), (11000, True), (12000, False), (750, False), (0, False), (11999, False), (12500, False)])
def test_step_rule(env, capsys, step, ok):
    make_ckpt(env, step=step); assert (run(env) == 0) is ok
    if not ok: assert "step" in capsys.readouterr().out


def test_expect_step(env, capsys):
    make_ckpt(env, step=11000); assert run(env, None, "--expect-step", "10500") == 2 and "--expect-step" in capsys.readouterr().out


def test_missing_last_pt_and_dir(env, capsys):
    assert run(env) == 2 and "last.pt does not exist" in capsys.readouterr().out
    shutil.rmtree(ROOT / OUT); assert run(env) == 2 and "does not exist" in capsys.readouterr().out


@pytest.mark.parametrize("drop", ["rng", "optimizer"])
def test_missing_keys(env, capsys, drop):
    make_ckpt(env, drop=(drop,)); assert run(env) == 2 and drop in capsys.readouterr().out


def test_cfg_mismatch(env, capsys):
    def patch(c): c["data"]["seed"] = 47; return c
    make_ckpt(env, cfg_patch=patch); assert run(env) == 2 and "does not resolve to the committed full config" in capsys.readouterr().out


def test_vits_shaped_checkpoint(env, capsys):
    make_ckpt(env, shape=(48, 384, 1, 1)); assert run(env) == 2 and "not ViT-L shaped" in capsys.readouterr().out


@pytest.mark.parametrize("rf", ["/abs/path/last.pt", f"{OUT}/../elsewhere/last.pt", "runs/other_dir/last.pt", f"{OUT}/step6000.pt", "~/x/last.pt"])
def test_resume_from_outside_out_dir(env, capsys, rf):
    make_ckpt(env); env["res"].write_text(f"extends: _t_full.yaml\ntraining:\n  resume_from: {rf}\n")
    assert run(env) == 2 and "resume_from" in capsys.readouterr().out


def test_resume_config_extra_difference(env, capsys):
    make_ckpt(env); env["res"].write_text(f"extends: _t_full.yaml\ntraining:\n  resume_from: {OUT}/last.pt\n  learning_rate: 0.0002\n")
    assert run(env) == 2
    o = capsys.readouterr().out
    assert "beyond training.resume_from" in o and "outside the allow-list" in o


def test_resume_flag_required_for_resume_config(env, capsys):
    make_ckpt(env)
    r = P.main(["--config", str(env["res"]), "--no-git", "--no-paths", "--out-dir", OUT, "--weights", str(env["w"]), "--weights-sha", env["sha"], "--weights-size", str(env["size"])])
    assert r == 2 and "outside the allow-list" in capsys.readouterr().out          # without --resume, resume_from is not allowed


def test_default_mode_still_refuses_existing_out_dir(env, capsys):
    r = P.main(["--config", str(env["full"]), "--no-git", "--no-paths", "--out-dir", OUT, "--weights", str(env["w"]), "--weights-sha", env["sha"], "--weights-size", str(env["size"])])
    assert r == 2 and "already exists" in capsys.readouterr().out


def test_committed_resume_configs_resolve():
    for v in ("", "_r3x", "_r2x"):
        r, f = train.load_config(CFG / f"ramen_l_c3_n46{v}_resume.yaml"), train.load_config(CFG / f"ramen_l_c3_n46{v}.yaml")
        assert r["training"]["resume_from"] == f"runs/ramen_l_c3_n46{v}/last.pt" == f["training"]["checkpoint_dir"] + "/last.pt"
        assert {k for k in P.flat(r) if P.flat(r)[k] != P.flat(f).get(k)} == {"training.resume_from"}
