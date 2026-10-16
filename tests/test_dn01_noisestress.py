"""DN-01 noise-stress ladder (bench_joint): additive cells; k=1 reproduces llb3 / st_llb3 bit for bit; existing cells and keys unchanged."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

import bench_joint as bj

BASE = "e564a845d3f8cd87e4dc8440d36513d8d3022aea"          # noise-kfix branch: the kittifix tip (already carries F3 / F1-UP / the KITTI fix); the ladder hooks are what this branch adds on top (the noise-stress branch itself pinned ec1a36b)


def _clean(n=4, h=28, w=42, seed=0):
    return torch.rand(n, 3, h, w, generator=torch.Generator().manual_seed(seed)) * 0.8 + 0.1


def test_existing_cells_and_constants_are_unchanged():
    assert bj.LL_CELLS == {"ll": dict(ev=bj.LL_EV_MEDIAN, blur=0.0), "llb3": dict(ev=bj.LL_EV_MEDIAN, blur=3.0), "llhard": dict(ev=bj.LL_EV_MEDIAN - 1.0, blur=3.0)}
    assert bj.LL_EV_MEDIAN == -1.68
    assert set(bj.STATIC_CELLS) == {"st_ll", "st_llb3", "st_llvar"} and bj.LL_KAPPA == 0.186 and bj.CELLS[-1] == "b3s25"
    assert bj.parse_cell("llb3") == (bj.LowLight("llb3"), 0.0) and bj.parse_cell("st_llvar")[0] == "llb3"
    for c in ("ll", "llb3", "llhard", "st_ll", "st_llb3", "st_llvar"):
        assert bj.seed_cell(c) == c                                   # existing seeds derive from the same names as before
    assert bj.ll_cfg("llb3")["kappa"] == bj.LL_KAPPA


def test_existing_source_outside_the_ladder_hooks_is_byte_identical():
    """Every line of bench_joint.py that differs from the base commit is one of the additive / hook lines (no existing cell definition edited)."""
    import difflib, pathlib, subprocess
    try:
        old = subprocess.check_output(["git", "show", f"{BASE}:bench_joint.py"], text=True, cwd=pathlib.Path(bj.__file__).parent)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("base commit not available")
    new = pathlib.Path(bj.__file__).read_text()
    removed = [l[1:] for l in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=0) if l.startswith("-") and not l.startswith("---")]
    allowed = ("def _lowlight", "c = LL_CELLS[cell]", "degrade_clip_lowlight(clean.numpy()", "base = STATIC_CELLS[cell]", "if cell in LL_CELLS:",
               "if c in STATIC_CELLS:", "noise_seed(ds, key, c)", "LL_CELLS[blur][\"ev\"]", "seed = noise_seed(", "\"cell\": c, \"noise_sigma_255\"",
               "    if cell != \"st_llvar\":", "    from data.elvis_noise",
                   # ultrareview 2026-10-02: the TEST guard in load_clips (no cell / scoring definition touched)
                   "def load_clips(dataset: str, split: str, limit: int | None = None)", "VDA manifest ``{dataset:", "clips = load_clips(ds, a.split, a.limit_clips)",
                   "if split != \"dev\" and model is not None and (\"ours:\" in model",
                   "raise SystemExit(f\"bench_joint: --split {split} refuses an ours: / ours_up: spec (",
                   # RE-BASED for protocol v3 (K96, WP7, 2026-10-04): the version constant, the v3 result-file suffix (art_path replaces two inline paths) and the llvar-draws subcommand dispatch.
                   # No cell / seed / degradation / scoring line is among them (those stay pinned by the AST test below and by test_scorer_is_byte_identical_to_the_audited_commit).
                   "BENCH_VERSION = 1", "todo = [c for c in a.cells if any(not (out_dir /", "\"table\": cmd_table, \"plot\": cmd_plot}[a.cmd](a)",
                   "keys = [k for k in next(iter(pv.values())) if k != \"frames\"]",
                   "x = degrade(clean, blur, sigma, seed).to(a.device)",       # unchanged call, one indent level deeper under the llvar branch (draw is None -> exactly the old path)
                   "path = out_dir / f\"{_slug(spec(v))}__{ds}__{c}__{a.split}.json\"",
                   # v3 dataset aliases (kitti_minus_0047: a clip filter; the loop variable / dataset field / --dataset choices are renamed, the base dataset still drives every loader and seed)
                   "for ds in a.dataset:", "\"model\": spec(v), \"dataset\": ds,", "s.add_argument(\"--dataset\", nargs=\"+\", default=DEFAULT_DATASETS, choices=list(DATASETS))",
                   # protocol v3 WP22/23: the per-frame fine-tuned specs (dav2_s_ft: / daac_ft:) join the depth-spec grammar (split_spec)
                   "i = next(i for i, s in enumerate(parts) if s.startswith((\"ours:\", \"ours_up:\")) or s in DEPTH_MODELS)",
                   # protocol v3 WP8 (R+D 2, merged for WP9): manifest key per dataset, the camera() fallback to evaluation/vda5_sets, and the score_slice hook (an identity except for ScanNet TAE,
                   # which scores frames [10:180] of the 192 the model sees): the scorer's INPUTS are routed through it, score_clip itself and every cell definition are unchanged (AST pin below)
                   "clips = [(k, v) for entry in json.load(f)[dataset] for k, v in entry.items()]", "keys = [k for entry in json.load(f)[dataset] for k in entry]",
                   "raise ValueError(f\"no camera poses for {ds}\")", "per_cell[c][v][key] = score_clip(dv, p[\"metric\"], gt, restored, x, clean.to(a.device),",
                   "proto[\"max_depth\"], proto[\"min_depth\"], cam, tofnet,",
                   # protocol v3 WP18 part 2 (Auditor B4/B5 + Table addendum e): the result file gains ckpt_sha256 / ckpt_basename (additive keys on the "split" line), and _results validates
                   # each file (check_aggregable) instead of a bare list comprehension. Result bookkeeping only; no scoring line.
                   "\"split\": a.split, \"limit_clips\": a.limit_clips, \"short_side\"", "DEPTH_POST_RE = re.compile(", "return [json.loads(p.read_text()) for p in sorted(Path(out).glob(","if not p.name.startswith(\"latency__\")]")
    stray = [l for l in removed if not any(a in l for a in allowed)]
    assert not stray, stray


V3_ASSIGNED = {"LLVAR_EV_RANGE", "LLVAR_KAPPA_MULTS", "LLVAR_BLUR", "LLVARF_KAPPA_MULT", "LLVARF_EV_JITTER", "LLVAR_VIDEO_CELLS", "LLVAR_DRAW_DIR"}


def _top_level(src: str):
    """{name: ast dump} of every top-level def / class / simple assignment (docstrings dropped) + the sorted dumps of the other top-level statements."""
    import ast
    named, other = {}, []
    for n in ast.parse(src).body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            if n.body and isinstance(n.body[0], ast.Expr) and isinstance(getattr(n.body[0], "value", None), ast.Constant) and isinstance(n.body[0].value.value, str):
                n.body = n.body[1:] or [ast.Pass()]
            named[n.name] = ast.dump(n)
        elif isinstance(n, ast.Assign) and all(isinstance(t, ast.Name) for t in n.targets):
            for t in n.targets:
                named[t.id] = ast.dump(n)
        elif (isinstance(n, ast.ImportFrom) and n.module == "evaluation" and [a.name for a in n.names] == ["vda5_sets"]) or \
                (isinstance(n, ast.Expr) and ast.unparse(n) == "_vda5.register(sys.modules[__name__])"):
            pass                                                                    # v3 WP8 (R+D 2): the registration of the VDA-five sets at import (the only two top-level statements it adds)
        elif isinstance(n, ast.Import) and {a.name for a in n.names} <= {"datetime", "os"}:
            pass                                                                    # v3 (WP7): the two stdlib imports of the TEST access log, re-based like CHANGED_OR_ADDED
        elif isinstance(n, ast.Assign) and {x.id for t in n.targets for x in ast.walk(t) if isinstance(x, ast.Name)} <= V3_ASSIGNED:
            pass                                                                    # v3 (WP7): the llvar constants (tuple-unpack assignments are not "named")
        elif not (isinstance(n, ast.Expr) and isinstance(getattr(n, "value", None), ast.Constant)):    # the module docstring is not code
            other.append(ast.dump(n))
    return named, sorted(other)


# The ONLY top-level names of bench_joint.py that differ from (or are absent in) the audited base: the ladder cells / hooks and the later ultrareview guards. Everything else, notably the
# scoring maths (score_clip, _depth_errors, degrade, ...), must be AST-identical to the base: the '-'-lines pin above cannot see a line INSERTED into such a function.
CHANGED_OR_ADDED = {"LADDER_CELLS", "LADDER_K", "STATIC_LADDER_CELLS", "depth_post_stages", "ll_cfg", "seed_cell",
                    "_lowlight", "check_split", "cmd_latency", "cmd_run", "degrade_static", "load_clips", "parse_cell",
                    # RE-BASED for protocol v3 (K96, WP7, 2026-10-04): the version constant, the moving-video llvar / llvarf cells (draw, degradation, pinned draw JSON), the v3 result-file name,
                    # the TEST access log, the llvar-draws subcommand and the argparse entry point. score_clip / _depth_errors / degrade and every existing cell definition stay AST-identical.
                    "BENCH_VERSION", "LLVAR_EV_RANGE", "LLVAR_KAPPA_MULTS", "LLVAR_BLUR", "LLVAR_VIDEO_CELLS", "LLVARF_KAPPA_MULT", "LLVARF_EV_JITTER", "LLVAR_DRAW_DIR",
                    "llvar_draw", "degrade_llvar", "llvar_noise_seed", "llvar_draw_table", "llvar_draws_file", "llvar_pin", "art_path", "log_test_access", "cmd_llvar_draws", "main", "DATASET_ALIASES", "select_clips", "camera",    # camera: v3 WP8 fallback to vda5_sets.camera (Sintel / ScanNet*)
                    "ckpt_info", "check_resumable", "check_aggregable", "_results",    # v3 WP18 part 2: checkpoint sha for the TEST log / results, resume + aggregate guards (bookkeeping, no scoring code)
                    "load_frames",    # v3 WP18 B1 (R+D 2): the protocol key ``swap_rb`` flips R/B of the stored frames at load (sintel_rgbfix, scannet_tae_rgbfix); input channel handling only, no scoring maths
                    "_vda_l", "_HOSTS", "DEPTH_MODELS", "DEPTH_POST_RE", "depth_post",    # v3 WP29: the released relative VDA-L spec (path + builder + registry entry) and the FBASE post tokens (kal / oneeuro / flowema -> evaluation/fbase_filters.py); depth_post's old branches are untouched
                    "PERFRAME_FT", "_perframe_ft", "split_spec", "build_pipeline", "FT_VDA", "_vda_ft", "_ours"}   # + protocol v3 WP22/23: the dav2_s_ft: / daac_ft: adapters (spec grammar only; no scoring code); _ours: DN-01 design (c) default-off branch (depth_input == 'restored')


def test_scoring_code_outside_the_ladder_hooks_is_ast_identical_to_the_base():
    """Added lines are constrained too: every top-level definition / constant of bench_joint.py except the whitelisted hook set has an AST identical to the base commit's (so no line can be
    inserted into score_clip / _depth_errors / degrade ...), no definition was removed, and the other top-level statements (imports, ...) are unchanged."""
    import pathlib, subprocess
    try:
        old = subprocess.check_output(["git", "show", f"{BASE}:bench_joint.py"], text=True, cwd=pathlib.Path(bj.__file__).parent)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("base commit not available")
    (o, oo), (n, no) = _top_level(old), _top_level(pathlib.Path(bj.__file__).read_text())
    assert not set(o) - set(n), sorted(set(o) - set(n))              # nothing removed
    assert not (set(n) - set(o)) - CHANGED_OR_ADDED, sorted((set(n) - set(o)) - CHANGED_OR_ADDED)
    assert {"score_clip", "_depth_errors", "degrade"} <= set(o) and not {"score_clip", "_depth_errors", "degrade"} & CHANGED_OR_ADDED
    changed = sorted(k for k in set(o) & set(n) if o[k] != n[k])
    assert not set(changed) - CHANGED_OR_ADDED, sorted(set(changed) - CHANGED_OR_ADDED)
    assert oo == no, "a top-level statement (import / guard / registration) differs from the base"


@pytest.mark.parametrize("ev_shape", [(4, 3, 28, 42)])
def test_k1_reproduces_llb3_bit_for_bit_moving_and_static(ev_shape):
    c = _clean()
    seed = bj.noise_seed("kitti", "clip0", bj.seed_cell("llb3x1"))
    assert seed == bj.noise_seed("kitti", "clip0", "llb3")
    a = bj.degrade(c, bj.parse_cell("llb3")[0], 0.0, bj.noise_seed("kitti", "clip0", bj.seed_cell("llb3")))
    b = bj.degrade(c, bj.parse_cell("llb3x1")[0], 0.0, seed)
    assert torch.equal(a, b)
    sc, _ = bj.static_clip(c, np.ones((4, 4, 4)), n=6)
    s0 = bj.degrade_static(sc, "st_llb3", bj.noise_seed("kitti", "clip0", bj.seed_cell("st_llb3")))
    s1 = bj.degrade_static(sc, "st_llb3x1", bj.noise_seed("kitti", "clip0", bj.seed_cell("st_llb3x1")))
    assert torch.equal(s0, s1)


def test_ladder_noise_grows_monotonically_with_k_and_is_deterministic():
    c = torch.full((4, 3, 48, 64), 0.9)                              # flat: the std of (x - reference) is the noise alone
    std = {}
    for n in bj.LADDER_CELLS:
        x = bj.degrade(c, bj.parse_cell(n)[0], 0.0, 11)
        assert torch.equal(x, bj.degrade(c, bj.parse_cell(n)[0], 0.0, 11)) and x.min() >= 0 and x.max() <= 1
        ref = torch.from_numpy(__import__("data.elvis_noise", fromlist=["x"]).lowlight_reference(c.numpy(), bj.LL_CELLS["llb3"]["ev"]))
        std[n] = float((x - ref).std())
    order = ["llb3x1", "llb3x2", "llb3x4", "llb3kappa1", "llb3x8", "llb3x16"]
    assert [std[n] for n in order] == sorted(std[n] for n in order) and std["llb3x16"] > 3 * std["llb3x1"]


def test_ladder_cell_table():
    assert [bj.LADDER_CELLS[f"llb3x{k}"]["kappa"] for k in (1, 2, 4, 8, 16)] == [bj.LL_KAPPA * k for k in (1, 2, 4, 8, 16)]
    assert bj.LADDER_CELLS["llb3kappa1"]["kappa"] == 1.0 and bj.ll_cfg("llb3x8")["ev"] == bj.LL_CELLS["llb3"]["ev"]
    assert all(bj.ll_cfg(n)["blur"] == 3.0 for n in bj.LADDER_CELLS)
    for n in bj.LADDER_CELLS:
        assert bj.parse_cell(n) == (bj.LowLight(n), 0.0) and bj.parse_cell("st_" + n)[0] == n
        assert bj.seed_cell("st_" + n) == "st_llb3"


def test_agf_grid_variant_tokens_parse():
    from evaluation.depth_probe import expand_grid, parse_variant
    assert expand_grid(["none", "agf0.2t0.06"]) == ["none", "agf0.2t0.06"] and bj.DEPTH_POST_RE.fullmatch("agf0.2t0.06")
    assert parse_variant("agf0.2t0.06") == ["agf0.2t0.06"]


# ------------------------------------------------------------------ every scorer entry: cmd_run (moving + static, grid) with a fake pipeline, cmd_table, the artifact keys
def _run_cmd(tmp_path, monkeypatch, cells, grid=None, n=40):
    import argparse
    clean = _clean(n=n, h=28, w=42)
    gt = np.random.RandomState(1).uniform(2, 30, (n, 28, 42))

    def fake_run(p, x, sigma, blur, out_hw):                              # depth = a function of the degraded input: noise -> depth change
        d = (x.mean(1) * 40 + 1).cpu().numpy().astype(np.float64)
        p["last_guides"] = {"i": x, "r": x}
        return d, x
    monkeypatch.setattr(bj, "build_pipeline", lambda spec, dev: dict(fwd=None, metric=True, window=32, weights={}))
    monkeypatch.setattr(bj, "run_pipeline", fake_run)
    monkeypatch.setattr(bj, "load_clips", lambda ds, split, lim=None: [("clipA", [{}] * n)])
    monkeypatch.setattr(bj, "load_frames", lambda ds, frames: (clean, (28, 42)))
    monkeypatch.setattr(bj, "load_gt", lambda ds, frames: gt)
    monkeypatch.setattr(bj, "_sha", lambda p: "0" * 64)
    monkeypatch.setattr(bj, "git_state", lambda r: "test")
    a = argparse.Namespace(model="metric_vda_s", out=str(tmp_path), dataset=["bonn"], cells=cells, split="dev", limit_clips=None, device="cpu", temporal=False, grid=grid, diag=False, allow_test=False)
    bj.cmd_run(a)
    return {c: json.loads(bj.art_path(tmp_path, "metric_vda_s", "bonn", c, "dev").read_text()) for c in cells}


def test_cmd_run_moving_and_static_ladder_end_to_end(tmp_path, monkeypatch):
    cells = ["llb3", "llb3x1", "llb3x16", "llb3kappa1", "st_llb3", "st_llb3x1", "st_llb3x16", "st_llb3kappa1"]
    r = _run_cmd(tmp_path, monkeypatch, cells)
    for base, rung in (("llb3", "llb3x1"), ("st_llb3", "st_llb3x1")):
        assert json.dumps(r[base]["per_video"], sort_keys=True) == json.dumps(r[rung]["per_video"], sort_keys=True)    # k = 1 == llb3 exactly (full scorer path)
        assert json.dumps(r[base]["video_macro"], sort_keys=True) == json.dumps(r[rung]["video_macro"], sort_keys=True)
    assert "noise_kappa" not in r["llb3"] and "noise_kappa" not in r["st_llb3"]                                            # existing artifacts keep their keys
    assert r["llb3x1"]["noise_kappa"] == bj.LL_KAPPA and r["st_llb3x16"]["noise_kappa_mult_vs_llb3"] == 16 and r["llb3kappa1"]["noise_kappa"] == 1.0
    assert r["llb3x16"]["video_macro"]["psnr_input_native"] < r["llb3kappa1"]["video_macro"]["psnr_input_native"] < r["llb3x1"]["video_macro"]["psnr_input_native"]
    assert r["st_llb3x16"]["video_macro"]["flicker"] > r["st_llb3x1"]["video_macro"]["flicker"]
    assert all(r[c]["noise_sigma_255"] == 0.0 for c in cells)
    for c in cells:
        for k in ("psnr_native", "ssim_native", "psnr_input_native", "abs_rel_aligned", "abs_rel_unaligned", "fit_scale"):
            assert np.isfinite(r[c]["video_macro"][k]), (c, k)


def test_cmd_run_grid_agf_on_ladder_cells(tmp_path, monkeypatch):
    r = _run_cmd(tmp_path, monkeypatch, ["st_llb3x8", "llb3x8"], grid=["none", "agf0.2t0.06"])
    assert bj.art_path(tmp_path, "metric_vda_s+agf0.2t0.06", "bonn", "st_llb3x8", "dev").exists() and bj.art_path(tmp_path, "metric_vda_s+agf0.2t0.06", "bonn", "llb3x8", "dev").exists()
    a = json.loads(bj.art_path(tmp_path, "metric_vda_s+agf0.2t0.06", "bonn", "st_llb3x8", "dev").read_text())
    assert a["video_macro"]["flicker"] < r["st_llb3x8"]["video_macro"]["flicker"] and a["depth_post"] == ["agf0.2t0.06"]


def test_cmd_table_reads_ladder_files(tmp_path, monkeypatch, capsys):
    import argparse
    _run_cmd(tmp_path, monkeypatch, ["llb3x2"])
    bj.cmd_table(argparse.Namespace(out=str(tmp_path)))
    assert "llb3x2" in capsys.readouterr().out
