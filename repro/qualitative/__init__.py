"""Qualitative paper items for generate_results.py (schema: repro/SCHEMA.md, kind: qualitative).

Contract: render_<item id with ':' / '-' -> '_'>(cfg_entry: dict, out_dir: Path) -> list[Path]  (GPU: runs the pinned scorer's inference path).
Scripts in this folder are VENDORED from the cleared packages (see README.md for the exact diffs); each wrapper only builds the CLI call, runs it with out_dir as the output
root, and re-encodes every GIF to H.264 MP4 (yuv420p, crf 24) next to the PNG stills.
cfg_entry['params'] (all optional): device (cuda), smoke (False), max_frames, only (selection idx), eval_clone is read from figures_config.json (must be a CLEAN checkout at 368b5c53;
bench_joint is imported from there - the scripts assert clean + commit).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable


def _run(cmd: list, env_extra: dict | None = None) -> None:
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", **(env_extra or {}))
    subprocess.run([PY, "-B", *map(str, cmd)], check=True, cwd=HERE, env=env)


def _host_args(cfg_entry: dict) -> list:
    """Path selection for the renderers: the scorer clone generate_results.py was given (still asserted at the scorer commit + clean by the renderer) and, on a GH200 with the per-key map active, the Isambard checkpoint paths."""
    args = []
    if (cfg_entry or {}).get("_scorer_clone"):
        args += ["--eval-clone", cfg_entry["_scorer_clone"]]
    try:
        import repro.config as C
        if C.detected_host() == "gh200" and not os.environ.get("DATA_ROOT"):
            args += ["--host", "gh200"]
    except Exception:  # noqa: BLE001 - host detection failure = local paths
        pass
    return args


def gif_to_mp4(gif: Path) -> Path:
    mp4 = gif.with_suffix(".mp4")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(gif), "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "24",
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4)], check=True)
    return mp4


def _figures(cfg_entry: dict, out_dir: Path, only) -> list[Path]:
    p = (cfg_entry or {}).get("params", {}) or {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [HERE / "render_c3_figures.py", "--out", out_dir, "--device", p.get("device", "cuda")]
    if only is not None:
        cmd += ["--only", only]
    if p.get("smoke"):
        cmd += ["--smoke"]
    if p.get("max_frames"):
        cmd += ["--max-frames", p["max_frames"]]
    if p.get("panel_h"):
        cmd += ["--panel-h", p["panel_h"]]
    if p.get("depth_display"):
        cmd += ["--depth-display", p["depth_display"]]                 # Ifig3: abs_gt (default) | v1_log_p2p98 | v2_disparity_p2p98, comma list, one inference pass
    cmd += _host_args(cfg_entry)
    _run(cmd)
    files = sorted(out_dir.glob("*_still.png")) + sorted(out_dir.glob("*_still.pdf")) + sorted(out_dir.glob("*.json")) + sorted(out_dir.glob("*_depth.npz")) + sorted(out_dir.glob("*_meanchange_*"))
    for g in sorted(out_dir.glob("*.gif")):
        files += [g, gif_to_mp4(g)]
    return files


def render_fig_teaser(cfg_entry: dict, out_dir: Path) -> list[Path]:
    """fig:teaser = Bonn median clip, static low-light cell (selection idx 99), TWO versions from one inference pass (commit Ifig):
    teaser_noGT_*   input | Metric-VDA-S FT | RAMEN + TAG | change FT | change RAMEN + TAG
    teaser_withGT_* the same with a 'Ground truth' panel (sensor GT as-is, invalid masked, same metric colour scale) after the input.
    Each: PNG still + PDF still + GIF + MP4 + sidecar JSON (the withGT sidecar carries the GT coverage %)."""
    return _figures(cfg_entry, Path(out_dir), 99)


def _teaser_many(cfg_entry: dict, out_dir: Path, mode: str) -> list[Path]:
    """Ifig4: one teaser per clip of params.clips (one renderer subprocess per clip, one sub-directory per clip). mode: 'dev' / 'test' (benchmark clips with GT, withGT version only) or 'real' (no GT, noGT version only).
    TEST clips: the renderer itself refuses unless --allow-test is passed here AND RAMEN_AUDITOR_TOKEN is set in the environment (qualitative items are not covered by the plan-level TEST gate)."""
    p = (cfg_entry or {}).get("params", {}) or {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    disp = p.get("depth_display", "v1_log_p2p98")
    files: list[Path] = []
    for c in p["clips"]:
        if mode == "real":
            try:
                import repro.config as C
                gh = C.detected_host() == "gh200" and not os.environ.get("DATA_ROOT")
            except Exception:  # noqa: BLE001
                gh = False
            vdir = c["dir_gh200"] if gh else c["dir_local"]
            sub = out_dir / f"real__{c['name']}"
            cmd = [HERE / "render_c3_figures.py", "--out", sub, "--device", p.get("device", "cuda"), "--video-dir", vdir, "--name", c["name"], "--start", c.get("start", 0), "--n", c.get("n", 64),
                   "--stride", c.get("stride", 1), "--depth-display", disp, "--versions", "noGT"]
        else:
            ds, key = c["dataset"], c["clip"]
            sub = out_dir / f"{ds}__{key}"
            cmd = [HERE / "render_c3_figures.py", "--out", sub, "--device", p.get("device", "cuda"), "--clip", f"{ds}:{key}", "--depth-display", disp, "--versions", "withGT"]
            if mode == "test":
                cmd += ["--allow-test"]
        if p.get("smoke"):
            cmd += ["--smoke"]
        if p.get("max_frames"):
            cmd += ["--max-frames", p["max_frames"]]
        if p.get("panel_h"):
            cmd += ["--panel-h", p["panel_h"]]
        if p.get("cumulative_still"):
            cmd += ["--cumulative-still"]                       # Ifig6: still change panels cumulative (user standing rule)
        if p.get("dump_all_depth"):
            cmd += ["--dump-all-depth"]                         # Ifig6: full per-frame depth archive per clip
        if p.get("dump_frames"):
            cmd += ["--dump-frames"]                            # Ifig7: lossless input + restored frames per clip
        cmd += _host_args(cfg_entry)
        sub.mkdir(parents=True, exist_ok=True)
        _run(cmd)
        files += sorted(sub.glob("*_still.png")) + sorted(sub.glob("*_still.pdf")) + sorted(sub.glob("*.json")) + sorted(sub.glob("*_depth.npz")) + sorted(sub.glob("*_meanchange_*")) + sorted(sub.glob("teaser_alldepth_*")) + sorted(sub.glob("teaser_frames_*"))
        for g in sorted(sub.glob("*.gif")):
            files += [g, gif_to_mp4(g)]
    return files


def render_fig_teaser_many_dev(cfg_entry: dict, out_dir: Path) -> list[Path]:
    """fig:teaser_many_dev = the teaser (v1 log display, with GT) on 20 DEV clips of the benchmark datasets (candidate teasers; the released fig:teaser is unchanged)."""
    return _teaser_many(cfg_entry, Path(out_dir), "dev")


def render_fig_teaser_many_test(cfg_entry: dict, out_dir: Path) -> list[Path]:
    """fig:teaser_many_test = the teaser on TEST clips (exposed exploratory, qualitative only; needs --allow-test + RAMEN_AUDITOR_TOKEN)."""
    return _teaser_many(cfg_entry, Path(out_dir), "test")


def render_fig_teaser_many_real(cfg_entry: dict, out_dir: Path) -> list[Path]:
    """fig:teaser_many_real = the teaser (noGT, v1 log display) on real low-light footage with NO ground truth and NO synthetic degradation (horse, BBC, BVI-RLV holdout); qualitative only."""
    return _teaser_many(cfg_entry, Path(out_dir), "real")


def render_fig_long_video(cfg_entry: dict, out_dir: Path) -> list[Path]:
    """fig:long_video (Ifig8) = per-clip DATA for the long-video qualitative figure (VDA Fig. 5 analogue): N=500 frames of ONE DEV clip, dynamic low-light cell llvar (the scored rows' severity draw + noise
    stream), FT n46 / RAMEN n46 raw / RAMEN n46 + TAG (+ DAv2-S FT / DA-AC FT when params.models lists dav2_ft / daac_ft and params.model_pins holds their real checkpoint path + sha256). One renderer subprocess
    per clip -> <out>/<ds>__<clip>/long_<ds>_<clip>.npz + .json (keys: ft, c3_raw, c3_tag, gt, gt_valid, rgb, restored [+ dav2_ft, daac_ft; extras clean, gt_all, gt_valid_all]). No GIF, no figure, no numbers."""
    p = (cfg_entry or {}).get("params", {}) or {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import repro.config as C
        gh = C.detected_host() == "gh200" and not os.environ.get("DATA_ROOT")
    except Exception:  # noqa: BLE001
        gh = False
    hk = "gh200" if gh else "local"
    models = p.get("models") or ["ft", "c3_raw", "c3_tag"]
    files: list[Path] = []
    for c in p["clips"]:
        ds, key = c["dataset"], c["clip"]
        sub = out_dir / f"{ds}__{key}"
        cmd = [HERE / "render_long_video.py", "--out", sub, "--device", p.get("device", "cuda"), "--clip", f"{ds}:{key}", "--start", c.get("start", 0), "--n", c.get("n", 500), "--models", ",".join(models)]
        man = (c.get("manifest") or {})
        if man:
            mp = man.get(hk) or man.get("local")
            mp = str(HERE / mp[len("repo:"):]) if str(mp).startswith("repo:") else mp
            cmd += ["--manifest", mp]
            if man.get("sha256"):
                cmd += ["--manifest-sha256", man["sha256"]]
        if (c.get("root") or {}).get(hk):
            cmd += ["--root", c["root"][hk]]
        if p.get("model_pins"):
            import json as _json
            cmd += ["--model-pins", _json.dumps(p["model_pins"])]
        if p.get("smoke"):
            cmd += ["--smoke"]
        cmd += _host_args(cfg_entry)
        sub.mkdir(parents=True, exist_ok=True)
        _run(cmd)
        files += sorted(sub.glob("*.npz")) + sorted(sub.glob("*.json"))
    return files


def render_fig_teaser_depthdump(cfg_entry: dict, out_dir: Path) -> list[Path]:
    """fig:teaser_depthdump = the SAME teaser clip/inference as fig:teaser (selection idx 99) but writing the FULL per-frame depth arrays of FT n46 / RAMEN n46 + TAG (teaser_alldepth_*.npz + .json) so
    alternative change-map renderings (e.g. cumulative) can be made on CPU. Writes the v1 noGT teaser strip only as a sanity reference; no chart."""
    p = (cfg_entry or {}).get("params", {}) or {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [HERE / "render_c3_figures.py", "--out", out_dir, "--device", p.get("device", "cuda"), "--only", 99, "--depth-display", "v1_log_p2p98", "--versions", "noGT", "--no-teaser-chart", "--dump-all-depth"]
    if p.get("smoke"):
        cmd += ["--smoke"]
    if p.get("max_frames"):
        cmd += ["--max-frames", p["max_frames"]]
    if p.get("panel_h"):
        cmd += ["--panel-h", p["panel_h"]]
    cmd += _host_args(cfg_entry)
    _run(cmd)
    return sorted(out_dir.glob("*_still.png")) + sorted(out_dir.glob("teaser_alldepth_*")) + sorted(out_dir.glob("*.json"))


def render_fig_qual(cfg_entry: dict, out_dir: Path) -> list[Path]:
    """fig:qual = median/failure panels over dev (CLIP_SELECTION.json idx 0..5; params.only selects one). Depth + per-pixel AbsRel rows, FT | RAMEN raw | RAMEN+TAG (n46; the released Metric-VDA-S column is retired)."""
    return _figures(cfg_entry, Path(out_dir), (cfg_entry.get("params") or {}).get("only"))


def render_fig_flicker_curve(cfg_entry: dict, out_dir: Path) -> list[Path]:
    """fig:flicker_curve = c_t per frame (t = 8..63) on the static DEV clips of CLIP_SELECTION.json, FT n46 / RAMEN n46 raw / RAMEN n46 + TAG (the released curve is retired; pre-declared definition,
    see render_flicker_curve.py and configs/paper_results.yaml). DEV only, descriptive. Writes flicker_curve.json + flicker_curve.png + flicker_curve.pdf."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = (cfg_entry or {}).get("params", {}) or {}
    cmd = [HERE / "render_flicker_curve.py", "--out", out_dir, "--device", p.get("device", "cuda")]
    if cfg_entry.get("_smoke"):
        cmd += ["--max-clips", 1]
    cmd += _host_args(cfg_entry)
    _run(cmd)
    return [f for f in (out_dir / "flicker_curve.json", out_dir / "flicker_curve.png", out_dir / "flicker_curve.pdf") if f.exists()]
