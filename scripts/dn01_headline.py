"""DN-01 protocol v3 WP9: the HEADLINE WRAPPER, one command per model, one output directory per model.

    python scripts/dn01_headline.py --model <bench_joint spec> --out-root DIR [--tag] [--ladder] [--limit-clips N] [--bvi-root R] [--plan]

Output: ``DIR/<slug(spec)>/`` holding ``RUN_MANIFEST.json`` (spec, checkpoint sha, commit, host, stages, hashes), ``bench/`` (bench_joint result files), ``heldout/`` (nuScenes-night TEST), ``fnoise/`` (BVI F_noise).
Stages (each skipped when every expected file already exists = resume by file existence; bench_joint / heldout_eval resume per cell on top of that):
  1 bench     the sets x {llvar, llvarf} with ``--temporal --split all --allow-test`` (psnr / ssim / tof_native + metric / aligned AbsRel, delta1, TAE come from the same scorer) + the static cells
              (st_llvar, st_llb3: static flicker) [+ the noise ladder llb3x1..16 with --ladder]. The sets are the VDA five (KITTI, Bonn, Sintel, ScanNet v2 (rgbfix twin) + its TAE twin, NYUv2) + ScanNet++
              (the fallback ScanNet* column, scored until parity decides) and ``kitti_minus_0047`` (derived, no GPU, from the kitti files: the KITTI row minus the VKITTI-replica drive). A set whose manifest is
              missing on this host is REPORTED as unavailable in the run manifest (never silently dropped).
  2 night     nuScenes-night TEST (real low light) through dn01_heldout_eval, ``--temporal``; its manifest / projections / RGB files are hash-checked first (R42).
  3 fnoise    BVI-RLV F_noise through dn01_bvi_fnoise (needs --bvi-root; without it the stage is recorded as SKIPPED, not hidden).
``--tag`` appends the TAG depth post-filter (agf0.2t0.06) to the spec unless it already has a post-filter token: ours = fine-tune + TAG + restorer is always evaluated with TAG (protocol v3, K96).
Every non-dev access is appended to TEST_access_log.jsonl by bench_joint (log_test_access)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

TAG = "agf0.2t0.06"
HEADLINE_SETS = ("kitti", "bonn", "sintel_rgbfix", "scannet_rgbfix", "scannet_tae_rgbfix", "nyuv2", "scannetpp")
NO_POSE_SETS = ("nyuv2", "scannet_rgbfix")                   # no per-frame poses in their manifests: TAE is blank (dry run 2026-10-04: scannet_rgbfix KeyError 'K')
MOVING_CELLS =("llvar", "llvarf")
STATIC_CELLS = ("st_llvar", "st_llb3")
LADDER_CELLS = tuple(f"llb3x{k}" for k in (1, 2, 4, 8, 16))
NIGHT_SET, NIGHT_CELL = "nuscenes_night_test", "real"
DERIVED = {"kitti_minus_0047": ("kitti", "2011_10_03_drive_0047_sync")}
REGISTRATION = REPO / "provenance" / "dn01_fnoise" / "registration.json"


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def with_tag(spec: str) -> str:
    """``spec`` with the TAG post-filter appended, unless one of its '+' tokens already is a depth post-filter (agf / ema / sgf / ...)."""
    import bench_joint as bj
    return spec if any(bj.DEPTH_POST_RE.fullmatch(t) for t in spec.split("+")) else f"{spec}+{TAG}"


def checkpoint_files(spec: str) -> dict[str, str]:
    """{token: path} of every checkpoint file a spec names (``ours:<ckpt>``, ``ours_up:``, ``<restorer>:<ckpt>``, ``*_ft:<ckpt>``)."""
    return {t: t.split(":", 1)[1] for t in spec.split("+") if ":" in t and Path(t.split(":", 1)[1]).is_file()}


def entry_main(argv: list[str]) -> None:
    """``--bj`` mode: register the WP8 sets + the ScanNet aliases in bench_joint, harden the long-clip paths exactly as the WP8 parity driver does, wire the TAE cameras, then run bench_joint's CLI."""
    import torch

    import bench_joint as bj
    # every set (incl. scannet_rgbfix, scannet_tae_rgbfix with its [10:180] scoring slice, scannetpp) is registered by bench_joint itself since WP8 (e1dba2d): nothing is re-registered or re-wrapped here
    orig_cam = bj.camera

    def camera(ds, key, frames):                              # TAE poses come from bench_joint (KITTI / Bonn / Sintel / scannet_tae_rgbfix / ScanNet++, WP8); NYU and the 90-frame scannet_rgbfix have none (TAE is blank there by protocol; the TAE column is the scannet_tae_rgbfix twin)
        return None if ds in NO_POSE_SETS else orig_cam(ds, key, frames)
    bj.camera = camera
    orig_lag, orig_run, orig_score = bj.lag_metrics, bj.run_pipeline, bj.score_clip

    def lag_or_skip(*a, **k):                                   # descriptive only; whole-clip tensors OOM a 24 GB card on 500-frame clips (WP8 finding)
        if a[0].shape[0] > 200:
            return {}
        try:
            return orig_lag(*a, **k)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            return {}

    def run_retry(*a, **k):                                     # an inference OOM caused by a neighbour on the shared card is retried, never silently skipped
        for i in range(8):
            try:
                return orig_run(*a, **k)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"run_pipeline: CUDA OOM, retry {i + 1}/8", flush=True)
                time.sleep(20 + 10 * i)
        return orig_run(*a, **k)

    def score_free(*a, **k):
        torch.cuda.empty_cache()
        return orig_score(*a, **k)
    bj.lag_metrics, bj.run_pipeline, bj.score_clip = lag_or_skip, run_retry, score_free
    sys.argv = ["bench_joint.py", *argv]
    bj.main()


def available_sets(wanted: list[str], workdir: Path) -> tuple[list[str], dict[str, str]]:
    """(sets whose manifest exists on this host, {set: reason} for the others), by the same registry bench_joint runs from."""
    import bench_joint as bj
    ok, missing = [], {}
    for ds in wanted:
        man = bj.PATHS.get(f"{ds}_manifest")
        if ds not in bj.DATASETS:
            missing[ds] = "dataset not registered"
        elif not man or not Path(man).is_file():
            missing[ds] = f"manifest missing: {man}"
        else:
            ok.append(ds)
    return ok, missing


def bench_expected(out_dir: Path, spec: str, sets: list[str], cells: list[str]) -> list[Path]:
    import bench_joint as bj
    return [bj.art_path(out_dir / "bench", spec, ds, c, "all") for ds in sets for c in cells]


def plan(spec: str, out_root: Path, sets: list[str], ladder: bool, limit: int | None, bvi_root: str | None, device: str = "cuda", static: bool = True) -> list[dict]:
    """The stages as {name, cmd, expected}: pure (nothing is run, nothing read from the model)."""
    import bench_joint as bj
    d = out_root / bj._slug(spec)
    cells = list(MOVING_CELLS) + (list(STATIC_CELLS) if static else []) + (list(LADDER_CELLS) if ladder else [])
    lim = ["--limit-clips", str(limit)] if limit else []
    py = sys.executable
    stages = [{"name": "bench", "cmd": [py, str(Path(__file__).resolve()), "--bj", "run", "--model", spec, "--dataset", *sets, "--cells", *cells, "--split", "all", "--allow-test",
                                        "--temporal", "--device", device, "--out", str(d / "bench"), *lim],
               "expected": [str(p) for p in bench_expected(d, spec, sets, cells)], "limit": limit},
              {"name": "night", "cmd": [py, str(REPO / "scripts" / "dn01_heldout_eval.py"), "run", "--model", spec, "--dataset", NIGHT_SET, "--cells", NIGHT_CELL, "--split", "test",
                                        "--allow-test", "--temporal", "--device", device, "--out", str(d / "heldout"), *lim],
               "expected": [str(d / "heldout" / f"{bj._slug(spec)}__{NIGHT_SET}__{NIGHT_CELL}__test.json")], "limit": limit}]
    if bvi_root:
        stages.append({"name": "fnoise", "cmd": [py, str(REPO / "scripts" / "dn01_bvi_fnoise.py"), "run", "--model", spec, "--root", bvi_root, "--registration", str(REGISTRATION),
                                                  "--device", device, "--out", str(d / "fnoise"), *lim],
                       "expected": [str(d / "fnoise" / f"{bj._slug(spec)}__bvi_rlv_test__fnoise.json")], "limit": limit})
    else:
        stages.append({"name": "fnoise", "cmd": None, "expected": [], "skipped": "no --bvi-root (BVI-RLV frames are not extracted on this host)"})
    return stages


def derive_alias(src_file: Path, dst_file: Path, alias: str, base: str, drop_token: str) -> dict:
    """``alias`` result file from the ``base`` result file: the clips whose key contains ``drop_token`` removed, ``video_macro`` recomputed (same nanmean over the remaining clips).
    Exact, because every clip is scored independently with the base dataset's seeds; costs no GPU."""
    import numpy as np
    art = json.loads(Path(src_file).read_text())
    pv = {k: v for k, v in art["per_video"].items() if drop_token not in k}
    keys = [k for k in next(iter(pv.values())) if k not in ("frames", "llvar_draw")]
    art.update({"dataset": alias, "source_dataset": base, "dropped_clips": [k for k in art["per_video"] if drop_token in k], "derived_from": {"file": Path(src_file).name, "sha256": sha256_file(src_file)},
                "num_clips": len(pv), "per_video": pv, "video_macro": {k: float(np.nanmean([v[k] for v in pv.values()])) for k in keys}})
    Path(dst_file).write_text(json.dumps(art, indent=2))
    return art


def derive_all(out_dir: Path, spec: str) -> list[str]:
    import bench_joint as bj
    done = []
    for alias, (base, tok) in DERIVED.items():
        for c in list(MOVING_CELLS) + list(STATIC_CELLS) + list(LADDER_CELLS):
            src, dst = bj.art_path(out_dir / "bench", spec, base, c, "all"), bj.art_path(out_dir / "bench", spec, alias, c, "all")
            if src.exists() and not dst.exists():
                derive_alias(src, dst, alias, base, tok)
                done.append(dst.name)
    return done


def verify_hashes(entries: list[tuple[Path, str]]) -> dict:
    """sha256-check ``(path, expected)`` pairs; raises on the first missing / differing file (a hash mismatch must stop the TEST read), returns the count."""
    for p, want in entries:
        if not Path(p).is_file():
            raise RuntimeError(f"R42: {p} is missing")
        if sha256_file(p) != want:
            raise RuntimeError(f"R42: {p}: sha256 differs from the committed list")
    return {"n_files": len(entries)}


def nuscenes_night_entries(limit: int | None = None) -> tuple[list[tuple[Path, str]], dict]:
    """(RGB + projection hash pairs of the night-TEST scenes this run reads, manifest record). The committed manifest sha is checked by ``night_test_manifest`` itself."""
    import dn01_heldout_eval as he
    man, proj_dir = he.night_test_manifest()
    B = he.benchmark()
    rgb = {ln.split(maxsplit=1)[1].strip(): ln.split()[0] for ln in (he.NIGHT_TEST_SRC / "TEST_RGB.sha256").read_text().splitlines() if ln.strip()}
    proj = {ln.split()[1]: ln.split()[0] for ln in (he.NIGHT_TEST_SRC / "PROJECTIONS.sha256").read_text().splitlines() if ln.strip()}
    scenes = set(man["selection"]["night"][:limit] if limit else man["selection"]["night"])
    entries = []
    for fr in man["frames"]:
        if fr["split"] == "night" and fr["scene"] in scenes:
            entries.append((Path(B.NUSCENES_ROOT) / fr["rgb"], rgb[fr["rgb"]]))
            entries.append((proj_dir / Path(fr["projection"]).name, proj[fr["projection"]]))
    rec = {"manifest_sha256": he.NIGHT_TEST_MANIFEST_SHA256, "rgb_list_sha256": sha256_file(he.NIGHT_TEST_SRC / "TEST_RGB.sha256"),
           "projections_list_sha256": sha256_file(he.NIGHT_TEST_SRC / "PROJECTIONS.sha256"), "scenes": len(scenes)}
    return entries, rec


def check_resumable(paths, limit: int | None) -> None:
    """Auditor WP18 B5: resume-by-existence only for result files of THIS run's ``limit_clips`` and ``bench_version`` (a --limit-clips smoke in the same out-root is never taken for the full result).
    Raises on the first mismatch; a file without a ``limit_clips`` key counts as None."""
    import bench_joint as bj
    for p in paths:
        if not Path(p).exists():
            continue
        art = json.loads(Path(p).read_text())
        if art.get("limit_clips") != limit or ("bench_version" in art and art["bench_version"] != bj.BENCH_VERSION):
            raise RuntimeError(f"REFUSED to resume {p}: limit_clips={art.get('limit_clips')!r} bench_version={art.get('bench_version')!r} "
                               f"vs this run limit_clips={limit!r} bench_version={bj.BENCH_VERSION!r} (use a clean out-root)")


def run_stage(st: dict) -> str:
    if st.get("skipped"):
        return "skipped"
    check_resumable(st["expected"], st.get("limit"))
    if st["expected"] and all(Path(p).exists() for p in st["expected"]):
        return "resumed"
    r = subprocess.run(st["cmd"], cwd=str(REPO))
    return "ok" if r.returncode == 0 else f"failed({r.returncode})"


def write_manifest(d: Path, man: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "RUN_MANIFEST.json").write_text(json.dumps(man, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--bj":
        entry_main(argv[1:])
        return 0
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="bench_joint spec, e.g. metric_vda_s, ours:<ckpt>, ours:<ckpt>+agf0.2t0.06, dav2_s_ft:<ckpt>")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--tag", action="store_true", help="append the TAG post-filter (agf0.2t0.06) unless the spec already has a post-filter")
    ap.add_argument("--ladder", action="store_true", help="add the noise ladder llb3x1..16 (supplement) on every set")
    ap.add_argument("--sets", nargs="+", default=None, help=f"override the set list (default {list(HEADLINE_SETS)})")
    ap.add_argument("--limit-clips", type=int, default=None, help="smoke use; recorded in the manifest and every result")
    ap.add_argument("--bvi-root", default=None, help="the extracted BVI-Lowlight-videos directory (F_noise stage)")
    ap.add_argument("--no-static", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--plan", action="store_true", help="print the stages and exit (no GPU, no model)")
    a = ap.parse_args(argv)
    import bench_joint as bj
    spec = with_tag(a.model) if a.tag else a.model
    out_root = Path(a.out_root)
    d = out_root / bj._slug(spec)
    wanted = list(a.sets or HEADLINE_SETS)             # the headline ScanNet is the 90-frame protocol (PROTOCOL_TABLE); the 500-frame twin is parity-only (scripts/dn01_v3_vda5_parity.py)
    sets, missing = available_sets(wanted, d)
    stages = plan(spec, out_root, sets, a.ladder, a.limit_clips, a.bvi_root, a.device, not a.no_static)
    if a.plan:
        for st in stages:
            print(st["name"], "->", st.get("skipped") or " ".join(st["cmd"]))
        print("unavailable sets:", missing)
        return 0
    man = {"spec": spec, "spec_as_given": a.model, **bj.ckpt_info(spec),"tag": any(t == TAG for t in spec.split("+")), "checkpoints": {t: {"path": p, "sha256": sha256_file(p)} for t, p in checkpoint_files(spec).items()},
           "ramen_commit": bj.git_state(REPO), "bench_version": bj.BENCH_VERSION, "host": socket.gethostname(), "argv": argv, "limit_clips": a.limit_clips, "ladder": a.ladder,
           "sets_run": sets, "sets_unavailable": missing, "stages": {}, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    prev = d / "RUN_MANIFEST.json"
    if prev.exists() and json.loads(prev.read_text()).get("limit_clips") != a.limit_clips:
        raise SystemExit(f"REFUSED: {prev} has limit_clips={json.loads(prev.read_text()).get('limit_clips')!r}, this run has {a.limit_clips!r} (use a clean out-root)")
    write_manifest(d, man)
    entries, rec = nuscenes_night_entries(a.limit_clips)
    man["nuscenes_night_test"] = {**rec, **verify_hashes(entries)}              # R42: stops here on a hash mismatch, before any night-TEST frame is scored
    for st in stages:
        man["stages"][st["name"]] = run_stage(st)
        if st["name"] == "bench":
            man["derived"] = derive_all(d, spec)
        write_manifest(d, man)
    man["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_manifest(d, man)
    print(json.dumps(man["stages"]), flush=True)
    return 0 if all(v in ("ok", "resumed", "skipped") for v in man["stages"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
