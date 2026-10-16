#!/usr/bin/env python3
"""C3 figure renders (AUD-C3-FIGURES-20261012). DEV clips only, chosen by CLIP_RULE.md (CLIP_SELECTION.json). Number-free panels (names only).

  python3 -B render_c3_figures.py --out /mnt/DATA/DegradedDepth/qualitative/12_c3_figures            # real run (4090), after latency
  python3 -B render_c3_figures.py --smoke --device cpu --max-frames 10 --only 0 --out <scratch>      # CPU plumbing smoke (NOT a figure)

Everything model-side goes through the pinned benchmark clone (bench_joint @368b5c53): load_clips('dev'), load_frames, load_gt, llvar_draw / degrade_llvar / llvar_noise_seed
(the SAME per-clip severity and noise stream as the scored rows), build_pipeline + run_pipeline (the scored inference path), apply_variant (TAG on the raw depth), score_slice.
Per-pixel AbsRel uses the benchmark's per-clip least-squares (scale, shift) on disparity (score_clip), checked against bj._depth_errors in dry_check.py."""
import argparse, hashlib, io, json, os, socket, subprocess, sys, time, zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fit_align(pred_depth, gt, max_depth, torch):
    """score_clip's alignment: one least-squares (scale, shift) on disparity over valid GT pixels of the whole clip (fp64 normal equations). Returns aligned depth (clamped), valid mask."""
    pd = torch.from_numpy(pred_depth).to(torch.float64)
    g = torch.from_numpy(gt)
    disp = (1.0 / pd.clamp_min(1e-3)).clamp_min(1e-3)
    valid = (g > 1e-3) & (g < max_depth)
    gd = torch.where(valid, 1.0 / (g + 1e-8), torch.zeros_like(g))
    p, t = disp[valid], gd[valid]
    n = p.new_tensor(float(p.numel()))
    A = torch.stack([torch.stack([(p * p).sum(), p.sum()]), torch.stack([p.sum(), n])])
    s, b = torch.linalg.solve(A, torch.stack([(p * t).sum(), t.sum()]))
    al = (1.0 / (s * disp + b).clamp_min(1e-3)).clamp(1e-3, max_depth)
    return al.numpy(), valid.numpy()


def abs_rel_map(aligned, gt, valid):
    out = np.full(gt.shape, np.nan, dtype=np.float32)
    out[valid] = (np.abs(aligned - gt) / gt)[valid]
    return out


def colour(img01, cmap):
    import matplotlib
    c = matplotlib.colormaps[cmap](np.clip(img01, 0, 1))[..., :3]
    return (c * 255).astype(np.uint8)


def depth_rgb(d, lo, hi):
    return colour((np.log(np.clip(d, lo, hi)) - np.log(lo)) / (np.log(hi) - np.log(lo)), "turbo")


def err_rgb(e, vmax):
    out = colour(np.nan_to_num(e, nan=0.0) / vmax, "magma")
    out[np.isnan(e)] = (70, 70, 70)
    return out


def header(names, w, h=26):
    import cv2
    assert not any(ch.isdigit() for n in names for ch in n), "panel names must be number-free (no digits in any title)"
    bar = np.full((h, w * len(names), 3), 255, np.uint8)
    for i, n in enumerate(names):
        cv2.putText(bar, n, (i * w + 6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return bar


def resize_h(img, h):
    import cv2
    return cv2.resize(img, (round(img.shape[1] * h / img.shape[0]), h), interpolation=cv2.INTER_AREA)


def save_pdf(img, path, dpi=200):
    """Paper-ready PDF of a still: the same RGB pixels as the PNG, embedded losslessly (raster in PDF) at `dpi`."""
    from PIL import Image
    Image.fromarray(img).save(path, "PDF", resolution=float(dpi))


def gt_rgb(g, lo, hi, max_depth, no_gt=(70, 70, 70)):
    """Sensor GT as-is (no densify / interpolate / inpaint): the SAME log-turbo metric colour scale as the depth panels (lo, hi); invalid pixels (<= 1e-3 after the loader's 0 -> -1, or >= max_depth) are MASKED grey.
    Returns the RGB image and the valid mask."""
    valid = (g > 1e-3) & (g < max_depth)
    out = depth_rgb(np.where(valid, g, lo), lo, hi)
    out[~valid] = no_gt
    return out, valid


def resize_h_nearest(img, h):
    import cv2
    return cv2.resize(img, (round(img.shape[1] * h / img.shape[0]), h), interpolation=cv2.INTER_NEAREST)


def save_gif(frames, path, fps=10):
    from PIL import Image
    ims = [Image.fromarray(f).convert("P", palette=Image.ADAPTIVE, colors=128) for f in frames]
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=int(1000 / fps), loop=0, optimize=False, disposal=1)

def apply_host_paths(cfg, host, eval_clone):
    """Path selection only (no value is computed here): --host gh200 swaps each checkpoint to its Isambard path (checkpoint sha256 pins stay as in the config and are asserted by the caller);
    --eval-clone overrides the scorer clone (the caller still asserts HEAD == eval_commit and a clean tree)."""
    if eval_clone:
        cfg["eval_clone"] = str(eval_clone)
    if host == "gh200":
        for c in [cfg["c3"]] + ([cfg["ft"][k] for k in cfg["ft"]] if "ft" in cfg else []):
            if c.get("ckpt_gh200"):
                c["ckpt"] = c["ckpt_gh200"]
            else:
                c["ckpt"] = None          # no Isambard path declared: the checkpoint assert fails loudly
    return cfg


def u8(x):
    return (x.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


DISPLAY_CMAP = "turbo"
DISPLAY_MODES = {"abs_gt": ("", "GT 2nd-98th percentile of the still frame's valid GT, log depth, turbo, shared by all depth panels (Ifig rule)"),
                 "v1_log_p2p98": ("v1", "each depth panel normalised to its OWN 2nd-98th percentile of log depth (GT: valid pixels), turbo, near=cool far=hot; same rule for every panel"),
                 "v2_disparity_p2p98": ("v2", "VDA-style relative inverse depth: each depth panel normalised to its OWN 2nd-98th percentile of disparity 1/depth (GT: valid pixels), turbo, near=hot far=cool; same rule for every panel")}


def write_meanchange_chart(out, stem, m_ft, m_c3, fps=10):
    """Per-frame mean relative depth change (%), mean over ALL pixels of 100*|D_t - D_{t-1}| / max(D_t, 1e-3) (the arrays the change panels colour, unclipped), frames 2..N.
    Layout follows the paper's current teaser curve (serif, grey baseline / blue RAMEN, light grid, legend upper left, y from 0). CSV is the source of every plotted value."""
    import csv, glob
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    for g in ("/usr/share/fonts/**/LiberationSerif-*.ttf", os.path.expanduser("~/.local/share/flatpak/runtime/*/x86_64/*/*/files/share/fonts/liberation-fonts/LiberationSerif-*.ttf")):
        for f in sorted(glob.glob(g, recursive=True)):
            fm.fontManager.addfont(f)
    names = {f.name for f in fm.fontManager.ttflist}
    serif = next((n for n in ("Liberation Serif", "TeX Gyre Termes") if n in names), "DejaVu Serif")
    frames = np.arange(2, 2 + len(m_ft))
    csvp = Path(out) / f"{stem}.csv"
    with open(csvp, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["frame", "time_s", "FT n46", "RAMEN n46 + TAG"])
        for fr, a_, b_ in zip(frames, m_ft, m_c3):
            w.writerow([int(fr), f"{(fr - 1) / fps:.1f}", f"{a_:.6f}", f"{b_:.6f}"])
    ink, muted, grey, blue = "#1A1A19", "#55544F", "#7B7B78", "#3578BC"
    with plt.rc_context({"pdf.fonttype": 42, "font.family": serif, "font.size": 7, "axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted, "ytick.color": muted,
                         "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 300}):
        fig = plt.figure(figsize=(3.4, 1.35))
        ax = fig.add_axes([0.16, 0.25, 0.82, 0.70])
        ax.plot(frames, m_ft, color=grey, lw=1.1, label="Metric-VDA-S FT")
        ax.plot(frames, m_c3, color=blue, lw=1.1, label="RAMEN + TAG")
        ax.set_xlim(2, max(3, int(frames[-1])))
        ax.set_ylim(0, 1.2 * max(float(np.max(m_ft)), float(np.max(m_c3)), 1e-6))
        ax.set_xlabel("Frame", fontsize=7, labelpad=1)
        ax.set_ylabel("Mean depth change (%)", fontsize=7, labelpad=1)
        ax.legend(loc="upper left", ncol=2, fontsize=7, handlelength=1.2, borderaxespad=0.1, columnspacing=1.0)
        ax.tick_params(length=2.2, width=0.5, pad=1.5, labelsize=6.5)
        ax.grid(True, color=muted, lw=0.4, alpha=0.2)
        ax.set_axisbelow(True)
        fig.savefig(Path(out) / f"{stem}.pdf", format="pdf", metadata={"CreationDate": None, "Creator": "render_c3_figures.py", "Producer": "matplotlib"})
        fig.savefig(Path(out) / f"{stem}.png", dpi=300, metadata={"Software": None})
        plt.close(fig)
    return {"csv": csvp.name, "csv_sha256": sha(csvp), "pdf": f"{stem}.pdf", "png": f"{stem}.png", "font": serif, "columns": ["frame", "time_s", "FT n46", "RAMEN n46 + TAG"],
            "definition": "per frame t=2..N (1-based): mean over all pixels of 100*|D_t - D_(t-1)| / max(D_t, 1e-3) in percent; FT = Metric-VDA-S FT n46 raw, RAMEN = RAMEN n46 + TAG; the same arrays the change panels colour (unclipped, native resolution, full frame); time_s = (frame-1)/fps"}


def write_depth_npz(path, arrays):
    """Deterministic (timestamp-free) compressed npz: members in sorted key order, fixed 1980-01-01 zip date, fixed attrs, deflate level 6 (zlib is deterministic for a given build).
    Readable with np.load. Returns the sidecar record {file, sha256, keys, shapes, dtypes}."""
    path = Path(path)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for k in sorted(arrays):
            buf = io.BytesIO()
            np.save(buf, np.asarray(arrays[k]), allow_pickle=False)
            zi = zipfile.ZipInfo(k + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type, zi.external_attr, zi.create_system = zipfile.ZIP_DEFLATED, 0o600 << 16, 3
            z.writestr(zi, buf.getvalue())
    return {"file": path.name, "sha256": sha(path), "keys": sorted(arrays), "shapes": {k: list(np.asarray(arrays[k]).shape) for k in sorted(arrays)},
            "dtypes": {k: str(np.asarray(arrays[k]).dtype) for k in sorted(arrays)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "figures_config.json"))
    ap.add_argument("--selection", default=str(HERE / "CLIP_SELECTION.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true", help="plumbing smoke: first --max-frames frames, output labelled SMOKE, not a figure")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--only", type=int, default=None, help="index into selection (0..5) or 99 for the teaser")
    ap.add_argument("--panel-h", type=int, default=240)
    ap.add_argument("--host", choices=["local", "gh200"], default="local", help="checkpoint path set (config ckpt_gh200 on gh200); default local = the config paths")
    ap.add_argument("--eval-clone", default=None, help="override the scorer clone path (still asserted at eval_commit and clean)")
    ap.add_argument("--no-depth-dump", action="store_true", help="skip the <stem>_depth.npz dump (default ON: per-panel float32 metre depth of the shown frame + GT validity mask + colour range)")
    ap.add_argument("--depth-display", default="abs_gt", help="teaser depth-panel display rule(s), comma list of: " + ", ".join(DISPLAY_MODES) + " (abs_gt = Ifig rule, default; v1/v2 write teaser_v1_* / teaser_v2_* from ONE inference pass)")
    ap.add_argument("--dump-all-depth", action="store_true", help="Ifig5: also write <tag>teaser_alldepth_<ds>_<key>.npz (+ .json) with the FULL per-frame float32 metre depth of FT n46 (raw) and RAMEN n46 + TAG (and RAMEN raw), all frames, native resolution, so any change-map variant can be re-rendered on CPU")
    ap.add_argument("--cumulative-still", action="store_true", help="Ifig6 (user standing rule 10-15: still images of change are CUMULATIVE): the STILL (png/pdf) change panels show S = sum_{t=t0..N-1} |D_t - D_(t-1)| / D_t (t0 = 8, or 1 for clips shorter than 10 frames), ONE linear inferno scale shared by both panels, vmax = p99 of the POOLED S of both models, titles 'Cumulative: <model>'. GIF/MP4 frames stay per-pair. Default off = the previous behaviour")
    ap.add_argument("--dump-frames", action="store_true", help="Ifig7: also write <tag>teaser_frames_<ds>_<key>.npz (+ .json): LOSSLESS uint8 RGB of the degraded input frames the models saw ('input') and RAMEN's restored frames ('c3_restored'), (N,H,W,3), model resolution (518 short side), the same u8() conversion the panels use")
    ap.add_argument("--no-teaser-chart", action="store_true", help="skip the per-frame mean-change line chart (default ON for the teaser: CSV + PDF + PNG)")
    ap.add_argument("--err-vmax", type=float, default=0.3, help="fixed AbsRel colour range 0..vmax (magma), recorded in every sidecar")
    ap.add_argument("--clip", default=None, help="Ifig4: render the TEASER for one benchmark clip 'dataset:key' (cell st_llvar, as the Bonn teaser). DEV clips freely; a TEST clip needs --allow-test AND RAMEN_AUDITOR_TOKEN (output labelled exposed exploratory)")
    ap.add_argument("--allow-test", action="store_true", help="Ifig4: permit a TEST clip in --clip (also needs the RAMEN_AUDITOR_TOKEN env var)")
    ap.add_argument("--video-dir", default=None, help="Ifig4: render the TEASER for a real, NO-GT image-sequence directory (sorted png/jpg): real frames, no degradation, noGT version only, v1/v2 display only")
    ap.add_argument("--name", default=None, help="Ifig4: clip name for --video-dir outputs")
    ap.add_argument("--start", type=int, default=0, help="Ifig4: --video-dir first frame index")
    ap.add_argument("--n", type=int, default=64, help="Ifig4: --video-dir number of frames")
    ap.add_argument("--stride", type=int, default=1, help="Ifig4: --video-dir frame stride")
    ap.add_argument("--versions", default="noGT,withGT", help="Ifig4: teaser versions to write, comma list of noGT,withGT (default both = the Ifig behaviour)")
    a = ap.parse_args()
    assert not (a.clip and a.video_dir), "--clip and --video-dir are exclusive"
    assert not a.video_dir or a.name, "--video-dir needs --name"
    vers_sel = [v_.strip() for v_ in a.versions.split(",") if v_.strip()]
    assert vers_sel and all(v_ in ("noGT", "withGT") for v_ in vers_sel), "--versions must be a comma list of noGT,withGT"
    cfg, selj = json.load(open(a.config)), json.load(open(a.selection))
    apply_host_paths(cfg, a.host, a.eval_clone)
    assert sha(HERE / "CLIP_RULE.md").startswith(selj["rule_sha256"][:16]) and selj["rule_sha256"].startswith("085611d2869a7c9e"), "CLIP_RULE.md / selection mismatch"
    if a.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, cfg["eval_clone"])
    import torch, cv2
    import bench_joint as bj
    clone_head = subprocess.check_output(["git", "-C", cfg["eval_clone"], "rev-parse", "HEAD"], text=True).strip()
    porcelain = subprocess.check_output(["git", "-C", cfg["eval_clone"], "status", "--porcelain"], text=True).strip()
    assert clone_head == cfg["eval_commit"] and not porcelain, f"eval clone not at {cfg['eval_commit']} or not clean"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ft = cfg["ft"][selj["FT_CHOICE"]]
    for nm, c in (("c3", cfg["c3"]), ("ft", ft)):
        assert c["ckpt"] and Path(c["ckpt"]).exists(), f"{nm} checkpoint missing: {c['ckpt']}"
        assert sha(c["ckpt"]) == c["sha256"], f"{nm} checkpoint sha mismatch"
    specs = {"ft": f"{bj.FT_VDA}{ft['ckpt']}", "c3": f"ours:{cfg['c3']['ckpt']}"}      # Ifig: the released Metric-VDA-S panel is retired; FT n46 and RAMEN n46 only
    dev = a.device
    items = [dict(s, idx=i) for i, s in enumerate(selj["selected"])] + [dict(selj["teaser"], kind="teaser", idx=99)]
    if a.only is not None:
        items = [s for s in items if s["idx"] == a.only]
    if a.clip:                                                       # Ifig4: any benchmark clip, same teaser path (st_llvar)
        assert ":" in a.clip, "--clip must be dataset:key"
        items = [{"dataset": a.clip.split(":", 1)[0], "clip": a.clip.split(":", 1)[1], "kind": "teaser", "cell": "st_llvar", "idx": -1}]
    elif a.video_dir:                                                # Ifig4: real, no-GT sequence
        items = [{"dataset": "real", "clip": a.name, "kind": "teaser", "cell": "real", "idx": -2, "video_dir": str(a.video_dir), "start": a.start, "n": a.n, "stride": a.stride}]
    script_sha, sel_sha = sha(__file__), sha(a.selection)

    def run_models(x, sigma, blur, raw_hw, names):
        res = {}
        for nm in names:
            p = bj.build_pipeline(specs[nm], dev)
            depth, rest = bj.run_pipeline(p, x.to(dev), sigma, blur, raw_hw)
            res[nm] = depth
            if nm == "c3":
                res["c3_restored"] = rest.detach().float().cpu() if hasattr(rest, "detach") else None      # Ifig7: RAMEN's restored frames (== input when the model has no restorer)
                res["c3_tag"] = bj.apply_variant(cfg["tag"], depth, p["last_guides"])
            res.setdefault("_weights", {})[nm] = {k: v for k, v in p["weights"].items()}
            del p
            if dev == "cuda":
                torch.cuda.empty_cache()
        return res

    for it in items:
        ds, key = it["dataset"], it["clip"]
        real = ds == "real"
        proto = None if real else bj.DATASETS[ds]
        split_label = "dev (bench_joint.load_clips('dev'); allow_test never passed)"
        if real:                                                    # Ifig4: real footage, NO GT, NO degradation (frames resized exactly as render_supervisor.prepare)
            paths = sorted(p_ for p_ in Path(it["video_dir"]).iterdir() if p_.suffix.lower() in (".png", ".jpg", ".jpeg"))[it["start"]:it["start"] + it["n"] * it["stride"]:it["stride"]]
            assert len(paths) >= 3, f"--video-dir {it['video_dir']}: need >= 3 frames, found {len(paths)}"
            bj._vda_path()
            from video_depth_anything.util.transform import Resize
            resize = Resize(width=518, height=518, resize_target=False, keep_aspect_ratio=True, ensure_multiple_of=14, resize_method="lower_bound", image_interpolation_method=cv2.INTER_CUBIC)
            rgb = [cv2.cvtColor(cv2.imread(str(p_)), cv2.COLOR_BGR2RGB) for p_ in paths]
            clean = torch.from_numpy(np.stack([np.clip(resize({"image": r_.astype(np.float32) / 255})["image"], 0, 1) for r_ in rgb])).permute(0, 3, 1, 2).contiguous()
            raw_hw, gt, frames, shown = tuple(rgb[0].shape[:2]), None, [{"image": str(p_)} for p_ in paths], {"image": str(paths[len(paths) // 2])}
            split_label = "real footage; no GT; no degradation (qualitative only)"
            if a.max_frames:
                clean = clean[:a.max_frames]
            assert "noGT" in vers_sel and "withGT" not in vers_sel, "real clips have no GT: use --versions noGT"
        elif a.clip:                                                # Ifig4: any benchmark clip; only the shown (middle) frame is read (same result as reading all of them, the clip is static)
            in_dev = it["clip"] in dict(bj.load_clips(ds, "dev"))
            if in_dev:
                frames = dict(bj.load_clips(ds, "dev"))[key]
            else:
                assert a.allow_test and os.environ.get("RAMEN_AUDITOR_TOKEN"), f"{ds}:{key} is not a DEV clip: a TEST clip needs --allow-test and the RAMEN_AUDITOR_TOKEN env var (refused)"
                frames = dict(bj.load_clips(ds, "test", allow_test=True))[key]
                split_label = "test (exposed exploratory; --allow-test + RAMEN_AUDITOR_TOKEN set; qualitative figure only)"
            assert not (ds.startswith("kitti") and "0047" in key), "KITTI drive_0047 is excluded"
            used = frames[:proto["max_len"]]
            shown = used[len(used) // 2]
            clean, raw_hw = bj.load_frames(ds, [shown])
            gt = bj.load_gt(ds, [shown])
            frames = used
            if a.max_frames:
                clean, gt = clean[:a.max_frames], gt[:a.max_frames]
        else:
            frames = dict(bj.load_clips(ds, "dev"))[key]            # DEV only (bench_joint refuses non-DEV without allow_test, which this script never passes)
            clean, raw_hw = bj.load_frames(ds, frames)
            gt = bj.load_gt(ds, frames)
            if a.max_frames:
                clean, gt = clean[:a.max_frames], gt[:a.max_frames]
            shown = frames[clean.shape[0] // 2]
        tag = "SMOKE_" if a.smoke else ""
        base = f"{tag}{it['kind']}_{ds}_{key}"
        t0 = time.time()
        if it["kind"] == "teaser":
            cell = it["cell"]
            if real:
                x, seed, blur, sigma, g_still, valid, lo, hi, gt_img, gt_cov = clean, None, 0.0, 0.0, None, None, None, None, None, None
            else:
                sclean, sgt = bj.static_clip(clean, gt) if not a.max_frames else bj.static_clip(clean, gt, n=a.max_frames)
                seed = bj.noise_seed(ds, key, bj.seed_cell(cell))
                x = bj.degrade_static(sclean, cell, seed)
                blur, sigma = bj.parse_cell(cell)
            r = run_models(x, sigma, blur, raw_hw, ["ft", "c3"])
            dr, dc = r["ft"].astype(np.float64), r["c3_tag"].astype(np.float64)
            chg = lambda d: np.abs(d[1:] - d[:-1]) / np.clip(d[1:], 1e-3, None)       # |D_t - D_{t-1}| / D_t
            cr, cc = chg(dr), chg(dc)
            N = x.shape[0]
            fi = N // 2
            if not real:
                g_still = sgt[sgt.shape[0] // 2]
                valid = (g_still > 1e-3) & (g_still < proto["max_depth"])
                gmid = g_still[valid]
                lo, hi = float(np.percentile(gmid, 2)), float(np.percentile(gmid, 98))
                gt_img, gt_valid = gt_rgb(g_still, lo, hi, proto["max_depth"])
                gt_cov = {"valid_pixels": int(gt_valid.sum()), "total_pixels": int(gt_valid.size), "valid_pct": 100.0 * float(gt_valid.mean()), "gt_shape": list(g_still.shape),
                          "gt_file": shown["gt_depth"], "definition": "sensor GT as loaded by bench_joint.load_gt (crop + factor, 0 -> invalid); valid = 1e-3 < GT < max_depth; static clip = this one frame repeated; no densified or interpolated GT; invalid pixels masked grey"}
            ph = a.panel_h
            in_title = "Input" if real else "Degraded input"                  # Ifig4: real footage is not degraded by us
            versions = {"noGT": [in_title, "Metric-VDA-S FT", "RAMEN + TAG", "Change: FT", "Change: RAMEN + TAG"],
                        "withGT": [in_title, "Ground truth", "Metric-VDA-S FT", "RAMEN + TAG", "Change: FT", "Change: RAMEN + TAG"]}
            ft_ = max(1, fi)
            modes = [m_.strip() for m_ in a.depth_display.split(",") if m_.strip()]
            assert modes and all(m_ in DISPLAY_MODES for m_ in modes), f"--depth-display must be a comma list of {list(DISPLAY_MODES)}"
            assert not (real and "abs_gt" in modes), "real (no-GT) clips: abs_gt needs GT, use --depth-display v1_log_p2p98 (and/or v2_disparity_p2p98)"
            # Ifig2 display modes (user 10-15: communication, not accuracy). abs_gt = the Ifig rule, unchanged. v1/v2 = per-panel adaptive (relative) display, SAME rule for every depth panel (GT included),
            # range fixed from the SHOWN still frame so the GIF scale does not move with time. Change maps (v1/v2): ONE shared range 0..vmax, vmax = p99 of the FT change map over the whole clip, sqrt stretch, inferno.
            chg_vmax = float(np.percentile(cr, 99))
            def panel_range(mode, d, ok=None):
                v = d[ok] if ok is not None else d.reshape(-1)
                v = np.clip(v.astype(np.float64), 1e-3, None)
                x = np.log(v) if mode == "v1_log_p2p98" else 1.0 / v
                return float(np.percentile(x, 2)), float(np.percentile(x, 98))
            def depth_view(mode, d, rng):                          # rng in the mode's own space (log depth / disparity); returns RGB
                if mode == "v1_log_p2p98":
                    return colour((np.log(np.clip(d, 1e-3, None)) - rng[0]) / (rng[1] - rng[0]), DISPLAY_CMAP)
                return colour((1.0 / np.clip(d, 1e-3, None) - rng[0]) / (rng[1] - rng[0]), DISPLAY_CMAP)
            cum_t0 = 8 if N > 9 else 1
            cum_ft, cum_c3 = cr[cum_t0 - 1:].sum(0), cc[cum_t0 - 1:].sum(0)             # S over pairs t = cum_t0..N-1 (cr[t-1] is the pair (t-1, t))
            cum_vmax = float(np.percentile(np.concatenate([cum_ft.ravel(), cum_c3.ravel()]), 99))
            cum_stats = {"pairs": f"t={cum_t0}..{N - 1}", "n_pairs": int(N - cum_t0), "pooled_p99_vmax": cum_vmax, "FT": {"p99": float(np.percentile(cum_ft, 99)), "max": float(cum_ft.max()), "mean": float(cum_ft.mean())},
                         "RAMEN_TAG": {"p99": float(np.percentile(cum_c3, 99)), "max": float(cum_c3.max()), "mean": float(cum_c3.mean())},
                         "rule": "S = sum |D_t - D_(t-1)| / D_t over the pairs; one linear inferno scale 0..vmax shared by both panels, vmax = p99 of the pooled S; black = no change"}
            def chg_view(mode, c):
                if mode == "abs_gt":
                    return colour(c / 0.03, "inferno")
                return colour(np.sqrt(np.clip(c / chg_vmax, 0, 1)), "inferno")
            sides = {}
            for mode in modes:
                if mode == "abs_gt":
                    gt_img_m, rngs = gt_img, None
                else:
                    rngs = {"ft": panel_range(mode, dr[ft_]), "c3_tag": panel_range(mode, dc[ft_])} if real else {"gt": panel_range(mode, g_still, valid), "ft": panel_range(mode, dr[ft_]), "c3_tag": panel_range(mode, dc[ft_])}
                    if real:
                        gt_img_m = None
                    else:
                        gt_img_m = depth_view(mode, np.where(valid, g_still, np.median(g_still[valid])), rngs["gt"])
                        gt_img_m[~valid] = (70, 70, 70)
                def comp(t, with_gt, mode=mode, rngs=rngs, gt_img_m=gt_img_m, cum=False):         # t index into 1..N-1; cum = still with the cumulative change panels
                    if mode == "abs_gt":
                        d_ft, d_c3 = depth_rgb(dr[t], lo, hi), depth_rgb(dc[t], lo, hi)
                    else:
                        d_ft, d_c3 = depth_view(mode, dr[t], rngs["ft"]), depth_view(mode, dc[t], rngs["c3_tag"])
                    row = [resize_h(u8(x[t]), ph)] + ([resize_h_nearest(gt_img_m, ph)] if with_gt else []) + [resize_h(d_ft, ph), resize_h(d_c3, ph),
                           resize_h(colour(cum_ft / cum_vmax, "inferno") if cum else chg_view(mode, cr[t - 1]), ph), resize_h(colour(cum_c3 / cum_vmax, "inferno") if cum else chg_view(mode, cc[t - 1]), ph)]
                    w = min(r_.shape[1] for r_ in row)
                    row = [cv2.resize(r_, (w, ph), interpolation=cv2.INTER_NEAREST) if (with_gt and i_ == 1) else cv2.resize(r_, (w, ph)) for i_, r_ in enumerate(row)]
                    names_ = versions["withGT" if with_gt else "noGT"]
                    if cum:
                        names_ = names_[:-2] + ["Cumulative: FT", "Cumulative: RAMEN + TAG"]
                    return np.concatenate([header(names_, w), np.concatenate(row, 1)], 0)
                mtag = "" if mode == "abs_gt" else DISPLAY_MODES[mode][0] + "_"
                for vname, with_gt in (("noGT", False), ("withGT", True)):
                    if vname not in vers_sel or (real and with_gt):
                        continue
                    still = comp(ft_, with_gt, cum=a.cumulative_still)
                    vbase = f"{tag}teaser_{mtag}{vname}_{ds}_{key}"
                    cv2.imwrite(str(out / f"{vbase}_still.png"), still[..., ::-1])
                    save_pdf(still, out / f"{vbase}_still.pdf")
                    save_gif([comp(t, with_gt) for t in range(1, N)], out / f"{vbase}.gif")
                    dd = {}
                    if not a.no_depth_dump:                                  # same arrays as plotted (dr/dc[t], g_still, valid), NATIVE model/GT resolution; display resize = resize_h(panel_h) afterwards
                        arrs = {"ft": dr[ft_].astype(np.float32), "c3_tag": dc[ft_].astype(np.float32), "c3_raw": r["c3"][ft_].astype(np.float32), "frame_index": np.int64(ft_),
                                "panel_h": np.int64(ph), "raw_hw": np.array(raw_hw, np.int64)}
                        if not real:
                            arrs.update({"gt": g_still.astype(np.float32), "gt_valid": valid, "color_range_m": np.array([lo, hi], np.float64)})
                        if rngs is not None:
                            arrs.update({f"display_range_{k_}": np.array(v_, np.float64) for k_, v_ in rngs.items()})
                        dd = {"depth_dump": dict(write_depth_npz(out / f"{vbase}_depth.npz", arrs), frame_index=int(ft_), color_range_m=None if real else [lo, hi], panel_h=ph, colour_scale="log, turbo, clip(depth, lo, hi)",
                                                 display_resize="each depth panel = cv2.resize(INTER_AREA) to panel_h rows, then to the common width; GT panel INTER_NEAREST; arrays are NATIVE resolution (no crop beyond the loader's)",
                                                 invalid_gt="gt_valid False = no GT (grey in the GT panel); gt array holds the loader value there")}
                    disp = {} if mode == "abs_gt" else {"depth_display": {"mode": mode, "rule": DISPLAY_MODES[mode][1], "colormap": DISPLAY_CMAP, "ranges_in_mode_space": rngs, "ranges_taken_from_frame_index": int(ft_),
                                                        "disclosure": "per-panel adaptive display normalisation: depth colours are RELATIVE within each panel (not comparable across panels, not metric); a display choice for communication, not an accuracy claim",
                                                        "change_rule": "one shared range 0..vmax for both change panels, vmax = p99 of the FT change map over the whole clip, sqrt stretch, inferno", "change_vmax_fraction": chg_vmax}}
                    if a.cumulative_still:
                        disp = {**disp, "cumulative_change_still": cum_stats, "cumulative_note": "STILL png/pdf change panels are cumulative (titles 'Cumulative: <model>'); the GIF/MP4 change panels remain per consecutive pair"}
                    sides[f"{mtag}{vname}"] = {"version": vname, "panels": versions[vname], "depth_range_m": None if real else [lo, hi], "depth_range_rule": "GT 2nd-98th percentile of the still frame's valid GT, log, turbo, identical in every depth panel (the GT panel included)" if mode == "abs_gt" else "NOT USED for colouring in this display mode (see depth_display); depth_range_m is the abs_gt reference range only", "change_range": "0-3 % of depth, inferno" if mode == "abs_gt" else "shared 0..vmax sqrt-stretched, inferno (see depth_display)", "tag": cfg["tag"],
                                    "cell": cell, "seed": seed, "frames": N, "still_frame_index": ft_, "fps": 10, "models": {k: specs[k] for k in ("ft", "c3")},
                                    "tag_states": {"Metric-VDA-S FT": "raw (no TAG)", "RAMEN + TAG": "TAG " + cfg["tag"]}, **({"gt_coverage": gt_cov} if with_gt else {}), **dd, **disp, "_stem": vbase}
            chart = {}
            if not a.no_teaser_chart:                               # per-frame mean relative depth change, from the SAME cr/cc arrays the change panels colour (full frame, unclipped)
                chart = write_meanchange_chart(out, f"{tag}teaser_meanchange_{ds}_{key}", 100.0 * cr.mean(axis=(1, 2)), 100.0 * cc.mean(axis=(1, 2)), fps=10)
            if a.dump_all_depth:                                      # Ifig5: the SAME arrays dr/dc/r["c3"] the panels were computed from (inference output, no post-processing), deterministic npz
                adn = f"{tag}teaser_alldepth_{ds}_{key}"
                allarr = {"ft": r["ft"].astype(np.float32), "c3_tag": r["c3_tag"].astype(np.float32), "c3_raw": r["c3"].astype(np.float32), "raw_hw": np.array(raw_hw, np.int64)}
                if not real:                                          # static clip: the one GT frame (repeated N times in the clip) and its validity mask
                    allarr.update({"gt": g_still.astype(np.float32), "gt_valid": valid})
                rec = write_depth_npz(out / f"{adn}.npz", allarr)
                json.dump({"PROVENANCE": "Ifig5 full per-frame depth dump", "npz": rec, "frames": int(N), "cell": cell, "seed": seed, "dataset": ds, "clip": key, "tag": cfg["tag"], "models": {k: specs[k] for k in ("ft", "c3")},
                           "ckpts": {"c3": cfg["c3"]["sha256"], "ft": ft["sha256"]}, "meaning": "gt, gt_valid = the static clip's single GT frame + mask (m); ft = Metric-VDA-S FT n46 raw depth (m); c3_tag = RAMEN n46 + TAG depth (m); c3_raw = RAMEN n46 raw depth (m); shape (N,H,W) native resolution; frame index 0..N-1 (the teaser GIF shows pairs t=1..N-1, still at N//2)",
                           "eval_clone_head": clone_head, "script_sha256": script_sha}, open(out / f"{adn}.json", "w"), indent=1, default=str)
            if a.dump_frames:                                         # Ifig7: input + restored frames, lossless uint8 (what the panels show), model resolution
                fdn = f"{tag}teaser_frames_{ds}_{key}"
                rst = r.get("c3_restored")
                frm = {"input": np.stack([u8(x[t_]) for t_ in range(x.shape[0])])}
                frm["c3_restored"] = np.stack([u8(rst[t_]) for t_ in range(rst.shape[0])]) if rst is not None else frm["input"]
                frec = write_depth_npz(out / f"{fdn}.npz", frm)
                json.dump({"PROVENANCE": "Ifig7 input + restored frame dump", "npz": frec, "frames": int(N), "cell": cell, "seed": seed, "dataset": ds, "clip": key, "restored_is_input": bool(rst is None or np.array_equal(frm["input"], frm["c3_restored"])),
                           "meaning": "input = the degraded (real: undegraded) frames fed to the depth models, uint8 RGB (N,H,W,3), model resolution; c3_restored = RAMEN n46's restored frames (restorer output) in the same conversion; ft has no restorer (its 'restored' is its input)",
                           "ckpts": {"c3": cfg["c3"]["sha256"], "ft": ft["sha256"]}, "eval_clone_head": clone_head, "script_sha256": script_sha}, open(out / f"{fdn}.json", "w"), indent=1, default=str)
            side = sides
        else:
            cell = cfg["cell"]
            drawn = bj.llvar_draw(ds, key)
            seed = bj.llvar_noise_seed(ds, key, cell)
            x = bj.degrade_llvar(clean, cell, drawn, seed)
            blur, sigma = bj.parse_cell(cell)
            r = run_models(x, sigma, blur, raw_hw, ["ft", "c3"])
            order = ["ft", "c3", "c3_tag"]
            sl = []
            for k in order:
                sd, sg, _, sx, sc, _ = bj._vda5.score_slice(proto, r[k], gt, x, x, clean, None)
                if sd.shape[-2:] != sg.shape[-2:]:
                    sd = np.stack([cv2.resize(d, (sg.shape[-1], sg.shape[-2])) for d in sd])
                sl.append((sd.astype(np.float64), sg, sx))
            gtv = sl[0][1]
            N = gtv.shape[0]
            fi = N // 2
            valid_all = (gtv > 1e-3) & (gtv < proto["max_depth"])
            lo, hi = float(np.percentile(gtv[valid_all], 2)), float(np.percentile(gtv[valid_all], 98))
            errs = []
            for (sd, sg, _), k in zip(sl, order):
                al, valid = fit_align(sd, sg, proto["max_depth"], torch)
                errs.append(abs_rel_map(al, sg, valid))
            names = ["Degraded input", "FT (raw)", "RAMEN (raw)", "RAMEN + TAG"]
            ph = a.panel_h
            def comp(t):
                rowd = [resize_h(u8(sl[0][2][t]), ph)] + [resize_h(depth_rgb(s[0][t], lo, hi), ph) for s in sl]
                rowe = [np.full_like(rowd[0], 255)] + [resize_h(err_rgb(e[t], a.err_vmax), ph) for e in errs]
                w = min(q.shape[1] for q in rowd + rowe)
                rowd = [cv2.resize(q, (w, ph)) for q in rowd]
                rowe = [cv2.resize(q, (w, ph), interpolation=cv2.INTER_NEAREST) for q in rowe]
                return np.concatenate([header(names, w), np.concatenate(rowd, 1), np.concatenate(rowe, 1)], 0)
            still = comp(fi)
            cv2.imwrite(str(out / f"{base}_still.png"), still[..., ::-1])
            save_pdf(still, out / f"{base}_still.pdf")
            save_gif([comp(t) for t in range(N)], out / f"{base}.gif")
            dd = {}
            if not a.no_depth_dump:
                arrs = {"gt": gtv[fi].astype(np.float32), "gt_valid": valid_all[fi], "ft": sl[0][0][fi].astype(np.float32), "c3_raw": sl[1][0][fi].astype(np.float32), "c3_tag": sl[2][0][fi].astype(np.float32),
                        "color_range_m": np.array([lo, hi], np.float64), "frame_index": np.int64(fi), "panel_h": np.int64(ph), "raw_hw": np.array(raw_hw, np.int64)}
                dd = {"depth_dump": dict(write_depth_npz(out / f"{base}_depth.npz", arrs), frame_index=int(fi), color_range_m=[lo, hi], panel_h=ph, colour_scale="log, turbo, clip(depth, lo, hi)",
                                         display_resize="score_slice-space arrays (as plotted), resized to panel_h for display", invalid_gt="gt_valid False = no GT")}
            side = {**dd, "panels": names, "rows": ["depth", "per-pixel AbsRel after the benchmark's per-clip scale+shift fit (valid GT only; grey = no GT)"],
                    "depth_range_m": [lo, hi], "depth_range_rule": "GT 2nd-98th percentile over the clip, log, turbo, identical in every depth panel",
                    "error_range": [0.0, a.err_vmax], "error_colormap": "magma, fixed", "tag": cfg["tag"], "cell": cell, "llvar_draw": drawn, "noise_seed": seed, "frames": N,
                    "still_frame_index": fi, "fps": 10, "models": {k: specs[k] for k in ("ft", "c3")}, "score_slice": proto.get("score_slice"), "selection_entry": {k: v for k, v in it.items()}}
        outs = [(sd_.pop("_stem"), dict(sd_, **({"mean_change_chart": chart} if chart else {}))) for sd_ in side.values()] if it["kind"] == "teaser" else [(base, side)]
        for obase, oside in outs:
            prov = {"PROVENANCE": "C3 figure render, AUD-C3-FIGURES-20261012", "smoke": bool(a.smoke), "item": {k: v for k, v in it.items()},
                    "script": "render_c3_figures.py", "script_sha256": script_sha, "selection_sha256": sel_sha, "rule_sha256": selj["rule_sha256"],
                    "ckpts": {"c3": {"path": cfg["c3"]["ckpt"], "sha256": cfg["c3"]["sha256"]}, "ft": {"name": ft["name"], "path": ft["ckpt"], "sha256": ft["sha256"], "choice": selj["FT_CHOICE"]}},
                    "base_weights": {nm: {k: {"path": v, "sha256": sha(v)} for k, v in w_.items()} for nm, w_ in r.get("_weights", {}).items()},
                    "frames_used": [f["image"] for f in frames[:clean.shape[0]]] if (it["kind"] != "teaser" or real) else [shown["image"]], "eval_clone": cfg["eval_clone"], "host_paths": a.host, "figures_config_sha256": sha(a.config), "eval_clone_head": clone_head, "eval_clone_porcelain_empty": not porcelain,
                    "torch": torch.__version__, "device": dev, "gpu": torch.cuda.get_device_name() if dev == "cuda" and torch.cuda.is_available() else "cpu", "host": socket.gethostname(),
                    "split": split_label, "seconds": round(time.time() - t0, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "numbers_on_images": False, "sidecar": oside, **({"real_footage": "no GT, no synthetic degradation; change maps include true scene motion as well as noise flicker"} if real else {})}
            json.dump(prov, open(out / f"{obase}.json", "w"), indent=1, default=str)
        print("rendered", ", ".join(o for o, _ in outs), flush=True)


if __name__ == "__main__":
    main()
