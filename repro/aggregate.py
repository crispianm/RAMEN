"""Aggregation: per-clip -> per-dataset -> mean, exactly as the released reads (video_macro = nanmean over clips of per_video[clip][metric])."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from .plan import Unit
from .util import jdump, nanmean

TEMPORAL_PREFIX = ("tae", "tof")
CLIP_KEYS = ("per_video", "per_scene", "per_sequence")      # bench/night | fnoise: per_scene (13 BVI scenes) is the rule's macro unit and sits before per_sequence (22 sequences, a different mean)
MACRO_KEYS = ("video_macro", "macro")
SKIP_CLIP_FIELDS = ("frames", "llvar_draw")


def split_macro(art: dict) -> dict:
    for k in MACRO_KEYS:
        if k in art:
            return art[k]
    if "per_frame_ms_p50" in art:                                         # latency
        return {"per_frame_ms_p50": art["per_frame_ms_p50"], "per_frame_ms_p95": art["per_frame_ms_p95"],
                **{k: art[k] for k in ("window_ms_p50", "ms_per_new_frame_w2_p50") if k in art},
                **{f"stage_ms_p50.{k}": v for k, v in art.get("stage_ms_p50", {}).items()}}
    return {}


def per_clip(art: dict) -> dict:
    for k in CLIP_KEYS:
        if k in art and isinstance(art[k], dict):
            return art[k]
    return {}


def derive_alias_art(base: dict, alias: str, drop: str) -> dict:
    """Exact, GPU-free: the base result with clips containing `drop` removed, macro recomputed (same as scripts/dn01_headline.derive_alias)."""
    pv = {k: v for k, v in per_clip(base).items() if drop not in k}
    keys = [k for k in next(iter(pv.values())) if k not in SKIP_CLIP_FIELDS] if pv else []
    art = dict(base)
    art.update({"dataset": alias, "source_dataset": base.get("dataset"), "dropped_clips": [k for k in per_clip(base) if drop in k], "num_clips": len(pv),
                "per_video": pv, "video_macro": {k: nanmean([v.get(k) for v in pv.values()]) for k in keys}})
    return art


def dataset_row(art: dict, metrics: list[str]) -> dict:
    """{metric: value, '_n_clips': n, '_macro_mismatch': [...]}: value recomputed from per-clip, cross-checked with the stored macro."""
    pv, macro = per_clip(art), split_macro(art)
    row, bad = {}, []
    for m in metrics:
        if pv and any(m in c for c in pv.values()):
            v = nanmean([c.get(m) for c in pv.values()])
            stored = macro.get(m)
            if stored is not None and not (isinstance(stored, float) and math.isnan(stored)) and not math.isnan(v) and abs(v - stored) > 1e-9 * max(1.0, abs(stored)):
                bad.append(m)
        else:
            v = macro.get(m, float("nan"))
            v = float("nan") if v is None else v
        row[m] = v
    row["_n_clips"] = len(pv) if pv else art.get("num_clips")
    row["_macro_mismatch"] = bad
    return row


def aggregate_item(iid: str, item: dict, rows_in: dict[tuple, dict]) -> dict:
    """rows_in[(model_id, dataset, cell)] = result artifact. Returns {'item', 'rows': [...], 'mean': [...]}."""
    metrics = item["metrics"]
    agg = item.get("aggregation") or {}
    tae_ds = set(agg.get("tae_datasets") or [])
    rows = []
    for (mid, ds, cell), art in rows_in.items():
        r = dataset_row(art, metrics)
        rows.append({"model": mid, "dataset": ds, "cell": cell, "split": art.get("split"), "n_clips": r.pop("_n_clips"),
                     "macro_mismatch": r.pop("_macro_mismatch"), "metrics": r, **({"samples": art["samples"]} if "samples" in art else {}),
                     **({"protocol_iters": art["protocol_iters"]} if "protocol_iters" in art else {})})
    rows += seed_mean_rows(rows, item.get("seed_mean") or [], metrics)
    mean_ds = agg.get("mean_over") or []
    means = []
    if mean_ds:
        for mid in dict.fromkeys(r["model"] for r in rows):
            for cell in dict.fromkeys(r["cell"] for r in rows):
                sel = [r for r in rows if r["model"] == mid and r["cell"] == cell and r["dataset"] in mean_ds]
                if not sel:
                    continue
                mm = {}
                for m in metrics:
                    use = [r for r in sel if not (m.startswith(TEMPORAL_PREFIX) and tae_ds and r["dataset"] not in tae_ds)]
                    mm[m] = nanmean([r["metrics"][m] for r in use])
                means.append({"model": mid, "dataset": "MEAN(" + ",".join(mean_ds) + ")", "cell": cell, "n_datasets": len(sel), "metrics": mm})
    return {"item": iid, "title": item.get("title"), "label": item.get("label"), "metrics": metrics, "rows": rows, "mean": means}


def seed_mean_rows(rows: list[dict], groups: list[dict], metrics: list[str]) -> list[dict]:
    """Per (dataset, cell): unweighted mean over the member seeds' per-dataset values (the paper's 'C3 mean'); keeps min/max as the seed range. Members missing a row are skipped (n_seeds says how many)."""
    out = []
    for g in groups:
        keys = list(dict.fromkeys((r["dataset"], r["cell"]) for r in rows if r["model"] in g["members"]))
        for ds, cell in keys:
            mem = [r for r in rows if r["model"] in g["members"] and r["dataset"] == ds and r["cell"] == cell]
            vals = {m: [r["metrics"][m] for r in mem] for m in metrics}
            out.append({"model": g["name"], "dataset": ds, "cell": cell, "split": mem[0].get("split"), "n_clips": mem[0]["n_clips"], "n_seeds": len(mem),
                        "macro_mismatch": [], "metrics": {m: nanmean(v) for m, v in vals.items()},
                        "seed_range": {m: [min((x for x in v if x == x), default=float("nan")), max((x for x in v if x == x), default=float("nan"))] for m, v in vals.items()}})
    return out


def _upper_median(v):
    s = sorted(v)
    return s[len(s) // 2]


def ladder_rows(session: Path, stack: str, rows: list[str]) -> dict[tuple, dict]:
    """Rows of a latency-ladder session dir (rows/<stack>__<id>.json) as latency artifacts keyed (row, '-', '-'). Row 'mvda_bracket' = the stack's six Metric-VDA-S bracket runs: its value is the
    upper median of the six runs (the harness's own reference_upper_median); every run's samples are kept. Window 1 /32 = ms per frame; window 2 /22 = ms per new frame (the harness's divisions)."""
    got: dict[tuple, dict] = {}
    rd = Path(session) / "rows"
    for r in rows:
        if r == "mvda_bracket":
            fs = sorted(rd.glob(f"{stack}__mvda_bracket[0-9].json"))
            runs = [json.loads(f.read_text()) for f in fs]
            if not runs:
                continue
            art = {"per_frame_ms_p50": _upper_median([x["window1"]["ms_per_frame_p50"] for x in runs]), "per_frame_ms_p95": _upper_median([x["window1"]["ms_per_frame_p95"] for x in runs]),
                   "window_ms_p50": _upper_median([x["window1"]["window_ms_p50"] for x in runs]), "ms_per_new_frame_w2_p50": _upper_median([x["window2"]["ms_per_new_frame_p50"] for x in runs]),
                   "n_runs": len(runs), "protocol_iters": runs[0]["protocol"]["iters"], "bracket_run_ms_per_frame_p50": [x["window1"]["ms_per_frame_p50"] for x in runs],
                   "peak_alloc_GiB": max(x["mem"]["peak_alloc_GiB"] for x in runs)}
            if all("samples" in x for x in runs):
                art["samples"] = {"runs": [x["samples"] for x in runs], "n_samples": sum(x["samples"]["n_samples"] for x in runs), "per_run_n_samples": [x["samples"]["n_samples"] for x in runs]}
            got[("metric_vda_s", "-", "-")] = art
            continue
        f = rd / f"{stack}__{r}.json"
        if not f.exists():
            continue
        x = json.loads(f.read_text())
        art = {"per_frame_ms_p50": x["window1"]["ms_per_frame_p50"], "per_frame_ms_p95": x["window1"]["ms_per_frame_p95"], "window_ms_p50": x["window1"]["window_ms_p50"],
               "ms_per_new_frame_w2_p50": x["window2"]["ms_per_new_frame_p50"], "peak_alloc_GiB": x["mem"]["peak_alloc_GiB"], "toggles": x.get("toggles"), "protocol_iters": x["protocol"]["iters"]}
        if "samples" in x:
            art["samples"] = x["samples"]
        got[(r, "-", "-")] = art
    return got


def collect_item(cfg_item: dict, units: list[Unit], out: Path, loader=None) -> dict[tuple, dict]:
    """Load the result files of an item's units (derived ones computed in memory from their base). Missing files are simply absent from the dict."""
    load = loader or (lambda p: json.loads(Path(p).read_text()))
    got: dict[tuple, dict] = {}
    for u in units:
        if u.harness:
            if u.out_path(out).exists():
                got.update(ladder_rows(u.out_path(out).parent, u.harness["stack"], u.harness["rows"]))
            continue
        if u.derive_from:
            bp = Path(out) / u.rel_dir() / u.filename().replace(f"__{u.dataset}__", f"__{u.derive_from['base']}__")
            if bp.exists():
                got[(u.row_id, u.dataset, u.cell)] = derive_alias_art(load(bp), u.dataset, u.derive_from["drop"])
            continue
        p = u.out_path(out)
        if p.exists():
            got[(u.row_id, u.dataset, u.cell)] = load(p)
    return got


def write_item(result: dict, out: Path, stem: str) -> list[Path]:
    out = Path(out)
    jp, cp = out / f"{stem}.json", out / f"{stem}.csv"
    jdump(result, jp)
    metrics = result["metrics"]
    has_v = any("verdict" in r for r in result["rows"])
    slots = result.get("host_slots") or {}
    with open(cp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "dataset", "cell", "split", "n_clips", *metrics, *(["verdict"] if has_v else []), *[v["label"] for v in slots.values()]])
        for r in result["rows"]:
            w.writerow([r["model"], r["dataset"], r["cell"], r.get("split") or "", r["n_clips"], *[_fmt(r["metrics"][m]) for m in metrics], *([r.get("verdict", "")] if has_v else []), *[(r.get("host_slots") or {}).get(k, "") for k in slots]])
        for r in result["mean"]:
            w.writerow([r["model"], r["dataset"], r["cell"], "", r["n_datasets"], *[_fmt(r["metrics"][m]) for m in metrics], *([""] if has_v else []), *[""] * len(slots)])
    written = [jp, cp]
    v = result.get("verdict")
    if v and "pairs" in v:                                      # eps_d reading: pairwise statements, each row carrying the disclosure
        pp = out / f"{stem}_pairs.csv"
        with open(pp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["a", "b", "result", "datasets", "disclosure"])
            for p in v["pairs"]:
                w.writerow([p["a"], p["b"], p["result"], " ".join(p["datasets"]), result.get("disclosure", "")])
        written.append(pp)
    return written


def _fmt(v) -> str:
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else repr(float(v))
