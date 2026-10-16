"""Protocol v3 WP10: the headline-table aggregator + the binding win rule (PROTOCOL_TABLE.md sha e0183d09, sections 2-4).

Reads bench_joint / dn01_heldout_eval / dn01_bvi_fnoise result JSONs (their EXISTING schemas) and writes Table 1 (CSV, LaTeX, HTML), the win summary, SELECTION.md and an appendix table. Cell names, row models and metrics are
CONFIG, not code (``--config``; ``--make-default-config`` writes the section-2 cell list with the headline / static cell names you give it), so WP7's llvar cells need no edit here.

THE RULES (PROTOCOL_TABLE.md):
  s2  * counted cells = the config's ``counted`` cells (default list: VDA five x llvar: KITTI / Bonn / ScanNet* / Sintel x {metric AbsRel, aligned AbsRel, delta1, TAE}, NYUv2 x 3 (TAE blank), nuScenes-night x 4, BVI-RLV
        F_noise, static flicker per set); KITTI-minus-drive-0047 is a ``counted: false`` disclosure; a ``reference`` row (DepthCrafter) is shown, never competes, never counted;
      * relative-output raw rows compete in ``relative_ok`` (aligned, scale-invariant) cells only;
      * "best" = STRICTLY best at full precision (exact float compare, no rounding): a tie for the best value has NO winner;
      * a cell is blank when no competing row has a value; ours wins the table iff ours is best in MORE THAN HALF of the counted non-blank cells (an ours-missing cell where another row has a value is a loss, flagged);
  s3  * selection: each row with several candidates (ours, each fine-tuned baseline) is chosen by the SAME rule: most counted non-blank cells won against the released raw rows (strict best in the pool
        {released rows + the candidate}), then the lower mean rank in that pool, then the earlier candidate; the candidate counts are written to SELECTION.md;
  s4  * TAG hygiene: no candidate of a non-ours main-table row may carry a depth post-filter token (ema / tmed / agf / sgf / rgf / dvp), and every ours candidate carries exactly the configured TAG token (asserted;
        appendix rows are exempt: ours+TAG vs released+TAG goes there).
Deterministic: sorted iteration, no timestamps, shortest round-trip float text; a re-run on the same inputs is byte-identical (the optional TEST access log is append-only and separate).
``python scripts/dn01_headline_table.py --results DIR --config cfg.json --out DIR``   |   ``--make-default-config cfg.json --headline-cell llvar --static-cell st_llvar``"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import sys
from pathlib import Path

POST_RE = re.compile(r"^(ema\d|ema0|tmed\d|agf|sgf|rgf|dvp|kal\d|kal\.|oneeuro|flowema)")
DEFAULT_DATASETS = {"kitti": "KITTI", "bonn": "Bonn", "scannet": "ScanNet*", "sintel": "Sintel", "nyuv2": "NYUv2"}
METRICS = [("abs_rel_unaligned", "AbsRel (m)", "min", False), ("abs_rel_aligned", "AbsRel (al.)", "min", True), ("delta1_aligned", "$\\delta_1$", "max", True), ("tae", "TAE", "min", True)]


# ------------------------------------------------------------------------------------------------ config
def make_default_config(headline_cell: str = "llvar", static_cell: str = "st_llvar", static_sets=("kitti", "bonn", "nyuv2", "scannet")) -> dict:
    """The PROTOCOL_TABLE section-2 cell list. ``source`` bench = a bench_joint / heldout result file (dataset, cell, ``video_macro.<field>``); bvi = dn01_bvi_fnoise (``macro.f_noise``)."""
    cells = []
    for ds, label in DEFAULT_DATASETS.items():
        for field, mlabel, direction, rel in METRICS:
            if ds == "nyuv2" and field == "tae":
                continue                                            # NYUv2: TAE blank (single-frame JSON, no poses): 3 cells
            cells.append({"id": f"{ds}|{headline_cell}|{field}", "group": label, "label": f"{label} {mlabel}", "source": "bench", "dataset": ds, "cell": headline_cell, "field": f"video_macro.{field}",
                          "direction": direction, "relative_ok": rel, "counted": True})
    for field, mlabel, direction, rel in METRICS:
        cells.append({"id": f"nuscenes_night|real|{field}", "group": "nuScenes-night", "label": f"nuScenes-night {mlabel}", "source": "bench", "dataset": "nuscenes_night_test", "cell": "real",
                      "field": f"video_macro.{field}", "direction": direction, "relative_ok": rel, "counted": True})
    cells.append({"id": "bvi_rlv|fnoise|f_noise", "group": "BVI-RLV", "label": "BVI-RLV $F_{noise}$", "source": "bvi", "dataset": "bvi_rlv", "cell": "fnoise", "field": "macro.f_noise", "direction": "min",
                  "relative_ok": False, "counted": True})
    for ds in static_sets:
        cells.append({"id": f"{ds}|{static_cell}|flicker", "group": DEFAULT_DATASETS.get(ds, ds), "label": f"{DEFAULT_DATASETS.get(ds, ds)} static flicker", "source": "bench", "dataset": ds, "cell": static_cell,
                      "field": "video_macro.flicker", "direction": "min", "relative_ok": False, "counted": True})
    cells.append({"id": f"kitti_minus_0047|{headline_cell}|abs_rel_aligned", "group": "KITTI-0047", "label": "KITTI minus drive 0047 AbsRel (al.) [disclosure]", "source": "bench", "dataset": "kitti_minus_0047",
                  "cell": headline_cell, "field": "video_macro.abs_rel_aligned", "direction": "min", "relative_ok": True, "counted": False})
    return {"tag_token": "agf0.2t0.06", "cells": cells, "rows": [], "ratio_ref": None, "title": "Table 1"}


def _get(obj, path):
    for k in path.split("."):
        if not isinstance(obj, dict) or k not in obj:
            return None
        obj = obj[k]
    return float(obj) if isinstance(obj, (int, float)) and not isinstance(obj, bool) and math.isfinite(obj) else None


# ------------------------------------------------------------------------------------------------ results
FILTER_RE = re.compile(r"^(ema\d|ema0|ema\.|tmed\d|kal\d|kal\.|oneeuro|flowema)")     # the post-hoc smoothing tokens (FBASE D7); TAG (agf) is governed by check_tag_hygiene, not by this


def filter_tokens(model: str, result: dict | None = None) -> list:
    toks = [t for t in str(model).split("+")] + list((result or {}).get("depth_post", []) or [])
    return sorted({t for t in toks if FILTER_RE.match(t)})


def declared_filter_models(cfg: dict) -> frozenset:
    """Model specs of the rows the config DECLARES as filter baselines (``filter_baseline_rows`` = row ids; each must be kind 'filter_baseline', and every kind 'filter_baseline' row must be listed)."""
    ids = set(cfg.get("filter_baseline_rows", []))
    rows = {r["id"]: r for r in cfg["rows"]}
    assert ids <= set(rows), f"filter_baseline_rows names unknown rows: {sorted(ids - set(rows))}"
    for r in cfg["rows"]:
        assert (r["kind"] == "filter_baseline") == (r["id"] in ids), f"row {r['id']!r}: kind 'filter_baseline' and membership of filter_baseline_rows must agree"
        assert not (r["kind"] == "filter_baseline" and r.get("competes", False)), f"row {r['id']!r}: a filter_baseline row never competes (competes: true is refused)"
    return frozenset(c["model"] for i in ids for c in rows[i]["candidates"])


def load_results(root: Path, declared_filters: frozenset = frozenset()) -> dict:
    """{(source, dataset, cell, model): result dict} over every *.json under ``root`` whose ``bench`` is a known scorer. Two files for one key are an error (no silent overwrite).
    A result whose spec carries a post-hoc filter token (ema / tmed / kal / oneeuro / flowema) is REFUSED unless that exact spec belongs to a declared filter-baseline row (``declared_filters``)."""
    out = {}
    for f in sorted(root.rglob("*.json")):
        try:
            d = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        b = str(d.get("bench", "")) if isinstance(d, dict) else ""
        if b.startswith("bench_joint") or b.startswith("dn01_heldout_eval"):
            key = ("bench", d.get("dataset"), d.get("cell"), d.get("model"))
        elif b.startswith("dn01_bvi_fnoise"):
            key = ("bvi", "bvi_rlv", "fnoise", d.get("model"))
        else:
            continue
        if key in out:
            raise ValueError(f"two result files for {key}: {out[key]['_file']} and {f}")
        if filter_tokens(d.get("model", ""), d) and d.get("model") not in declared_filters:
            raise ValueError(f"REFUSED {f}: spec {d.get('model')!r} carries a post-filter token {filter_tokens(d.get('model', ''), d)} but is not a declared filter-baseline row (TAG hygiene)")
        if d.get("limit_clips") is not None:      # Auditor WP18 B5 / part-1 1b: smoke results and unpinned llvar results never enter the table
            raise ValueError(f"REFUSED {f}: limit_clips={d['limit_clips']!r} (a smoke result)")
        if d.get("cell") in ("llvar", "llvarf") and d.get("llvar_draws_sha256") is None:
            raise ValueError(f"REFUSED {f}: {d.get('cell')} result with a null llvar_draws_sha256")
        d["_file"] = str(f)
        out[key] = d
    return out


def tokens_of(model: str, result: dict | None) -> list:
    toks = [t for t in str(model).split("+")] + list((result or {}).get("depth_post", []) or [])
    return sorted({t for t in toks if POST_RE.match(t)})


def check_tag_hygiene(cfg: dict) -> None:
    """s4: TAG only on ours (main table). Ours candidates carry exactly the configured TAG token."""
    tag = cfg["tag_token"]
    for row in cfg["rows"]:
        if row.get("table", "main") != "main":
            continue
        for cand in row["candidates"]:
            t = tokens_of(cand["model"], None)
            if row["kind"] == "ours":
                assert t == [tag], f"ours row {row['id']!r} candidate {cand['model']!r} must carry exactly the TAG token {tag!r}, has {t}"
                assert not filter_tokens(cand["model"]), f"ours row {row['id']!r} candidate {cand['model']!r} carries a smoothing filter"
            elif row["kind"] == "filter_baseline":       # declared filter baseline (declared_filter_models checks the declaration): a filter token and no TAG / probe token
                assert filter_tokens(cand["model"]) and not [x for x in t if x not in filter_tokens(cand["model"])], f"filter-baseline row {row['id']!r} candidate {cand['model']!r} must carry a smoothing filter and nothing else, has {t}"
            else:
                assert not t, f"non-ours main-table row {row['id']!r} candidate {cand['model']!r} carries a depth post-filter {t}: TAG appears only on ours rows (PROTOCOL_TABLE s4)"


# ------------------------------------------------------------------------------------------------ evaluation
def value(res: dict, cell: dict, row: dict, model: str):
    r = res.get((cell["source"], cell["dataset"], cell["cell"], model))
    if r is None:
        return None
    if row.get("output") == "relative" and not cell.get("relative_ok", False):
        return None                                                      # relative-output rows compete in aligned cells only
    return _get(r, cell["field"])


def better(a: float, b: float, direction: str) -> bool:
    return a < b if direction == "min" else a > b


def cell_winner(vals: dict, direction: str):
    """({row: value} of the competitors with a value) -> the strictly best row or None (blank or tie). Exact float comparison."""
    if not vals:
        return None, "blank"
    best = min(vals.values()) if direction == "min" else max(vals.values())
    top = sorted(r for r, v in vals.items() if v == best)
    return (top[0], "win") if len(top) == 1 else (None, "tie:" + "+".join(top))


def rank_of(v: float, others, direction: str) -> int:
    return 1 + sum(1 for o in others if better(o, v, direction))


def select(cfg: dict, res: dict) -> tuple[dict, list]:
    """s3: chosen candidate per row (same rule everywhere) + the SELECTION.md lines."""
    counted = [c for c in cfg["cells"] if c.get("counted", True)]
    released = [r for r in cfg["rows"] if r["kind"] == "released_raw" and r.get("competes", True) and r.get("table", "main") == "main"]
    chosen, lines = {}, ["# SELECTION (PROTOCOL_TABLE s3)", "",
                         "Rule, identical for ours and every fine-tuned baseline: for each candidate, count the counted non-blank cells it wins STRICTLY against the released raw rows (pool = released rows + the candidate); most wins, "
                         "then the lower mean rank in that pool, then the earlier listed candidate. Rows with one candidate are not selections (count 1).", ""]
    lines += ["| row | kind | candidates | chosen | wins vs released | mean rank |", "|---|---|---|---|---|---|"]
    detail = []
    for row in cfg["rows"]:
        cands = row["candidates"]
        if row.get("table", "main") != "main":
            chosen[row["id"]] = cands[0]["model"] if cands else None
            continue
        if len(cands) == 1 or row["kind"] in ("released_raw", "reference", "filter_baseline"):
            chosen[row["id"]] = cands[0]["model"]
            lines.append(f"| {row['id']} | {row['kind']} | {len(cands)} | {cands[0]['label']} | - | - |")
            continue
        scored = []
        for i, cand in enumerate(cands):
            wins, ranks = 0, []
            for cell in counted:
                pool = {r["id"]: value(res, cell, r, r["candidates"][0]["model"]) for r in released}
                pool = {k: v for k, v in pool.items() if v is not None}
                mine = value(res, cell, row, cand["model"])
                allv = dict(pool)
                if mine is not None:
                    allv["__cand__"] = mine
                if not allv:
                    continue                                              # blank cell
                w, _ = cell_winner(allv, cell["direction"])
                wins += int(w == "__cand__")
                ranks.append(rank_of(mine, pool.values(), cell["direction"]) if mine is not None else len(pool) + 1)
            scored.append((-wins, sum(ranks) / len(ranks) if ranks else float("inf"), i, cand, wins))
        scored.sort(key=lambda t: (t[0], t[1], t[2]))
        best = scored[0]
        chosen[row["id"]] = best[3]["model"]
        lines.append(f"| {row['id']} | {row['kind']} | {len(cands)} | {best[3]['label']} | {best[4]} | {best[1]:.4f} |")
        detail.append(f"\n## {row['id']} ({len(cands)} candidates)\n\n| candidate | wins vs released | mean rank |\n|---|---|---|")
        detail += [f"| {s[3]['label']}{' **(chosen)**' if s is best else ''} | {s[4]} | {s[1]:.4f} |" for s in sorted(scored, key=lambda t: t[2])]
    return chosen, lines + detail


def evaluate(cfg: dict, res: dict, chosen: dict) -> dict:
    rows = [r for r in cfg["rows"] if r.get("table", "main") == "main"]
    ours = next((r["id"] for r in rows if r["kind"] == "ours"), None)
    assert ours is not None, "the config needs one row of kind 'ours'"
    cells, n_nb, wins = [], 0, 0
    for cell in sorted(cfg["cells"], key=lambda c: c["id"]):
        vals_all = {r["id"]: value(res, cell, r, chosen[r["id"]]) for r in rows}
        comp = {r["id"]: vals_all[r["id"]] for r in rows if r.get("competes", r["kind"] not in ("reference", "filter_baseline")) and vals_all[r["id"]] is not None}
        w, status = cell_winner(comp, cell["direction"])
        counted = bool(cell.get("counted", True))
        blank = not comp
        entry = {"id": cell["id"], "label": cell["label"], "group": cell.get("group", ""), "counted": counted, "direction": cell["direction"], "values": vals_all, "winner": w, "status": status, "blank": blank,
                 "ours_missing": (not blank) and vals_all[ours] is None and ours not in comp}
        if counted and not blank:
            n_nb += 1
            wins += int(w == ours)
        cells.append(entry)
    return {"ours": ours, "cells": cells, "n_counted_nonblank": n_nb, "ours_wins": wins, "table_win": n_nb > 0 and 2 * wins > n_nb}


# ------------------------------------------------------------------------------------------------ output
def fnum(v) -> str:
    return "" if v is None else repr(float(v))


def disp(v) -> str:
    return "--" if v is None else f"{v:.4f}"


def write_outputs(cfg: dict, ev: dict, chosen: dict, sel_lines: list, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in cfg["rows"] if r.get("table", "main") == "main"]
    cells = ev["cells"]
    lab = {r["id"]: r["label"] for r in rows}
    # CSV wide + long
    wide = ["row," + ",".join(c["id"] for c in cells)] + [f"{r['id']}," + ",".join(fnum(c["values"][r["id"]]) for c in cells) for r in rows]
    (out / "table1.csv").write_text("\n".join(wide) + "\n")
    long = ["cell,counted,row,value,rank,is_best,status"]
    for c in cells:
        comp = {k: v for k, v in c["values"].items() if v is not None and next(r for r in rows if r["id"] == k).get("competes", True) and next(r for r in rows if r["id"] == k)["kind"] not in ("reference", "filter_baseline")}
        for r in rows:
            v = c["values"][r["id"]]
            rk = rank_of(v, comp.values(), c["direction"]) if r["id"] in comp else ""
            long.append(f"{c['id']},{int(c['counted'])},{r['id']},{fnum(v)},{rk},{int(c['winner'] == r['id'])},{c['status']}")
    (out / "cells.csv").write_text("\n".join(long) + "\n")
    # win summary
    summ = [f"WIN COUNT: ours = {ev['ours']}: best in {ev['ours_wins']} of {ev['n_counted_nonblank']} counted non-blank cells (needs > half): TABLE {'WON' if ev['table_win'] else 'NOT WON'}", ""]
    for c in cells:
        tag = "" if c["counted"] else " [not counted]"
        summ.append(f"{c['id']}{tag}: " + ("blank" if c["blank"] else f"winner={c['winner'] or '-'} ({c['status']})" + (" [ours missing]" if c["ours_missing"] else "")))
    (out / "wins.txt").write_text("\n".join(summ) + "\n")
    (out / "wins.json").write_text(json.dumps({"ours": ev["ours"], "n_counted_nonblank": ev["n_counted_nonblank"], "ours_wins": ev["ours_wins"], "table_win": ev["table_win"],
                                               "cells": {c["id"]: {"counted": c["counted"], "blank": c["blank"], "winner": c["winner"], "status": c["status"], "ours_missing": c["ours_missing"]} for c in cells}}, indent=1, sort_keys=True) + "\n")
    (out / "SELECTION.md").write_text("\n".join(sel_lines) + "\n")
    # ratio ours / reference
    ref = cfg.get("ratio_ref")
    if ref:
        rl = ["cell,ours,ref,ratio"]
        for c in cells:
            o, r_ = c["values"].get(ev["ours"]), c["values"].get(ref)
            rl.append(f"{c['id']},{fnum(o)},{fnum(r_)},{fnum(o / r_) if o is not None and r_ not in (None, 0.0) else ''}")
        (out / "ratio.csv").write_text("\n".join(rl) + "\n")
    # LaTeX + HTML
    def cellfmt(c, rid, tex):
        v = c["values"][rid]
        s = disp(v)
        if v is not None and c["winner"] == rid:
            return f"\\textbf{{{s}}}" if tex else f"<b>{s}</b>"
        return s
    cols = "l" + "c" * len(cells)
    tex = ["\\begin{tabular}{" + cols + "}", "\\toprule", "Method & " + " & ".join(c["label"].replace("&", "\\&") + ("" if c["counted"] else "$^\\dagger$") for c in cells) + " \\\\", "\\midrule"]
    for r in rows:
        tex.append(lab[r["id"]].replace("&", "\\&").replace("_", "\\_") + " & " + " & ".join(cellfmt(c, r["id"], True) for c in cells) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}", f"% ours wins {ev['ours_wins']}/{ev['n_counted_nonblank']} counted non-blank cells; best = strictly best; $^\\dagger$ = shown, not counted"]
    (out / "table1.tex").write_text("\n".join(tex) + "\n")
    h = ["<table>", "<thead><tr><th>Method</th>" + "".join(f"<th>{html.escape(c['label'])}{'' if c['counted'] else '&dagger;'}</th>" for c in cells) + "</tr></thead>", "<tbody>"]
    for r in rows:
        h.append(f"<tr><td>{html.escape(lab[r['id']])}</td>" + "".join(f"<td>{cellfmt(c, r['id'], False)}</td>" for c in cells) + "</tr>")
    h += ["</tbody>", f"<caption>ours best in {ev['ours_wins']} of {ev['n_counted_nonblank']} counted non-blank cells (strictly best; ties are not wins); &dagger; shown, not counted</caption>", "</table>"]
    (out / "table1.html").write_text("\n".join(h) + "\n")
    # appendix (ours+TAG vs released+TAG etc.): the rows with table == appendix, no win rule
    arows = [r for r in cfg["rows"] if r.get("table") == "appendix"]
    if arows:
        al = ["row," + ",".join(c["id"] for c in cells)] + [f"{r['id']}," + ",".join(fnum(value(ev["_res"], c0, r, r['candidates'][0]['model'])) for c0 in sorted(cfg['cells'], key=lambda c: c['id'])) for r in arows]
        (out / "appendix.csv").write_text("\n".join(al) + "\n")


def run(cfg: dict, results: Path, out: Path, access_log: Path | None = None) -> dict:
    declared = declared_filter_models(cfg)
    check_tag_hygiene(cfg)
    res = load_results(results, declared)
    chosen, sel = select(cfg, res)
    ev = evaluate(cfg, res, chosen)
    ev["_res"] = res
    write_outputs(cfg, ev, chosen, sel, out)
    if access_log is not None:                                              # s3: every TEST access is appended (separate, append-only, carries a time; excluded from the byte-identity of the table)
        import time
        test_files = sorted(r["_file"] for r in res.values() if str(r.get("split", "")) in ("test", "all") or r.get("allow_test"))
        with access_log.open("a") as f:
            f.write(json.dumps({"t_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tool": "dn01_headline_table", "n_test_files": len(test_files),
                                "files": [{"path": p, "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in test_files]}) + "\n")
    ev.pop("_res")
    return ev


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results")
    ap.add_argument("--config")
    ap.add_argument("--out")
    ap.add_argument("--access-log")
    ap.add_argument("--make-default-config")
    ap.add_argument("--headline-cell", default="llvar")
    ap.add_argument("--static-cell", default="st_llvar")
    a = ap.parse_args()
    if a.make_default_config:
        Path(a.make_default_config).write_text(json.dumps(make_default_config(a.headline_cell, a.static_cell), indent=1) + "\n")
        print("wrote", a.make_default_config)
        return
    cfg = json.loads(Path(a.config).read_text())
    ev = run(cfg, Path(a.results), Path(a.out), Path(a.access_log) if a.access_log else None)
    print((Path(a.out) / "wins.txt").read_text())
    sys.exit(0 if ev["n_counted_nonblank"] else 2)


if __name__ == "__main__":
    main()
