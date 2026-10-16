"""Protocol v3 WP35: the paper's Table 1 + supplementary-table generators, built on the WP10 aggregator (scripts/dn01_headline_table.py).

The WP10 aggregator owns every RULE (strict best / ties are not wins / more-than-half / selection symmetry / TAG hygiene) and its input (a directory of bench_joint / dn01_heldout_eval / dn01_bvi_fnoise result JSONs
+ a config). This script only (1) fixes the v3 cell->dataset mapping, (2) lays the result out (B5: split by metric group; rows = released raw, FT baselines, cascades, ours [+TAG]; best bold, second underlined,
wins/counted column, footnotes, metric arrows), (3) writes the supplement tables. Nothing here is scored from a real file in the tests: ``--make-mock DIR`` writes clearly-marked MOCK inputs (random numbers).

DATASET MAP (Manager 2026-10-04): kitti (+ kitti_minus_0047 disclosure, not counted), bonn, scannet_rgbfix (depth) + scannet_tae_rgbfix (TAE) [or scannetpp for all ScanNet cells under ``--scannet-column scannetpp``],
sintel_rgbfix, nyuv2, nuscenes_night_test, BVI. The LITERAL scannet / scannet_tae / sintel (R/B-swapped frames) are parity-only and must never reach Table 1 (asserted by ``assert_no_literal``).

RULES (Auditor audit/TABLE1_display_rules_20261004.md): WIN = strict unique best at FULL precision (tie = no winner), printed in its own Wins column (caption says so). DISPLAY bold = best at the PRINTED (4-decimal)
precision, every tied row bold; underline = second best at printed precision only when exactly one row is bold, tied seconds all underlined; cells where bold and win disagree go to display_vs_win.md/json. kitti_minus_0047
is never marked or counted. Parity TAE tolerance +-0.02 on VDA's scale (harness-to-VDA factor shown; VDA-L < VDA-S checked; a miss is a footnote, not a block). Selection: wins -> mean rank -> lower geometric mean of
per-cell ours/released ratios (delta1 as released/ours) -> declared candidate order; the deciding step is reported, step 3/4 = DISCLOSE. The pre-declared SELECTION.md must exist, be sha+commit-pinned and its COMMIT time (git) must precede the earliest candidate entry (limit_clips smoke runs and released models ignored) of the merged --test-access-log; mtime is a warning only. Old wording: older than the earliest
TEST score file (check_selection_guard; CLI refuses unless --no-selection-guard, MOCK only). Other decisions: reference rows never marked; 4-decimal display; llvarf supplement reuses llvar-selected checkpoints.
Usage: python scripts/dn01_v3_table1.py --results DIR --config cfg.json --out DIR [--scannet-column scannet|scannetpp] [--latency GLOB] [--parity parity.json] [--mass MASS_TABLE.md]
       python scripts/dn01_v3_table1.py --make-mock DIR   (writes DIR/results/*.json + DIR/config.json + DIR/parity.json, all MOCK)"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dn01_headline_table as ht  # noqa: E402  (the WP10 aggregator: rules + reader)

TAG = "agf0.2t0.06"
LITERAL_FORBIDDEN = {"scannet", "scannet_tae", "sintel"}                      # R/B-swapped literal frames: parity only
FIELDS = {"abs_rel_unaligned": ("AbsRel", "min", "metric"), "abs_rel_aligned": ("AbsRel", "min", "aligned"), "delta1_aligned": ("$\\delta_1$", "max", "aligned"), "tae": ("TAE", "min", "")}
PANELS = [("A", ["abs_rel_unaligned", "abs_rel_aligned", "delta1_aligned"]), ("B", ["tae", "static_flicker", "f_noise", "kitti0047"])]
MG_TITLE = {"abs_rel_unaligned": ("AbsRel (metric)", "$\\downarrow$"), "abs_rel_aligned": ("AbsRel (aligned)", "$\\downarrow$"), "delta1_aligned": ("$\\delta_1$", "$\\uparrow$"), "tae": ("TAE", "$\\downarrow$"),
            "static_flicker": ("Static flicker", "$\\downarrow$"), "f_noise": ("BVI-RLV $F_{noise}$", "$\\downarrow$"), "kitti0047": ("KITTI$-$0047 AbsRel (al.)", "$\\downarrow$")}
MD_ARROW = {"$\\downarrow$": "↓", "$\\uparrow$": "↑"}
GROUP_ORDER = ["KITTI", "Bonn", "ScanNet", "ScanNet++", "Sintel", "NYUv2", "nuScenes-night", "BVI-RLV", "KITTI-0047"]
SECTIONS = [("released_raw", "Released (raw)"), ("ft_baseline", "Fine-tuned baselines"), ("cascade", "Cascades"), ("ours", "Ours"), ("reference", "Reference (not competing)")]
NOTE_TEXT = {
    "scannetpp": "ScanNet++ fallback column (ScanNet v2 access did not land), never compared with the VDA-paper ScanNet numbers; SYN+REAL rows trained on the disjoint ScanNet++ sem\\_test scenes of the same capture setup (evaluation: nvs\\_sem\\_val + nvs\\_test\\_iphone).",
    "irs": "Pre-v3 Isambard runs trained on an IRS subset (Restaurant ${\\sim}$50\\%, Office ${\\sim}$78\\% of files); v3 runs use the complete copy. The comparison with released rows carries this data-mix difference.",
    "kitti0047": "KITTI minus drive 0047 (VKITTI Scene20 replicates this VDA eval drive): disclosure column, shown, not counted.",
    "r50": "Real share 50\\% of the mix (r50), which oversamples the 50 ScanNet++ train scenes ${\\sim}7\\times$; the variant is chosen on TEST by the same rule for every row (SELECTION.md).",
    "tae": "ScanNet TAE: sorted first 20 scenes, frames [10:180]; VDA's list comes from an unsorted glob, so its exact 20 scenes are unrecoverable (disclosed).",
    "rel": "Relative-output rows compete in aligned cells only (\\texttt{--} elsewhere).",
    "taepar": "TAE: our re-implementation of VDA's TAE protocol; released VDA scores {X} here vs {Y} in its paper (parity miss, see the parity table).",
    "ref": "Reference row: shown, never competes, never counted.",
}
NOTE_ORDER = ["scannetpp", "irs", "kitti0047", "r50", "tae", "taepar", "rel", "ref"]
# VDA-paper references for the parity table (PAPER_TODO_V3.md section 7b, 7b corrections): (set, model, metric) -> (value, paper table / row source)
VDA_REFS = {
    ("ScanNet-500", "vda_s", "abs_rel"): (0.110, "VDA paper Table 1, ScanNet row, VDA-S, up-to-500-frame videos"),
    ("ScanNet-500", "vda_s", "delta1"): (0.876, "VDA paper Table 1, ScanNet row, VDA-S, up-to-500-frame videos"),
    ("ScanNet-500", "vda_l", "abs_rel"): (0.089, "VDA paper Table 1, ScanNet row, VDA-L, up-to-500-frame videos"),
    ("ScanNet-500", "vda_l", "delta1"): (0.926, "VDA paper Table 1, ScanNet row, VDA-L, up-to-500-frame videos"),
    ("ScanNet-TAE-170", "vda_s", "tae"): (0.703, "VDA paper TAE table (ScanNet, 170 frames [10:180], eval_tae.sh); table id as cited in PAPER_TODO_V3 7b"),
    ("ScanNet-TAE-170", "vda_l", "tae"): (0.570, "VDA paper TAE table (ScanNet, 170 frames [10:180], eval_tae.sh); table id as cited in PAPER_TODO_V3 7b"),
    ("ScanNet-90", "vda_l", "abs_rel"): (0.075, "VDA paper Table 7, VDA-L, 90-frame (the only 90-frame reference; not a Table 1 parity)"),
    ("ScanNet-90", "vda_l", "delta1"): (0.954, "VDA paper Table 7, VDA-L, 90-frame (the only 90-frame reference; not a Table 1 parity)"),
}
PARITY_TOL = {"abs_rel": 0.005, "delta1": 0.005, "tae": 0.02}                 # PAPER_TODO_V3 s7; TAE +-0.02 absolute on VDA's scale (Auditor TABLE1_display_rules 4)


# ------------------------------------------------------------------------------------------------ config
def v3_cells(scannet_column: str = "scannet", headline_cell: str = "llvar", static_cell: str = "st_llvar") -> list:
    """The PROTOCOL_TABLE s2 counted cells (via the WP10 default config) with the v3 dataset map applied and layout attributes (``mgroup``, ``panel``) added."""
    assert scannet_column in ("scannet", "scannetpp"), scannet_column
    base = ht.make_default_config(headline_cell, static_cell)["cells"]
    sn_label = "ScanNet++" if scannet_column == "scannetpp" else "ScanNet"
    cells = []
    for c in base:
        c = dict(c)
        ds = c["dataset"]
        field = c["field"].split(".")[-1]
        if ds == "scannet":
            c["dataset"] = "scannetpp" if scannet_column == "scannetpp" else ("scannet_tae_rgbfix" if field == "tae" else "scannet_rgbfix")
            c["group"] = sn_label
        elif ds == "sintel":
            c["dataset"] = "sintel_rgbfix"
        elif ds == "nuscenes_night_test":
            c["group"] = "nuScenes-night"
        elif ds == "kitti_minus_0047":
            c["group"] = "KITTI-0047"
        c["label"] = c["label"].replace("ScanNet*", sn_label)
        c["mgroup"] = "static_flicker" if c["cell"] == static_cell else ("f_noise" if field == "f_noise" else ("kitti0047" if ds == "kitti_minus_0047" else field))
        c["panel"] = "B" if c["mgroup"] in PANELS[1][1] else "A"
        cells.append(c)
    return cells


def assert_no_literal(cells: list) -> None:
    """Literal scannet / scannet_tae / sintel never reach Table 1 (R/B-swapped frames: parity only)."""
    bad = sorted({c["dataset"] for c in cells if c["dataset"] in LITERAL_FORBIDDEN})
    assert not bad, f"literal (R/B-swapped) dataset(s) {bad} in the Table 1 cell list: use the *_rgbfix variants (parity-only sets never enter Table 1)"


def build_cfg(user: dict, scannet_column: str = "scannet", headline_cell: str = "llvar", static_cell: str = "st_llvar") -> dict:
    cells = v3_cells(scannet_column, headline_cell, static_cell)
    assert_no_literal(cells)
    cfg = {"tag_token": user.get("tag_token", TAG), "cells": cells, "rows": user["rows"], "ratio_ref": user.get("ratio_ref"), "step3_ref_row": user.get("step3_ref_row", "mvda_s"), "title": ("MOCK (random numbers, not results) " if user.get("MOCK") else "") + "Table 1", "scannet_column": scannet_column}
    for r in cfg["rows"]:
        for cand in r["candidates"]:
            assert not (LITERAL_FORBIDDEN & set(cand.get("datasets", []))), f"row {r['id']}: literal dataset in candidate"
    return cfg


# ------------------------------------------------------------------------------------------------ rendering helpers
def tex_esc(s: str) -> str:
    return s.replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#")


def disp(v) -> str:
    return "--" if v is None else f"{v:.4f}"


def printed(v: float) -> float:
    """The value as printed (4 decimals): the display precision (disp)."""
    return float(f"{v:.4f}")


def marks(cell_ev: dict, rows_by_id: dict) -> dict:
    """DISPLAY marks (Auditor TABLE1_display_rules 1-2): bold = best at PRINTED precision, every tied row bold; underline = second best at printed precision, only when exactly one row is bold, every tied
    second underlined. Counted, non-blank cells only; reference rows never marked. NOT the win (the win is the full-precision strict unique best in cell_ev['winner'])."""
    out = {}
    if not cell_ev["counted"] or cell_ev["blank"]:
        return out
    comp = {k: printed(v) for k, v in cell_ev["values"].items() if v is not None and rows_by_id[k]["kind"] != "reference" and rows_by_id[k].get("competes", True)}
    if not comp:
        return out
    pick = min if cell_ev["direction"] == "min" else max
    best = pick(comp.values())
    bold = [k for k, v in comp.items() if v == best]
    for k in bold:
        out[k] = "bold"
    if len(bold) == 1:
        rest = [v for k, v in comp.items() if k not in bold]
        if rest:
            sec = pick(rest)
            for k, v in comp.items():
                if v == sec:
                    out[k] = "under"
    return out


def disagreements(ev: dict, rows_by_id: dict) -> list:
    """Counted non-blank cells where the display-bold set differs from the full-precision win (winner None = tie)."""
    out = []
    for c in ev["cells"]:
        if not c["counted"] or c["blank"]:
            continue
        bold = sorted(k for k, m in marks(c, rows_by_id).items() if m == "bold")
        win = [c["winner"]] if c["winner"] else []
        if bold != win:
            out.append({"cell": c["id"], "display_bold": bold, "full_precision_winner": c["winner"], "status": c["status"],
                        "values": {k: c["values"][k] for k in sorted(set(bold) | set(win))}})
    return out


def disagreement_md(items: list) -> str:
    head = "## Appendix: cells where display bold and the full-precision win disagree\n\nBold is best at printed (4-decimal) precision, every tied row bold; the win is the strict unique best at full precision (a tie has no winner).\n\n"
    if not items:
        return head + "None.\n"
    return head + md_table(["cell", "display bold", "full-precision winner", "status", "full-precision values"],
                           [[i["cell"], ", ".join(i["display_bold"]) or "-", i["full_precision_winner"] or "- (tie)", i["status"], "; ".join(f"{k}={v!r}" for k, v in i["values"].items())] for i in items])


def fmt(v, mark, tex: bool) -> str:
    s = disp(v)
    if v is None or mark is None:
        return s
    if tex:
        return f"\\textbf{{{s}}}" if mark == "bold" else f"\\underline{{{s}}}"
    return f"**{s}**" if mark == "bold" else f"<u>{s}</u>"


def row_notes(row: dict, chosen_cand: dict) -> list:
    n = set(row.get("notes", []))
    if (chosen_cand.get("axes", {}) or {}).get("mix") == "r50":
        n.add("r50")
    if row.get("output") == "relative":
        n.add("rel")
    if row["kind"] == "reference":
        n.add("ref")
    return [k for k in NOTE_ORDER if k in n]


def cand_of(row: dict, model: str) -> dict:
    return next(c for c in row["candidates"] if c["model"] == model)


class Notes:
    """Footnote letters assigned in NOTE_ORDER over the keys actually used."""

    def __init__(self):
        self.used = set()

    def use(self, k):
        self.used.add(k)

    def letter(self, k):
        order = [x for x in NOTE_ORDER if x in self.used]
        return chr(ord("a") + order.index(k))

    def sup(self, keys, tex=True):
        keys = [k for k in NOTE_ORDER if k in keys]
        for k in keys:
            self.used.add(k)
        return "" if not keys else (f"$^{{{','.join(self.letter(k) for k in keys)}}}$" if tex else "<sup>" + ",".join(self.letter(k) for k in keys) + "</sup>")


def panel_columns(cells: list, mgroups: list) -> list:
    """[(mgroup, [cell, ...] in GROUP_ORDER)] for the cells of the given metric groups."""
    out = []
    for mg in mgroups:
        cs = [c for c in cells if c["mgroup"] == mg]
        cs.sort(key=lambda c: (GROUP_ORDER.index(c["group"]) if c["group"] in GROUP_ORDER else 99, c["id"]))
        if cs:
            out.append((mg, cs))
    return out


def render_table1(cfg: dict, ev: dict, chosen: dict) -> tuple[str, str]:
    """(LaTeX, markdown) of Table 1: one tabular per metric panel (A accuracy, B temporal / real / disclosure), shared rows, section rules."""
    rows = [r for r in cfg["rows"] if r.get("table", "main") == "main"]
    rows_by_id = {r["id"]: r for r in rows}
    cells = {c["id"]: c for c in ev["cells"]}
    cfg_cells = {c["id"]: c for c in cfg["cells"]}
    mk = {cid: marks(c, rows_by_id) for cid, c in cells.items()}
    notes = Notes()
    sn_label = "ScanNet++" if cfg.get("scannet_column") == "scannetpp" else "ScanNet"
    tex_parts, md_parts = [], []
    for pname, mgroups in PANELS:
        cols = panel_columns([dict(cfg_cells[i], **{}) for i in cfg_cells if cfg_cells[i]["panel"] == pname], mgroups)
        flat = [c for _, cs in cols for c in cs]
        if not flat:
            continue
        spec = "l" + "".join("c" * len(cs) + "|" for _, cs in cols)[:-1] + "|c"
        t = [f"% panel {pname}", "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{" + spec + "}", "\\toprule"]
        # header 1: metric groups with arrows
        h1, cmid, pos = ["Method"], [], 2
        for mg, cs in cols:
            title, arrow = MG_TITLE[mg]
            extra = notes.sup(["tae"] + (["taepar"] if cfg.get("tae_parity_miss") else [])) if mg == "tae" else ""
            h1.append(f"\\multicolumn{{{len(cs)}}}{{c}}{{{title} {arrow}{extra}}}")
            cmid.append(f"\\cmidrule(lr){{{pos}-{pos + len(cs) - 1}}}")
            pos += len(cs)
        h1.append("Wins")
        t += [" & ".join(h1) + " \\\\", "".join(cmid)]
        h2 = [""]
        for _, cs in cols:
            for c in cs:
                lab = "KITTI$-$0047" if c["group"] == "KITTI-0047" else tex_esc(c["group"])
                ex = ""
                if c["group"] == sn_label and cfg.get("scannet_column") == "scannetpp":
                    ex = notes.sup(["scannetpp"])
                if not c.get("counted", True):
                    ex += notes.sup(["kitti0047"])
                h2.append(lab + ex)
        h2.append("/ counted" if pname == "B" else "")
        t += [" & ".join(h2) + " \\\\", "\\midrule"]
        # panel counts
        pc = [c for c in flat if cells[c["id"]]["counted"] and not cells[c["id"]]["blank"]]
        first = True
        for kind, title in SECTIONS:
            srows = [r for r in rows if r["kind"] == kind]
            if not srows:
                continue
            if not first:
                t.append("\\midrule")
            first = False
            for r in srows:
                cand = cand_of(r, chosen[r["id"]])
                lab = tex_esc(r["label"]) + notes.sup(row_notes(r, cand))
                body = [fmt(cells[c["id"]]["values"][r["id"]], mk[c["id"]].get(r["id"]) if c.get("counted", True) else None, True) for c in flat]
                if r["kind"] == "reference":
                    wn = "--"
                else:
                    pw = sum(1 for c in pc if cells[c["id"]]["winner"] == r["id"])
                    tot = sum(1 for c in ev["cells"] if c["counted"] and not c["blank"] and c["winner"] == r["id"])
                    wn = f"{pw}/{len(pc)}" if pname == "A" else f"{pw}/{len(pc)} (all: {tot}/{ev['n_counted_nonblank']})"
                t.append(" & ".join([lab] + body + [wn]) + " \\\\")
        t += ["\\bottomrule", "\\end{tabular}}"]
        tex_parts.append("\n".join(t))
        # markdown
        hdr = ["Method"] + [f"{MG_TITLE[mg][0].replace('$', '').replace(chr(92) + 'delta_1', 'δ1').replace('F_{noise}', 'F_noise')} {MD_ARROW[MG_TITLE[mg][1]]}: {c['group']}" for mg, cs in cols for c in cs] + ["Wins"]
        md = ["", f"### Panel {pname}", "", "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
        for kind, title in SECTIONS:
            for r in [r for r in rows if r["kind"] == kind]:
                cand = cand_of(r, chosen[r["id"]])
                body = [fmt(cells[c["id"]]["values"][r["id"]], mk[c["id"]].get(r["id"]) if c.get("counted", True) else None, False) for c in flat]
                wn = "--" if r["kind"] == "reference" else f"{sum(1 for c in pc if cells[c['id']]['winner'] == r['id'])}/{len(pc)}"
                md.append("| " + " | ".join([r["label"] + notes.sup(row_notes(r, cand), False)] + body + [wn]) + " |")
        md_parts.append("\n".join(md))
    foot_keys = [k for k in NOTE_ORDER if k in notes.used]
    cap = (f"Ours is best in {ev['ours_wins']} of {ev['n_counted_nonblank']} counted non-blank cells (ours wins the table iff more than half; ties are not wins). "
           "Wins are counted at full precision (a tie has no winner); the Wins column is that count. Bold = best at the printed precision (all tied rows bold), underline = second best at the printed precision (only when one row is bold).")
    xy = cfg.get("taepar_xy", ("X", "Y"))
    foot = [f"$^{{{notes.letter(k)}}}$ " + NOTE_TEXT[k].replace("{X}", str(xy[0])).replace("{Y}", str(xy[1])) for k in foot_keys]
    tex = ("% generated by scripts/dn01_v3_table1.py\n\\begin{table*}[t]\n\\centering\\small\n" + "\n\\medskip\n".join(tex_parts) + "\n\\par\\smallskip\n{\\footnotesize " + cap + " \\\\\n" + " \\\\\n".join(foot) + "}\n"
           f"\\caption{{{cfg.get('title', 'Table 1')}}}\n\\label{{tab:main}}\n\\end{{table*}}\n")
    mdt = f"## {cfg.get('title', 'Table 1')}\n" + "\n".join(md_parts) + "\n\n" + re.sub(r"\\textbackslash\{\}|\\", "", cap) + "\n\n" + "\n".join(f"- ({notes.letter(k)}) " + re.sub(r"\$\^?\{?\\?sim\}?\\?times\$", "~7x", NOTE_TEXT[k]).replace("\\%", "%").replace("\\_", "_").replace("\\texttt{--}", "--").replace("${\\sim}7\\times$", "~7x") for k in foot_keys) + "\n"
    return tex, mdt


# ------------------------------------------------------------------------------------------------ generic tables
def md_to_tex_table(header: list, body: list) -> str:
    def cx(s):
        s = str(s)
        b = re.fullmatch(r"\*\*(.*)\*\*", s)
        return f"\\textbf{{{tex_esc(b.group(1))}}}" if b else tex_esc(s)
    return "\\begin{tabular}{l" + "r" * (len(header) - 1) + "}\n\\toprule\n" + " & ".join(tex_esc(h) for h in header) + " \\\\\n\\midrule\n" + "\n".join(" & ".join(cx(x) for x in r) + " \\\\" for r in body) + "\n\\bottomrule\n\\end{tabular}\n"


def md_table(header: list, body: list) -> str:
    return "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n" + "\n".join("| " + " | ".join(str(x) for x in r) + " |" for r in body) + "\n"


def rows_x_cols(cfg: dict, res: dict, chosen: dict, rows: list, cols: list, field_dir: dict | None = None, mark: bool = True):
    """Generic value grid (rows x cols of cell dicts) with strict-best bold among ``rows`` (competing, same relative rule). -> (header, body_md, body_tex)."""
    header = ["Method"] + [c["label"] for c in cols]
    md, tx = [], []
    mk = []
    for c in cols:
        vals = {r["id"]: ht.value(res, c, r, chosen.get(r["id"], r["candidates"][0]["model"])) for r in rows}
        comp = {k: v for k, v in vals.items() if v is not None and next(r for r in rows if r["id"] == k)["kind"] != "reference"}
        w, _ = ht.cell_winner(comp, c["direction"])
        mk.append((vals, w if mark else None))
    for r in rows:
        md.append([r["label"]] + [fmt(v[r["id"]], "bold" if w == r["id"] else None, False) for v, w in mk])
        tx.append([r["label"]] + [fmt(v[r["id"]], "bold" if w == r["id"] else None, True) for v, w in mk])
    return header, md, tx


def ladder_tables(cfg: dict, res: dict, chosen: dict, prefix: str = "llb3x", levels=(1, 2, 4, 8, 16), datasets=None) -> tuple[str, str]:
    """Noise ladder: per dataset, aligned AbsRel at each llb3x level for every main row + the ours/released-VDA-S gap ratio (success = gap vs released)."""
    rows = [r for r in cfg["rows"] if r.get("table", "main") == "main"]
    ours = next(r for r in rows if r["kind"] == "ours")
    rel = next((r for r in rows if r["kind"] == "released_raw"), None)
    datasets = datasets or sorted({(c["dataset"], c["group"]) for c in cfg["cells"] if c["mgroup"] == "abs_rel_aligned" and c["counted"] and c["dataset"] != "nuscenes_night_test"})
    tex, md = [], []
    for ds, grp in datasets:
        cols = [{"id": f"{ds}|{prefix}{k}", "label": f"x{k}", "source": "bench", "dataset": ds, "cell": f"{prefix}{k}", "field": "video_macro.abs_rel_aligned", "direction": "min", "relative_ok": True} for k in levels]
        header, bmd, btx = rows_x_cols(cfg, res, chosen, rows, cols)
        ratio = ["ours / " + (rel["label"] if rel else "ref")]
        for c in cols:
            o, r_ = ht.value(res, c, ours, chosen[ours["id"]]), (ht.value(res, c, rel, chosen[rel["id"]]) if rel else None)
            ratio.append("--" if o is None or not r_ else f"{o / r_:.3f}")
        md.append(f"### Noise ladder, {grp}: aligned AbsRel (lower is better)\n\n" + md_table(header, bmd + [ratio]))
        tex.append(f"% {grp}\n" + md_to_tex_table(header, btx + [ratio]))
    return "\n".join(tex), "\n".join(md)


def appendix_tables(cfg: dict, res: dict, chosen: dict) -> tuple[str, str]:
    """+/-TAG appendix: rows with table == 'appendix' (released + TAG, ours + TAG, and their no-TAG partners), counted cells only, strict best bold among the appendix rows. Kept OUT of Table 1."""
    arows = [r for r in cfg["rows"] if r.get("table") == "appendix"]
    if not arows:
        return "", ""
    cols = [c for c in sorted(cfg["cells"], key=lambda c: c["id"]) if c.get("counted", True)]
    header, md, tx = rows_x_cols(cfg, res, {}, arows, cols)
    return md_to_tex_table(header, tx), "## Appendix: +TAG on every row (not part of Table 1)\n\n" + md_table(header, md)


def ratio_gmean(res: dict, cells: list, row: dict, model: str, ref_row: dict):
    """Step 3 of the selection: geometric mean over the counted cells of the per-cell gap vs the FIXED reference row (released Metric-VDA-S raw: no TAG, no post-filter; the model ours is fine-tuned from; the row id is
    pinned in the pre-declared SELECTION.md as ``STEP3_REFERENCE_ROW: <id>``). Lower-better cells use ours / ref; delta1 (direction max) uses ref / ours, so LOWER is better everywhere. Cells where either side is
    missing (e.g. the reference is blank) or non-positive are dropped. -> (geometric mean, number of cells used)."""
    import math
    logs = []
    for c in cells:
        o = ht.value(res, c, row, model)
        rv = ht.value(res, c, ref_row, ref_row["candidates"][0]["model"])
        if o is None or rv is None or o <= 0 or rv <= 0:
            continue
        logs.append(math.log(o / rv) if c["direction"] == "min" else math.log(rv / o))
    return (math.exp(sum(logs) / len(logs)) if logs else float("inf")), len(logs)


def read_step3_ref(selection_md: Path | None, cfg: dict) -> str:
    """The step-3 reference row id: ``STEP3_REFERENCE_ROW: <id>`` in the pre-declared SELECTION.md (guarded runs); MOCK runs without a file fall back to cfg['step3_ref_row'] or 'mvda_s'."""
    if selection_md is not None and Path(selection_md).exists():
        m = re.search(r"^STEP3_REFERENCE_ROW:\s*(\S+)\s*$", Path(selection_md).read_text(), re.M)
        if not m:
            raise SystemExit("REFUSED: the pre-declared SELECTION.md does not pin 'STEP3_REFERENCE_ROW: <row id>'")
        return m.group(1)
    return cfg.get("step3_ref_row", "mvda_s")


def select_v3(cfg: dict, res: dict, ref_id: str = "mvda_s"):
    """Selection per Auditor TABLE1_display_rules 5, identical for ours and every FT baseline: (1) most counted non-blank cells won STRICTLY (full precision) against the released raw rows, (2) lower mean rank,
    (3) LOWER geometric mean of per-cell ours/ref ratios vs the fixed row ``ref_id`` (ratio_gmean), (4) the candidate's position in the pre-declared SELECTION.md candidate order. -> (chosen {row: model}, info {row: dict}); info records
    the deciding step; step 3 or 4 sets disclose=True."""
    counted = [c for c in cfg["cells"] if c.get("counted", True)]
    released = [r for r in cfg["rows"] if r["kind"] == "released_raw" and r.get("competes", True) and r.get("table", "main") == "main"]
    chosen, info = {}, {}
    ref_row = next((r for r in cfg["rows"] if r["id"] == ref_id and r.get("table", "main") == "main"), None)
    assert ref_row is not None and ref_row["kind"] == "released_raw" and not ht.tokens_of(ref_row["candidates"][0]["model"], None), f"step-3 reference row {ref_id!r} must be a released raw main row without TAG/post-filter"
    for row in cfg["rows"]:
        cands = row["candidates"]
        if row.get("table", "main") != "main" or len(cands) == 1 or row["kind"] in ("released_raw", "reference"):
            chosen[row["id"]] = cands[0]["model"]
            info[row["id"]] = {"candidates": [], "decided_by": "single" if len(cands) == 1 else "n/a", "disclose": False, "row": row, "ref": ref_id}
            continue
        scored = []
        for i, cand in enumerate(cands):
            wins, ranks = 0, []
            for cell in counted:
                pool = {r["id"]: ht.value(res, cell, r, r["candidates"][0]["model"]) for r in released}
                pool = {k: v for k, v in pool.items() if v is not None}
                mine = ht.value(res, cell, row, cand["model"])
                allv = dict(pool)
                if mine is not None:
                    allv["__cand__"] = mine
                if not allv:
                    continue
                w, _ = ht.cell_winner(allv, cell["direction"])
                wins += int(w == "__cand__")
                ranks.append(ht.rank_of(mine, pool.values(), cell["direction"]) if mine is not None else len(pool) + 1)
            mr = sum(ranks) / len(ranks) if ranks else float("inf")
            gm, ng = ratio_gmean(res, counted, row, cand["model"], ref_row)
            scored.append({"cand": cand, "idx": i, "wins": wins, "mean_rank": mr, "gmean": gm, "n_ratio": ng, "key": (-wins, mr, gm, i)})
        scored.sort(key=lambda d: d["key"])
        a, b = scored[0], scored[1]
        step = next((n for n, (x, y) in zip(("wins", "mean_rank", "geomean_ratio", "candidate_order"), zip(a["key"], b["key"])) if x != y), "candidate_order")
        chosen[row["id"]] = a["cand"]["model"]
        info[row["id"]] = {"candidates": sorted(scored, key=lambda d: d["idx"]), "chosen": a, "decided_by": step, "disclose": step in ("geomean_ratio", "candidate_order"), "row": row, "ref": ref_id}
    return chosen, info


def sha256_file(p: Path) -> str:
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _git(repo: Path, *args: str) -> str:
    import subprocess
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"REFUSED: git {' '.join(args[:2])} failed in {repo}: {r.stderr.strip()[:200]}")
    return r.stdout.strip()


def pin_selection(selection_md: Path, pin: Path) -> None:
    """Record the sha256 AND the git commit of the pre-declared SELECTION.md (candidate order). The file must be committed and unmodified: the commit time is the audit evidence."""
    f = Path(selection_md).resolve()
    if _git(f.parent, "status", "--porcelain", "--", f.name):
        raise SystemExit("REFUSED: SELECTION.md is not committed (untracked or modified): commit it before pinning")
    commit = _git(f.parent, "log", "-1", "--format=%H", "--", f.name)
    if not commit:
        raise SystemExit("REFUSED: SELECTION.md has no commit")
    pin.write_text(json.dumps({"file": f.name, "sha256": sha256_file(f), "commit": commit}) + "\n")


def check_all_mock(results: Path) -> None:
    """--no-selection-guard is for MOCK runs only: refused unless EVERY *.json under the results dir carries ``"MOCK": true`` (and there is at least one)."""
    files = sorted(Path(results).rglob("*.json"))
    bad = []
    for f in files:
        try:
            d = json.loads(f.read_text())
        except (OSError, ValueError):
            bad.append(f.name)
            continue
        if not (isinstance(d, dict) and d.get("MOCK") is True):
            bad.append(f.name)
    if not files or bad:
        raise SystemExit(f"REFUSED: the no-selection-guard (MOCK) mode needs every input JSON to carry MOCK=true; offending: {bad[:5]}{' (no JSON at all)' if not files else ''}")


def is_mock_marked(d) -> bool:
    return (isinstance(d, dict) and d.get("MOCK") is True) or (isinstance(d, dict) and str(d.get("source", "")).upper() == "MOCK")


def check_no_mock_inputs(results: Path, latency, parity) -> None:
    """A GUARDED (real) run refuses any input carrying MOCK: a results JSON, a latency JSON or a parity entry (``MOCK: true`` or source == MOCK)."""
    bad = []
    for f in sorted(Path(results).rglob("*.json")):
        try:
            if is_mock_marked(json.loads(f.read_text())):
                bad.append(f.name)
        except (OSError, ValueError):
            pass
    for f in latency or []:
        if is_mock_marked(json.loads(Path(f).read_text())):
            bad.append(Path(f).name)
    bad += [f"parity:{m.get('set')}/{m.get('model')}/{m.get('metric')}" for m in (parity or []) if is_mock_marked(m)]
    if bad:
        raise SystemExit(f"REFUSED: a guarded (real) run must not read MOCK inputs; MOCK-marked: {bad[:5]}")


def _ts(x: str):
    from datetime import datetime, timezone
    t = datetime.fromisoformat(str(x).replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def _norm_model(m: str) -> str:
    return "+".join(t for t in str(m).split("+") if not ht.POST_RE.match(t))


def read_access_logs(paths) -> list:
    """Merge one or more TEST_access_log.jsonl files (append-only JSON lines) -> list of entry dicts."""
    out = []
    for p in paths:
        for ln in Path(p).read_text().splitlines():
            if ln.strip():
                out.append(json.loads(ln))
    return out


def _spec_key(model: str) -> str:
    """Post-filter-stripped spec with every checkpoint path reduced to its basename (cross-host: /projects/b5dh/x/final.pt == /local/y/final.pt)."""
    toks = []
    for t in _norm_model(model).split("+"):
        k, sep, path = t.partition(":")
        toks.append(f"{k}:{Path(path).name}" if sep else t)
    return "+".join(toks)


def _shas(x) -> set:
    if not x:
        return set()
    return {str(v) for v in x} if isinstance(x, (list, tuple, set)) else {str(x)}


def _result_shas(f: Path, d: dict) -> set:
    """Checkpoint sha256(s) of a result: its own ckpt_sha256 / checkpoint_sha256 field, else the nearest RUN_MANIFEST.json above it (checkpoints.<tag>.sha256, or ckpt_sha256)."""
    out = _shas(d.get("ckpt_sha256"))
    if out:
        return out
    for par in list(f.parents)[:4]:
        m = par / "RUN_MANIFEST.json"
        if m.exists():
            return _manifest_shas(json.loads(m.read_text()))
    return set()


def _manifest_shas(man: dict) -> set:
    return _shas(man.get("ckpt_sha256")) | {str(v["sha256"]) for v in (man.get("checkpoints") or {}).values() if isinstance(v, dict) and v.get("sha256")}


def check_selection_guard(selection_md: Path | None, pin: Path | None, results: Path, access_logs=None, candidates=None, allow_spec_fallback: bool = False) -> dict:
    """Refuse unless: the pre-declared SELECTION.md exists, is pinned (sha256 + git commit) with a matching sha (working file AND the committed blob), a TEST access log is supplied, the candidate list is
    non-empty, every candidate that has a full (no limit_clips) TEST/all-split result among the inputs has a matching log entry, and the COMMIT time of the pinned file (git log -1 --format=%cI <commit> -- SELECTION.md)
    is EARLIER than the earliest ts_utc of a matching candidate entry (released models and limit_clips smoke runs ignored). File mtimes are only a warning.
    MATCHING per scored candidate, in priority order: (a) checkpoint sha256 == the log record's ckpt_sha256; (b) for records without ckpt_sha256: the record's ``out`` joined, by out-dir basename (cross-host paths), to
    a RUN_MANIFEST.json under the results dir, whose checkpoint sha256 is compared; (c) last resort, ONLY when neither the log record nor the result/manifest has a ckpt_sha256: post-filter-stripped spec with checkpoint basenames. Method (c) is accepted only with allow_spec_fallback=True (MOCK runs / unit tests); run() never passes it, so a GUARDED run needs (a) or (b). Same spec with both shas present and different = refusal (checkpoint sha mismatch). -> {candidate: {"method": a|b|c, "ts_utc": ...}}."""
    import hashlib
    import subprocess
    import warnings
    if selection_md is None or not Path(selection_md).exists():
        raise SystemExit("REFUSED: the pre-declared SELECTION.md does not exist")
    if pin is None or not Path(pin).exists():
        raise SystemExit("REFUSED: SELECTION.md is unpinned (no sha record)")
    rec = json.loads(Path(pin).read_text())
    if rec.get("sha256") != sha256_file(selection_md):
        raise SystemExit("REFUSED: SELECTION.md does not match its recorded sha (changed after pinning)")
    if not access_logs or not all(Path(p).exists() for p in access_logs):
        raise SystemExit("REFUSED: no TEST access log supplied (--test-access-log): the selection-before-TEST evidence needs it")
    if not candidates:
        raise SystemExit("REFUSED: empty candidate list: the guard would pass vacuously")
    if not rec.get("commit"):
        raise SystemExit("REFUSED: the pin carries no git commit (re-pin with the committed SELECTION.md)")
    f = Path(selection_md).resolve()
    top = Path(_git(f.parent, "rev-parse", "--show-toplevel"))
    rel = str(f.relative_to(top.resolve()))
    blob = subprocess.run(["git", "-C", str(top), "show", f"{rec['commit']}:{rel}"], capture_output=True)
    if blob.returncode != 0 or hashlib.sha256(blob.stdout).hexdigest() != rec["sha256"]:
        raise SystemExit("REFUSED: the pinned commit does not contain SELECTION.md with the pinned sha256")
    ctime = _ts(_git(top, "log", "-1", "--format=%cI", rec["commit"], "--", rel))
    cands = {_spec_key(m) for m in candidates}
    entries = [e for e in read_access_logs(access_logs) if not e.get("limit_clips")]
    manifests = {}                                                                    # out-dir basename -> checkpoint shas
    for m in sorted(Path(results).rglob("RUN_MANIFEST.json")):
        manifests.setdefault(m.parent.name, set()).update(_manifest_shas(json.loads(m.read_text())))
    # scored candidates: full TEST/all-split results among the inputs
    scored, test_mtimes = {}, []
    for fj in sorted(Path(results).rglob("*.json")):
        try:
            d = json.loads(fj.read_text())
        except (OSError, ValueError):
            continue
        if not (isinstance(d, dict) and (str(d.get("split", "")) in ("test", "all") or d.get("allow_test"))):
            continue
        test_mtimes.append(fj.stat().st_mtime)
        if d.get("limit_clips") or _spec_key(d.get("model", "")) not in cands:
            continue
        sc = scored.setdefault(_spec_key(d["model"]), {"shas": set(), "keys": set()})
        sc["shas"] |= _result_shas(fj, d)
        sc["keys"].add(_spec_key(d["model"]))
    matched, ts = {}, []
    for cand, sc in sorted(scored.items()):
        hit = None
        for e in entries:                                                              # (a) log record ckpt_sha256
            if _shas(e.get("ckpt_sha256")) & sc["shas"]:
                hit = ("a", e)
                break
        if hit is None:                                                                # (b) record without ckpt_sha256: out dir -> RUN_MANIFEST.json (by basename)
            for e in entries:
                if e.get("ckpt_sha256") or not e.get("out"):
                    continue
                o = Path(str(e["out"]).rstrip("/"))
                if any(manifests.get(n, set()) & sc["shas"] for n in (o.name, o.parent.name)):
                    hit = ("b", e)
                    break
        if hit is None:                                                                # (c) ONLY when neither side carries a checkpoint sha; both sides with differing shas is a refusal
            for e in entries:
                if _spec_key(e.get("model", "")) not in sc["keys"]:
                    continue
                esha = _shas(e.get("ckpt_sha256"))
                if not esha and e.get("out"):
                    o = Path(str(e["out"]).rstrip("/"))
                    for n in (o.name, o.parent.name):
                        esha |= manifests.get(n, set())
                if esha and sc["shas"]:
                    raise SystemExit(f"REFUSED: checkpoint sha mismatch for {cand}: result {sorted(sc['shas'])[0][:12]} vs log {sorted(esha)[0][:12]} (same spec, different checkpoint)")
                if not esha and not sc["shas"]:
                    if not allow_spec_fallback:
                        raise SystemExit(f"REFUSED: scored but unlogged: spec mismatch or missing log (candidate {cand!r} matches only by spec + checkpoint basename, method c, which a guarded run does not accept: a ckpt_sha256 or an out-dir RUN_MANIFEST.json is required)")
                    hit = ("c", e)
                    break
        if hit is None:
            raise SystemExit(f"REFUSED: scored but unlogged: spec mismatch or missing log (candidate {cand!r} has a full TEST/all-split result and no matching access-log entry)")
        matched[cand] = {"method": hit[0], "ts_utc": hit[1]["ts_utc"]}
        ts.append(_ts(hit[1]["ts_utc"]))
    ts += [_ts(e["ts_utc"]) for e in entries if _spec_key(e.get("model", "")) in cands]
    if ts and not ctime < min(ts):
        raise SystemExit(f"REFUSED: SELECTION.md commit time {ctime.isoformat()} is not earlier than the earliest candidate TEST access {min(ts).isoformat()}")
    if test_mtimes and Path(selection_md).stat().st_mtime > min(test_mtimes):
        warnings.warn("SELECTION.md mtime is newer than the earliest TEST score file (warning only; the commit time vs the access log decides)")
    return matched


def selection_md(cfg: dict, res: dict, info: dict) -> str:
    """SELECTION.md (result): per row every candidate with wins, mean rank, geometric-mean ratio, the step that decided, a DISCLOSE flag when step 3 (geomean ratio) or 4 (candidate order) decided, then every
    variant considered per row with the axis counts (natural vs r50, A vs B, llvar vs approx ...; candidate 'axes' is free-form)."""
    out = ["# SELECTION (Auditor TABLE1_display_rules 5; same rule for ours and every fine-tuned baseline)", "",
           "Order: (1) most counted non-blank cells won strictly (full precision) vs the released raw rows, (2) lower mean rank, (3) lower geometric mean of per-cell ours/reference ratios vs the FIXED row `" + (next(iter(info.values()))["ref"] if info else "?") + "` = released Metric-VDA-S raw "
           "(lower-better cells ours/ref; delta1 as ref/ours; cells where the reference is blank are dropped and the cells used are listed), (4) the pre-declared candidate order (the order the candidates are listed in the committed SELECTION.md). If step 3 or 4 decides, it is disclosed.", "",
           "| row | kind | candidates | chosen | wins | mean rank | geomean ratio (cells) | decided by | disclose |", "|---|---|---|---|---|---|---|---|---|"]
    for rid, i in info.items():
        r = i["row"]
        if r.get("table", "main") != "main":
            continue
        if i["candidates"]:
            c = i["chosen"]
            out.append(f"| {rid} | {r['kind']} | {len(i['candidates'])} | {c['cand']['label']} | {c['wins']} | {c['mean_rank']:.4f} | {c['gmean']:.4f} ({c['n_ratio']}) | {i['decided_by']} | {'DISCLOSE' if i['disclose'] else ''} |")
        else:
            out.append(f"| {rid} | {r['kind']} | {len(r['candidates'])} | {r['candidates'][0]['label']} | - | - | - | {i['decided_by']} | |")
    for rid, i in info.items():
        if i["candidates"]:
            out += ["", f"## {rid} ({len(i['candidates'])} candidates; decided by {i['decided_by']})", "", "| candidate | wins | mean rank | geomean ratio | cells used |", "|---|---|---|---|---|"]
            out += [f"| {d['cand']['label']}{' **(chosen)**' if d is i['chosen'] else ''} | {d['wins']} | {d['mean_rank']:.4f} | {d['gmean']:.4f} | {d['n_ratio']} |" for d in i["candidates"]]
    out += ["", "# VARIANTS CONSIDERED PER ROW", "", "Every candidate listed (axes are the row's declared variant axes); the count per axis value is the number of candidates considered, whether or not chosen.", ""]
    for r in cfg["rows"]:
        if r.get("table", "main") != "main":
            continue
        axes = {}
        for c in r["candidates"]:
            for k, v in (c.get("axes") or {}).items():
                axes.setdefault(k, {}).setdefault(str(v), 0)
                axes[k][str(v)] += 1
        out += [f"## {r['id']}: {len(r['candidates'])} candidate(s)", ""]
        if axes:
            out += ["axes: " + "; ".join(f"{k} = " + ", ".join(f"{v} x{n}" for v, n in sorted(vs.items())) for k, vs in sorted(axes.items())), ""]
        out += ["| candidate | model | axes |", "|---|---|---|"] + [f"| {c['label']} | {c['model']} | {json.dumps(c.get('axes') or {}, sort_keys=True)} |" for c in r["candidates"]] + [""]
    return "\n".join(out) + "\n"


def latency_tables(paths: list) -> tuple[str, str]:
    """From runs/WP24/latency_v3*.json: one row per timed model; x VDA-S as the file records it (a file without a reference reports '--')."""
    body = []
    for p in sorted(paths):
        d = json.loads(Path(p).read_text())
        for m in d.get("models", []):
            x = m.get("x_vda_s")
            body.append([m["model"], f"{m['per_frame_ms_p50']:.2f}", f"{m['per_frame_ms_p95']:.2f}", "--" if x is None else f"{x:.2f}", Path(p).name])
    header = ["Model", "p50 ms/frame", "p95 ms/frame", "x VDA-S", "source file"]
    return md_to_tex_table(header, body), "## Latency (fp16, 518x518, 32-frame window)\n\n" + md_table(header, body)


def parity_tables(measured: list) -> tuple[str, str]:
    """measured = [{set, model, metric, value | (raw, scale), variant, source}]. TAE is converted to VDA's units first: value = raw * scale (the harness-to-VDA factor, e.g. 0.01) and the factor is shown.
    Tolerance 0.005 (AbsRel, delta1), 0.02 (TAE, absolute on VDA's scale); a miss is a footnote (NOTE 'taepar'), never a block. TAE also checks VDA-L < VDA-S."""
    body, tae = [], {}
    for m in measured:
        ref = VDA_REFS.get((m["set"], m["model"], m["metric"]))
        refv, src = ref if ref else (None, "no VDA reference recorded")
        val = m["raw"] * m["scale"] if "raw" in m and "scale" in m else m["value"]
        d = None if refv is None else abs(val - refv)
        tol = PARITY_TOL.get(m["metric"])
        verdict = "n/a" if d is None or tol is None else ("pass" if d <= tol + 1e-12 else "MISS")
        if m["metric"] == "tae":
            tae[m["model"]] = val
        body.append([m["set"], m["model"], m["metric"], m.get("variant", ""), f"{val:.3f}", "--" if m.get("scale") is None else f"x{m['scale']:g}", "--" if refv is None else f"{refv:.3f}",
                     "--" if d is None else f"{d:.3f}", "--" if tol is None else f"{tol:g}", verdict, src, m.get("source", "")])
    header = ["Set", "Model", "Metric", "Variant", "Ours (VDA units)", "Harness-to-VDA scale", "VDA paper", "|d|", "Tol", "Parity", "Paper table / row", "Measured from"]
    out = md_table(header, body)
    if "vda_s" in tae and "vda_l" in tae:
        ok = tae["vda_l"] < tae["vda_s"]
        out += f"\nTAE ordering check, VDA-L < VDA-S: {'pass' if ok else 'MISS'} (VDA-L {tae['vda_l']:.3f}, VDA-S {tae['vda_s']:.3f}).\n"
    miss = [b for b in body if b[9] == "MISS" and b[2] == "tae"]
    if miss or ("vda_s" in tae and "vda_l" in tae and not tae["vda_l"] < tae["vda_s"]):
        out += "\nFOOTNOTE (not a block): " + NOTE_TEXT["taepar"].replace("{X}", ", ".join(f"{b[1]} {b[4]}" for b in miss) or "X").replace("{Y}", ", ".join(f"{b[6]}" for b in miss) or "Y") + "\n"
    return md_to_tex_table(header[:10], [b[:10] for b in body]), "## Parity vs the VDA paper (clean data)\n\n" + out


def mass_tables(mass_md: Path) -> tuple[str, str]:
    """The markdown tables of runs/WP27_prep/MASS_TABLE.md (the pipe tables only), converted; the prose is not copied."""
    tabs, cur = [], []
    for ln in mass_md.read_text().splitlines():
        if ln.startswith("|"):
            cur.append(ln)
        elif cur:
            tabs.append(cur)
            cur = []
    if cur:
        tabs.append(cur)
    tex, md = [], []
    for t in tabs:
        cells = [[x.strip() for x in ln.strip().strip("|").split("|")] for ln in t if not re.fullmatch(r"\|[-| ]+\|", ln.strip())]
        tex.append(md_to_tex_table(cells[0], cells[1:]))
        md.append("\n".join(t) + "\n")
    return "\n".join(tex), "## Data mass (natural vs r50)\n\n" + "\n".join(md)


# ------------------------------------------------------------------------------------------------ driver
def run(user_cfg: dict, results: Path, out: Path, scannet_column: str = "scannet", latency=None, parity=None, mass: Path | None = None, ladder_levels=(1, 2, 4, 8, 16),
        selection_md_in: Path | None = None, selection_pin: Path | None = None, mock_mode: bool = False, guard: bool | None = None, parity_path: Path | None = None, access_logs=None) -> dict:
    if guard is None:
        guard = not mock_mode                                                      # the SELECTION.md guard applies unless this is an all-MOCK run
    match_info = {}
    if mock_mode:
        check_all_mock(results)
    if guard:
        check_no_mock_inputs(results, latency, parity)
        cands = [c["model"] for r in user_cfg["rows"] if r.get("table", "main") == "main" and r["kind"] not in ("released_raw", "reference") for c in r["candidates"]]
        match_info = check_selection_guard(selection_md_in, selection_pin, results, access_logs, cands)
    out.mkdir(parents=True, exist_ok=True)

    def wr(name: str, text: str) -> None:
        """Write one output. MOCK runs (--no-selection-guard) are watermarked: MOCK_ file name + MOCK first line in every CSV / TeX / MD."""
        if mock_mode:
            ext = Path(name).suffix
            line = {".tex": "% MOCK: random numbers, not results\n", ".md": "MOCK: random numbers, not results\n\n", ".csv": "MOCK\n"}.get(ext, "")
            name, text = "MOCK_" + name, line + text
        (out / name).write_text(text)

    cfg = build_cfg(user_cfg, scannet_column)
    ht.check_tag_hygiene(cfg)
    res = ht.load_results(results)
    inputs = {"results": {str(Path(f).relative_to(results)): sha256_file(f) for f in sorted(Path(results).rglob("*.json"))},
              "latency": {str(f): sha256_file(f) for f in sorted(latency or [])}, "parity": ({str(parity_path): sha256_file(parity_path)} if parity_path else {}),
              "selection_md": ({str(selection_md_in): sha256_file(selection_md_in)} if selection_md_in and Path(selection_md_in).exists() else {}), "test_access_logs": {str(f): sha256_file(f) for f in sorted(access_logs or []) if Path(f).exists()}, "selection_log_match": match_info, "mock_mode": bool(mock_mode)}
    wr("INPUTS.json", json.dumps(inputs, indent=1, sort_keys=True) + "\n")
    chosen, info = select_v3(cfg, res, read_step3_ref(selection_md_in, cfg))
    ev = ht.evaluate(cfg, res, chosen)
    if parity:
        pmiss = [m for m in parity if m["metric"] == "tae" and "vda" in m["model"]]
        _, pmd = parity_tables(parity)
        if "FOOTNOTE (not a block)" in pmd:
            cfg["tae_parity_miss"] = True
            cfg["taepar_xy"] = (", ".join(f"{m['model']} {(m['raw'] * m['scale'] if 'raw' in m and 'scale' in m else m['value']):.3f}" for m in pmiss), ", ".join(f"{VDA_REFS[(m['set'], m['model'], 'tae')][0]:.3f}" for m in pmiss if (m["set"], m["model"], "tae") in VDA_REFS))
    tex, md = render_table1(cfg, ev, chosen)
    dis = disagreements(ev, {r["id"]: r for r in cfg["rows"] if r.get("table", "main") == "main"})
    wr("display_vs_win.md", disagreement_md(dis))
    wr("display_vs_win.json", json.dumps(dis, indent=1, sort_keys=True) + "\n")
    wr("table1.tex", tex)
    wr("table1.md", md)
    wr("SELECTION.md", selection_md(cfg, res, info))
    wr("wins.json", json.dumps({"ours": ev["ours"], "ours_wins": ev["ours_wins"], "n_counted_nonblank": ev["n_counted_nonblank"], "table_win": ev["table_win"],
                                               "cells": {c["id"]: {"counted": c["counted"], "blank": c["blank"], "winner": c["winner"], "status": c["status"]} for c in ev["cells"]}}, indent=1, sort_keys=True) + "\n")
    # llvarf supplement: same rows / selected checkpoints, cell name llvarf, never selects
    cfg_f = build_cfg(user_cfg, scannet_column, headline_cell="llvarf", static_cell="st_llvarf")
    ev_f = ht.evaluate(cfg_f, res, chosen)
    tf, mf = render_table1(dict(cfg_f, title=("MOCK " if user_cfg.get("MOCK") else "") + "Supplement: per-frame degradation (llvarf)"), ev_f, chosen)
    wr("supp_llvarf.tex", tf)
    wr("supp_llvarf.md", mf)
    lt, lm = ladder_tables(cfg, res, chosen, levels=ladder_levels)
    wr("supp_noise_ladder.tex", lt)
    wr("supp_noise_ladder.md", lm)
    at, am = appendix_tables(cfg, res, chosen)
    if at:
        wr("supp_tag_appendix.tex", at)
        wr("supp_tag_appendix.md", am)
    if latency:
        lt2, lm2 = latency_tables(latency)
        wr("supp_latency.tex", lt2)
        wr("supp_latency.md", lm2)
    if parity:
        pt, pm = parity_tables(parity)
        wr("supp_parity.tex", pt)
        wr("supp_parity.md", pm)
    if mass:
        mt, mm = mass_tables(mass)
        wr("supp_mass.tex", mt)
        wr("supp_mass.md", mm)
    return ev


# ------------------------------------------------------------------------------------------------ MOCK inputs
def mock_rows() -> list:
    def r(rid, label, kind, model, output="metric", **kw):
        return {"id": rid, "label": label, "kind": kind, "output": output, "candidates": [{"label": model, "model": model}], **kw}
    rows = [r("vda_s", "VDA-S", "released_raw", "vda_s", "relative"), r("vda_l", "VDA-L", "released_raw", "vda_l", "relative"), r("mvda_s", "Metric-VDA-S", "released_raw", "metric_vda_s"),
            r("mvda_l", "Metric-VDA-L", "released_raw", "metric_vda_l"), r("dav2_s", "DA-V2-S", "released_raw", "dav2_s"), r("daac", "DA-AC", "released_raw", "daac", notes=["irs"]),
            r("casc1", "FastDVDnet $\\to$ MVDA-S", "cascade", "fastdvdnet_blind->metric_vda_s"), r("casc2", "PocketDVDnet $\\to$ MVDA-S", "cascade", "pocketdvdnet->metric_vda_s"),
            r("depthcrafter", "DepthCrafter", "reference", "depthcrafter", "relative")]
    for rid, lab in (("ft_mvda", "FT Metric-VDA-S"), ("ft_dav2", "FT DA-V2-S"), ("ft_daac", "FT DA-AC")):
        cands = [{"label": f"{rid} {mix} step{st}", "model": f"{rid}_{mix}_{st}", "axes": {"mix": mix, "step": st}} for mix in ("natural", "r50") for st in (6000, "final")]
        rows.append({"id": rid, "label": lab, "kind": "ft_baseline", "output": "metric", "candidates": cands})
    cands = [{"label": f"ours {fr} {data} {mix} {st}", "model": f"ours_{fr}_{data}_{mix}_{st}+{TAG}", "axes": {"framing": fr, "data": data, "mix": mix, "step": st, "degr": "llvar"}}
             for fr in ("A", "B") for data, mix in (("SYN", "natural"), ("SYN+REAL", "natural"), ("SYN+REAL", "r50")) for st in (6000, "final")]
    rows.append({"id": "ours", "label": "Ours (+TAG)", "kind": "ours", "output": "metric", "candidates": cands})
    rows.append({"id": "mvda_s_tag", "label": "Metric-VDA-S + TAG", "kind": "released_raw", "output": "metric", "table": "appendix", "candidates": [{"label": "mvda_s+tag", "model": f"metric_vda_s+{TAG}"}]})
    rows.append({"id": "ours_tag", "label": "Ours + TAG", "kind": "ours", "output": "metric", "table": "appendix", "candidates": [{"label": "ours+tag", "model": f"ours_B_SYN+REAL_r50_final+{TAG}"}]})
    rows.append({"id": "ours_notag", "label": "Ours (no TAG)", "kind": "ours", "output": "metric", "table": "appendix", "candidates": [{"label": "ours", "model": "ours_B_SYN+REAL_r50_final"}]})
    return rows


def make_mock(dest: Path, seed: int = 20261004, scannet_column: str = "scannet") -> None:
    """Random numbers only, never derived from any real file. Every file carries MOCK: true."""
    rng = random.Random(seed)
    user = {"rows": mock_rows(), "tag_token": TAG, "MOCK": True}
    cfg = build_cfg(user, scannet_column)
    rd = dest / "results"
    rd.mkdir(parents=True, exist_ok=True)
    (dest / "config.json").write_text(json.dumps(user, indent=1) + "\n")
    models = sorted({c["model"] for r in user["rows"] for c in r["candidates"]})
    keys = {}
    for c in cfg["cells"]:
        keys.setdefault((c["source"], c["dataset"], c["cell"]), set()).add(c["field"])
    for c in cfg_cells_twin(cfg, "llvar", "llvarf"):                                    # llvarf twin of every headline cell
        keys.setdefault((c["source"], c["dataset"], c["cell"]), set()).add(c["field"])
    for ds in sorted({c["dataset"] for c in cfg["cells"] if c["mgroup"] == "abs_rel_aligned" and c["dataset"] != "nuscenes_night_test"}):
        for k in (1, 2, 4, 8, 16):
            keys.setdefault(("bench", ds, f"llb3x{k}"), set()).add("video_macro.abs_rel_aligned")
    i = 0
    for (src, ds, cn), fields in sorted(keys.items()):
        for m in models:
            vm = {}
            for f in sorted(fields):
                leaf = f.split(".")[-1]
                base = {"abs_rel_unaligned": 0.25, "abs_rel_aligned": 0.12, "delta1_aligned": 0.86, "tae": 0.9, "flicker": 0.05, "f_noise": 0.04}[leaf]
                v = base * (0.7 + 0.6 * rng.random())
                if leaf == "delta1_aligned":
                    v = 0.6 + 0.38 * rng.random()
                vm[leaf] = round(v, 6) if leaf != "f_noise" else v
            d = {"bench": "dn01_bvi_fnoise" if src == "bvi" else "bench_joint", "MOCK": True, "NOTE": "MOCK random numbers; not a real result", "model": m, "dataset": ds, "cell": cn, "split": "mock"}
            d["macro" if src == "bvi" else "video_macro"] = vm
            (rd / f"mock_{i:05d}.json").write_text(json.dumps(d, sort_keys=True))
            i += 1
    parity = []
    for (s_, mo, me), (refv, _) in sorted(VDA_REFS.items()):
        v = round(refv + rng.uniform(-0.009, 0.009), 4)
        parity.append({"set": s_, "model": mo, "metric": me, "variant": "rgbfix", "source": "MOCK", "MOCK": True, **({"raw": round(v * 100, 3), "scale": 0.01} if me == "tae" else {"value": v})})
    (dest / "parity.json").write_text(json.dumps(parity, indent=1) + "\n")


def cfg_cells_twin(cfg: dict, a: str, b: str) -> list:
    out = []
    for c in cfg["cells"]:
        c2 = dict(c)
        if c["cell"] == a:
            c2["cell"] = b
        elif c["cell"] == "st_" + a:
            c2["cell"] = "st_" + b
        out.append(c2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results")
    ap.add_argument("--config")
    ap.add_argument("--out")
    ap.add_argument("--scannet-column", default="scannet", choices=["scannet", "scannetpp"])
    ap.add_argument("--latency", help="glob, e.g. runs/WP24/latency_v3*.json")
    ap.add_argument("--parity", help="JSON list of measured parity rows")
    ap.add_argument("--mass")
    ap.add_argument("--make-mock")
    ap.add_argument("--selection-md", help="the PRE-DECLARED SELECTION.md (candidate order), committed before any TEST score")
    ap.add_argument("--test-access-log", nargs="+", help="one or more TEST_access_log.jsonl files (merged); REQUIRED unless --no-selection-guard")
    ap.add_argument("--selection-pin", help="JSON {sha256} of that file; write it with --pin-selection")
    ap.add_argument("--pin-selection", help="write the pin for --selection-md to this path and exit")
    ap.add_argument("--no-selection-guard", action="store_true", help="MOCK runs only")
    a = ap.parse_args()
    if a.pin_selection:
        pin_selection(Path(a.selection_md), Path(a.pin_selection))
        print("pinned", a.selection_md)
        return
    if a.make_mock:
        make_mock(Path(a.make_mock), scannet_column=a.scannet_column)
        print("wrote MOCK inputs to", a.make_mock)
        return
    ev = run(json.loads(Path(a.config).read_text()), Path(a.results), Path(a.out), a.scannet_column, sorted(glob.glob(a.latency)) if a.latency else None,
             json.loads(Path(a.parity).read_text()) if a.parity else None, Path(a.mass) if a.mass else None,
             parity_path=Path(a.parity) if a.parity else None,
             selection_md_in=Path(a.selection_md) if a.selection_md else None, selection_pin=Path(a.selection_pin) if a.selection_pin else None, mock_mode=a.no_selection_guard, access_logs=a.test_access_log)
    print(f"ours best in {ev['ours_wins']} of {ev['n_counted_nonblank']} counted non-blank cells: {'WON' if ev['table_win'] else 'NOT WON'}")


if __name__ == "__main__":
    main()
