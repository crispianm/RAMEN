"""Protocol v3 WP10: the headline-table aggregator (scripts/dn01_headline_table.py) on SYNTHETIC fixtures only (no real v3 score exists or is opened): strict-best / ties, blanks, relative rows, the reference row, non-counted cells,
the more-than-half rule, TAG hygiene, selection, byte-identical re-runs, config-driven cell names."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import dn01_headline_table as ht  # noqa: E402

TAG = "agf0.2t0.06"
OURS = f"ours:/c/final.pt+{TAG}"


def put(d: Path, model, dataset, cell, **fields):
    """A bench_joint-schema result file: video_macro.<field>."""
    n = len(list(d.glob("*.json")))
    (d / f"r{n:04d}.json").write_text(json.dumps({"bench": "bench_joint", "model": model, "dataset": dataset, "cell": cell, "split": "dev", "llvar_draws_sha256": "ab", "video_macro": fields}))


def cell(cid, field, direction="min", rel=True, counted=True, ds=None, cn="llvar"):
    ds = ds or cid                                                                    # a distinct (dataset, cell) per fixture cell unless given: each is its own result file
    return {"id": cid, "label": cid, "group": ds, "source": "bench", "dataset": ds, "cell": cn, "field": f"video_macro.{field}", "direction": direction, "relative_ok": rel, "counted": counted}


def row(rid, kind, model, output="metric", **kw):
    return {"id": rid, "label": rid, "kind": kind, "output": output, "candidates": [{"label": model, "model": model}], **kw}


def cfg(cells, rows, **kw):
    return {"tag_token": TAG, "cells": cells, "rows": rows, "ratio_ref": None, **kw}


def fixture(tmp_path, table: dict, cells, rows, **kw):
    """table = {model: {cell_id: value}}; one file per (model, cell) with the cell's dataset / cell name / field."""
    res = tmp_path / "res"
    res.mkdir(exist_ok=True)
    by_id = {c["id"]: c for c in cells}
    for model, vals in table.items():                                       # ONE file per (model, dataset, cell) holding all its metrics, as the scorers write them
        files = {}
        for cid, v in vals.items():
            c = by_id[cid]
            files.setdefault((c["dataset"], c["cell"]), {})[c["field"].split(".")[1]] = v
        for (ds, cn), fields in sorted(files.items()):
            put(res, model, ds, cn, **fields)
    return cfg(cells, rows, **kw), res


def go(tmp_path, table, cells, rows, **kw):
    c, res = fixture(tmp_path, table, cells, rows, **kw)
    return ht.run(c, res, tmp_path / "out"), tmp_path / "out"


R2 = [row("ours", "ours", OURS), row("mvda", "released_raw", "metric_vda_s")]
C3 = [cell(f"c{i}", "abs_rel_aligned") for i in range(3)]


def test_ours_wins_when_strictly_best_in_more_than_half(tmp_path):
    ev, out = go(tmp_path, {OURS: {"c0": 0.10, "c1": 0.10, "c2": 0.30}, "metric_vda_s": {"c0": 0.20, "c1": 0.20, "c2": 0.20}}, C3, R2)
    assert (ev["ours_wins"], ev["n_counted_nonblank"], ev["table_win"]) == (2, 3, True)
    assert "winner=ours" in (out / "wins.txt").read_text() and "TABLE WON" in (out / "wins.txt").read_text()


@pytest.mark.parametrize("wins,n,expect", [(2, 4, False), (3, 4, True), (1, 2, False), (2, 3, True), (1, 3, False), (0, 1, False), (1, 1, True)])
def test_majority_edge_cases_exactly_half_is_not_a_win(tmp_path, wins, n, expect):
    cells = [cell(f"c{i}", "abs_rel_aligned") for i in range(n)]
    tbl = {OURS: {f"c{i}": (0.1 if i < wins else 0.9) for i in range(n)}, "metric_vda_s": {f"c{i}": 0.5 for i in range(n)}}
    ev, _ = go(tmp_path, tbl, cells, R2)
    assert (ev["ours_wins"], ev["n_counted_nonblank"], ev["table_win"]) == (wins, n, expect)


def test_ties_are_not_wins_and_full_precision_breaks_them(tmp_path):
    cells = [cell("tie", "abs_rel_aligned"), cell("eps", "abs_rel_aligned")]
    tbl = {OURS: {"tie": 0.1234567, "eps": 0.12345670000001}, "metric_vda_s": {"tie": 0.1234567, "eps": 0.12345671}}
    ev, _ = go(tmp_path, tbl, cells, R2)
    c = {x["id"]: x for x in ev["cells"]}
    assert c["tie"]["winner"] is None and c["tie"]["status"].startswith("tie") and c["eps"]["winner"] == "ours"            # a 1e-8 gap at full precision IS strictly better
    assert ev["ours_wins"] == 1 and ev["n_counted_nonblank"] == 2 and ev["table_win"] is False                              # 1 of 2: not more than half


def test_a_three_way_tie_and_a_tie_that_excludes_ours(tmp_path):
    rows = R2 + [row("b3", "ft_baseline", "x_ft:/c/b.pt")]
    cells = [cell("c0", "abs_rel_aligned")]
    ev, _ = go(tmp_path, {OURS: {"c0": 0.2}, "metric_vda_s": {"c0": 0.1}, "x_ft:/c/b.pt": {"c0": 0.1}}, cells, rows)
    assert ev["cells"][0]["winner"] is None and ev["ours_wins"] == 0 and ev["n_counted_nonblank"] == 1                        # the best is shared by two baselines: nobody wins, ours does not either


def test_blank_cells_are_not_counted_but_an_ours_missing_cell_is_a_loss(tmp_path):
    cells = [cell("has", "abs_rel_aligned"), cell("blank_all", "tae"), cell("ours_missing", "abs_rel_aligned", ds="bonn")]
    tbl = {OURS: {"has": 0.1}, "metric_vda_s": {"has": 0.2, "ours_missing": 0.2}}
    ev, out = go(tmp_path, tbl, cells, R2)
    c = {x["id"]: x for x in ev["cells"]}
    assert c["blank_all"]["blank"] and c["ours_missing"]["blank"] is False and c["ours_missing"]["ours_missing"] and c["ours_missing"]["winner"] == "mvda"
    assert (ev["ours_wins"], ev["n_counted_nonblank"], ev["table_win"]) == (1, 2, False)                                    # NYU-TAE-style blank dropped; the missing-ours cell stays and is lost
    assert "[ours missing]" in (out / "wins.txt").read_text()


def test_relative_raw_rows_compete_in_aligned_cells_only(tmp_path):
    cells = [cell("metric_abs", "abs_rel_unaligned", rel=False), cell("aligned_abs", "abs_rel_aligned", rel=True)]
    rows = R2 + [row("vda_rel", "released_raw", "vda_s", output="relative")]
    tbl = {OURS: {"metric_abs": 0.30, "aligned_abs": 0.30}, "metric_vda_s": {"metric_abs": 0.50, "aligned_abs": 0.50}, "vda_s": {"metric_abs": 0.01, "aligned_abs": 0.01}}
    ev, _ = go(tmp_path, tbl, cells, rows)
    c = {x["id"]: x for x in ev["cells"]}
    assert c["metric_abs"]["winner"] == "ours" and c["metric_abs"]["values"]["vda_rel"] is None                              # the relative row is blank in the metric cell, so it cannot beat ours there
    assert c["aligned_abs"]["winner"] == "vda_rel"                                                                           # ...but it competes (and wins) in the aligned one


def test_reference_row_and_non_counted_cells_never_count(tmp_path):
    cells = [cell("c0", "abs_rel_aligned"), cell("kitti_minus_0047", "abs_rel_aligned", counted=False)]
    rows = R2 + [row("depthcrafter", "reference", "depthcrafter")]
    tbl = {OURS: {"c0": 0.2, "kitti_minus_0047": 0.9}, "metric_vda_s": {"c0": 0.3, "kitti_minus_0047": 0.1}, "depthcrafter": {"c0": 0.01, "kitti_minus_0047": 0.01}}
    ev, out = go(tmp_path, tbl, cells, rows)
    c = {x["id"]: x for x in ev["cells"]}
    assert c["c0"]["winner"] == "ours" and c["c0"]["values"]["depthcrafter"] == 0.01                                          # shown, never competes (a better reference value does not take the win)
    assert (ev["ours_wins"], ev["n_counted_nonblank"], ev["table_win"]) == (1, 1, True)                                      # the disclosure cell is not counted either way
    assert "[not counted]" in (out / "wins.txt").read_text() and "0.01" in (out / "table1.csv").read_text()


def test_direction_higher_is_better_for_delta1(tmp_path):
    cells = [cell("d1", "delta1_aligned", direction="max")]
    ev, _ = go(tmp_path, {OURS: {"d1": 0.9}, "metric_vda_s": {"d1": 0.8}}, cells, R2)
    assert ev["cells"][0]["winner"] == "ours"


def test_tag_on_a_baseline_fails_and_ours_must_carry_exactly_the_tag(tmp_path):
    cells = [cell("c0", "abs_rel_aligned")]
    for i, bad in enumerate(("metric_vda_s+agf0.2t0.06", "metric_vda_s+ema0.7", "x_ft:/c/f.pt+agf0.3t0.06")):
        d = tmp_path / f"case{i}"
        d.mkdir()
        with pytest.raises(AssertionError, match="TAG appears only on ours"):
            go(d, {OURS: {"c0": 0.1}}, cells, [row("ours", "ours", OURS), row("b", "released_raw", bad)])
    with pytest.raises(AssertionError, match="exactly the TAG token"):
        ht.check_tag_hygiene(cfg(cells, [row("ours", "ours", "ours:/c/final.pt")]))                                              # ours without TAG
    with pytest.raises(AssertionError, match="exactly the TAG token"):
        ht.check_tag_hygiene(cfg(cells, [row("ours", "ours", "ours:/c/final.pt+ema0.7")]))                                       # ours with a different filter
    ht.check_tag_hygiene(cfg(cells, [row("ours", "ours", OURS), row("a", "released_raw", "metric_vda_s", table="appendix"), row("a2", "released_raw", f"metric_vda_s+{TAG}", table="appendix")]))   # appendix rows may carry TAG


def test_a_result_file_with_a_post_filter_is_visible_to_the_token_check():
    assert ht.tokens_of("metric_vda_s", {"depth_post": ["agf0.2t0.06"]}) == ["agf0.2t0.06"] and ht.tokens_of("ours:/p/a.pt", None) == [] and ht.tokens_of("dav2_s", None) == []


def test_selection_is_the_same_rule_for_ours_and_a_baseline_and_counts_are_written(tmp_path):
    cells = [cell(f"c{i}", "abs_rel_aligned") for i in range(4)]
    cand = lambda *ms: [{"label": f"cand{i}", "model": m} for i, m in enumerate(ms)]
    ours_c, base_c = [f"ours:/c/a{i}.pt+{TAG}" for i in range(3)], [f"x_ft:/c/b{i}.pt" for i in range(2)]
    rows = [{"id": "ours", "label": "ours", "kind": "ours", "output": "metric", "candidates": cand(*ours_c)},
            {"id": "base", "label": "base", "kind": "ft_baseline", "output": "metric", "candidates": cand(*base_c)}, row("mvda", "released_raw", "metric_vda_s")]
    # released = 0.5 everywhere. ours a0: wins 1; a1: wins 3; a2: wins 3 but worse mean rank is impossible (same pool) -> tie on wins & rank -> the earlier (a1). base b0: wins 0, b1: wins 2.
    tbl = {"metric_vda_s": {f"c{i}": 0.5 for i in range(4)}, ours_c[0]: {"c0": 0.1, "c1": 0.9, "c2": 0.9, "c3": 0.9}, ours_c[1]: {"c0": 0.1, "c1": 0.1, "c2": 0.1, "c3": 0.9}, ours_c[2]: {"c0": 0.2, "c1": 0.2, "c2": 0.2, "c3": 0.9},
           base_c[0]: {f"c{i}": 0.9 for i in range(4)}, base_c[1]: {"c0": 0.1, "c1": 0.1, "c2": 0.9, "c3": 0.9}}
    ev, out = go(tmp_path, tbl, cells, rows)
    sel = (out / "SELECTION.md").read_text()
    assert "| ours | ours | 3 | cand1 | 3 |" in sel and "| base | ft_baseline | 2 | cand1 | 2 |" in sel and "| mvda | released_raw | 1 |" in sel      # candidate counts per row + chosen + wins
    assert "(chosen)" in sel and sel.count("## ") == 2
    # the chosen models are the ones in the table
    csv = (out / "table1.csv").read_text().splitlines()
    assert csv[1].startswith("ours,0.1,0.1,0.1,0.9")                                                                                   # cand1's values


def test_selection_ties_on_wins_fall_to_mean_rank_then_listing_order(tmp_path):
    cells = [cell(f"c{i}", "abs_rel_aligned") for i in range(2)]
    c0, c1 = f"ours:/c/a0.pt+{TAG}", f"ours:/c/a1.pt+{TAG}"
    rows = [{"id": "ours", "label": "ours", "kind": "ours", "output": "metric", "candidates": [{"label": "a0", "model": c0}, {"label": "a1", "model": c1}]}, row("r1", "released_raw", "m1"), row("r2", "released_raw", "m2")]
    # both candidates win 1 cell vs {m1: .5, m2: .6}; a0 loses c1 by being 3rd (rank 3), a1 loses c1 by being 2nd (rank 2): a1 has the better mean rank
    tbl = {"m1": {"c0": 0.5, "c1": 0.5}, "m2": {"c0": 0.6, "c1": 0.6}, c0: {"c0": 0.1, "c1": 0.9}, c1: {"c0": 0.1, "c1": 0.55}}
    _, out = go(tmp_path, tbl, cells, rows)
    assert "| ours | ours | 2 | a1 | 1 | 1.5000 |" in (out / "SELECTION.md").read_text()


def test_outputs_are_byte_identical_on_rerun_and_carry_the_win_line(tmp_path):
    tbl = {OURS: {"c0": 0.10, "c1": 0.10, "c2": 0.30}, "metric_vda_s": {"c0": 0.20, "c1": 0.20, "c2": 0.20}}
    c, res = fixture(tmp_path, tbl, C3, R2, ratio_ref="mvda")
    a, b = tmp_path / "o1", tmp_path / "o2"
    ht.run(c, res, a)
    ht.run(c, res, b)
    names = sorted(p.name for p in a.iterdir())
    assert names == ["SELECTION.md", "cells.csv", "ratio.csv", "table1.csv", "table1.html", "table1.tex", "wins.json", "wins.txt"] == sorted(p.name for p in b.iterdir())
    assert all((a / n).read_bytes() == (b / n).read_bytes() for n in names)
    tex, htm = (a / "table1.tex").read_text(), (a / "table1.html").read_text()
    assert "\\textbf{0.1000}" in tex and "<b>0.1000</b>" in htm and "ours wins 2/3" in tex and "ours best in 2 of 3" in htm
    assert "c0,0.1,0.2,0.5" in (a / "ratio.csv").read_text()


def test_cell_names_come_from_the_config_not_the_code(tmp_path):
    cells = ht.make_default_config("llvarZ", "st_llvarZ")["cells"]
    assert {c["cell"] for c in cells if c["source"] == "bench" and c["dataset"] in ht.DEFAULT_DATASETS} == {"llvarZ", "st_llvarZ"}
    one = [c for c in cells if c["id"] == "kitti|llvarZ|abs_rel_aligned"]
    ev, _ = go(tmp_path, {OURS: {"kitti|llvarZ|abs_rel_aligned": 0.1}, "metric_vda_s": {"kitti|llvarZ|abs_rel_aligned": 0.2}}, one, R2)
    assert ev["table_win"] is True


def test_default_cell_list_is_the_protocol_s2_list():
    cs = ht.make_default_config("llvar", "st_llvar")["cells"]
    counted = [c for c in cs if c["counted"]]
    per = {}
    for c in counted:
        per.setdefault(c["group"], []).append(c["field"])
    assert len(per["KITTI"]) == len(per["Bonn"]) == len(per["ScanNet*"]) == 5 and len(per["Sintel"]) == 4 and len(per["NYUv2"]) == 4          # 4 metrics + 1 static flicker (Sintel: no static clips; NYUv2: 3 + 1, TAE blank)
    assert not any(c["field"].endswith("tae") for c in counted if c["group"] == "NYUv2")
    assert len(per["nuScenes-night"]) == 4 and len(per["BVI-RLV"]) == 1 and len(counted) == 19 + 4 + 1 + 4 == 28
    assert [c["id"] for c in cs if not c["counted"]] == ["kitti_minus_0047|llvar|abs_rel_aligned"]
    rel = {c["field"].split(".")[1] for c in counted if c["relative_ok"]}
    assert rel == {"abs_rel_aligned", "delta1_aligned", "tae"}                                                                  # relative rows: aligned / scale-invariant cells only


def test_bvi_results_are_read_through_their_own_schema(tmp_path):
    res = tmp_path / "res"
    res.mkdir()
    (res / "a.json").write_text(json.dumps({"bench": "dn01_bvi_fnoise", "model": OURS, "macro": {"f_noise": 0.1}}))
    (res / "b.json").write_text(json.dumps({"bench": "dn01_bvi_fnoise", "model": "metric_vda_s", "macro": {"f_noise": 0.3}}))
    bvi = {"id": "bvi", "label": "BVI", "group": "BVI-RLV", "source": "bvi", "dataset": "bvi_rlv", "cell": "fnoise", "field": "macro.f_noise", "direction": "min", "relative_ok": False, "counted": True}
    ev = ht.run(cfg([bvi], R2), res, tmp_path / "out")
    assert ev["cells"][0]["winner"] == "ours" and ev["table_win"]


def test_two_result_files_for_one_cell_are_an_error(tmp_path):
    res = tmp_path / "res"
    res.mkdir()
    put(res, "metric_vda_s", "kitti", "llvar", abs_rel_aligned=0.1)
    put(res, "metric_vda_s", "kitti", "llvar", abs_rel_aligned=0.2)
    with pytest.raises(ValueError, match="two result files"):
        ht.load_results(res)


def test_the_test_access_log_is_append_only_and_records_test_files(tmp_path):
    res = tmp_path / "res"
    res.mkdir()
    (res / "t.json").write_text(json.dumps({"bench": "bench_joint", "model": OURS, "dataset": "kitti", "cell": "llvar", "split": "test", "llvar_draws_sha256": "ab", "video_macro": {"abs_rel_aligned": 0.1}}))
    (res / "d.json").write_text(json.dumps({"bench": "bench_joint", "model": "metric_vda_s", "dataset": "kitti", "cell": "llvar", "split": "dev", "llvar_draws_sha256": "ab", "video_macro": {"abs_rel_aligned": 0.2}}))
    log = tmp_path / "TEST_access_log.jsonl"
    c = cfg([cell("c0", "abs_rel_aligned")], R2)
    ht.run(c, res, tmp_path / "o", log)
    ht.run(c, res, tmp_path / "o", log)
    lines = [json.loads(ln) for ln in log.read_text().splitlines()]
    assert len(lines) == 2 and all(ln["n_test_files"] == 1 and ln["files"][0]["path"].endswith("t.json") and len(ln["files"][0]["sha256"]) == 64 for ln in lines)


# ------------------------------------------------------------------ protocol v3 WP29: post-filter tokens enter only as DECLARED filter-baseline rows
FB = "metric_vda_s+kal0.05"


def test_a_filtered_spec_is_refused_unless_it_is_a_declared_filter_baseline_row(tmp_path):
    rows = R2 + [row("fb", "released_raw", FB)]
    c, res = fixture(tmp_path, {OURS: {"c0": 0.1}, "metric_vda_s": {"c0": 0.2}, FB: {"c0": 0.15}}, C3[:1], rows)
    with pytest.raises(AssertionError):                                      # a filtered spec as a plain released row: hygiene refuses it
        ht.run(c, res, tmp_path / "out")
    with pytest.raises(ValueError, match="not a declared filter-baseline"):  # and a stray filtered result file for an undeclared spec: the loader refuses it
        ht.load_results(res)
    assert ht.load_results(res, frozenset({FB}))                             # declared: accepted


def test_a_declared_filter_baseline_is_shown_but_never_competes_or_counts_as_ours_or_released(tmp_path):
    rows = R2 + [row("fb", "filter_baseline", FB)]
    ev, out = go(tmp_path, {OURS: {"c0": 0.3}, "metric_vda_s": {"c0": 0.2}, FB: {"c0": 0.05}}, C3[:1], rows, filter_baseline_rows=["fb"])
    c0 = next(c for c in ev["cells"] if c["id"] == "c0")
    assert c0["values"]["fb"] == 0.05 and c0["winner"] == "mvda"           # the filter row has the best value but does not win, and ours did not beat released


def test_filter_baseline_declaration_must_agree_with_the_row_kind():
    with pytest.raises(AssertionError):                                      # kind filter_baseline but not listed
        ht.declared_filter_models(cfg(C3[:1], R2 + [row("fb", "filter_baseline", FB)]))
    with pytest.raises(AssertionError):                                      # listed but another kind
        ht.declared_filter_models(cfg(C3[:1], R2 + [row("fb", "released_raw", FB)], filter_baseline_rows=["fb"]))
    with pytest.raises(AssertionError):                                      # a filter-baseline row must carry a filter token
        ht.check_tag_hygiene(cfg(C3[:1], R2 + [row("fb", "filter_baseline", "metric_vda_s")], filter_baseline_rows=["fb"]))
    with pytest.raises(AssertionError):                                      # ours never carries a smoothing filter
        ht.check_tag_hygiene(cfg(C3[:1], [row("ours", "ours", f"ours:/c/final.pt+{TAG}+ema0.7"), row("mvda", "released_raw", "metric_vda_s")]))
    assert ht.declared_filter_models(cfg(C3[:1], R2 + [row("fb", "filter_baseline", FB)], filter_baseline_rows=["fb"])) == frozenset({FB})


def test_a_filter_baseline_row_cannot_be_made_competing(tmp_path):
    rows = R2 + [row("fb", "filter_baseline", FB, competes=True)]
    with pytest.raises(AssertionError, match="never competes"):
        ht.declared_filter_models(cfg(C3[:1], rows, filter_baseline_rows=["fb"]))
    with pytest.raises(AssertionError, match="never competes"):                  # and the full run refuses it before reading any result
        go(tmp_path, {OURS: {"c0": 0.3}, "metric_vda_s": {"c0": 0.2}, FB: {"c0": 0.05}}, C3[:1], rows, filter_baseline_rows=["fb"])
    ht.declared_filter_models(cfg(C3[:1], R2 + [row("fb", "filter_baseline", FB, competes=False)], filter_baseline_rows=["fb"]))      # explicit false is fine
