"""Forced-fail controls for check_a4_518.py: a synthetic PASSing 518 result is derived from the real 378 release row; each mutation must flip a named check."""
import copy, json, sys
from pathlib import Path
import pytest
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import check_a4_518 as C
REF = json.load(open(HERE.parents[1] / "LR_DEV" / "ddad_release" / "ddad_scores.json")); SHA = "e" * 64
LOG = "[ddad_scorer] first clip: model input (100, 3, 518, 826) -> depth (100, 518, 826) (same HxW, identity resample)\n"


def good():
    d = copy.deepcopy(REF); d["meta"].update(short_side=518, label=C.L518, ddad_scorer_sha256=SHA); d["label"] = C.L518; return d


def failed(d, log=LOG):
    return [n for n, ok in C.checks(d, log, SHA, REF) if not ok]


def test_synthetic_518_passes():
    assert failed(good()) == []


def test_real_378_row_fails_as_518():   # natural negative control: the existing 378 release row must not pass
    f = failed(copy.deepcopy(REF), "[ddad_scorer] first clip: model input (100, 3, 378, 602) -> depth (100, 378, 602) (same HxW, identity resample)")
    assert {"short_side == 518 (meta)", "label exact (meta+top)", "scorer sha == A4 pin", "shape log: first clip (N,3,518,826) -> depth (N,518,826)"} <= set(f)


@pytest.mark.parametrize("mut,name", [
    (lambda d: d["meta"].update(short_side=378), "short_side == 518 (meta)"),
    (lambda d: d.update(label="x"), "label exact (meta+top)"),
    (lambda d: d["meta"].update(ddad_scorer_sha256="0" * 64), "scorer sha == A4 pin"),
    (lambda d: d["meta"]["release_weights"].update(sha256="0" * 64), "release weights sha"),
    (lambda d: d["meta"].update(precision="fp32"), "every other meta field == the 378 release row"),
    (lambda d: d.update(n_scored_frames=394), "395 frames / 50 scenes, 0 non-finite"),
    (lambda d: d["bins"][next(iter(d["bins"]))].update(n_px=1), "same GT px/frames/scenes per bin as the 378 row"),
    (lambda d: d["bins"][next(iter(d["bins"]))].update(median_pred_over_gt=float("nan")), "all metrics finite")])
def test_forced_fail_each_check(mut, name):
    d = good(); mut(d); assert name in failed(d)


@pytest.mark.parametrize("log", ["", "first clip: model input (100, 3, 518, 826) -> depth (100, 378, 602) (same HxW, identity resample)",
                                 "first clip: model input (100, 3, 378, 602) -> depth (100, 378, 602) (same HxW, identity resample)"])
def test_forced_fail_shape_log(log):
    assert "shape log: first clip (N,3,518,826) -> depth (N,518,826)" in failed(good(), log)
