"""END-TO-END synthetic test (no AV2 data, no real model): fake feathers/jpgs for 2 logs (+ decoy pre-R1 jpgs and sweeps in the data dir), the REAL build_av2_masks.main()
(fake segmenters via make_sky_mask._Seg) and the REAL av2_scorer.main() (fake bench_joint pipeline). Needs pyarrow/pandas/cv2/scipy/torch (run with ~/.venvs/ramen_seg2; skipped otherwise).
It fails on the old behaviour (scorer re-running select_frames on the dir glob; GT-capped LiDAR exclusion; drop rule routing; mask-meta checks)."""
import contextlib, json, shutil, sys
from pathlib import Path
import numpy as np
import pytest

for m in ("pyarrow", "pandas", "cv2", "scipy", "torch"):
    pytest.importorskip(m)
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
AV2_DIR = HERE.parent
STAGE1_CFG = AV2_DIR.parent / "stage1" / "configs"
START_OF = {"s50": "n46", "s51": "n47", "s52": "n48"}
sys.path.insert(0, str(HERE))
import synth_av2 as SY
import av2_dataset as A
import av2_scorer as S
import build_av2_masks as B
import make_sky_mask as M


class FakeSeg:
    """Stands in for make_sky_mask._Seg: sky = top 700 rows if the frame is bright (mean > 150) else top 20 rows (-> mask-dropped frame)."""
    instances = []

    def __init__(self, repo, revision, expect, kind, device="cpu", use_safetensors=None):
        self.kind, self.use_safetensors = kind, use_safetensors
        FakeSeg.instances.append(self)

    def sky_maps(self, img):
        rows = SY.SKY_ROWS if img.mean() > 150 else 20
        p = np.full(img.shape[:2], 0.02, np.float32); m = np.full(img.shape[:2], -0.96, np.float32)
        p[:rows] = 0.99; m[:rows] = 0.98
        return p, m


ORIG_AUTOCAST = S.autocast_ctx


def sha(p):
    return A.sha256_file(p)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("w")
    w = SY.build_world(tmp)
    mp = pytest.MonkeyPatch()
    try:
        segA, segB = tmp / "segA", tmp / "segB"
        for d, c in ((segA, b"A-weights"), (segB, b"B-weights")):
            d.mkdir(); (d / "pytorch_model.bin").write_bytes(c)
        mp.setitem(B.SEG_A, "bin_sha", sha(segA / "pytorch_model.bin")); mp.setitem(B.SEG_B, "bin_sha", sha(segB / "pytorch_model.bin"))
        mp.setattr(M, "_Seg", FakeSeg)
        for k, f in (("frames_sha", "frames"), ("clips_sha", "clips"), ("objects_sha", "objects")):
            mp.setitem(A.REGISTERED_SHAS, k, sha(w[f]))
        w["mask"] = tmp / "sky_masks.npz"
        B.main(["--root", str(w["root"]), "--frames", str(w["frames"]), "--clips", str(w["clips"]), "--objects", str(w["objects"]), "--out", str(w["mask"]),
                "--seg-a", str(segA), "--seg-b", str(segB), "--device", "cpu"])
        w["segA"], w["segB"] = segA, segB
        w["registered"] = dict(A.REGISTERED_SHAS)
        yield w
    finally:
        mp.undo()


def mask_meta(w):
    return json.load(open(Path(w["mask"]).with_suffix(".meta.json")))


def compute_pins(w, mask=None):
    sys.path.insert(0, str(REPO))
    import bench_joint as bj, ddad_scorer as DS
    mask = Path(mask or w["mask"]); mp_ = mask.with_suffix(".meta.json"); mm = json.load(open(mp_))
    return dict(logs_sha=sha(w["logs"]), frames_sha=sha(w["frames"]), clips_sha=sha(w["clips"]), objects_sha=sha(w["objects"]), mask_sha=sha(mask), meta_sha=sha(mp_),
                dataset_module_sha=sha(A.__file__), ddad_scorer_sha=sha(DS.__file__), bench_joint_sha=sha(bj.__file__),
                builder_sha=sha(AV2_DIR / "build_av2_masks.py"), make_sky_mask_sha=sha(AV2_DIR.parent / "seg" / "make_sky_mask.py"),
                segmenter_a_sha=mm["segmenter_a_sha"], segmenter_b_sha=mm["segmenter_b_sha"], sums_sha=sha(w["sums"]))


def declared_cfg(pair, arm):
    """The declared Stage 1 resolved config of (pair, arm) (outdoor_debug/parts/confirm/stage1/configs), as train.py would store it in ck['cfg']."""
    import yaml
    return yaml.safe_load(open(STAGE1_CFG / f"arm_S1_{START_OF[pair]}_{arm}.yaml"))


def make_ckpt(path, arm, step=14000, cfg="declared", pair="s50", salt=""):
    import torch
    c = declared_cfg(pair, arm) if cfg == "declared" else cfg
    d = {"trainable_ema": {}, "step": step, "tag": arm + salt}
    if c is not None:
        d["cfg"] = c
    torch.save(d, path)
    return path


def registry_text(entries):
    """entries: {(pair, arm): sha or None(=UNSET)}. Six CKPT_SHA lines as in RUNBOOK_STAGE1.md."""
    return "".join(f"CKPT_SHA {p} {a} = {entries.get((p, a)) or 'UNSET'}\n" for p in S.PAIRS for a in ("H300", "TS2"))


def full_registry(w, pair, arm, ck_sha):
    ent = {(p, a): sha(w["tmp"] / "SHA256SUMS")[:60] + f"{i:04x}" for i, (p, a) in enumerate((p, a) for p in S.PAIRS for a in ("H300", "TS2"))}
    ent[(pair, arm)] = ck_sha
    return ent


def run_scorer(w, monkeypatch, arm, out, pair="s50", step=14000, pins=None, mask=None, expect_sha="auto", gpu="NVIDIA GH200 120GB", patch_autocast=True, extra=(), frames=None,
               cfg="declared", cfg_arm=None, cfg_pair=None, registry="ok", sums="default", ckpt_salt="", ckpt=None, registry_path=None):
    """cfg: 'declared' (the declared Stage 1 cfg of cfg_pair/cfg_arm, default the CLI labels), a dict, or None (no cfg in the checkpoint). registry: 'ok' (the six registered shas, this run's own right),
    'unset' (all UNSET), 'wrong' (this pair+arm registered with another sha), or a dict of entries."""
    sys.path.insert(0, str(REPO))
    import torch, bench_joint as bj
    tag = f"{arm}_{pair}_{step}_{cfg_arm}_{cfg_pair}_{ckpt_salt}_{abs(hash(str(cfg))) % 10**6}"
    ck = ckpt or make_ckpt(w["tmp"] / f"ckpt_{tag}.pt", arm, step, cfg=(declared_cfg(cfg_pair or pair, cfg_arm or arm) if cfg == "declared" else cfg), salt=ckpt_salt)
    ck_sha = sha(ck)
    ent = {"ok": lambda: full_registry(w, pair, arm, ck_sha), "unset": lambda: {}, "wrong": lambda: full_registry(w, pair, arm, "e" * 64)}.get(registry, lambda: registry)() if isinstance(registry, str) else registry
    reg = registry_path or (w["tmp"] / f"registry_{tag}_{registry if isinstance(registry, str) else 'custom'}.md")
    if not registry_path:
        reg.write_text("EXPECT_SCORER_SHA = <not used here>\n" + registry_text(ent))
    w["last_registry"] = reg
    mask = mask or w["mask"]
    monkeypatch.setattr(S, "PINS", dict(compute_pins(w, mask) if pins is None else pins))
    monkeypatch.setattr(S, "gpu_name", lambda: gpu)
    monkeypatch.setattr(S, "autocast_ctx", (lambda dev: contextlib.nullcontext()) if patch_autocast else ORIG_AUTOCAST)
    monkeypatch.setattr(bj, "build_pipeline", lambda spec, dev: dict(metric=True, spec=spec))

    def fake_run(pipe, x, s, b, out_hw):
        d = torch.full((x.shape[0], *out_hw), 30.0)
        d[:, : int(np.ceil(SY.SKY_ROWS * out_hw[0] / A.NATIVE_HW[0]))] = 400.0 if arm == "H300" else 150.0
        return d, None
    monkeypatch.setattr(bj, "run_pipeline", fake_run)
    argv = ["--ckpt", str(ck), "--pair", pair, "--arm", arm, "--out", str(out), "--root", str(w["root"]), "--mask", str(mask), "--frames", str(frames or w["frames"]),
            "--clips", str(w["clips"]), "--objects", str(w["objects"]), "--logs", str(w["logs"]), "--device", "cpu", "--repo", str(REPO),
            *(["--sums", str(w["sums"] if sums == "default" else sums)] if sums is not None else []), "--registry", str(reg), *extra]
    if expect_sha == "auto":
        argv += ["--expect-scorer-sha", sha(S.__file__)]
    elif expect_sha:
        argv += ["--expect-scorer-sha", expect_sha]
    S.main(argv)
    return json.load(open(out))


@pytest.fixture(scope="module")
def runs(world):
    """Two real main() runs (pair s50, both arms) against ONE registry file (so they share registry_sha)."""
    mp = pytest.MonkeyPatch()
    try:
        cks = {arm: make_ckpt(world["tmp"] / f"runs_{arm}.pt", arm, pair="s50") for arm in ("H300", "TS2")}
        reg = world["tmp"] / "runs_registry.md"
        reg.write_text(registry_text(full_registry(world, "s50", "H300", sha(cks["H300"])) | {("s50", "TS2"): sha(cks["TS2"])}))
        r = {arm: run_scorer(world, mp, arm, world["tmp"] / f"{arm}.json", ckpt=cks[arm], registry_path=reg) for arm in ("H300", "TS2")}
    finally:
        mp.undo()
    return r


@pytest.fixture(scope="module")
def six(world):
    """All six Stage 1 runs through the real main() against ONE registry (3 pairs x 2 arms, declared cfgs); returns (json paths by (pair, arm), registry path)."""
    mp = pytest.MonkeyPatch()
    try:
        cks = {(p, a): make_ckpt(world["tmp"] / f"six_{p}_{a}.pt", a, pair=p) for p in S.PAIRS for a in ("H300", "TS2")}
        reg = world["tmp"] / "six_registry.md"
        reg.write_text(registry_text({k: sha(v) for k, v in cks.items()}))
        paths = {}
        for (p, a), ck in cks.items():
            paths[(p, a)] = world["tmp"] / f"six_{p}_{a}.json"
            run_scorer(world, mp, a, paths[(p, a)], pair=p, ckpt=ck, registry_path=reg)
    finally:
        mp.undo()
    return paths, reg


# ------------------------------------------------------------------ builder
def test_old_dir_glob_selection_differs_from_registered(world):
    """The decoys really change what a directory glob would select (the old scorer main re-ran select_frames on it)."""
    for log in SY.LOGS:
        d = world["root"] / log
        cams = sorted(int(p.stem) for p in (d / "sensors" / "cameras" / A.CAMERA).glob("*.jpg")); lids = sorted(int(p.stem) for p in (d / "sensors" / "lidar").glob("*.feather"))
        old = A.select_frames(cams, lids)
        reg = A.load_registered_inputs(world["frames"], world["clips"], world["objects"], world["root"], registered=world["registered"])[log]
        assert old["clip"] != reg["clip"] and [s[0] for s in old["scored"]] != [r[4] for r in reg["rows"]] or [s[1] for s in old["scored"]] != [r[1] for r in reg["rows"]]
        assert len(reg["clip"]) == 20 and [r[4] for r in reg["rows"]] == [0, 10]


def test_build_meta_and_store(world):
    m = mask_meta(world)
    assert m["status"] == "OK" and m["pooled_iou"] == 1.0 and m["n_candidates"] == 4
    tiny = A.frame_id(SY.TINY_FRAME[0], A.read_clips_tsv(world["clips"])[SY.TINY_FRAME[0]][1][SY.TINY_FRAME[1]])
    assert m["dropped"] == [tiny]
    st = A.MaskStore(world["mask"])
    c, a = st.get(tiny)                                           # R1 M4: the P-dropped frame STAYS in the store (guards need its `any`)
    assert a.any() and c.sum() < 0.01 * c.size
    assert m["frames_sha"] == sha(world["frames"]) and m["clips_sha"] == sha(world["clips"]) and m["objects_sha"] == sha(world["objects"])
    assert m["segmenter_a"]["repo"] == B.SEG_A["repo"] and m["segmenter_a"]["revision"] == B.SEG_A["revision"] and m["segmenter_b"]["revision"] == B.SEG_B["revision"]
    assert m["lidar_xyz_dtypes"] == ["float16", "float32"]


def test_segmenters_loaded_without_safetensors(world):
    assert FakeSeg.instances and all(s.use_safetensors is False for s in FakeSeg.instances)


def test_lidar_exclusion_uses_returns_beyond_200m(world):
    """A return at 300 m inside the sky excludes its pixels (the capped GT map would not): the certain mask has a hole of radius 9 around FAR_SKY_UV, and none beyond."""
    st = A.MaskStore(world["mask"])
    log = SY.LOGS[0]; cam = A.read_clips_tsv(world["clips"])[log][1][0]
    c, a = st.get(A.frame_id(log, cam))
    u, v = SY.FAR_SKY_UV
    assert not c[v, u] and not c[v, u + 8] and not c[v - 8, u] and a[v, u]
    assert c[v, u + 10] and c[v - 10, u] and c[v + 8, u + 8]
    assert not c[SY.NEAR_SKY_UV[1], SY.NEAR_SKY_UV[0]]
    L = A.Log(world["root"], log)
    assert (L.sparse_gt(cam, A.read_frames_tsv(world["frames"])[0][2]) > 250).sum() == 0     # the capped GT map does not even contain the 300 m return


# ------------------------------------------------------------------ scorer
def test_scorer_routing_drop_affects_P_only(runs):
    h = runs["H300"]
    assert h["n_scored_frames"] == 4                                  # guards: ALL scored frames, mask-dropped included
    assert h["meta"]["n_mask_dropped_frames"] == 1
    assert h["sky"]["n_frames_seen"] == 3 and len(h["sky"]["kept_frame_ids"]) == 3 and h["sky"]["n_frames_dropped"] == 0     # P: the mask-dropped frame never reached it
    assert h["sky"]["P_framebalanced_median"] == pytest.approx(400.0) and runs["TS2"]["sky"]["P_framebalanced_median"] == pytest.approx(150.0)
    assert h["n_lidar_px_sky_excluded"] > 0 and h["meta"]["lidar_xyz_dtypes"] == ["float16", "float32"]


def test_scorer_meta_records_pins_and_facts(runs, world):
    m = runs["H300"]["meta"]
    assert m["scorer_sha"] == sha(S.__file__) and m["ckpt_step"] == 14000 and m["expected_seed"] == 50 and m["arm"] == "H300" and m["seed_label"] == "s50" and m["stage"] == 1
    assert m["ckpt_sha"] != runs["TS2"]["meta"]["ckpt_sha"] and "GH200" in m["host_gpu"] and m["pins_asserted"] and "single seed" not in m["label"].lower()
    for k in ("guards carry the information", "TS2+H300 was chosen after the screen", "NOT BETTER on its primary", "THREE paired"):
        assert k in m["disclosure"]


def test_compare_runs_and_disclosure_per_stage(runs):
    c1 = S.compare(runs["H300"], runs["TS2"], stage=1)
    assert c1["d_ln_PH300_over_PTS2"] == pytest.approx(np.log(400 / 150)) and "THREE paired" in c1["disclosure"] and "SINGLE SEED" not in c1["disclosure"]
    h2, t2 = json.loads(json.dumps(runs["H300"])), json.loads(json.dumps(runs["TS2"]))
    for r in (h2, t2):
        r["meta"]["seed_label"] = "stage2"; r["meta"]["stage"] = 2
    c2 = S.compare(h2, t2, stage=2)
    assert c2["disclosure"].startswith("SINGLE SEED (50): one paired difference cannot size seed noise; the A0/A0b screen pair showed 26% spread on a guard. Sign and magnitude only.")
    assert "THREE paired" not in c2["disclosure"] and "single seed 50" in c2["label"]
    assert c1["verdict"].startswith("NOT CONFIRMED")                  # synthetic bins are UNEVALUABLE (< 20,000 px): never a pass


def _pairs(runs):
    out = {}
    for i, k in enumerate(S.PAIRS):
        h, t = json.loads(json.dumps(runs["H300"])), json.loads(json.dumps(runs["TS2"]))
        for r in (h, t):
            r["meta"]["seed_label"] = k
        h["meta"]["ckpt_sha"] += str(i); t["meta"]["ckpt_sha"] += str(i)
        out[k] = (h, t)
    return out


def test_compare_pairs_and_forced_fails(runs):
    ok = _pairs(runs)
    r = S.compare_pairs(ok)
    assert r["d_min"] == pytest.approx(np.log(400 / 150)) and "THREE paired" in r["disclosure"] and "single seed" not in r["label"].lower()
    sw = dict(ok); sw["s51"] = (ok["s51"][1], ok["s51"][0])                               # arms swapped: TS2 given as H300
    with pytest.raises(AssertionError, match="argument order"):
        S.compare_pairs(sw)
    with pytest.raises(AssertionError, match="argument order"):
        S.compare(runs["TS2"], runs["H300"], stage=1)
    same = _pairs(runs); same["s50"][1]["meta"]["ckpt_sha"] = same["s50"][0]["meta"]["ckpt_sha"]
    with pytest.raises(AssertionError, match="different checkpoints"):
        S.compare_pairs(same)
    for key, val in (("kept_frame_ids", ["x"]), ("frame_mask_px", [1, 2, 3]), ("dropped_frame_ids", ["x"])):             # P11: dropped ids too
        bad = _pairs(runs); bad["s52"][1]["sky"][key] = val
        with pytest.raises(AssertionError, match=key):
            S.compare_pairs(bad)
    bad = _pairs(runs); bad["s50"][1]["n_lidar_px_sky_excluded"] += 1
    with pytest.raises(AssertionError, match="n_lidar_px_sky_excluded"):
        S.compare_pairs(bad)
    bad = _pairs(runs); bad["s50"][0]["meta"]["stage"] = 2
    with pytest.raises(AssertionError):
        S.compare_pairs(bad)
    bad = _pairs(runs); bad["s50"][1]["n_scored_frames"] += 1                                                   # P12
    with pytest.raises(AssertionError, match="n_scored_frames"):
        S.compare_pairs(bad)
    h, t = (json.loads(json.dumps(runs[a])) for a in ("H300", "TS2")); t["meta"]["stage"] = 2                    # P13: reaches check_pair_runs directly
    with pytest.raises(AssertionError, match="pair label and stage"):
        S.compare(h, t, stage=1)
    for ov in (dict(stage=2), dict(seed_label="s51")):                                                          # P14: BOTH runs of s50 consistent with each other, but not s50/stage 1
        bad = _pairs(runs)
        for r in bad["s50"]:
            r["meta"].update(ov)
        with pytest.raises(AssertionError, match="must carry stage 1 and the pair label"):
            S.compare_pairs(bad)


def test_N2_compare_pairs_asserts_identical_pins_scorer_and_six_distinct_shas(runs):
    ok = _pairs(runs)
    S.compare_pairs(ok)
    for key in ("pins", "scorer_sha", "registry_sha"):                       # one run differs (or lacks the key)
        for how in ("differ", "missing"):
            bad = _pairs(runs)
            if how == "differ":
                bad["s52"][0]["meta"][key] = {"x": 1} if key == "pins" else "f" * 64
            else:
                del bad["s52"][0]["meta"][key]
            with pytest.raises(AssertionError, match=r"meta\." + key):
                S.compare_pairs(bad)
    # arms of ONE pair with different pins / scorer sha are refused by check_pair_runs itself
    h, t = (json.loads(json.dumps(runs[a])) for a in ("H300", "TS2")); t["meta"]["scorer_sha"] = "0" * 64
    with pytest.raises(AssertionError, match="scorer_sha"):
        S.compare(h, t, stage=1)
    # one H300 checkpoint scored under three pair labels (and one TS2 likewise): only 2 distinct shas
    dup = _pairs(runs)
    for k in S.PAIRS:
        for r in dup[k]:
            r["meta"]["ckpt_sha"] = r["meta"]["ckpt_sha"][:-1] + "0"
    # make the three H300 shas equal and the three TS2 shas equal (but H300 != TS2)
    for k in S.PAIRS:
        dup[k][0]["meta"]["ckpt_sha"] = "a" * 64; dup[k][1]["meta"]["ckpt_sha"] = "b" * 64
    with pytest.raises(AssertionError, match="6 DISTINCT"):
        S.compare_pairs(dup)


def test_N2_main_compare_pairs_requires_expect_sha_registry_and_registered_shas(six, world, monkeypatch, capsys, tmp_path):
    paths, reg = six
    files = [str(paths[(p, a)]) for p in S.PAIRS for a in ("H300", "TS2")]
    good = sha(S.__file__)
    S.main(["--compare-pairs", *files, "--registry", str(reg), "--expect-scorer-sha", good])                                      # positive control
    res = json.loads(capsys.readouterr().out)
    assert res["verdict"].startswith("NOT CONFIRMED") and set(res["d_per_pair"]) == set(S.PAIRS)
    with pytest.raises(SystemExit, match="expect-scorer-sha"):
        S.main(["--compare-pairs", *files, "--registry", str(reg)])
    with pytest.raises(AssertionError, match="scorer sha"):
        S.main(["--compare-pairs", *files, "--registry", str(reg), "--expect-scorer-sha", "0" * 64])
    with pytest.raises(SystemExit, match="registry"):
        S.main(["--compare-pairs", *files, "--expect-scorer-sha", good])
    other = tmp_path / "other.md"; other.write_text(reg.read_text() + "# edited after scoring\n")
    with pytest.raises(AssertionError, match="different registry"):                  # a registry file that is not the one the runs were scored against
        S.main(["--compare-pairs", *files, "--registry", str(other), "--expect-scorer-sha", good])
    # a run json whose checkpoint sha is not the registered one for its pair+arm (six still distinct, same registry file, same pins): only the registry check can see it
    forged = tmp_path / "forged_s51_H300.json"
    j = json.load(open(paths[("s51", "H300")])); j["meta"]["ckpt_sha"] = "9" * 64; forged.write_text(json.dumps(j))
    ff = list(files); ff[2] = str(forged)
    with pytest.raises(AssertionError, match="registered"):
        S.main(["--compare-pairs", *ff, "--registry", str(reg), "--expect-scorer-sha", good])
    with pytest.raises(AssertionError, match="registered"):
        S.main(["--compare", ff[2], ff[3], "--registry", str(reg), "--expect-scorer-sha", good])
    with pytest.raises(SystemExit, match="expect-scorer-sha"):
        S.main(["--compare", files[0], files[1], "--registry", str(reg)])
    S.main(["--compare", files[0], files[1], "--registry", str(reg), "--expect-scorer-sha", good])                                  # positive control for --compare
    assert "d_ln_PH300_over_PTS2" in json.loads(capsys.readouterr().out)


# ---- scorer refusals (each reaches main() and must stop BEFORE writing any output)
def _refuses(world, monkeypatch, tmp_path, exc, match=None, **kw):
    out = tmp_path / "o.json"
    with pytest.raises(exc, match=match):
        run_scorer(world, monkeypatch, "H300", out, **kw)
    assert not out.exists()


def test_refuses_wrong_or_missing_expected_scorer_sha(world, monkeypatch, tmp_path):
    _refuses(world, monkeypatch, tmp_path, AssertionError, "scorer sha", expect_sha="0" * 64)
    _refuses(world, monkeypatch, tmp_path, SystemExit, "expect-scorer-sha", expect_sha=None)


def test_refuses_wrong_unset_pin(world, monkeypatch, tmp_path):
    p = compute_pins(world); p["mask_sha"] = "0" * 64
    _refuses(world, monkeypatch, tmp_path, AssertionError, "pin mask_sha", pins=p)
    p = compute_pins(world); p["meta_sha"] = S.UNSET
    _refuses(world, monkeypatch, tmp_path, AssertionError, "UNSET", pins=p)
    p = compute_pins(world); p["meta_sha"] = "1" * 64                                         # meta file edited after the pin was filled
    _refuses(world, monkeypatch, tmp_path, AssertionError, "pin meta_sha", pins=p)


def _edited_mask(world, tmp_path, **edit):
    d = tmp_path / "m"; d.mkdir()
    shutil.copy(world["mask"], d / "sky_masks.npz")
    m = mask_meta(world); m.update(edit)
    (d / "sky_masks.meta.json").write_text(json.dumps(m, sort_keys=True, indent=1))
    return d / "sky_masks.npz"


@pytest.mark.parametrize("edit,match", [(dict(pooled_iou=0.5), "pooled_iou"), (dict(status="REJECTED"), "status"), (dict(pooled_iou=float("nan")), "pooled_iou"),
                                        (dict(pooled_iou=True), "pooled_iou"), (dict(dropped=None), "dropped")])
def test_main_runs_check_mask_meta(world, monkeypatch, tmp_path, edit, match):
    """Pins are recomputed for the edited sidecar, so only check_mask_meta can reject it."""
    _refuses(world, monkeypatch, tmp_path, RuntimeError, match, mask=_edited_mask(world, tmp_path, **edit))


def test_main_checks_meta_frames_sha(world, monkeypatch, tmp_path):
    _refuses(world, monkeypatch, tmp_path, AssertionError, "frames_sha", mask=_edited_mask(world, tmp_path, frames_sha="0" * 64))


def test_main_checks_step_host_and_cuda(world, monkeypatch, tmp_path):
    _refuses(world, monkeypatch, tmp_path, AssertionError, "step", step=13999)
    _refuses(world, monkeypatch, tmp_path, AssertionError, "GH200", gpu="NVIDIA A100")
    _refuses(world, monkeypatch, tmp_path, AssertionError, "requires cuda", patch_autocast=False)          # the real autocast_ctx refuses a non-cuda device


def test_main_requires_pair_and_arm(world, monkeypatch, tmp_path):
    with pytest.raises(AssertionError, match="pair"):
        S.main(["--ckpt", "x.pt", "--out", str(tmp_path / "o.json"), "--root", "r", "--mask", "m", "--frames", "f", "--clips", "c", "--objects", "o", "--logs", "l", "--sums", "s",
                "--expect-scorer-sha", sha(S.__file__)])


def test_registered_inputs_refuse_tampered_clip_and_objects(world, tmp_path):
    reg = world["registered"]
    t = tmp_path / "clips.tsv"
    lines = open(world["clips"]).read().split("\n")
    bad = lines[0].split("\t"); ts = bad[3].split(","); ts[0], ts[10] = ts[10], ts[0]; bad[3] = ",".join(ts)       # sorted-order violation
    t.write_text("\t".join(bad) + "\n" + "\n".join(lines[1:]))
    with pytest.raises(AssertionError):
        A.load_registered_inputs(world["frames"], t, world["objects"], world["root"], registered=dict(reg, clips_sha=sha(t)))
    # a clip whose index 10 is not the scored camera ts
    ts = lines[0].split("\t")[3].split(","); ts[10] = str(int(ts[10]) + 1); bad = lines[0].split("\t"); bad[3] = ",".join(ts)
    t.write_text("\t".join(bad) + "\n" + "\n".join(lines[1:]))
    with pytest.raises(AssertionError, match="clip\\[10\\]"):
        A.load_registered_inputs(world["frames"], t, world["objects"], world["root"], registered=dict(reg, clips_sha=sha(t)))
    # a clip jpg that is not in objects.tsv
    o = tmp_path / "objects.tsv"
    o.write_text("".join(l for l in open(world["objects"]) if "sensors/cameras" not in l or not l.split("\t")[0].endswith(f"{SY.cam_ts()[2]}.jpg")))
    with pytest.raises(AssertionError, match="not in objects.tsv"):
        A.load_registered_inputs(world["frames"], world["clips"], o, world["root"], registered=dict(reg, objects_sha=sha(o)))
    # sha not pinned
    with pytest.raises(AssertionError, match="clips_sha"):
        A.load_registered_inputs(world["frames"], t, world["objects"], world["root"], registered=reg)
    # missing file on disk
    f = world["root"] / SY.LOGS[0] / "sensors" / "lidar"
    victim = sorted(f.glob("*.feather"))[0]; keep = victim.read_bytes(); victim.unlink()
    try:
        with pytest.raises(FileNotFoundError):
            A.load_registered_inputs(world["frames"], world["clips"], world["objects"], world["root"], registered=reg)
    finally:
        victim.write_bytes(keep)


def test_decode_sanity_on_synthetic(world):
    import decode_sanity as DSN
    out = DSN.main(world["root"], world["frames"], 5, world["logs"])
    assert [o["log"] for o in out] == SY.LOGS and all(o["ok"] for o in out)


def test_decode_sanity_negative_controls(world):
    import decode_sanity as DSN, cv2
    row = A.read_frames_tsv(world["frames"])[0]
    jpg = A.Log.image_path_of(world["root"], row[0], row[1]); keep = jpg.read_bytes()
    try:
        cv2.imwrite(str(jpg), np.full((*A.NATIVE_HW, 3), 90, np.uint8))                              # constant image
        with pytest.raises(AssertionError, match="constant"):
            DSN.check_frame(world["root"], row[0], row[1], row[2])
        cv2.imwrite(str(jpg), np.zeros((100, 100, 3), np.uint8))                                       # wrong shape
        with pytest.raises(AssertionError, match="image"):
            DSN.check_frame(world["root"], row[0], row[1], row[2])
    finally:
        jpg.write_bytes(keep)
    DSN.check_frame(world["root"], row[0], row[1], row[2])


def test_panel_on_synthetic(world, tmp_path):
    import make_panel as P
    o = tmp_path / "p.png"
    P.main(["--root", str(world["root"]), "--frames", str(world["frames"]), "--masks", str(world["mask"]), "--out", str(o)])
    assert o.stat().st_size > 0


# ------------------------------------------------------------------ N1: the checkpoint's own cfg + the registered sha decide, never the CLI labels
def test_N1_declared_cfgs_pass_for_all_six_and_record_facts():
    for p in S.PAIRS:
        for a in ("H300", "TS2"):
            f = S.assert_ckpt_cfg(declared_cfg(p, a), p, a)
            assert f["cfg_data_seed"] == S.SEED_OF[p] and f["cfg_resume_from"].endswith(f"c3_{START_OF[p]}/step12000.pt")
            assert (f["cfg_sky_hinge_weight"], f["cfg_sky_hinge_min_depth"]) == ((0.1, 300.0) if a == "H300" else (0.0, None))


def test_N1_cfg_mutations_each_refuse():
    import copy
    base = declared_cfg("s50", "H300")
    def mut(fn, pair="s50", arm="H300", match=None):
        c = copy.deepcopy(declared_cfg(pair, arm)); fn(c)
        with pytest.raises(AssertionError, match=match):
            S.assert_ckpt_cfg(c, pair, arm)
    mut(lambda c: c["loss"].__setitem__("sky_hinge_min_depth", 30.0), match="arm H300")
    mut(lambda c: c["loss"].__setitem__("sky_hinge_weight", 0.2), match="arm H300")
    mut(lambda c: c["loss"].pop("sky_hinge_min_depth"), match="arm H300")
    mut(lambda c: c["loss"].__setitem__("sky_hinge_weight", 0.1), arm="TS2", match="arm TS2")                       # control with the hinge weight
    mut(lambda c: c["loss"].__setitem__("sky_hinge_min_depth", 300.0), arm="TS2", match="arm TS2")
    mut(lambda c: c["data"].__setitem__("seed", 51), match="data.seed")
    mut(lambda c: c["training"].__setitem__("resume_reseed", 51), match="resume_reseed")
    mut(lambda c: c["training"].pop("resume_reseed"), match="resume_reseed")
    mut(lambda c: c["training"].__setitem__("resume_from", S.START_FMT.format(n="47")), match="resume_from")
    mut(lambda c: c["training"].__setitem__("resume_from", None), match="resume_from")
    mut(lambda c: c["training"].__setitem__("max_steps", 12050), match="max_steps")
    mut(lambda c: [s.__setitem__("ts2_sky", False) for s in c["data"]["sources"] if s.get("type") == "tartanair"], match="ts2_sky")
    for bad in (None, {}, {"model": {}}):
        with pytest.raises(AssertionError, match="no resolved cfg"):
            S.assert_ckpt_cfg(bad, "s50", "H300")
    with pytest.raises(AssertionError, match="Stage 2"):
        S.assert_ckpt_cfg(base, "stage2", "H300")
    with pytest.raises(AssertionError):
        S.assert_ckpt_cfg(base, "s50", "H301")


def test_N1_main_refuses_ts2_checkpoint_scored_as_H300_and_other_label_cfg_mismatches(world, monkeypatch, tmp_path):
    out = tmp_path / "o.json"
    with pytest.raises(AssertionError, match="arm H300"):                          # THE critical case: TS2 checkpoint, --arm H300
        run_scorer(world, monkeypatch, "H300", out, cfg_arm="TS2"); 
    _refuses(world, monkeypatch, tmp_path, AssertionError, "pair s51", pair="s51", cfg_pair="s50")                     # wrong pair label (cfg is the s50 one)
    c = declared_cfg("s50", "H300"); c["data"]["seed"] = 99
    _refuses(world, monkeypatch, tmp_path, AssertionError, "data.seed", cfg=c)                                          # wrong seed
    c = declared_cfg("s50", "H300"); c["training"]["resume_from"] = S.START_FMT.format(n="47")
    _refuses(world, monkeypatch, tmp_path, AssertionError, "resume_from", cfg=c)                                        # wrong start checkpoint
    _refuses(world, monkeypatch, tmp_path, AssertionError, "no resolved cfg", cfg=None)                                # no cfg in the checkpoint
    _refuses(world, monkeypatch, tmp_path, AssertionError, "stage2", pair="stage2", cfg_pair="s50") if False else None


def test_N1_registry_parse_and_refusals(world, monkeypatch, tmp_path):
    full = {(p, a): f"{i:064x}" for i, (p, a) in enumerate((p, a) for p in S.PAIRS for a in ("H300", "TS2"))}
    reg = S.parse_registry(registry_text(full))
    assert reg == full
    for bad, m in ((registry_text(full).replace("CKPT_SHA s50 H300", "CKPT_SHA s50 H301"), "unexpected"), (registry_text(full) + "CKPT_SHA s50 H300 = " + "0" * 64 + "\n", "duplicate"),
                   (registry_text(full).replace("CKPT_SHA s52 TS2 = " + full[("s52", "TS2")], "CKPT_SHA s52 TS2 = XYZ"), "neither UNSET"),
                   (registry_text(full).split("\n", 1)[1], "exactly the six")):
        with pytest.raises(AssertionError, match=m):
            S.parse_registry(bad)
    assert S.assert_registered_ckpt(full, "s51", "TS2", full[("s51", "TS2")]) == full[("s51", "TS2")]
    with pytest.raises(AssertionError, match="UNSET"):
        S.assert_registered_ckpt(S.parse_registry(registry_text({})), "s50", "H300", "a" * 64)               # unset list
    part = dict(full); part[("s52", "TS2")] = None
    with pytest.raises(AssertionError, match="UNSET"):
        S.assert_registered_ckpt(S.parse_registry(registry_text(part)), "s50", "H300", full[("s50", "H300")])    # ONE unset entry refuses every run
    with pytest.raises(AssertionError, match="registered"):
        S.assert_registered_ckpt(full, "s50", "H300", full[("s51", "H300")])                                    # ckpt sha of the wrong pair
    with pytest.raises(AssertionError, match="registered"):
        S.assert_registered_ckpt(full, "s50", "H300", full[("s50", "TS2")])                                     # swapped arm
    dup = dict(full); dup[("s51", "H300")] = dup[("s50", "H300")]
    with pytest.raises(AssertionError, match="distinct"):
        S.assert_registered_ckpt(dup, "s50", "H300", dup[("s50", "H300")])
    # through main(): unset list, wrong-pair sha, swapped arm
    _refuses(world, monkeypatch, tmp_path, AssertionError, "UNSET", registry="unset")
    _refuses(world, monkeypatch, tmp_path, AssertionError, "registered", registry="wrong")
    h, t = make_ckpt(tmp_path / "h.pt", "H300", pair="s50"), make_ckpt(tmp_path / "t.pt", "TS2", pair="s50")
    swapped = full_registry(world, "s50", "H300", sha(t)); swapped[("s50", "TS2")] = sha(h)                   # arms swapped in the registry
    _refuses(world, monkeypatch, tmp_path, AssertionError, "registered", ckpt=h, registry=swapped)
    with pytest.raises(SystemExit, match="registry"):
        run_scorer(world, monkeypatch, "H300", tmp_path / "o2.json", registry_path=None, extra=("--registry", ""))


def test_N1_registry_and_ckpt_facts_recorded_in_the_meta(runs):
    m = runs["H300"]["meta"]
    assert m["cfg_checked"] is True and m["cfg_sky_hinge_weight"] == 0.1 and m["cfg_sky_hinge_min_depth"] == 300.0 and m["cfg_data_seed"] == 50 and m["registry_sha"]
    assert runs["TS2"]["meta"]["cfg_sky_hinge_weight"] == 0.0 and runs["TS2"]["meta"]["registry_sha"] == m["registry_sha"]


# ------------------------------------------------------------------ N3: explicit --sums
def test_N3_sums_path_is_explicit_and_pinned(world, monkeypatch, tmp_path):
    _refuses(world, monkeypatch, tmp_path, AssertionError, "--sums", sums=None)
    _refuses(world, monkeypatch, tmp_path, FileNotFoundError, None, sums=tmp_path / "does_not_exist")
    other = tmp_path / "SHA256SUMS"; other.write_text("tampered\n")
    _refuses(world, monkeypatch, tmp_path, AssertionError, "pin sums_sha", sums=other)                       # the pinned file, not just any file
    assert "meta_R1/SHA256SUMS" in (AV2_DIR / "RUNBOOK_STAGE1.md").read_text() and "--sums" in (AV2_DIR / "RUNBOOK_STAGE1.md").read_text()


# ------------------------------------------------------------------ N4: mask meta provenance + rule
def test_N4_mask_meta_records_code_shas_and_the_scorer_checks_them(world):
    m = mask_meta(world)
    assert m["rule"] == S.REGISTERED_RULE == json.loads(json.dumps(M.RULE))
    assert m["builder_sha"] == sha(AV2_DIR / "build_av2_masks.py") and m["make_sky_mask_sha"] == sha(AV2_DIR.parent / "seg" / "make_sky_mask.py") and m["dataset_module_sha"] == sha(A.__file__)
    assert m["git_commit"] is None or len(m["git_commit"]) == 40


@pytest.mark.parametrize("edit,match", [
    (dict(rule=dict(json.loads(json.dumps(M.RULE)), prob_thresh=0.8)), "rule"),            # tampered rule constant (also IoU gate, erosion, ... same path)
    (dict(rule=dict(json.loads(json.dumps(M.RULE)), iou_min=0.5)), "rule"),
    (dict(rule={k: v for k, v in json.loads(json.dumps(M.RULE)).items() if k != "seed"}), "rule"),
    (dict(rule=None), "rule"),
    (dict(builder_sha="0" * 64), "builder_sha"),                                            # built by different code than the one on disk
    (dict(make_sky_mask_sha="0" * 64), "make_sky_mask_sha"),
    (dict(dataset_module_sha="0" * 64), "dataset_module_sha"),
    (dict(builder_sha=None), "builder_sha")])
def test_N4_main_refuses_tampered_rule_or_unrecorded_code(world, monkeypatch, tmp_path, edit, match):
    _refuses(world, monkeypatch, tmp_path, AssertionError, match, mask=_edited_mask(world, tmp_path, **edit))


# ------------------------------------------------------------------ N5: survivors of the re-review (P6 P8 P9 P11-P14 [above] P17 P18 P19)
def _edited_frames_and_registered(world, tmp_path, fn):
    rows = [l.split("\t") for l in open(world["frames"]).read().splitlines()]
    hdr = rows[0] if not rows[0][0][:1].isalnum() or rows[0][0] == "log" else None
    return rows, hdr


def test_P6_P8_registered_inputs_check_sweep_in_objects_and_dt(world, tmp_path):
    reg = world["registered"]
    # P6: a PAIRED SWEEP missing from objects.tsv (all clip jpgs still listed)
    lid = [r[2] for r in A.read_frames_tsv(world["frames"])][0]
    o = tmp_path / "objects_nosweep.tsv"
    o.write_text("".join(l for l in open(world["objects"]) if not l.split("\t")[0].endswith(f"sensors/lidar/{lid}.feather")))
    with pytest.raises(AssertionError, match=r"sensors/lidar/\d+\.feather not in objects.tsv"):
        A.load_registered_inputs(world["frames"], world["clips"], o, world["root"], registered=dict(reg, objects_sha=sha(o)))
    # P8: a frames-tsv row whose dt exceeds 25 ms
    lines = open(world["frames"]).read().split("\n")
    hdr_cols = lines[0].split("\t")
    f = tmp_path / "frames_bigdt.tsv"
    bad = A.read_frames_tsv(world["frames"])
    idx = next(i for i, c in enumerate(hdr_cols) if c in ("dt_ns", "dt")) if any(c in ("dt_ns", "dt") for c in hdr_cols) else 3
    cols = lines[1].split("\t") if hdr_cols[0] in ("log", "log_id") else lines[0].split("\t")
    first = 1 if hdr_cols[0] in ("log", "log_id") else 0
    cols[idx] = str(A.MAX_PAIR_DT_NS + 1)
    lines[first] = "\t".join(cols)
    f.write_text("\n".join(lines))
    with pytest.raises(AssertionError, match=r"dt \d+ ns > "):
        A.load_registered_inputs(f, world["clips"], world["objects"], world["root"], registered=dict(reg, frames_sha=sha(f)))


def test_P9_clip_must_be_strictly_ascending(world, tmp_path):
    reg = world["registered"]
    t = tmp_path / "clips_unsorted.tsv"
    lines = open(world["clips"]).read().split("\n")
    row = lines[0].split("\t"); ts = row[3].split(","); ts[1], ts[2] = ts[2], ts[1]; row[3] = ",".join(ts)       # indices 0 and 10 (the scored ones) untouched: only sortedness is wrong
    t.write_text("\t".join(row) + "\n" + "\n".join(lines[1:]))
    with pytest.raises(AssertionError, match="ascending"):
        A.load_registered_inputs(world["frames"], t, world["objects"], world["root"], registered=dict(reg, clips_sha=sha(t)))


def test_P17_decode_sanity_gt_range_is_checked(world, monkeypatch):
    import decode_sanity as DSN
    row = A.read_frames_tsv(world["frames"])[0]
    orig = A.Log.sparse_gt
    for label, fn in (("above 200", lambda g: np.where(np.isfinite(g), g + 250.0, g).astype(np.float32)), ("zero", lambda g: np.where(np.isfinite(g), 0.0, g).astype(np.float32))):
        monkeypatch.setattr(A.Log, "sparse_gt", lambda self, cam, lid, fn=fn: fn(orig(self, cam, lid)))
        with pytest.raises(AssertionError, match="GT valid-range"):
            DSN.check_frame(world["root"], row[0], row[1], row[2])
    monkeypatch.setattr(A.Log, "sparse_gt", orig)
    DSN.check_frame(world["root"], row[0], row[1], row[2])


def test_decode_sanity_logs_are_the_first_five_of_the_registered_list(world, tmp_path):
    import decode_sanity as DSN
    reg_logs = AV2_DIR / "av2_val_logs_selected50.txt"
    sel = [l.strip() for l in open(reg_logs) if l.strip()]
    assert sel == sorted(sel) and sel == [l.strip() for l in open(AV2_DIR / "av2_val_logs_stage1_50.txt") if l.strip()]
    out = DSN.main(world["root"], world["frames"], 5, world["logs"])                                          # synthetic world: logs.txt == its two sorted logs
    assert [o["log"] for o in out] == SY.LOGS
    wrong = tmp_path / "other_logs.txt"; wrong.write_text("zzz\nlogB\n")
    with pytest.raises(AssertionError, match="registered list"):
        DSN.main(world["root"], world["frames"], 5, wrong)
    with pytest.raises(SystemExit):
        DSN.main(world["root"], world["frames"], 5, None)


def test_P18_builder_asserts_native_image_shape(world, monkeypatch, tmp_path):
    monkeypatch.setattr(A, "load_rgb", lambda p: np.zeros((100, 100, 3), np.uint8))
    out = tmp_path / "m.npz"
    with pytest.raises(AssertionError, match="!= native"):
        B.main(["--root", str(world["root"]), "--frames", str(world["frames"]), "--clips", str(world["clips"]), "--objects", str(world["objects"]), "--out", str(out),
                "--seg-a", str(world["segA"]), "--seg-b", str(world["segB"])])
    assert not out.exists()


def test_P19_builder_main_verifies_segmenter_weights_before_building(world, tmp_path):
    badA = tmp_path / "badA"; badA.mkdir(); (badA / "pytorch_model.bin").write_bytes(b"tampered weights")
    out = tmp_path / "m.npz"
    with pytest.raises(RuntimeError, match="sha256"):
        B.main(["--root", str(world["root"]), "--frames", str(world["frames"]), "--clips", str(world["clips"]), "--objects", str(world["objects"]), "--out", str(out),
                "--seg-a", str(badA), "--seg-b", str(world["segB"])])
    assert not out.exists() and not out.with_suffix(".meta.json").exists()
