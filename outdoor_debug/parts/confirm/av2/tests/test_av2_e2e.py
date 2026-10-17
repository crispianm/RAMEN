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
                segmenter_a_sha=mm["segmenter_a_sha"], segmenter_b_sha=mm["segmenter_b_sha"], sums_sha=sha(w["sums"]))


def make_ckpt(path, arm, step=14000):
    import torch
    torch.save({"trainable_ema": {}, "step": step, "cfg": {"model": {"arch": "fake"}}, "tag": arm}, path)
    return path


def run_scorer(w, monkeypatch, arm, out, pair="s50", step=14000, pins=None, mask=None, expect_sha="auto", gpu="NVIDIA GH200 120GB", patch_autocast=True, extra=(), frames=None):
    sys.path.insert(0, str(REPO))
    import torch, bench_joint as bj
    ck = make_ckpt(w["tmp"] / f"ckpt_{arm}_{pair}_{step}.pt", arm, step)
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
            "--clips", str(w["clips"]), "--objects", str(w["objects"]), "--logs", str(w["logs"]), "--device", "cpu", "--repo", str(REPO), *extra]
    if expect_sha == "auto":
        argv += ["--expect-scorer-sha", sha(S.__file__)]
    elif expect_sha:
        argv += ["--expect-scorer-sha", expect_sha]
    S.main(argv)
    return json.load(open(out))


@pytest.fixture(scope="module")
def runs(world):
    mp = pytest.MonkeyPatch()
    try:
        r = {arm: run_scorer(world, mp, arm, world["tmp"] / f"{arm}.json") for arm in ("H300", "TS2")}
    finally:
        mp.undo()
    return r


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
    for key, val in (("kept_frame_ids", ["x"]), ("frame_mask_px", [1, 2, 3])):
        bad = _pairs(runs); bad["s52"][1]["sky"][key] = val
        with pytest.raises(AssertionError, match=key):
            S.compare_pairs(bad)
    bad = _pairs(runs); bad["s50"][1]["n_lidar_px_sky_excluded"] += 1
    with pytest.raises(AssertionError, match="n_lidar_px_sky_excluded"):
        S.compare_pairs(bad)
    bad = _pairs(runs); bad["s50"][0]["meta"]["stage"] = 2
    with pytest.raises(AssertionError):
        S.compare_pairs(bad)


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
        S.main(["--ckpt", "x.pt", "--out", str(tmp_path / "o.json"), "--root", "r", "--mask", "m", "--frames", "f", "--clips", "c", "--objects", "o", "--logs", "l",
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
    out = DSN.main(world["root"], world["frames"], 5)
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
