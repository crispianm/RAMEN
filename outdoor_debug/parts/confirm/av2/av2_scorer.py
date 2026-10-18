#!/usr/bin/env python3
"""AV2 Sensor VAL scorer for the STRICT confirmatory run TS2+H300 vs TS2 (AV2_REGISTRATION.md). DRAFT: the model/real-data path has NEVER run.

Per checkpoint (one JSON):
  PRIMARY statistic input (prediction only, no GT): for each scored frame, the median PREDICTED depth over CERTAIN-SKY pixels (frozen mask, nearest-resized to the model grid,
    frames with < 2000 certain-sky px at the model grid are dropped and counted); P = median over the kept frames (frame-balanced). Also: pooled median (description), frac of sky px > 200 / 300 m.
  GUARDS: unaligned pixel-pooled AbsRel and pooled median pred:gt in the bins [0,20) [20,50) [50,80) [80,120) [120,200] m over LiDAR-projected px with the ANY-SKY mask excluded
    (A1: prediction bilinearly resized to native 2048x1550 and sampled at the projected LiDAR px; GT never resized; no alignment, no clamp). Overall AbsRel 0-200.
Model input mirrors ddad_scorer (A2): bench_joint Resize lower_bound, multiple of 14, INTER_CUBIC, short side 378, fp16 autocast; portrait 2048x1550 -> 504x378 (Resize rounding).
Decision (Stage 1, Add. 48): per pair d_i = ln(P_H300 / P_TS2) for pairs s50/s51/s52; CONFIRMED iff all three d_i AND their mean >= ln 1.5 AND all guards pass in every pair; REFUTED iff any d_i <= 0 (Auditor O39); unevaluable guard never passes.
Inputs (Auditor O45): the model clip and the scored frames come ONLY from the pinned av2_clips.tsv + av2_frames.tsv (cross-checked against objects.tsv); the data dir is never globbed.
The scorer's OWN sha256 is pinned OUTSIDE the file: --expect-scorer-sha <literal from RUNBOOK_STAGE1.md> is REQUIRED and compared with the sha of the running file.
Usage:  python -I av2_scorer.py --ckpt <ours .pt> --pair s50|s51|s52|stage2 --arm TS2|H300 | --spec metric_vda_s  --out r.json --root <av2_val dir> --mask <sky_masks.npz>
            --frames <av2_frames.tsv> --clips <av2_clips.tsv> --objects <objects.tsv> --logs <logs.txt> --sums <$DEST/meta_R1/SHA256SUMS>
            --registry av2_stage1_ckpts.txt --expect-scorer-sha <sha256>   (use run_score.py: it adds --expect-scorer-sha from the runbook and --registry av2_stage1_ckpts.txt)
        python -I av2_scorer.py --compare H300.json TS2.json --registry av2_stage1_ckpts.txt --expect-scorer-sha <sha256> --run-shas av2_stage1_run_shas.txt [--out verdict.json]
        python -I av2_scorer.py --compare-pairs <6 jsons> --registry av2_stage1_ckpts.txt --expect-scorer-sha <sha256> --run-shas av2_stage1_run_shas.txt [--out verdict.json]
        python -I av2_scorer.py --selftest
"""
from __future__ import annotations

import argparse, json, math, re, socket, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "LR_DEV"))
sys.path.insert(0, str(HERE.parent.parent / "scorer"))
import av2_dataset as A            # noqa: E402

UNSET = "UNSET"
# ---- pins: filled in the registration BEFORE the first prediction. UNSET makes assert_pins fail (a scorer with unset pins cannot run on real data).
# The scorer's OWN sha is NOT in this dict (it cannot contain its own hash): it is passed from outside as --expect-scorer-sha (check_scorer_sha).
PINS = dict(logs_sha="f7f508f958e8e9d173772677d239c7a53165a0fb67520a68a4bfcf134592e370", **A.REGISTERED_SHAS,        # frames_sha, clips_sha, objects_sha
            mask_sha=UNSET, meta_sha=UNSET, sums_sha=UNSET,                                                      # fill after the mask build / download
            registry_sha=UNSET,                                    # D1: sha256 of av2_stage1_ckpts.txt; filled AFTER the six runs exist and the six CKPT_SHA lines are committed (RUNBOOK_STAGE1.md order)
            dataset_module_sha="52b3eb9c3d63479cb8efe5a61f15078fbbcb68b349c2771a2b2609f7a9fd31ae",      # av2_dataset.py (test_pins_equal_actual_file_shas)
            ddad_scorer_sha="8e71511b1192a1a15797a71482c40b547bac7d905641b7d2f145ad1da079a503",       # outdoor_debug/parts/LR_DEV/ddad_scorer.py
            builder_sha="718be61453e894356f3c10b3111af688840f5bdb1911cc0e2729cf1cad5ae2bc",            # build_av2_masks.py
            make_sky_mask_sha="31b18f51aed452f575019eee1ad463ee3b6d277ba9945ee505a1baaea16f29e1",       # seg/make_sky_mask.py
            segmenter_a_sha="a3dbf2d7a9912203078f304d88df8cfee9fd752ca2d2413f82036bdb02c5e7df",
            segmenter_b_sha="876855252aa674c7ad6e920c268d6722add811a927806372811faa724f19856c",
            bench_joint_sha="5d3d048ce7b26fb53e1e6a5aadf8265cb92c19dafc7340bc2e04cc6bcdf6e7ef")
# N4: the registered mask RULE (AV2_REGISTRATION.md R1; literal copy, NOT imported from make_sky_mask.py): the mask meta's `rule` must equal this dict exactly.
REGISTERED_RULE = dict(prob_thresh=0.9, erode_ref_px=8, erode_ref_short=900, lidar_ref_px=5, drop_frac=0.01, any_p=0.5, iou_min=0.85, work_short=512, work_mult=32, seed=0,
                       ade_sky=["sky", 2], city_sky=["sky", 10])
# N1: the declared Stage 1 training facts per pair label (outdoor_debug/parts/confirm/stage1: arms_stage1.py VALUES, SUBMIT.md). The checkpoint's OWN cfg must match them.
START_N = dict(s50="46", s51="47", s52="48")
START_FMT = "/projects/b5dh/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n{n}/step12000.pt"
HINGE_W, HINGE_MIN = 0.1, 300.0
SKY_VDA_ZD = True                    # D2: declared in all six stage1 yamls AND in the c3_n46/47/48 start checkpoints (VideoDepthLoss sees the sky at disparity 0 in BOTH arms); must be identical in both arms
# D2: the ONLY cfg paths in which the two arms of a pair may differ (recursive key diff of the stored ck['cfg']): the hinge keys and the checkpoint dir.
ALLOWED_PAIR_DIFF = frozenset({"loss.sky_hinge_weight", "loss.sky_hinge_min_depth", "training.checkpoint_dir"})
EXPECT_STEP = 14000                  # registration 'Pins': checkpoint step 14000 (read from the checkpoint file)
HOST_GPU = "GH200"                   # registration 'Pins': host type GH200 (substring of the CUDA device name)
IOU_MIN = 0.85                       # R1 M2: the mask meta's pooled segmenter IoU must be >= this
BINS_M = A.BINS_M
MAX_M = A.MAX_DEPTH_M
MIN_SKY_PX = 2000                    # certain-sky px at the model grid for a frame to count in P
LN_MARGIN = math.log(1.5)
GUARD_TOL = 0.05                     # H300 AbsRel <= (1 + 0.05) * same-seed TS2 AbsRel, per bin
GUARD_MIN_PX = 20000                 # a bin with fewer pooled px in either arm is reported "UNEVALUABLE" (not silently passed): the verdict says so
PAIRS = ("s50", "s51", "s52")           # Stage 1 pairs: c3_n46/seed 50, c3_n47/seed 51, c3_n48/seed 52 (Add. 48)
PAIR_LABELS = PAIRS + ("stage2",)        # 'stage2' = the single from-scratch seed-50 pair of Stage 2
SEED_OF = dict(s50=50, s51=51, s52=52, stage2=50)
ARMS = ("H300", "TS2")
# Registered disclosure strings, one per stage (AV2_REGISTRATION.md 'Stage structure' and 'Metrics'); compare()/compare_pairs() print the matching one.
STAGE1_NOTE = ("Stage 1: THREE paired 2000-step continuations (n46/s50, n47/s51, n48/s52); 3 pairs size the pair-to-pair spread only roughly; the A0/A0b screen pair showed 26% spread on a guard. "
               "P measures what the H300 hinge enforces by design (a PASS shows transfer of the hinge's effect, not accuracy; the guards carry the information). "
               "TS2+H300 was chosen after the screen on a descriptive sky secondary and was NOT BETTER on its primary (the screen primary).")
STAGE2_NOTE = ("SINGLE SEED (50): one paired difference cannot size seed noise; the A0/A0b screen pair showed 26% spread on a guard. Sign and magnitude only. "
               "Framing (O35): P measures what the hinge enforces by design; a pass is near built-in and is NOT an accuracy claim; the guards carry the information.")
LABEL_STAGE1 = "confirmatory STRICT, Stage 1 (AV2 sensor val, held out); three paired starts s50/s51/s52; primary = predicted sky depth by design, NOT accuracy"
LABEL_STAGE2 = "confirmatory STRICT, Stage 2 (AV2 sensor val, held out); single seed 50; primary = predicted sky depth by design, NOT accuracy"
REF_LABEL = "REFERENCE row, description only: released Metric-VDA-S at the registered 378 input, fp16; never enters the decision"


def bin_labels() -> list[str]:
    return [f"[{lo:g},{hi:g}{']' if i == len(BINS_M) - 1 else ')'}" for i, (lo, hi) in enumerate(BINS_M)]


def bin_masks(g: np.ndarray) -> list[np.ndarray]:
    return [(g >= lo) & ((g <= hi) if i == len(BINS_M) - 1 else (g < hi)) for i, (lo, hi) in enumerate(BINS_M)]


def assert_pins(actual: dict, registered: dict | None = None) -> dict:
    reg = PINS if registered is None else registered
    for k, want in reg.items():
        if want == UNSET:
            raise AssertionError(f"pin {k} is UNSET in the scorer: the registration has not been completed")
        got = actual.get(k)
        if got != want:
            raise AssertionError(f"pin {k}: {got} != registered {want}")
    return dict(actual)


def check_scorer_sha(expected: str | None, path=None) -> str:
    """D1: the running scorer file must hash to the sha passed from OUTSIDE (a literal committed in RUNBOOK_STAGE1.md). Absent / mismatch -> raises."""
    if not expected:
        raise SystemExit("--expect-scorer-sha is required (the literal sha256 of the final committed av2_scorer.py, from RUNBOOK_STAGE1.md)")
    got = A.sha256_file(path or __file__)
    if got != expected:
        raise AssertionError(f"scorer sha {got} != expected {expected}: refusing to run")
    return got


def assert_ckpt_facts(ck_meta: dict, pair: str) -> dict:
    """Registered checkpoint facts: step == 14000 (read from the checkpoint file by ddad_scorer.model_meta); the seed is carried by the pair label (s50/s51/s52 -> 50/51/52; stage2 -> 50).
    The label is cross-checked against the checkpoint's own cfg and the registered sha by assert_ckpt_cfg / assert_registered_ckpt (N1)."""
    if ck_meta.get("step") != EXPECT_STEP:
        raise AssertionError(f"checkpoint step {ck_meta.get('step')!r} != registered {EXPECT_STEP}")
    if pair not in SEED_OF:
        raise AssertionError(f"pair label {pair!r} not in {sorted(SEED_OF)}")
    return dict(ckpt_step=ck_meta["step"], expected_seed=SEED_OF[pair])


def _tartanair_ts2(cfg: dict):
    return [bool(s.get("ts2_sky")) for s in (cfg.get("data", {}) or {}).get("sources", []) if s.get("type") == "tartanair"]


def assert_ckpt_cfg(cfg, pair: str, arm: str) -> dict:
    """N1: a CLI label must never decide d. The checkpoint's own resolved cfg (train.py saves it as ck['cfg']) must match the declared Stage 1 facts of (pair, arm):
    H300: loss.sky_hinge_weight 0.1 AND sky_hinge_min_depth 300; TS2: weight 0 / absent and NO sky_hinge_min_depth; both: TS2 labels on the TartanAir source (ts2_sky), data.seed ==
    training.resume_reseed == pair seed (50/51/52), training.resume_from == the declared start checkpoint path of that pair (n46/n47/n48), max_steps 14000. Missing cfg / key -> raises."""
    if pair not in START_N:
        raise AssertionError(f"pair {pair!r}: only {sorted(START_N)} have declared training facts (Stage 2 needs a new scorer version)")
    if arm not in ARMS:
        raise AssertionError(f"arm {arm!r} not in {ARMS}")
    if not isinstance(cfg, dict) or not all(isinstance(cfg.get(k), dict) for k in ("loss", "training", "data")):
        raise AssertionError("checkpoint has no resolved cfg with loss/training/data sections: refused")
    loss, tr, data = cfg["loss"], cfg["training"], cfg["data"]
    seed = SEED_OF[pair]
    if arm == "H300":
        if loss.get("sky_hinge_weight") != HINGE_W or loss.get("sky_hinge_min_depth") != HINGE_MIN:
            raise AssertionError(f"arm H300: cfg sky_hinge_weight/min_depth {loss.get('sky_hinge_weight')!r}/{loss.get('sky_hinge_min_depth')!r} != {HINGE_W}/{HINGE_MIN}")
    else:
        if loss.get("sky_hinge_weight", 0) not in (0, 0.0) or "sky_hinge_min_depth" in loss:
            raise AssertionError(f"arm TS2: cfg carries the hinge (weight {loss.get('sky_hinge_weight')!r}, min_depth {loss.get('sky_hinge_min_depth')!r})")
    if loss.get("sky_hinge_max_depth") is not None:                      # D2: train.py:1231 reads it (upper side of a two-sided band); the declared design has none
        raise AssertionError(f"cfg loss.sky_hinge_max_depth {loss.get('sky_hinge_max_depth')!r} must be absent or null")
    if loss.get("sky_vda_zero_disparity") is not SKY_VDA_ZD:             # D2: the declared value (true in both arms); the pair diff additionally forbids it differing between the arms
        raise AssertionError(f"cfg loss.sky_vda_zero_disparity {loss.get('sky_vda_zero_disparity')!r} != declared {SKY_VDA_ZD}")
    if data.get("seed") != seed or tr.get("resume_reseed") != seed:
        raise AssertionError(f"pair {pair}: cfg data.seed/resume_reseed {data.get('seed')!r}/{tr.get('resume_reseed')!r} != {seed}")
    want_start = START_FMT.format(n=START_N[pair])
    if tr.get("resume_from") != want_start:
        raise AssertionError(f"pair {pair}: cfg resume_from {tr.get('resume_from')!r} != {want_start}")
    if tr.get("max_steps") != EXPECT_STEP:
        raise AssertionError(f"cfg max_steps {tr.get('max_steps')!r} != {EXPECT_STEP}")
    if _tartanair_ts2(cfg) != [True]:
        raise AssertionError(f"cfg: TartanAir ts2_sky must be true on exactly one tartanair source (got {_tartanair_ts2(cfg)})")
    return dict(cfg_checked=True, cfg_sky_hinge_weight=loss.get("sky_hinge_weight"), cfg_sky_hinge_min_depth=loss.get("sky_hinge_min_depth"), cfg_data_seed=data["seed"],
                cfg_resume_reseed=tr["resume_reseed"], cfg_resume_from=tr["resume_from"])


def load_ckpt_cfg(path):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg") if isinstance(ck, dict) else None
    del ck
    return cfg


def cfg_canonical(cfg) -> str:
    """D2: canonical json of the full stored ck['cfg'] (sorted keys, no spaces); raises on a non-json value (tuples become lists, as in the run json)."""
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"))


def cfg_sha256(cfg) -> str:
    import hashlib
    return hashlib.sha256(cfg_canonical(cfg).encode()).hexdigest()


def cfg_diff(a, b, prefix="") -> set:
    """D2: recursive key diff: the set of dotted paths at which a and b differ (a key present in only one side counts; lists and scalars are compared as whole values)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = set()
        for k in set(a) | set(b):
            path = f"{prefix}.{k}" if prefix else str(k)
            if k not in a or k not in b:
                out.add(path)
            else:
                out |= cfg_diff(a[k], b[k], path)
        return out
    return set() if (a == b and type(a) is type(b)) else {prefix}


def check_run_cfg(r: dict) -> dict:
    """D2/D3 at compare: the run json carries the full stored cfg; its hash matches meta.cfg_sha256; the cfg satisfies the declared facts of (meta.seed_label, meta.arm)
    (so a forged arm label contradicts the cfg's own hinge fields), and the cfg_* facts in the meta equal the ones re-derived from the cfg."""
    m = r["meta"]
    cfg = m.get("cfg")
    if cfg is None or not m.get("cfg_sha256"):
        raise AssertionError("run has no stored cfg / cfg_sha256 in its meta")
    if cfg_sha256(cfg) != m["cfg_sha256"]:
        raise AssertionError("meta.cfg does not hash to meta.cfg_sha256")
    facts = assert_ckpt_cfg(cfg, m.get("seed_label"), m.get("arm"))
    for k, v in facts.items():
        if m.get(k) != v:
            raise AssertionError(f"meta.{k} {m.get(k)!r} != {v!r} re-derived from the stored cfg")
    return facts


def check_pair_cfgs(h: dict, t: dict) -> set:
    """D2: the two cfgs of a pair differ ONLY in the declared hinge keys plus the checkpoint dir."""
    for r in (h, t):
        check_run_cfg(r)
    def norm(c):                                                  # absent == null for sky_hinge_max_depth (assert_ckpt_cfg already forbids a value)
        c = json.loads(json.dumps(c)); c["loss"].pop("sky_hinge_max_depth", None) if c["loss"].get("sky_hinge_max_depth") is None else None
        return c
    d = cfg_diff(norm(h["meta"]["cfg"]), norm(t["meta"]["cfg"]))
    if not d <= ALLOWED_PAIR_DIFF:
        raise AssertionError(f"the two arms' cfgs differ outside the declared hinge keys + checkpoint dir: {sorted(d - ALLOWED_PAIR_DIFF)}")
    return d


def check_registry_pin(reg_sha: str, pins=None) -> None:
    """D1: the registry file's sha must equal PINS.registry_sha (refused while UNSET); asserted at scoring (via assert_pins) AND at compare."""
    want = (PINS if pins is None else pins).get("registry_sha")
    if want in (None, UNSET):
        raise AssertionError("pin registry_sha is UNSET: the registry file av2_stage1_ckpts.txt has not been filled and pinned")
    if reg_sha != want:
        raise AssertionError(f"registry file sha {reg_sha} != pinned registry_sha {want}")


def check_run_provenance(r: dict, expect_scorer_sha: str, pins=None) -> None:
    """D4: every run was produced by THE pinned scorer file and under exactly the current PINS."""
    m = r["meta"]
    if m.get("scorer_sha") != expect_scorer_sha:
        raise AssertionError(f"run meta.scorer_sha {m.get('scorer_sha')} != --expect-scorer-sha {expect_scorer_sha}")
    reg = PINS if pins is None else pins
    p = m.get("pins")
    if not isinstance(p, dict) or set(p) != set(reg):
        raise AssertionError("run meta.pins is missing or has other keys than the current PINS")
    assert_pins(p, reg)                                   # refuses UNSET and any differing value


def write_run_sha(out_path) -> str:
    """D3: sha256 of a written run json -> `<out>.sha256` (sha256sum format) and stdout; the RUNBOOK commits the six lines (av2_stage1_run_shas.txt)."""
    out_path = Path(out_path)
    h = A.sha256_file(out_path)
    side = out_path.with_name(out_path.name + ".sha256")
    side.write_text(f"{h}  {out_path.name}\n")
    print(f"sha256 {h}  {out_path.name}", flush=True)
    return h


def load_run_shas(path) -> dict:
    if not path:
        raise SystemExit("--run-shas <file> is required (the committed sha256 of every run json, sha256sum format: `<64 hex>  <name>`)")
    out = {}
    for ln in Path(path).read_text().splitlines():
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        m = re.fullmatch(r"([0-9a-f]{64})\s+\*?(\S+)", ln.strip())
        if not m or m.group(2) in out:
            raise AssertionError(f"run-shas: malformed or duplicate line {ln!r}")
        out[m.group(2)] = m.group(1)
    return out


def check_run_shas(paths, listing: dict, exact: bool = False) -> None:
    """D3: every run json is listed in the committed shas file and hashes to the listed value (a json edited after it was written is refused); exact: the listing has no other names."""
    names = [Path(p).name for p in paths]
    if len(set(names)) != len(names):
        raise AssertionError("run json basenames are not unique")
    for p, n in zip(paths, names):
        if n not in listing:
            raise AssertionError(f"run json {n} is not in the committed run-shas file")
        if A.sha256_file(p) != listing[n]:
            raise AssertionError(f"run json {n}: sha256 {A.sha256_file(p)} != committed {listing[n]}")
    if exact and set(listing) != set(names):
        raise AssertionError(f"run-shas file lists {sorted(set(listing) ^ set(names))} beyond/besides the given runs")


REGISTRY_KEYS = [(p, a) for p in PAIRS for a in ("H300", "TS2")]


def parse_registry(text: str) -> dict:
    """N1 (Auditor): the registered Stage 1 checkpoint table in RUNBOOK_STAGE1.md: six lines `CKPT_SHA <pair> <arm> = <64 hex | UNSET>`, exactly once each."""
    found = re.findall(r"^CKPT_SHA\s+(\S+)\s+(\S+)\s*=\s*(\S+)\s*$", text, re.M)
    reg = {}
    for p, a, v in found:
        if (p, a) not in REGISTRY_KEYS or (p, a) in reg:
            raise AssertionError(f"registry: unexpected or duplicate line CKPT_SHA {p} {a}")
        if v != UNSET and not re.fullmatch(r"[0-9a-f]{64}", v):
            raise AssertionError(f"registry: CKPT_SHA {p} {a} = {v!r} is neither UNSET nor 64 lowercase hex")
        reg[(p, a)] = v
    if sorted(reg) != sorted(REGISTRY_KEYS):
        raise AssertionError(f"registry: need exactly the six lines {REGISTRY_KEYS}, found {sorted(reg)}")
    return reg


def assert_registered_ckpt(reg: dict, pair: str, arm: str, ckpt_sha: str) -> str:
    """The run's checkpoint sha must be THE registered one for (pair, arm); the registry must be complete (no UNSET) with six distinct shas."""
    unset = [k for k, v in reg.items() if v == UNSET]
    if unset:
        raise AssertionError(f"registered checkpoint list has UNSET entries {unset}: the Stage 1 runs have not been registered")
    if len(set(reg.values())) != len(reg):
        raise AssertionError("registered checkpoint list: the six shas are not distinct")
    if (pair, arm) not in reg:
        raise AssertionError(f"({pair}, {arm}) is not a registered Stage 1 run")
    if reg[(pair, arm)] != ckpt_sha:
        raise AssertionError(f"checkpoint sha {ckpt_sha} != registered {reg[(pair, arm)]} for ({pair}, {arm})")
    return reg[(pair, arm)]


def load_registry(path) -> tuple[dict, str]:
    if not path:
        raise SystemExit("--registry <RUNBOOK_STAGE1.md> is required (the registered checkpoint sha list)")
    return parse_registry(Path(path).read_text()), A.sha256_file(path)


def check_runs_against_registry(runs, reg: dict) -> None:
    """compare / compare-pairs: every run's ckpt sha is the registered one for its (pair label, arm)."""
    for r in runs:
        m = r["meta"]
        assert_registered_ckpt(reg, m.get("seed_label"), m.get("arm"), m.get("ckpt_sha"))


def check_mask_provenance(m: dict, rule=None) -> None:
    """N4: the mask meta's rule dict must equal the registered rule exactly, and it must record the shas of the three builder files (plus the commit if available)."""
    want = REGISTERED_RULE if rule is None else rule
    if m.get("rule") != want:
        raise AssertionError(f"mask meta rule {m.get('rule')} != registered {want}")
    for k in ("builder_sha", "make_sky_mask_sha", "dataset_module_sha"):
        if not isinstance(m.get(k), str) or len(m[k]) != 64:
            raise AssertionError(f"mask meta has no {k}: the mask was built by unrecorded code")


def assert_host(gpu_name: str) -> str:
    if HOST_GPU not in str(gpu_name):
        raise AssertionError(f"host GPU {gpu_name!r} does not contain {HOST_GPU!r} (registered host type)")
    return gpu_name


def gpu_name() -> str:
    import torch
    return torch.cuda.get_device_name()


def autocast_ctx(dev: str):
    """Registered precision: fp16 autocast on cuda ONLY. A non-cuda device raises (tests monkeypatch this function, not the check)."""
    import torch
    if not str(dev).startswith("cuda"):
        raise AssertionError("fp16 autocast requires cuda (registered precision)")
    return torch.autocast(device_type="cuda", dtype=torch.float16)


# ----------------------------------------------------------------------------- guards (LiDAR px, sky excluded)
class Accumulator:
    def __init__(self):
        n = len(BINS_M)
        self.n = [0] * n; self.s = [0.0] * n
        self.ratios = [[] for _ in range(n)]
        self.frames = [0] * n; self.logs = [set() for _ in range(n)]
        self.scored_frames = 0; self.logs_seen = set()
        self.n_sky_excluded = 0; self.n_lidar_valid_before_sky = 0; self.n_nonfinite_pred = 0

    def add(self, p, g, log, sky_at_px=None):
        """p, g: 1-D at the valid LiDAR px of one frame. sky_at_px: bool 1-D (any-sky) or None (variant that includes sky: for tests / description only)."""
        p = np.asarray(p, np.float64).ravel(); g = np.asarray(g, np.float64).ravel()
        assert p.shape == g.shape
        with np.errstate(invalid="ignore"):
            v = np.isfinite(g) & (g > 0) & (g <= MAX_M)
        self.n_lidar_valid_before_sky += int(v.sum())
        if sky_at_px is not None:
            sk = np.asarray(sky_at_px, bool).ravel(); assert sk.shape == g.shape
            self.n_sky_excluded += int((v & sk).sum())
            v = v & ~sk
        p, g = p[v], g[v]
        self.n_nonfinite_pred += int((~np.isfinite(p)).sum())
        self.scored_frames += 1; self.logs_seen.add(log)
        for i, m in enumerate(bin_masks(g)):
            if not m.any():
                continue
            pm, gm = p[m], g[m]
            self.n[i] += int(pm.size); self.s[i] += float((np.abs(pm - gm) / gm).sum())
            self.ratios[i].append(pm / gm); self.frames[i] += 1; self.logs[i].add(log)

    def result(self) -> dict:
        bins = {}
        for i, lab in enumerate(bin_labels()):
            r = np.concatenate(self.ratios[i]) if self.ratios[i] else np.zeros(0)
            bins[lab] = dict(n_px=self.n[i], absrel_unaligned_pooled=(self.s[i] / self.n[i]) if self.n[i] else float("nan"),
                             median_pred_over_gt=float(np.median(r)) if r.size else float("nan"), n_frames=self.frames[i], n_logs=len(self.logs[i]))
        tn = sum(self.n)
        return dict(bins=bins, overall_absrel_0_200=(sum(self.s) / tn) if tn else float("nan"), overall_n_px=tn, n_scored_frames=self.scored_frames,
                    n_logs=len(self.logs_seen), n_lidar_px_sky_excluded=self.n_sky_excluded, n_lidar_px_valid_before_sky=self.n_lidar_valid_before_sky,
                    n_nonfinite_pred_px=self.n_nonfinite_pred)


def sample_prediction(pred: np.ndarray, gt: np.ndarray):
    """A1: bilinear (align_corners=False) pred (h,w) metres -> native 2048x1550; return (pred_at_valid, gt_at_valid, valid_mask). GT never resized (shape asserted by the caller)."""
    import torch, torch.nn.functional as F
    m = np.isfinite(gt) & (gt > 0)
    p = torch.from_numpy(np.ascontiguousarray(pred, dtype=np.float32))[None, None]
    if tuple(p.shape[-2:]) != tuple(A.NATIVE_HW):
        p = F.interpolate(p, size=A.NATIVE_HW, mode="bilinear", align_corners=False)
    return p[0, 0].numpy()[m], gt[m], m


def score_frame_guard(acc: Accumulator, pred, gt, log, any_sky=None):
    if tuple(gt.shape) != tuple(A.NATIVE_HW):
        raise ValueError(f"GT shape {gt.shape} != native {A.NATIVE_HW}: the registered rule forbids resizing GT")
    p, g, m = sample_prediction(pred, gt)
    sk = None
    if any_sky is not None:
        if tuple(any_sky.shape) != tuple(A.NATIVE_HW):
            raise ValueError("any-sky mask must be at native resolution")
        sk = any_sky[m]
    acc.add(p, g, log, sk)


# ----------------------------------------------------------------------------- primary (prediction only)
def mask_to_grid(mask_native: np.ndarray, hw) -> np.ndarray:
    import cv2
    return cv2.resize(mask_native.astype(np.uint8), (int(hw[1]), int(hw[0])), interpolation=cv2.INTER_NEAREST).astype(bool)


def weighted_median(v: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(v, kind="stable"); v, w = v[o], w[o]
    c = np.cumsum(w)
    return float(v[int(np.searchsorted(c, 0.5 * c[-1]))])


class SkyAccumulator:
    def __init__(self):
        self.frame_medians = []; self.frame_n = []; self.dropped = 0; self.seen = 0
        self.kept_ids = []; self.dropped_ids = []; self.mask_px = []
        self.n_gt200 = 0; self.n_gt300 = 0; self.n_px = 0; self._samples = []; self._weights = []

    def add(self, pred: np.ndarray, certain_native: np.ndarray, fid=None):
        fid = fid if fid is not None else f"#{self.seen}"
        self.seen += 1
        m = mask_to_grid(certain_native, pred.shape[-2:])
        self.mask_px.append(int(m.sum()))
        v = np.asarray(pred, np.float64)[m]
        v = v[np.isfinite(v)]
        if v.size < MIN_SKY_PX:
            self.dropped += 1; self.dropped_ids.append(fid); return
        self.kept_ids.append(fid)
        self.frame_medians.append(float(np.median(v))); self.frame_n.append(int(v.size))
        self.n_px += v.size; self.n_gt200 += int((v > 200).sum()); self.n_gt300 += int((v > 300).sum())
        rng = np.random.default_rng(0)
        smp = v if v.size <= 20000 else rng.choice(v, 20000, replace=False)
        self._samples.append(smp); self._weights.append(np.full(smp.size, v.size / smp.size))     # weights restore the true pixel pooling

    def result(self) -> dict:
        ids = dict(kept_frame_ids=list(self.kept_ids), dropped_frame_ids=list(self.dropped_ids), frame_mask_px=list(self.mask_px))
        if not self.frame_medians:
            return dict(P_framebalanced_median=float("nan"), n_frames_kept=0, n_frames_dropped=self.dropped, n_frames_seen=self.seen, **ids)
        return dict(P_framebalanced_median=float(np.median(self.frame_medians)), n_frames_kept=len(self.frame_medians), n_frames_dropped=self.dropped,
                    n_frames_seen=self.seen, pooled_median_description=weighted_median(np.concatenate(self._samples), np.concatenate(self._weights)),
                    frac_sky_px_gt200=self.n_gt200 / self.n_px, frac_sky_px_gt300=self.n_gt300 / self.n_px, n_sky_px=int(self.n_px),
                    frame_medians=self.frame_medians, **ids)


def route_frame(acc: Accumulator, sky: SkyAccumulator, fid: str, pred, certain, any_sky, gt, log: str, mask_dropped) -> bool:
    """Per scored frame (R1 M4): a mask-dropped frame (< 1% certain-sky px, listed in the mask meta) is removed from the PRIMARY P ONLY (the sky accumulator never sees it);
    the guard accumulator ALWAYS sees every scored frame. Returns True iff the frame was mask-dropped."""
    dropped = is_mask_dropped(fid, mask_dropped)
    if not dropped:
        sky.add(pred, certain, fid)
    score_frame_guard(acc, pred, gt, log, any_sky)
    return dropped


# ----------------------------------------------------------------------------- decision (Auditor O36, single seed)
def decide(d: float, guards_pass: bool, unevaluable=()) -> str:
    if not np.isfinite(d):
        return "INVALID (non-finite d)"
    if d <= 0:
        return "REFUTED"
    if unevaluable:             # Auditor O37: an unevaluable guard NEVER passes by default (the far guard is the informative leg)
        return "NOT CONFIRMED (" + ", ".join(sorted(unevaluable)) + " guard unevaluable)"
    if d >= LN_MARGIN and guards_pass:
        return "CONFIRMED"
    return "NOT CONFIRMED"      # 0 < d < ln 1.5, or d >= ln 1.5 with a failed guard


def guard_check(h_bins: dict, t_bins: dict, tol: float = GUARD_TOL, min_px: int = GUARD_MIN_PX) -> dict:
    """Per bin: H300 AbsRel <= (1+tol) * TS2 AbsRel (same seed). Bins with < min_px in either arm -> status UNEVALUABLE (reported; the verdict line says so)."""
    want = bin_labels()
    for tag, b in (("H300", h_bins), ("TS2", t_bins)):             # D6: an empty / short / extra / renamed bin table would make all() vacuously True
        if not isinstance(b, dict) or list(b) != want:
            raise AssertionError(f"guard bins of arm {tag} {list(b) if isinstance(b, dict) else b!r} != the five registered labels {want}")
    out = {}
    for lab in h_bins:
        h, t = h_bins[lab], t_bins[lab]
        if h["n_px"] < min_px or t["n_px"] < min_px:
            out[lab] = dict(status="UNEVALUABLE", n_px_H300=h["n_px"], n_px_TS2=t["n_px"]); continue
        ratio = h["absrel_unaligned_pooled"] / t["absrel_unaligned_pooled"]
        out[lab] = dict(status="PASS" if ratio <= 1.0 + tol else "FAIL", absrel_ratio_H300_over_TS2=ratio, n_px_H300=h["n_px"], n_px_TS2=t["n_px"])
    return out


def decide_pairs(ds, guards_pass_all: bool, unevaluable=()) -> str:
    """Stage 1 (Add. 48): CONFIRMED iff ALL pair d_i >= ln 1.5 AND mean(d) >= ln 1.5 AND all guards pass in every pair.
    REFUTED iff ANY d_i <= 0 (Auditor O39). An unevaluable guard in any pair never passes (Auditor O37); else NOT CONFIRMED. min/max d_i = pair spread."""
    ds = list(ds)
    if len(ds) != len(PAIRS) or not all(np.isfinite(ds)):
        return "INVALID (need 3 finite pair d values)"
    mean = float(np.mean(ds))
    if min(ds) <= 0:                       # Auditor O39: any d_i <= 0 = REFUTED
        return "REFUTED"
    if unevaluable:
        return "NOT CONFIRMED (" + ", ".join(sorted(set(unevaluable))) + " guard unevaluable)"
    if min(ds) >= LN_MARGIN and mean >= LN_MARGIN and guards_pass_all:
        return "CONFIRMED"
    return "NOT CONFIRMED"


def check_pair_runs(h: dict, t: dict) -> None:
    """D11: argument order (first = H300, second = TS2), different checkpoints, one pair label / stage, and IDENTICAL kept/dropped frame ids, certain-sky px counts per frame,
    scored-frame and LiDAR px counts in both arms (the drop sets and the pixel sets must be the same: same mask file, same frames)."""
    for tag, r in (("first", h), ("second", t)):
        if "meta" not in r:
            raise AssertionError(f"{tag} run has no meta")
    if h["meta"].get("arm") != "H300" or t["meta"].get("arm") != "TS2":
        raise AssertionError(f"argument order: first must be arm H300, second arm TS2 (got {h['meta'].get('arm')!r}, {t['meta'].get('arm')!r})")
    if not h["meta"].get("ckpt_sha") or h["meta"]["ckpt_sha"] == t["meta"].get("ckpt_sha"):
        raise AssertionError("the two arms must use different checkpoints (ckpt sha equal or missing)")
    if h["meta"].get("seed_label") != t["meta"].get("seed_label") or h["meta"].get("stage") != t["meta"].get("stage"):
        raise AssertionError("a pair shares its pair label and stage")
    for k in ("pins", "scorer_sha"):                      # N2: both arms were scored with the same pinned inputs and the same scorer file
        if not h["meta"].get(k) or h["meta"].get(k) != t["meta"].get(k):
            raise AssertionError(f"arms differ in (or lack) meta.{k}: both runs must carry identical pins and scorer_sha")
    for k in ("kept_frame_ids", "dropped_frame_ids", "frame_mask_px"):
        if h["sky"].get(k) != t["sky"].get(k):
            raise AssertionError(f"arms differ in sky.{k}: the frame/mask sets must be identical across the two arms")
    for k in ("n_scored_frames", "n_lidar_px_valid_before_sky", "n_lidar_px_sky_excluded"):
        if h.get(k) != t.get(k):
            raise AssertionError(f"arms differ in {k}: {h.get(k)} vs {t.get(k)}")


def compare(h: dict, t: dict, stage: int) -> dict:
    """Per-pair comparison. `stage` selects the registered disclosure string (1 or 2)."""
    assert stage in (1, 2)
    check_pair_runs(h, t)
    d = math.log(h["sky"]["P_framebalanced_median"] / t["sky"]["P_framebalanced_median"])
    g = guard_check(h["bins"], t["bins"])
    ok = all(v["status"] == "PASS" for v in g.values())      # UNEVALUABLE is not a pass
    unev = [k for k, v in g.items() if v["status"] == "UNEVALUABLE"]
    return dict(label=LABEL_STAGE1 if stage == 1 else LABEL_STAGE2, stage=stage, disclosure=STAGE1_NOTE if stage == 1 else STAGE2_NOTE, d_ln_PH300_over_PTS2=d,
                ln_margin=LN_MARGIN, P_H300=h["sky"]["P_framebalanced_median"], P_TS2=t["sky"]["P_framebalanced_median"], guards=g, guards_pass=ok,
                guards_unevaluable=unev, verdict=decide(d, ok, unev),
                rule="CONFIRMED iff d>=ln1.5 AND all guards pass; 0<d<ln1.5 NOT CONFIRMED; d<=0 REFUTED")


def compare_pairs(pairs: dict) -> dict:
    """pairs: {'s50': (h_json, t_json), 's51': ..., 's52': ...}; per-pair compare() plus the Stage 1 rule over the three."""
    assert tuple(sorted(pairs)) == PAIRS, "need exactly the pairs s50, s51, s52"
    for k in PAIRS:
        for r in pairs[k]:
            assert r["meta"].get("stage") == 1 and r["meta"].get("seed_label") == k, f"pair {k}: runs must carry stage 1 and the pair label"
    allruns = [r for k in PAIRS for r in pairs[k]]
    for key in ("pins", "scorer_sha", "registry_sha"):          # N2: all six runs share pins, scorer file and registry
        vals = {json.dumps(r["meta"].get(key), sort_keys=True) for r in allruns}
        if len(vals) != 1 or "null" in vals:
            raise AssertionError(f"the six runs differ in (or lack) meta.{key}")
    for k in PAIRS:
        check_pair_cfgs(*pairs[k])                                 # D2/D3: stored cfgs agree with the arm labels and differ only in the declared keys
    per = {k: compare(*pairs[k], stage=1) for k in PAIRS}
    shas = [r["meta"].get("ckpt_sha") for r in allruns]
    if len(set(shas)) != 6 or not all(shas):
        raise AssertionError(f"Stage 1 needs 6 DISTINCT checkpoint shas (3 pairs x 2 arms); got {len(set(shas))} distinct of {len(shas)}")
    ds = [per[k]["d_ln_PH300_over_PTS2"] for k in PAIRS]
    unev = [f"{k}:{b}" for k in PAIRS for b in per[k]["guards_unevaluable"]]
    gp = all(per[k]["guards_pass"] for k in PAIRS)
    return dict(label=LABEL_STAGE1, stage=1, disclosure=STAGE1_NOTE, d_per_pair=dict(zip(PAIRS, ds)), d_mean=float(np.mean(ds)), d_min=float(np.min(ds)), ln_margin=LN_MARGIN,
                guards_pass_all=gp, guards_unevaluable=unev, per_pair=per, verdict=decide_pairs(ds, gp, unev),
                rule="CONFIRMED iff all 3 pair d >= ln1.5 AND mean d >= ln1.5 AND all guards pass in every pair; REFUTED iff any d_i <= 0; unevaluable guard never passes; else NOT CONFIRMED; min/max d_i = pair spread")


# ----------------------------------------------------------------------------- model path (NEVER run) [UNVERIFIED until data exists]
def model_input(paths):
    """Same as ddad_scorer.model_input (A2) with the AV2 jpg reader."""
    import cv2, torch
    import bench_joint as bj
    bj._vda_path()
    from video_depth_anything.util.transform import Resize
    rs = Resize(width=A.SHORT_SIDE, height=A.SHORT_SIDE, resize_target=False, keep_aspect_ratio=True, ensure_multiple_of=14,
                resize_method="lower_bound", image_interpolation_method=cv2.INTER_CUBIC)
    ims = [rs({"image": A.load_rgb(p).astype(np.float32) / 255.0})["image"] for p in paths]
    return torch.from_numpy(np.clip(np.stack(ims), 0, 1)).permute(0, 3, 1, 2).contiguous()


def check_mask_meta(m: dict) -> None:
    """The mask store's sidecar must say OK and carry a pooled segmenter IoU >= 0.85 (R1 M2); anything else refuses to score.
    bool, NaN, inf, non-numbers and values outside [0, 1] are rejected explicitly."""
    if m.get("status") != "OK":
        raise RuntimeError(f"mask meta status {m.get('status')!r} != 'OK'")
    iou = m.get("pooled_iou")
    if isinstance(iou, bool) or not isinstance(iou, (int, float)):
        raise RuntimeError(f"mask meta pooled_iou {iou!r} is not a number: mask set rejected")
    if math.isnan(iou):
        raise RuntimeError("mask meta pooled_iou is NaN: mask set rejected")
    if iou < 0.0 or iou > 1.0:                       # IoU lives in [0, 1] (also rejects +-inf)
        raise RuntimeError(f"mask meta pooled_iou {iou!r} outside [0, 1]: mask set rejected")
    if iou < IOU_MIN:
        raise RuntimeError(f"mask meta pooled_iou {iou!r} < {IOU_MIN}: mask set rejected")
    if not isinstance(m.get("dropped"), list):
        raise RuntimeError("mask meta has no dropped list")


def is_mask_dropped(fid: str, dropped: set) -> bool:
    return fid in dropped


def load_mask_meta(mask_path, frames_path) -> tuple[dict, str]:
    """Reads the sidecar next to the mask store, runs check_mask_meta, and asserts meta.frames_sha == sha256(frames tsv) (the mask was built for THESE frames).
    Returns (meta, sha256 of the sidecar file: the meta_sha pin)."""
    mp = Path(mask_path).with_suffix(".meta.json")
    mmeta = json.load(open(mp))
    check_mask_meta(mmeta)
    fs = A.sha256_file(frames_path)
    if mmeta.get("frames_sha") != fs:
        raise AssertionError(f"mask meta frames_sha {mmeta.get('frames_sha')} != frames tsv sha {fs}")
    return mmeta, A.sha256_file(mp)


def to_numpy(x) -> np.ndarray:
    """Model output (torch tensor, possibly cuda/half) -> float32 numpy."""
    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy()
    return np.asarray(x, np.float32)


def selftest() -> dict:
    """Synthetic frames only: pred = 0.9 * smooth field -> 0.9 / 0.1 in every bin; sky region predicted at 450 m -> P = 450."""
    H, W = A.NATIVE_HW
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    field = (1.0 + 199.0 * (xx / (W - 1))).astype(np.float32)
    rng = np.random.default_rng(0)
    acc = Accumulator(); sk = SkyAccumulator()
    sky = np.zeros((H, W), bool); sky[: H // 4] = True
    for log in ("a", "b"):
        gt = np.full((H, W), np.nan, np.float32); idx = rng.choice(H * W, 40000, replace=False); gt.flat[idx] = field.flat[idx]
        pred = 0.9 * field; pred[sky] = 450.0
        score_frame_guard(acc, pred, np.where(sky, np.nan, gt), log, sky)
        sk.add(pred[::4, ::4], sky)
    return dict(label="selftest (synthetic)", **acc.result(), sky=sk.result())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt"); ap.add_argument("--spec", default=None); ap.add_argument("--out")
    ap.add_argument("--root"); ap.add_argument("--mask"); ap.add_argument("--frames"); ap.add_argument("--clips"); ap.add_argument("--objects"); ap.add_argument("--logs")
    ap.add_argument("--expect-scorer-sha", help="REQUIRED for a scoring run: the literal sha256 of the final committed av2_scorer.py (from RUNBOOK_STAGE1.md)")
    ap.add_argument("--sums", help="REQUIRED for a scoring run: the download's SHA256SUMS file (= $DEST/meta_R1/SHA256SUMS), pinned as PINS.sums_sha")
    ap.add_argument("--registry", help="REQUIRED with --ckpt, --spec, --compare, --compare-pairs: av2_stage1_ckpts.txt (its CKPT_SHA table = the six registered Stage 1 checkpoint shas; its sha is PINS.registry_sha)")
    ap.add_argument("--run-shas", help="REQUIRED with --compare / --compare-pairs: the committed file listing the sha256 of every run json (`<64 hex>  <name>`, written as <out>.sha256 by each scoring run)")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[4]))
    ap.add_argument("--spec-prefix", default="ours:")
    ap.add_argument("--compare", nargs=2, metavar=("H300_JSON", "TS2_JSON"))
    ap.add_argument("--compare-pairs", nargs=6, metavar=("H300_s50", "TS2_s50", "H300_s51", "TS2_s51", "H300_s52", "TS2_s52"))
    ap.add_argument("--pair", choices=PAIR_LABELS, help="pair label of this run (required with --ckpt): s50 = c3_n46 start/seed 50, s51 = n47/51, s52 = n48/52, stage2 = Stage 2 seed 50")
    ap.add_argument("--arm", choices=ARMS, help="arm of this run (required with --ckpt): H300 = TS2+H300 (treatment), TS2 = control")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        print(json.dumps(selftest(), indent=1, default=float)); return
    if a.compare_pairs or a.compare:
        check_scorer_sha(a.expect_scorer_sha)                                  # N2: the verdict is also produced by the pinned scorer file
        reg, reg_sha = load_registry(a.registry)
        check_registry_pin(reg_sha)                                            # D1: the registry file is THE pinned one (refuses while PINS.registry_sha is UNSET)
        run_listing = load_run_shas(a.run_shas)                                # D3: required
    if a.compare_pairs:
        check_run_shas(a.compare_pairs, run_listing, exact=True)
        js = [json.load(open(p)) for p in a.compare_pairs]
        for r in js:
            check_run_provenance(r, a.expect_scorer_sha)                       # D4
        pairs = {}
        for i, k in enumerate(PAIRS):
            h, t = js[2 * i], js[2 * i + 1]
            for r in (h, t):
                assert r["meta"]["pins_asserted"] and r["meta"]["status"] == "FULL" and r["meta"]["seed_label"] == k, f"pair {k}: FULL pinned runs with the pair label"
            pairs[k] = (h, t)
        for r in js:
            assert r["meta"].get("registry_sha") == reg_sha, "run was scored against a different registry file than the one given now"
        check_runs_against_registry(js, reg)
        res = compare_pairs(pairs)
        print(json.dumps(res, indent=1, default=float))
        if a.out:
            assert not Path(a.out).exists(); Path(a.out).write_text(json.dumps(res, indent=1, default=float))
        return
    if a.compare:
        check_run_shas(a.compare, run_listing)
        h, t = (json.load(open(p)) for p in a.compare)
        for r in (h, t):
            check_run_provenance(r, a.expect_scorer_sha)                       # D4
            assert r["meta"]["pins_asserted"] and r["meta"]["status"] == "FULL", "compare needs FULL runs with pins asserted"
            assert r["meta"]["seed_label"] in PAIR_LABELS, "pair label"
        assert h["meta"]["seed_label"] == t["meta"]["seed_label"], "a pair shares its start and seed"
        assert h["meta"]["seed_label"] in START_N, "--compare is only defined for the Stage 1 pair labels here; Stage 2 needs a new scorer version (new registration + sha)"
        for r in (h, t):
            assert r["meta"].get("registry_sha") == reg_sha, "run was scored against a different registry file than the one given now"
        check_runs_against_registry([h, t], reg)
        check_pair_cfgs(h, t)                                                  # D2/D3
        res = compare(h, t, stage=2 if h["meta"]["seed_label"] == "stage2" else 1)
        print(json.dumps(res, indent=1, default=float))
        if a.out:
            assert not Path(a.out).exists(); Path(a.out).write_text(json.dumps(res, indent=1, default=float))
        return
    assert a.out and (a.ckpt or a.spec) and a.root and a.mask and a.frames and a.clips and a.objects and a.logs and a.sums, "--sums (the download's SHA256SUMS) and the other inputs are required"
    assert not (a.ckpt and a.spec)
    scorer_sha = check_scorer_sha(a.expect_scorer_sha)                       # D1: pinned OUTSIDE the file
    if a.ckpt:
        assert a.pair and a.arm, "--pair and --arm are required with --ckpt"
    else:
        assert not a.pair and not a.arm, "the reference row has no pair/arm"
    out_path = Path(a.out); assert not out_path.exists(), f"{out_path} exists (never overwrite)"
    import torch
    sys.path.insert(0, str(Path(a.repo).resolve()))
    import bench_joint as bj
    import ddad_scorer as DS
    ckreg, reg_sha = load_registry(a.registry)                                 # N1: parsed here, enforced right after the checkpoint is read; D1: its sha is PINS.registry_sha (assert_pins below)
    mmeta, meta_sha = load_mask_meta(a.mask, a.frames)                       # check_mask_meta + frames_sha == sha(frames tsv)
    check_mask_provenance(mmeta)                                             # N4: rule == registered RULE, builder shas recorded
    seg_dir = HERE.parent / "seg"
    on_disk = dict(builder_sha=A.sha256_file(HERE / "build_av2_masks.py"), make_sky_mask_sha=A.sha256_file(seg_dir / "make_sky_mask.py"), dataset_module_sha=A.sha256_file(A.__file__))
    for k, v in on_disk.items():
        if mmeta[k] != v:
            raise AssertionError(f"mask meta {k} {mmeta[k]} != sha of the file on disk {v}: the mask was built by different code than the one scoring")
    actual = dict(logs_sha=A.sha256_file(a.logs), frames_sha=A.sha256_file(a.frames), clips_sha=A.sha256_file(a.clips), objects_sha=A.sha256_file(a.objects),
                  mask_sha=A.sha256_file(a.mask), meta_sha=meta_sha, dataset_module_sha=A.sha256_file(A.__file__), ddad_scorer_sha=A.sha256_file(DS.__file__),
                  builder_sha=mmeta["builder_sha"], make_sky_mask_sha=mmeta["make_sky_mask_sha"],
                  bench_joint_sha=A.sha256_file(bj.__file__), segmenter_a_sha=mmeta["segmenter_a_sha"], segmenter_b_sha=mmeta["segmenter_b_sha"],
                  sums_sha=A.sha256_file(a.sums), registry_sha=reg_sha)
    pins = assert_pins(actual)
    reg = A.load_registered_inputs(a.frames, a.clips, a.objects, a.root, registered={k: PINS[k] for k in ("frames_sha", "clips_sha", "objects_sha")})
    bj_spec, mm = DS.model_meta(a.ckpt, a.spec, bj.PATHS)
    if a.ckpt:
        facts = dict(assert_ckpt_facts(mm["ckpt"], a.pair), arm=a.arm, ckpt_sha=mm["ckpt"]["sha256"])
        ck_cfg = load_ckpt_cfg(a.ckpt)
        facts.update(assert_ckpt_cfg(ck_cfg, a.pair, a.arm))                          # N1: the cfg stored IN the checkpoint decides, not the CLI labels
        facts.update(cfg=json.loads(cfg_canonical(ck_cfg)), cfg_sha256=cfg_sha256(ck_cfg))          # D2: the full stored cfg + its hash go into the run meta (compare diffs the pair's cfgs)
        assert_registered_ckpt(ckreg, a.pair, a.arm, facts["ckpt_sha"])               # N1: and the sha must be the registered one for this pair+arm
        facts["registry_sha"] = reg_sha
    else:
        facts = dict(arm="reference", ckpt_sha=None)
    gpu = assert_host(gpu_name())
    dev = a.device
    ctx = lambda: autocast_ctx(dev)                                            # raises on a non-cuda device (registered fp16)
    ctx()
    pipe = bj.build_pipeline(bj_spec or (a.spec_prefix + a.ckpt), dev); assert pipe["metric"]
    masks = A.MaskStore(a.mask)
    mask_dropped = set(mmeta["dropped"]); n_mask_dropped = 0
    acc, sky = Accumulator(), SkyAccumulator(); t0 = time.time(); dtypes = set()
    for log, d in reg.items():
        L = A.Log(a.root, log)
        x = model_input([L.image_path(c) for c in d["clip"]]).to(dev)
        with ctx():
            dep, _ = bj.run_pipeline(pipe, x, 0.0, 0.0, tuple(x.shape[-2:]))
        DS.assert_same_hw(dep, x)
        for (_, c, l, dt, i) in d["rows"]:
            fid = A.frame_id(log, c)
            cert, anys = masks.get(fid)
            n_mask_dropped += int(route_frame(acc, sky, fid, to_numpy(dep[i]), cert, anys, L.sparse_gt(c, l), log, mask_dropped))
        dtypes |= L.xyz_dtypes
        print(f"{log} {len(d['rows'])} frames {time.time() - t0:.0f}s", flush=True)
    res = acc.result()
    stage = 2 if a.pair == "stage2" else 1
    meta = dict(label=REF_LABEL if a.spec else (LABEL_STAGE1 if stage == 1 else LABEL_STAGE2), status="FULL", pins_asserted=True, pins=pins, scorer_sha=scorer_sha,
                seed_label=(a.pair if a.ckpt else "reference"), stage=(stage if a.ckpt else None), host_gpu=gpu, **facts, **mm,
                precision="fp16", short_side=A.SHORT_SIDE, resolution_rule="A1 bilinear pred->2048x1550 sampled at projected LiDAR px; GT never resized; sky = any-sky mask excluded",
                n_mask_dropped_frames=n_mask_dropped, mask_pooled_iou=mmeta["pooled_iou"], mask_n_candidates=mmeta.get("n_candidates"), mask_builder_shas={k: mmeta[k] for k in on_disk}, mask_git_commit=mmeta.get("git_commit"), mask_segmenters=dict(a=mmeta.get("segmenter_a"), b=mmeta.get("segmenter_b")),
                lidar_xyz_dtypes=sorted(dtypes | set(mmeta.get("lidar_xyz_dtypes", []))), host=socket.gethostname(), seconds=time.time() - t0, bins_m=bin_labels(),
                disclosure=(STAGE1_NOTE if stage == 1 else STAGE2_NOTE) if a.ckpt else REF_LABEL)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dict(meta=meta, sky=sky.result(), **res), indent=1, default=float))
    print("wrote", out_path)
    write_run_sha(out_path)                                                    # D3: commit this sha (RUNBOOK_STAGE1.md section 7)


if __name__ == "__main__":
    main()
