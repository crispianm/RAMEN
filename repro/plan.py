"""Unit planner: one unit = model x dataset x cell x split (x kind). Pure logic, no GPU, no torch."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from . import config as C
from .util import REPO, item_stem, slug

HEADLINE = REPO / "scripts" / "dn01_headline.py"
HELDOUT = REPO / "scripts" / "dn01_heldout_eval.py"
FNOISE = REPO / "scripts" / "dn01_bvi_fnoise.py"
VENDORED = REPO / "repro" / "vendored"
REGISTRATION = REPO / "provenance" / "dn01_fnoise" / "registration.json"
NIGHT_SET, NIGHT_CELL = "nuscenes_night_test", "real"
FNOISE_SET, FNOISE_CELL = "bvi_rlv_test", "fnoise"

# rough GPU minutes per clip on one GH200/4090 (dn01 logs); only used for --dry-run estimates, never for decisions.
MIN_PER_CLIP = {"bench": 0.6, "night": 0.4, "fnoise": 0.3, "latency": 0.0}
MIN_PER_UNIT_FIXED = {"latency": 2.0}
SLOW = (("vitl", 3.0), ("metric_vda_l", 3.0), ("vda_l", 3.0), ("fastdvdnet", 1.6), ("pocketdvdnet", 1.6))
ENV = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
NHR_PER_GPU_H = 0.25                                      # Isambard ledger observed rate per GPU-hour (1 GPU)
SMOKE_MIN = 1.0                                           # heuristic minutes per unit in --smoke (1 clip + model load)


@dataclass
class Unit:
    kind: str
    model_id: str
    spec: str
    dataset: str
    cell: str
    split: str
    temporal: bool
    items: list = field(default_factory=list)       # item ids sharing this unit
    derive_from: dict | None = None                 # {base, drop}: no GPU
    n_clips: int | None = None
    est_min: float = 0.0
    host: str = "gh200"                             # gh200 | 4090 | any
    runner: str | None = None                       # None | "fidelity_init" (vendored script, restorer-only PSNR/SSIM)
    arm: dict | None = None                         # {name, coupling, stage2}: vendored bench_overrides.py eval-time knobs
    tag_backend: str = "reference"                  # reference | fast
    extra: dict = field(default_factory=dict)       # runner inputs (ckpt path/sha) resolved at plan time
    base_spec: str = ""                             # spec WITHOUT the TAG token (what --grid needs)
    variant: str = "none"                           # "none" | TAG token: which depth-post variant this unit's row is
    grid: list = field(default_factory=list)        # non-empty on the unit that EXECUTES a grid pass: --model base_spec --grid <variants>
    group: list = field(default_factory=list, compare=False, repr=False)   # the other units whose result files this pass also writes
    run_with: tuple | None = None                   # key of the unit that produces this unit's file (this unit is not executed itself)
    harness: dict | None = None                     # latency ladder unit: {stack, rows, warmup, iters, script, ...} (one unit = one single-session ladder run of that stack)

    @property
    def row_id(self) -> str:
        return self.model_id if not self.arm else f"{self.model_id}@{self.arm['name']}"

    @property
    def key(self) -> tuple:
        return (self.kind, self.spec, self.dataset, self.cell, self.split, self.temporal, self.arm['name'] if self.arm else None, self.runner, self.tag_backend)

    @property
    def slug(self) -> str:
        return slug(self.spec)

    def rel_dir(self) -> str:
        if self.harness:
            return f"per_clip/latency_ladder_{self.harness['stack']}"
        if self.runner:
            return f"per_clip/{self.runner}"
        base = slug(self.base_spec) if (self.kind == "bench" and self.base_spec) else self.slug      # bench grid passes write raw + TAG files into ONE dir
        return f"per_clip/{base}" + (f"/{self.arm['name']}" if self.arm else "")

    def all_paths(self, out: Path) -> list[Path]:
        return [self.out_path(out)] + [m.out_path(out) for m in self.group]

    def filename(self) -> str:
        s = self.slug
        if self.harness:
            return "LADDER_SUMMARY.json"
        if self.runner == "fidelity_init":
            return f"restorer_temp2_init__{self.dataset}__{self.cell}__{self.split}__v3.json"
        if self.kind == "bench":
            return f"{s}__{self.dataset}__{self.cell}__{self.split}__v3.json"
        if self.kind == "night":
            return f"{s}__{NIGHT_SET}__{NIGHT_CELL}__test.json"
        if self.kind == "fnoise":
            return f"{s}__{FNOISE_SET}__fnoise.json"
        return f"latency__{s}.json"

    def out_path(self, out: Path) -> Path:
        return Path(out) / self.rel_dir() / self.filename()

    def label(self) -> str:
        return f"{self.row_id}/{self.dataset}/{self.cell}/{self.split}"

    def cwd(self, scorer_clone: Path | None) -> Path:
        """Vendored wrapper scripts import bench_joint from their cwd / --bench-root: run them in the pristine scorer-commit clone; headline units run in this repo (scorer files byte-identical)."""
        return Path(scorer_clone) if (scorer_clone and (self.arm or self.runner)) else REPO

    def command(self, out: Path, py: str = sys.executable, smoke: bool = False, allow_test: bool = False, bvi_root: str | None = None,
                scorer_clone: Path | None = None) -> list[str] | None:
        d = str(Path(out) / self.rel_dir())
        lim = ["--limit-clips", "1"] if smoke else []
        if self.runner == "fidelity_init":
            return [py, str(VENDORED / "fidelity_init_score.py"), "--mode", "init", "--restorer-ckpt", self.extra["ckpt_path"], "--expect-sha", self.extra["ckpt_sha256"],
                    "--dataset", self.dataset, "--split", self.split, *(["--allow-test"] if self.split != "dev" else []), "--device", "cuda",
                    "--bench-root", str(scorer_clone or REPO), "--out", d, *lim]
        if self.arm:
            return [py, str(VENDORED / "bench_overrides.py"), "--coupling", repr(float(self.arm["coupling"])), "--stage2", repr(float(self.arm["stage2"])), "--",
                    "run", "--model", self.base_spec or self.spec, "--dataset", self.dataset, "--cells", self.cell, "--split", self.split, *(["--allow-test"] if self.split != "dev" else []),
                    *(["--temporal"] if self.temporal else []), "--grid", *(self.grid or ["none"]), "--diag", "--device", "cuda", "--out", d, *lim]
        if self.kind == "bench":
            if self.derive_from:
                return None
            return [py, str(HEADLINE), "--bj", "run", "--model", self.base_spec if self.grid else self.spec, "--dataset", self.dataset, "--cells", self.cell, "--split", self.split,
                    *(["--allow-test"] if self.split != "dev" else []), *(["--temporal"] if self.temporal else []), *(["--grid", *self.grid] if self.grid else []),
                    "--device", "cuda", "--out", d, *lim]
        if self.kind == "night":
            return [py, str(HELDOUT), "run", "--model", self.spec, "--dataset", NIGHT_SET, "--cells", NIGHT_CELL, "--split", "test", "--allow-test",
                    *(["--temporal"] if self.temporal else []), "--device", "cuda", "--out", d, *lim]
        if self.kind == "fnoise":
            return [py, str(FNOISE), "run", "--model", self.spec, "--root", bvi_root or "${BVI_ROOT}", "--registration", str(REGISTRATION), "--device", "cuda", "--out", d, *lim]
        if self.kind == "latency":
            h = self.harness
            warm, iters = (1, 3) if smoke else (h["warmup"], h["iters"])           # smoke numbers are plumbing, never a latency measurement
            lock = os.environ.get("RAMEN_GPU_LOCK") or str(Path(out) / "GPU_LOCK")
            return [os.environ.get("RAMEN_LADDER_PYTHON") or py, str(REPO / h["script"]), "--stacks", h["stack"], "--skip-standalone", "--out", d, "--lock", lock,
                    "--warmup", str(warm), "--iters", str(iters)]
        raise ValueError(self.kind)


SMOKE_FRAMES = 110                                         # bench_joint caps every clip at <=110 frames (DATASETS max_len); 1 clip per dataset in smoke


def latency_basis(cfg: dict) -> dict:
    """Per-frame model ms from the RELEASED 4090 latency read (tab:latency reference): {ms_per_frame, source}; falls back to flat SMOKE_MIN per unit.
    Reads LATENCY_READ.json (C3_LATENCY layout) or the ladder session layout (rows/S0__*.json, window 1)."""
    import json
    try:
        ref = C.reference_dir(cfg, "tab:latency", next(iter(cfg["items"]["tab:latency"]["models"])))
        if (ref / "LATENCY_READ.json").exists():
            j = json.loads((ref / "LATENCY_READ.json").read_text())
            ms = sorted(r["per_frame_ms_p50"] for r in j["rows"].values() if isinstance(r, dict) and "per_frame_ms_p50" in r)
            return {"ms_per_frame": ms[-1], "source": f"max per_frame_ms_p50 over {len(ms)} released rows of {ref / 'LATENCY_READ.json'} (RTX 4090, fp16)"}
        ms = sorted(json.loads(f.read_text())["window1"]["ms_per_frame_p50"] for f in (ref / "rows").glob("S0__*.json"))
        return {"ms_per_frame": ms[-1], "source": f"max window-1 ms/frame over {len(ms)} S0 rows of the released ladder session {ref} (RTX 4090, fp16)"}
    except Exception as e:  # noqa: BLE001
        return {"ms_per_frame": None, "source": f"fallback flat {SMOKE_MIN} min/unit ({type(e).__name__})"}


def smoke_unit_min(unit: "Unit", basis: dict | None) -> float:
    if not basis or basis.get("ms_per_frame") is None:
        return SMOKE_MIN
    f = 1.0
    for tok, mult in SLOW:
        if tok in unit.spec:
            f = max(f, mult)
    return round((SMOKE_FRAMES * basis["ms_per_frame"] * f / 1000.0 + basis.get("overhead_s", 90.0)) / 60.0, 2)


def _est(unit: Unit, item: dict, smoke: bool = False, basis: dict | None = None) -> float:
    if unit.derive_from:
        return 0.0
    ovr = (item.get("est_gpu_min_per_unit") or {})
    if unit.dataset in ovr:
        return float(ovr[unit.dataset])
    if unit.kind == "latency":
        return MIN_PER_UNIT_FIXED["latency"]
    if smoke:
        return smoke_unit_min(unit, basis)
    f = 1.0
    for tok, mult in SLOW:
        if tok in unit.spec:
            f = max(f, mult)
    return round((unit.n_clips or 10) * MIN_PER_CLIP[unit.kind] * f, 1)


def efficiency_primary_file(cfg: dict, it: dict) -> Path | None:
    """Path of the condition file of a conditional item (env RAMEN_EFFICIENCY_PRIMARY, else condition.file under DATA_ROOT; absolute and repo: forms accepted)."""
    c = it.get("condition")
    if not c:
        return None
    return Path(os.environ["RAMEN_EFFICIENCY_PRIMARY"]) if os.environ.get("RAMEN_EFFICIENCY_PRIMARY") else C.resolve_path(cfg, c["file"])


def condition_status(cfg: dict, iid: str) -> tuple[bool, str]:
    """(included, note). An item with `condition.file` is included iff that file exists (pre-declared; independent of any output of this run)."""
    it = cfg["items"][iid]
    f = efficiency_primary_file(cfg, it)
    if f is None:
        return True, ""
    if f.is_file():
        return True, f"included: {f.name} present"
    return False, f"SKIPPED {iid}: condition file {f} does not exist ({(it['condition'].get('note') or 'conditional item')})"


LADDER_STACKS = ("S0", "X", "S1", "S2", "S3", "S4", "C", "AZ")


def ladder_harness(cfg: dict, iid: str) -> dict:
    """The item's harness dict with the stack resolved (stack_from_file: first non-comment word of the condition file)."""
    it = cfg["items"][iid]
    h = dict(it["harness"])
    h.setdefault("script", "repro/efficiency/latency/ladder_samples.py")
    h.setdefault("warmup", 5)
    h.setdefault("iters", 30)
    if h.get("stack_from_file"):
        f = efficiency_primary_file(cfg, it)
        words = [w for ln in (f.read_text().splitlines() if f and f.is_file() else []) if not ln.lstrip().startswith("#") for w in ln.split()]
        stack = words[0] if words else None
        if stack not in LADDER_STACKS:
            raise C.ConfigError(f"{iid}: {f} must name a ladder stack {LADDER_STACKS} as its first word, got {stack!r}")
        h["stack"] = stack
    elif h["stack"] not in LADDER_STACKS:
        raise C.ConfigError(f"{iid}: unknown ladder stack {h['stack']!r}")
    return h


def unit_host(it: dict, smoke: bool, tag_backend: str) -> str:
    """Where a unit runs: the item's `host` (default gh200). Smoke and fast-TAG items go to the 4090; latency is always 4090."""
    h = str(it.get("host") or ("4090" if it["kind"] == "latency" else "gh200"))
    if smoke or tag_backend == "fast":
        return "4090" if h != "any" else "any"
    return h


def plan_item_units(cfg: dict, iid: str, smoke: bool = False, tag_backend_override: str | None = None, stats: dict | None = None, basis: dict | None = None) -> list[Unit]:
    it = cfg["items"][iid]
    kind = it["kind"]
    if kind == "qualitative":
        return []
    tb = C.tag_backend(cfg, iid, tag_backend_override)
    host = unit_host(it, smoke, tb)
    arms = [dict(name=n, **v) for n, v in (it.get("arms") or {}).items()] if it.get("override_script") else [None]
    sm = C.smoke_cfg(cfg) if smoke else None
    stats = stats if stats is not None else {}
    if smoke and arms != [None]:
        stats["dropped_arms"] = stats.get("dropped_arms", 0) + max(0, len(arms) - sm["arms_per_item"])
        arms = arms[:sm["arms_per_item"]]
    out = []
    if kind == "latency":
        h = ladder_harness(cfg, iid)
        mid = it["models"][0]
        u = Unit("latency", mid, f"ladder:{h['stack']}", "-", "-", "-", False, [iid], host=host, tag_backend=tb, harness=h)
        u.est_min = _est(u, it, smoke, basis)
        return [u]
    for mid in it["models"]:
        if smoke and mid not in sm["models"]:
            continue
        m = cfg["models"][mid]
        spec = C.model_spec(cfg, mid)
        base_spec, variant = C.model_spec(cfg, mid, with_tag=False), (cfg["tag"]["token"] if m.get("tag") else "none")
        runner = m.get("runner")
        if kind == "latency":
            continue                                               # latency = one ladder unit per item (below)
        split = it.get("split", "dev")
        temporal = bool(it.get("temporal", False))
        for ds in it["datasets"]:
            if smoke and ds not in sm["datasets"]:
                continue
            dcfg = cfg["datasets"][ds]
            cells = it["cells"] if kind == "bench" else [NIGHT_CELL if kind == "night" else FNOISE_CELL]
            for cell in cells:
                if smoke and kind == "bench" and cell not in sm["cells"]:
                    continue
                for arm in arms:
                    usplit = "test" if kind in ("night", "fnoise") else split
                    if smoke and usplit != "dev":                      # smoke is DEV-only: TEST-split units are dropped, never run
                        stats["dropped_test"] = stats.get("dropped_test", 0) + 1
                        continue
                    ex = {}
                    if runner:
                        ck = m["ckpt"]
                        ex = {"ckpt_path": str(C.resolve_path(cfg, ck["path"])), "ckpt_sha256": ck["sha256"]}
                    u = Unit(kind, mid, spec, ds, cell, "test" if kind in ("night", "fnoise") else split, temporal, [iid],
                             derive_from=dcfg.get("derive_from") if not (runner or arm) else None, n_clips=dcfg.get("n_clips"),
                             host=host, runner=runner, arm=arm, tag_backend=tb, extra=ex, base_spec=base_spec, variant=variant)
                    u.est_min = _est(u, it, smoke, basis)
                    out.append(u)
    return out


def normalise_only(cfg: dict, only: list[str] | None) -> list[str]:
    """--only accepts 'tab:static' or 'tab_static'; returns item ids in config order. Unknown -> ConfigError."""
    ids = list(cfg["items"])
    if not only:
        return ids
    by_stem = {item_stem(i): i for i in ids}
    sel = []
    for o in only:
        i = o if o in cfg["items"] else by_stem.get(item_stem(o))
        if i is None:
            raise C.ConfigError(f"--only {o!r}: no such item; known: {', '.join(ids)}")
        sel.append(i)
    return [i for i in ids if i in sel]


def plan(cfg: dict, only: list[str] | None = None, smoke: bool = False, tag_backend: str | None = None, stats: dict | None = None) -> tuple[list[Unit], list[str]]:
    """(deduplicated units in first-use order, selected item ids). Derived units depend on their base unit; the base is added if absent."""
    items = normalise_only(cfg, only)
    kept = []
    for iid in items:
        ok, note = condition_status(cfg, iid)
        if stats is not None and note:
            stats.setdefault("conditions", {})[iid] = note
        if ok:
            kept.append(iid)
    items = kept
    units: dict[tuple, Unit] = {}
    basis = None
    if smoke:
        basis = {**latency_basis(cfg), "overhead_s": C.smoke_cfg(cfg)["overhead_s"]}
        if stats is not None:
            stats["basis"] = basis
    for iid in items:
        for u in plan_item_units(cfg, iid, smoke, tag_backend, stats, basis):
            if u.key in units:
                if iid not in units[u.key].items:
                    units[u.key].items.append(iid)
                if units[u.key].host != u.host:
                    units[u.key].host = "any"                          # one run serves items planned for different hosts: either host may produce it
            else:
                units[u.key] = u
    # derived datasets need their base unit (same model/cell/split)
    for u in list(units.values()):
        if u.derive_from:
            bkey = (u.kind, u.spec, u.derive_from["base"], u.cell, u.split, u.temporal, None, None, u.tag_backend)
            if bkey not in units:
                b = Unit(u.kind, u.model_id, u.spec, u.derive_from["base"], u.cell, u.split, u.temporal, list(u.items),
                         n_clips=cfg["datasets"][u.derive_from["base"]].get("n_clips"), host=u.host, tag_backend=u.tag_backend)
                b.est_min = _est(b, cfg["items"][u.items[0]], smoke, basis)
                units[bkey] = b
    ordered = sorted(units.values(), key=lambda x: x.derive_from is not None)
    group_grid_passes(ordered)
    one_host_per_item(ordered, items)
    return ordered, items


def one_host_per_item(units: list[Unit], items: list[str]) -> None:
    """Run labels are per host and never mixed in one table: every item's units must be planned for ONE host (units shared with an item of another host are 'any')."""
    for iid in items:
        hs = {u.host for u in units if iid in u.items} - {"any"}
        if len(hs) > 1:
            raise C.ConfigError(f"{iid}: units planned for more than one host ({sorted(hs)}); a table is regenerated on one host")


def group_grid_passes(units: list[Unit]) -> None:
    """Raw + TAG rows of one base model/dataset/cell/split share ONE scoring pass (bench_joint --grid none <tag>: one result file per variant, named as a single-token run).
    --grid refuses a spec that already carries a depth-post token, so the pass runs the BASE spec. TAG-only units and every override-arm unit also go through --grid;
    a lone raw headline unit stays a plain run (as released). The first member (raw preferred) executes; the others point at it via run_with."""
    groups: dict[tuple, list[Unit]] = {}
    for u in units:
        if u.kind == "bench" and not u.derive_from and not u.runner and u.base_spec:
            groups.setdefault((u.base_spec, u.dataset, u.cell, u.split, u.temporal, u.arm["name"] if u.arm else None, u.tag_backend), []).append(u)
    for mem in groups.values():
        if len(mem) == 1 and mem[0].variant == "none" and not mem[0].arm:
            continue
        mem.sort(key=lambda x: x.variant != "none")
        rep = mem[0]
        rep.grid = [x.variant for x in mem]
        rep.group = mem[1:]
        rep.est_min = max(x.est_min for x in mem)
        for x in mem[1:]:
            x.run_with = rep.key
            x.est_min = 0.0


def qualitative_items(cfg: dict, items: list[str]) -> list[str]:
    return [i for i in items if cfg["items"][i]["kind"] == "qualitative"]


def needs_test(units: list[Unit]) -> list[Unit]:
    return [u for u in units if u.kind in ("night", "fnoise") or (u.kind == "bench" and u.split != "dev")]


def test_gate(units: list[Unit], allow_test: bool, env: dict) -> str | None:
    """None when allowed, else the refusal message. TEST reads need --allow-test AND an Auditor token (env RAMEN_AUDITOR_TOKEN non-empty, or RAMEN_AUDITOR_TOKEN_FILE existing)."""
    t = needs_test(units)
    if not t:
        return None
    tok = bool(env.get("RAMEN_AUDITOR_TOKEN")) or (bool(env.get("RAMEN_AUDITOR_TOKEN_FILE")) and Path(env["RAMEN_AUDITOR_TOKEN_FILE"]).is_file())
    if allow_test and tok:
        return None
    miss = [x for x, ok in (("--allow-test", allow_test), ("RAMEN_AUDITOR_TOKEN or RAMEN_AUDITOR_TOKEN_FILE", tok)) if not ok]
    return f"{len(t)} unit(s) read TEST clips (exposed exploratory); missing: {', '.join(miss)}"


def hours_by_host(units: list[Unit]) -> dict[str, float]:
    h: dict[str, float] = {}
    for u in units:
        h[u.host] = h.get(u.host, 0.0) + u.est_min / 60.0
    return h


def format_plan(units: list[Unit], items: list[str], cfg: dict, out: Path, smoke: bool = False, qual_present: dict | None = None, stats: dict | None = None) -> str:
    lines = [f"items: {', '.join(items)}", f"units: {len(units)} ({sum(1 for u in units if u.derive_from)} derived, no GPU){'  [SMOKE: 1 clip per dataset]' if smoke else ''}"]
    per_item: dict[str, float] = {}
    for u in units:
        for i in u.items:
            per_item[i] = per_item.get(i, 0.0) + u.est_min / max(1, len(u.items))
    lines += ["", f"{'item':<20}{'host':>7}{'was':>7}{'units':>6}{'est GPU min':>13}"]
    for i in items:
        n = sum(1 for u in units if i in u.items)
        hs = sorted({u.host for u in units if i in u.items}) or ["4090" if smoke and cfg["items"][i]["kind"] == "qualitative" else cfg["items"][i].get("host", "gh200")]
        prev = cfg["items"][i].get("previous_host") or "-"
        lines.append(f"{i:<20}{'/'.join(hs):>7}{prev:>7}{n:>6}{per_item.get(i, 0.0):>13.0f}")
    tot = sum(u.est_min for u in units)
    lines.append(f"{'TOTAL (deduplicated)':<20}{'':>7}{'':>7}{len(units):>6}{tot:>13.0f}")
    hh = hours_by_host(units)
    lines.append("GPU hours per host (heuristic, sequential, one GPU per host): " + ", ".join(f"{h}={v:.1f} h" for h, v in sorted(hh.items())))
    if not smoke:
        gh = hh.get("gh200", 0.0)
        moved = sum(u.est_min for u in units for i in u.items if cfg["items"][i].get("previous_host") not in (None, "gh200") and (u.host == "gh200")) / 60.0
        lines.append(f"Isambard NHR estimate: {gh:.1f} GPU-h x {NHR_PER_GPU_H} NHR/GPU-h (1 GPU; observed ledger rate, e.g. 1.45 NHR for a 5.7 h scoring job) = {gh * NHR_PER_GPU_H:.1f} NHR "
                     f"[planner minutes are heuristic; a 4-GPU node run would take ~1/4 of the wall time at the same GPU-hours]")
        lines.append(f"Addendum 30 delta vs the previous plan (items marked 'was' != host moved to the GH200): +{moved:.1f} GPU-h on the GH200 = +{moved * NHR_PER_GPU_H:.1f} NHR; the 4090 keeps inference timing only (tab:latency, tab:latency_eff, the bridge as a separate invocation).")
    if smoke:
        lines.append("SMOKE = plumbing only, on the 4090 (DEV, 1 clip per dataset): the outputs are not numbers and are never quoted. Real numbers come from the GH200 (Addendum 30).")
    stats = stats or {}
    if smoke:
        lines.append(f"smoke: DEV only. Dropped {stats.get('dropped_test', 0)} TEST-split unit(s) (not run, not planned) and {stats.get('dropped_arms', 0)} extra arm(s); "
                     f"models {C.smoke_cfg(cfg)['models']}, datasets {C.smoke_cfg(cfg)['datasets']}, cells {C.smoke_cfg(cfg)['cells']}, first arm only, 1 clip per dataset.")
        b = stats.get("basis") or {}
        lines.append(f"smoke time basis: unit = {SMOKE_FRAMES} frames x ms/frame [{b.get('source', 'n/a')}] x model factor + {b.get('overhead_s', 90)} s fixed (model load, data, scoring: ASSUMED, not measured); latency units 2 min.")
    q = qualitative_items(cfg, items)
    if q and smoke:
        lines.append(f"renderers add ~{C.smoke_cfg(cfg)['qual_min']:.0f} min each at their cheapest smoke setting on the 4090 (assumed): +{len(q) * C.smoke_cfg(cfg)['qual_min'] / 60:.2f} h not included in the totals above")
    if q:
        lines.append("renderers (repro.qualitative): " + ", ".join(f"{i}[{'present' if (qual_present or {}).get(i) else 'MISSING'}]" for i in q))
    lines.append("")
    for u in units:
        mark = "derive" if u.derive_from else ("timing" if u.kind == "latency" else ("TEST" if u.split != "dev" or u.kind in ("night", "fnoise") else "dev "))
        done = "done" if u.out_path(out).exists() else "todo"
        lines.append(f"  [{done}] {u.host:<5} {mark:<6} {u.kind:<8} {u.label():<58} {u.est_min:>6.1f} min  -> {u.rel_dir()}/")
    return "\n".join(lines)


def plan_json(units: list[Unit], items: list[str], cfg: dict, out: Path, smoke: bool, stats: dict, extra: dict) -> dict:
    """The plan as data (for the Auditor package): every unit with host, TEST/dev/timing label, estimate, expected output and command."""
    def label(u):
        return "derived" if u.derive_from else ("timing" if u.kind == "latency" else ("TEST" if u.split != "dev" or u.kind in ("night", "fnoise") else "dev"))
    return {"smoke": smoke, "items": items, "stats": stats, "hours_by_host": hours_by_host(units), "total_est_min": sum(u.est_min for u in units),
            "qualitative": {i: {"host": unit_host(cfg["items"][i], smoke, C.tag_backend(cfg, i)), "smoke_params": cfg["items"][i].get("smoke_params")} for i in qualitative_items(cfg, items)},
            "units": [{"items": u.items, "kind": u.kind, "row": u.row_id, "model": u.model_id, "spec": u.spec, "dataset": u.dataset, "cell": u.cell, "split": u.split, "label": label(u),
                       "host": u.host, "est_min": u.est_min, "tag_backend": u.tag_backend, "output": f"{u.rel_dir()}/{u.filename()}",
                       "command": u.command(out, smoke=smoke, allow_test=True, bvi_root="${BVI_ROOT}", scorer_clone=Path("${SCORER_CLONE}"))} for u in units], **extra}
