"""Loader + validator for configs/paper_results.yaml (schema: repro/SCHEMA.md). Pure python + pyyaml."""
from __future__ import annotations

import os
from pathlib import Path

from .util import REPO, expand

SCHEMA_VERSION = 1
KINDS = ("bench", "night", "fnoise", "latency", "qualitative")
SPLITS = ("dev", "test", "all")
TOP_KEYS = {"schema_version", "tag", "host_paths", "models", "datasets", "items", "scorer_commit", "tag_backend", "vendored", "smoke"}
RUNNERS = ("fidelity_init",)
OVERRIDE_SCRIPTS = ("bench_overrides",)
HOSTS = ("gh200", "4090", "any", "mac")
TAG_BACKENDS = ("reference", "fast")
DEFAULT_SCORER_COMMIT = "368b5c53b37ebd22f5d44ce0c9d514122efc88da"
MODEL_KEYS = {"label", "spec", "tag", "ckpt", "weights", "reference_spec", "note", "runner"}
CKPT_KEYS = {"path", "sha256", "download", "download_url"}
DATASET_KEYS = {"n_clips", "clip_list", "clip_list_sha256", "manifest_sha256", "llvar_draws_sha256", "derive_from", "note"}
ITEM_KEYS = {"kind", "title", "paper_ref", "models", "datasets", "cells", "split", "temporal", "metrics", "aggregation", "seeds",
             "reference", "label", "est_gpu_min_per_unit", "host_gpu_contains", "params", "note", "host", "arms", "override_script", "tag_backend", "seed_mean", "smoke_params", "harness", "condition", "reading", "host_slots", "previous_host", "verify_class", "by_hand_reference"}
AGG_KEYS = {"per_dataset", "mean_over", "tae_datasets"}
HEX = set("0123456789abcdef")
HARNESS_KEYS = {"runner", "stack", "stack_from_file", "script", "min_commit", "rows", "samples", "warmup", "iters", "protocol", "smoke"}
HARNESS_RUNNERS = ("ladder",)
CONDITION_KEYS = {"file", "note"}
READING_KINDS = ("clean_noninferiority", "eps_d_every_dataset")
FROZEN_MARK = "FROZEN (this commit)"


class ConfigError(ValueError):
    pass


def _is_hex(s, min_len=8, max_len=64) -> bool:
    return isinstance(s, str) and min_len <= len(s) <= max_len and set(s.lower()) <= HEX


def _unknown(where: str, d: dict, allowed: set, errs: list) -> None:
    for k in d:
        if k not in allowed:
            errs.append(f"{where}: unknown key {k!r}")


def validate(cfg: dict) -> list[str]:
    """Return a list of error strings (empty = valid)."""
    errs: list[str] = []
    if not isinstance(cfg, dict):
        return ["config is not a mapping"]
    _unknown("top", cfg, TOP_KEYS, errs)
    if cfg.get("schema_version") != SCHEMA_VERSION:
        errs.append(f"schema_version must be {SCHEMA_VERSION}, got {cfg.get('schema_version')!r}")
    tag = cfg.get("tag") or {}
    for k in ("alpha", "tau", "token"):
        if k not in tag:
            errs.append(f"tag.{k} missing")
    if "token" in tag and tag.get("alpha") is not None and tag.get("tau") is not None:
        want = f"agf{tag['alpha']}t{tag['tau']}"
        if tag["token"] != want:
            errs.append(f"tag.token {tag['token']!r} != agf<alpha>t<tau> = {want!r}")
    sm = cfg.get("smoke") or {}
    _unknown("smoke", sm, {"models", "datasets", "cells", "arms_per_item", "qual_min", "overhead_s"}, errs)
    for k in ("models", "datasets"):
        for x in sm.get(k, []):
            if x not in (cfg.get(k) or {}):
                errs.append(f"smoke.{k}: {x!r} not in {k}")
    hp = cfg.get("host_paths") or {}
    if hp.get("gh200") is not None or hp.get("gh200_local_only"):
        gm, lo, keys = hp.get("gh200") or {}, hp.get("gh200_local_only") or [], path_keys(cfg)
        errs += [f"host_paths.gh200: path key {k!r} is neither mapped nor in gh200_local_only (no silent local fallback on the GH200)" for k in sorted(keys) if k not in gm and k not in lo]
        errs += [f"host_paths.gh200: {k!r} is not a path key of the config" for k in gm if k not in keys]
        errs += [f"host_paths.gh200_local_only: {k!r} is not a path key of the config" for k in lo if k not in keys]
        errs += [f"host_paths.gh200[{k!r}] must be an absolute path (null = refused)" for k, v in gm.items() if v is not None and not str(v).startswith("/")]
    models, datasets, items = cfg.get("models") or {}, cfg.get("datasets") or {}, cfg.get("items") or {}
    if not models:
        errs.append("models: empty")
    if not items:
        errs.append("items: empty")
    for mid, m in models.items():
        w = f"models.{mid}"
        if not isinstance(m, dict):
            errs.append(f"{w}: not a mapping")
            continue
        _unknown(w, m, MODEL_KEYS, errs)
        if "spec" not in m:
            errs.append(f"{w}: spec missing")
            continue
        if "{ckpt}" in m["spec"]:
            ck = m.get("ckpt")
            if not isinstance(ck, dict) or not ck.get("path"):
                errs.append(f"{w}: spec uses {{ckpt}} but ckpt.path is missing")
            else:
                _unknown(f"{w}.ckpt", ck, CKPT_KEYS, errs)
                if not _is_hex(ck.get("sha256", "")):
                    errs.append(f"{w}.ckpt.sha256 must be hex (>=8 chars, 64 for a full pin)")
        if m.get("runner") is not None and m["runner"] not in RUNNERS:
            errs.append(f"{w}.runner must be one of {RUNNERS}")
        if m.get("runner") and not (m.get("ckpt") or {}).get("path"):
            errs.append(f"{w}: runner {m['runner']} needs ckpt.path")
        if "tag" in m and not isinstance(m["tag"], bool):
            errs.append(f"{w}.tag must be bool")
    for did, d in datasets.items():
        w = f"datasets.{did}"
        if not isinstance(d, dict):
            errs.append(f"{w}: not a mapping")
            continue
        _unknown(w, d, DATASET_KEYS, errs)
        df = d.get("derive_from")
        if df is not None:
            if not (isinstance(df, dict) and set(df) == {"base", "drop"}):
                errs.append(f"{w}.derive_from must be {{base, drop}}")
            elif df["base"] not in datasets:
                errs.append(f"{w}.derive_from.base {df['base']!r} not in datasets")
        for k in ("clip_list_sha256", "manifest_sha256", "llvar_draws_sha256"):
            if k in d and not _is_hex(d[k]):
                errs.append(f"{w}.{k} must be hex")
    for iid, it in items.items():
        w = f"items.{iid}"
        if not isinstance(it, dict):
            errs.append(f"{w}: not a mapping")
            continue
        _unknown(w, it, ITEM_KEYS, errs)
        kind = it.get("kind")
        if kind not in KINDS:
            errs.append(f"{w}.kind must be one of {KINDS}")
            continue
        for mid in it.get("models", []):
            if mid not in models:
                errs.append(f"{w}: model {mid!r} not in models")
        if kind != "qualitative" and not it.get("models"):
            errs.append(f"{w}: models empty")
        if kind in ("bench", "night", "fnoise"):
            for did in it.get("datasets", []):
                if did not in datasets:
                    errs.append(f"{w}: dataset {did!r} not in datasets")
            if not it.get("datasets"):
                errs.append(f"{w}: datasets empty")
            if kind == "bench" and not it.get("cells"):
                errs.append(f"{w}: cells empty")
            if it.get("split", "dev") not in SPLITS:
                errs.append(f"{w}.split must be one of {SPLITS}")
            if not it.get("metrics"):
                errs.append(f"{w}: metrics empty")
        if str(it.get("host", "gh200")) not in HOSTS:
            errs.append(f"{w}.host must be one of {HOSTS}")
        if it.get("tag_backend", "reference") not in TAG_BACKENDS:
            errs.append(f"{w}.tag_backend must be one of {TAG_BACKENDS}")
        ref = it.get("reference")
        if isinstance(ref, dict):
            if set(ref) - {"default", "by_model"}:
                errs.append(f"{w}.reference: only default / by_model allowed")
            for mid in (ref.get("by_model") or {}):
                if mid not in models:
                    errs.append(f"{w}.reference.by_model: {mid!r} not in models")
        elif ref is not None and not isinstance(ref, str):
            errs.append(f"{w}.reference must be a dir, null or {{default, by_model}}")
        if it.get("override_script") is not None:
            if it["override_script"] not in OVERRIDE_SCRIPTS:
                errs.append(f"{w}.override_script must be one of {OVERRIDE_SCRIPTS}")
            for an, av in (it.get("arms") or {}).items():
                if not isinstance(av, dict) or set(av) - {"coupling", "stage2"}:
                    errs.append(f"{w}.arms.{an}: keys must be coupling / stage2")
            if not it.get("arms"):
                errs.append(f"{w}: override_script needs arms")
        elif it.get("arms"):
            errs.append(f"{w}: arms needs override_script")
        for mid in it.get("models", []):
            if (models.get(mid) or {}).get("runner") == "fidelity_init" and list(it.get("cells", [])) != ["llvar"]:
                errs.append(f"{w}: runner fidelity_init (model {mid}) only scores cells == [llvar]")
        for grp in it.get("seed_mean") or []:
            if not (isinstance(grp, dict) and set(grp) == {"name", "members"}):
                errs.append(f"{w}.seed_mean entries need {{name, members}}")
                continue
            for mid in grp["members"]:
                if mid not in it.get("models", []):
                    errs.append(f"{w}.seed_mean {grp['name']!r}: member {mid!r} not in item models")
        errs += validate_item_extras(w, it, cfg)
        agg = it.get("aggregation") or {}
        _unknown(f"{w}.aggregation", agg, AGG_KEYS, errs)
        if agg.get("per_dataset", "macro") != "macro":
            errs.append(f"{w}.aggregation.per_dataset must be 'macro'")
        for k in ("mean_over", "tae_datasets"):
            for did in agg.get(k, []):
                if did not in (it.get("datasets") or []):
                    errs.append(f"{w}.aggregation.{k}: {did!r} not in item datasets")
    return errs


def validate_item_extras(w: str, it: dict, cfg: dict) -> list[str]:
    """harness (latency ladder), condition (conditional item), reading (pre-declared verdict rule)."""
    errs: list[str] = []
    h = it.get("harness")
    if h is not None:
        _unknown(f"{w}.harness", h, HARNESS_KEYS, errs)
        if it.get("kind") != "latency":
            errs.append(f"{w}: harness only on kind latency")
        if h.get("runner") not in HARNESS_RUNNERS:
            errs.append(f"{w}.harness.runner must be one of {HARNESS_RUNNERS}")
        if not (h.get("stack") or h.get("stack_from_file")):
            errs.append(f"{w}.harness needs stack or stack_from_file")
        if not isinstance(h.get("rows"), list) or not h["rows"]:
            errs.append(f"{w}.harness.rows must be a non-empty list (stack-less row ids)")
        for k in ("samples", "warmup", "iters"):
            if k in h and not (isinstance(h[k], int) and h[k] > 0):
                errs.append(f"{w}.harness.{k} must be a positive int")
        if h.get("samples") is not None and h.get("iters") is not None and h["samples"] != h["iters"]:
            errs.append(f"{w}.harness: samples ({h['samples']}) must equal iters ({h['iters']}): every timed pass is stored")
    elif it.get("kind") == "latency":
        errs.append(f"{w}: latency items run on the ladder harness (harness: ...); the bench_joint latency path is retired")
    for sk, sv in (it.get("host_slots") or {}).items():
        if sk != "mac" or not isinstance(sv, dict) or sv.get("authorised") is not False or sv.get("value") != "not measured" or not sv.get("label"):
            errs.append(f"{w}.host_slots.{sk}: only a 'mac' slot with {{label, value: 'not measured', authorised: false}} is allowed (no run authorised)")
    if it.get("previous_host") is not None and it["previous_host"] not in HOSTS:
        errs.append(f"{w}.previous_host must be one of {HOSTS}")
    if it.get("verify_class") not in (None, "cross_host"):
        errs.append(f"{w}.verify_class must be cross_host")
    bh = it.get("by_hand_reference")
    if bh is not None and (set(bh) - {"file", "sha256", "compare", "note"} or bh.get("compare") != "p1_table" or not _is_hex(bh.get("sha256", ""), 64, 64) or not bh.get("file")):
        errs.append(f"{w}.by_hand_reference needs {{file, sha256 (64 hex), compare: p1_table}}")
    cond = it.get("condition")
    if cond is not None:
        _unknown(f"{w}.condition", cond, CONDITION_KEYS, errs)
        if not cond.get("file"):
            errs.append(f"{w}.condition.file missing")
    rd = it.get("reading")
    if rd is not None:
        if rd.get("kind") not in READING_KINDS:
            errs.append(f"{w}.reading.kind must be one of {READING_KINDS}")
        elif rd["kind"] == "clean_noninferiority":
            for k in ("reference_model", "subjects", "datasets", "ratio_metrics", "abs_metrics", "rel_tol"):
                if k not in rd:
                    errs.append(f"{w}.reading.{k} missing")
            for mid in list(rd.get("subjects", [])) + [rd.get("reference_model")]:
                if mid not in (it.get("models") or []):
                    errs.append(f"{w}.reading: model {mid!r} not in item models")
        elif rd["kind"] == "eps_d_every_dataset":
            if rd.get("metric") not in (it.get("metrics") or []):
                errs.append(f"{w}.reading.metric must be one of the item metrics")
            if not rd.get("disclosure"):
                errs.append(f"{w}.reading.disclosure missing (the output must carry it)")
    return errs


def frozen_header_problems(text: str) -> list[str]:
    """The config's leading comment block must say FROZEN (this commit) and must not say DRAFT. The freeze IS the git commit: the run records HEAD and refuses an untracked/modified config."""
    head = []
    for ln in text.splitlines():
        if ln.startswith("#") or not ln.strip():
            head.append(ln)
        else:
            break
    h = "\n".join(head)
    errs = []
    if "DRAFT" in h:
        errs.append("config header says DRAFT")
    if FROZEN_MARK not in h:
        errs.append(f"config header lacks {FROZEN_MARK!r}")
    return errs


def load(path) -> dict:
    import yaml
    cfg = yaml.safe_load(Path(path).read_text())
    errs = validate(cfg)
    if errs:
        raise ConfigError(f"{path}: {len(errs)} schema error(s):\n  " + "\n  ".join(errs))
    return cfg


_SMI: list = []


def detected_host() -> str | None:
    """gh200 | 4090 | None. RAMEN_HOST overrides (tests); else nvidia-smi (cached)."""
    if os.environ.get("RAMEN_HOST"):
        return os.environ["RAMEN_HOST"]
    if not _SMI:
        import subprocess
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
            g = (r.stdout.strip().splitlines() or [""])[0].lower() if r.returncode == 0 else ""
        except (OSError, subprocess.TimeoutExpired):
            g = ""
        _SMI.append("4090" if "4090" in g else ("gh200" if "gh200" in g else None))
    return _SMI[0]


class UnmappedPath(ConfigError):
    pass


def data_root(cfg: dict) -> Path:
    """Local DATA_ROOT: env DATA_ROOT (or --data-root) > host_paths.DATA_ROOT > /mnt/DATA/DegradedDepth."""
    hp = cfg.get("host_paths") or {}
    return Path(os.environ.get("DATA_ROOT") or hp.get("DATA_ROOT") or "/mnt/DATA/DegradedDepth")


def gh200_map_active(cfg: dict) -> bool:
    """The explicit GH200 per-key map is used on a GH200 when the config has one and no DATA_ROOT override is given."""
    return bool((cfg.get("host_paths") or {}).get("gh200")) and not os.environ.get("DATA_ROOT") and detected_host() == "gh200"


def resolve_path(cfg: dict, p: str) -> Path:
    """Absolute path as is; 'repo:<rel>' = inside this clone; else a relative key: on a GH200 through host_paths.gh200 (unmapped / null = UnmappedPath, except gh200_local_only keys, which stay local), elsewhere relative to data_root(cfg)."""
    if p.startswith("repo:"):
        return REPO / p[5:]
    q = Path(expand(p, {**{k: str(v) for k, v in (cfg.get("host_paths") or {}).items() if isinstance(v, (str, int, float))}, **os.environ}))
    if q.is_absolute():
        return q
    if gh200_map_active(cfg):
        hp = cfg["host_paths"]
        if p not in (hp.get("gh200_local_only") or []):
            v = (hp["gh200"] or {}).get(p)
            if not v:
                raise UnmappedPath(f"path key {p!r} has no Isambard path in host_paths.gh200 (refusing to fall back to a local path)")
            return Path(v)
    return data_root(cfg) / q


def path_keys(cfg: dict) -> set[str]:
    """Every relative path key of the config that goes through resolve_path (checkpoints, clip lists, reference dirs, condition files, bvi_root)."""
    keys = set()
    for m in (cfg.get("models") or {}).values():
        pth = (m.get("ckpt") or {}).get("path")
        if pth:
            keys.add(pth)
    for d in (cfg.get("datasets") or {}).values():
        if d.get("clip_list"):
            keys.add(d["clip_list"])
    for it in (cfg.get("items") or {}).values():
        ref = it.get("reference")
        for r in ([ref] if isinstance(ref, str) else ([ref.get("default"), *(ref.get("by_model") or {}).values()] if isinstance(ref, dict) else [])):
            if r:
                keys.add(r)
        if (it.get("condition") or {}).get("file"):
            keys.add(it["condition"]["file"])
        if (it.get("by_hand_reference") or {}).get("file"):
            keys.add(it["by_hand_reference"]["file"])
    if (cfg.get("host_paths") or {}).get("bvi_root"):
        keys.add(cfg["host_paths"]["bvi_root"])
    return {k for k in keys if not k.startswith(("repo:", "/"))}


def model_spec(cfg: dict, mid: str, with_tag: bool = True) -> str:
    """The bench_joint --model string for a registry model: {ckpt} substituted, +TAG token appended when tag: true."""
    m = cfg["models"][mid]
    spec = m["spec"]
    if "{ckpt}" in spec:
        spec = spec.replace("{ckpt}", str(resolve_path(cfg, m["ckpt"]["path"])))
    if m.get("tag") and with_tag:
        spec = f"{spec}+{cfg['tag']['token']}"
    return spec


def reference_spec(cfg: dict, mid: str) -> str:
    """The spec string the RELEASED read ran under (e.g. the Isambard /projects/... checkpoint path); falls back to model_spec."""
    m = cfg["models"][mid]
    rs = m.get("reference_spec")
    if rs is None:
        return model_spec(cfg, mid)
    return f"{rs}+{cfg['tag']['token']}" if m.get("tag") else rs


def reference_dir(cfg: dict, iid: str, mid: str) -> Path | None:
    """The released-read dir for one model's rows of an item: reference str (all rows), or {default, by_model}; None = no reference."""
    ref = cfg["items"][iid].get("reference")
    if isinstance(ref, dict):
        ref = (ref.get("by_model") or {}).get(mid, ref.get("default"))
    return resolve_path(cfg, ref) if ref else None


def tag_backend(cfg: dict, iid: str, override: str | None = None) -> str:
    return override or cfg["items"][iid].get("tag_backend") or cfg.get("tag_backend") or "reference"


DEFAULT_SMOKE = {"models": ["earlier_s45", "earlier_s45_tag", "metric_vda_s", "ft_n46"], "datasets": ["kitti", "bonn", "scannetpp"],
                 "cells": ["llvar", "st_llvar", "b0s0"], "arms_per_item": 1, "qual_min": 3.0, "overhead_s": 90.0}


def smoke_cfg(cfg: dict) -> dict:
    """The smoke selection: DEV only, 1 clip per dataset, a model/dataset/cell subset, first arm only. b0s0 (the DEV clean cell) is kept so tab:clean is smoked too."""
    return {**DEFAULT_SMOKE, **(cfg.get("smoke") or {})}
