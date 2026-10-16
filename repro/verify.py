"""--verify <ref_dir>: compare freshly produced item JSONs against a reference (a previous paper_results/ dir, or the released results_v3 layout)."""
from __future__ import annotations

import json
import math
from pathlib import Path

from . import aggregate as A
from .plan import Unit

# (kind of tolerance, value). rel = |a-b|/max(|b|,tiny); abs = |a-b|
CROSS_HOST = {"depth": ("rel", 5e-3), "tae": ("rel", 5e-3), "flicker": ("rel", 5e-3), "tof": ("rel", 1.5e-2), "psnr": ("abs", 0.02), "ssim": ("abs", 3e-4)}
SAME_HOST_MACRO_REL, SAME_HOST_CLIP_REL = 1e-4, 1e-3
LATENCY_FLAG_REL = 0.10


def family(metric: str) -> str:
    m = metric.lower()
    if m.startswith("psnr"):
        return "psnr"
    if m.startswith("ssim"):
        return "ssim"
    if m.startswith("tof"):
        return "tof"
    if m.startswith("tae"):
        return "tae"
    if m.startswith(("flicker", "eps_", "joint_p95", "abs_rel_static")):
        return "flicker"
    return "depth"


def within(metric: str, a: float, b: float, same_host: bool, per_clip: bool = False, kind: str = "bench") -> tuple[bool, float]:
    """(ok, deviation) for candidate a vs reference b."""
    if a is None or b is None or math.isnan(a) or math.isnan(b):
        return (math.isnan(a if a is not None else float("nan")) and math.isnan(b if b is not None else float("nan"))), 0.0
    if kind == "latency":
        dev = abs(a - b) / max(abs(b), 1e-12)
        return dev <= LATENCY_FLAG_REL, dev
    if same_host:
        tol = SAME_HOST_CLIP_REL if per_clip else SAME_HOST_MACRO_REL
        dev = abs(a - b) / max(abs(b), 1e-12)
        return dev <= tol, dev
    typ, tol = CROSS_HOST[family(metric)]
    dev = abs(a - b) if typ == "abs" else abs(a - b) / max(abs(b), 1e-12)
    return dev <= tol, dev


def ref_rows_from_released(ref: Path, units: list[Unit], ref_specs: dict[str, str], item: dict) -> dict[tuple, dict]:
    """Read the released layout: <ref>/<slug(ref_spec)>/{bench,heldout,fnoise}/<file>. Uses plan.Unit.filename with the reference slug."""
    from .util import slug
    got = {}
    for u in units:
        rs = ref_specs.get(u.model_id)
        if u.arm or u.runner:
            continue                                              # wrapper units have no released bench layout
        if rs is None:
            continue
        ru = Unit(u.kind, u.model_id, rs, u.dataset, u.cell, u.split, u.temporal, derive_from=u.derive_from)
        sub = {"bench": "bench", "night": "heldout", "fnoise": "fnoise", "latency": "."}[u.kind]
        base_ds = u.derive_from["base"] if u.derive_from else u.dataset
        fn = ru.filename().replace(f"__{u.dataset}__", f"__{base_ds}__") if u.derive_from else ru.filename()
        p = Path(ref) / slug(rs) / sub / fn
        if p.exists():
            art = json.loads(p.read_text())
            got[(u.row_id, u.dataset, u.cell)] = A.derive_alias_art(art, u.dataset, u.derive_from["drop"]) if u.derive_from else art
    return got


def compare_results(cand: dict, ref: dict, same_host: bool, kind: str = "bench") -> dict:
    """cand/ref = aggregate_item outputs (rows + mean). Returns {'status': PASS|FLAG|FAIL, 'fails': [...], 'flags': [...], 'n_compared': n}."""
    fails, flags, n = [], [], 0

    def idx(res):
        d = {}
        for r in res["rows"] + res["mean"]:
            d[(r["model"], r["dataset"], r["cell"])] = r
        return d
    ci, ri = idx(cand), idx(ref)
    for side, res in (("cand", cand), ("ref", ref)):                      # a stored macro that the per-clip recomputation does not reproduce is a defect of the aggregation, never a pass
        for r in res["rows"]:
            if r.get("macro_mismatch"):
                fails.append({"row": [r["model"], r["dataset"], r["cell"]], "why": f"macro_mismatch ({side})", "metrics": list(r["macro_mismatch"])})
    for k in ri:
        if k not in ci:
            fails.append({"row": list(k), "why": "missing row"})
    for k in ci:
        if k not in ri:
            fails.append({"row": list(k), "why": "extra row"})
    for k in ri:
        if k not in ci:
            continue
        for m, rv in ri[k]["metrics"].items():
            cv = ci[k]["metrics"].get(m)
            ok, dev = within(m, cv, rv, same_host, kind=kind)
            n += 1
            if not ok:
                (flags if kind == "latency" else fails).append({"row": list(k), "metric": m, "cand": cv, "ref": rv, "dev": dev})
    # rank / sign flips, among models of the same (dataset, cell, metric), only for pairs whose reference gap exceeds the tolerance
    by = {}
    for k, r in ri.items():
        for m, v in r["metrics"].items():
            by.setdefault((k[1], k[2], m), []).append((k[0], v, ci.get(k, {"metrics": {}})["metrics"].get(m)))
    for (ds, cell, m), lst in by.items():
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                (ma, ra, ca), (mb, rb, cb) = lst[i], lst[j]
                if None in (ra, rb, ca, cb) or any(math.isnan(x) for x in (ra, rb, ca, cb)):
                    continue
                okgap, _ = within(m, ra, rb, same_host, kind=kind)
                if okgap:
                    continue                                              # reference pair within tolerance: order is not claimed
                if (ra - rb) * (ca - cb) < 0:
                    flags.append({"row": [ma, mb, ds, cell], "metric": m, "why": "rank/sign flip", "ref_diff": ra - rb, "cand_diff": ca - cb})
    status = "FAIL" if fails else ("FLAG" if flags else "PASS")
    return {"status": status, "fails": fails, "flags": flags, "n_compared": n}


def parse_sums(text: str) -> dict[str, str]:
    """sha256sum-format lines -> {basename: sha}."""
    out = {}
    for ln in text.splitlines():
        parts = ln.strip().split(None, 1)
        if len(parts) == 2 and len(parts[0]) == 64:
            out[Path(parts[1].lstrip("* ")).name] = parts[0]
    return out


def compare_sha_list(ref_dir: Path, out_dir: Path) -> dict:
    """Figure verify: compare the released SHA256SUMS list with the files rendered into out_dir. Byte equality of stills/GIFs is NOT required (encoder/host dependent),
    so differing bytes are reported, not failed; a released file name with no rendered counterpart is a FAIL."""
    from .util import sha256_file
    ref = parse_sums((Path(ref_dir) / "SHA256SUMS").read_text())
    have = {p.name: p for p in Path(out_dir).rglob("*") if p.is_file()} if Path(out_dir).exists() else {}
    same = sorted(n for n, h in ref.items() if n in have and sha256_file(have[n]) == h)
    differ = sorted(n for n in ref if n in have and n not in same)
    missing = sorted(n for n in ref if n not in have)
    extra = sorted(n for n in have if n not in ref)
    status = "FAIL" if missing else ("PASS" if not differ else "PASS")
    return {"status": status, "n_ref": len(ref), "byte_identical": same, "bytes_differ": differ, "missing": missing, "extra": extra,
            "fails": [{"row": [n], "why": "released figure file not produced"} for n in missing], "flags": [], "n_compared": len(ref)}


# ---- by-hand reference: P1_TABLE.md (released P1 read, a 4090 read; no bench-layout JSONs exist for it) -----------------------------------------------------
P1_COLS = {"tae_gtvalid": ("tae_gtvalid", "llvar"), "AbsRel_al": ("abs_rel_aligned", "llvar"), "AbsRel_unal": ("abs_rel_unaligned", "llvar"), "d1_al": ("delta1_aligned", "llvar"),
           "sky_ge80": ("sky_ge80", "llvar"), "sky_med": ("sky_med", "llvar"), "sky_nearer_p95": ("sky_nearer_than_scene_p95", "llvar"), "st flicker": ("flicker", "st_llvar")}


def parse_p1_table(text: str) -> dict:
    """P1_TABLE.md -> {(variant 'raw'|'tag', dataset, arm): {column: (value, n_decimals)}} (nan kept as nan)."""
    out, var, cols = {}, None, None
    for ln in text.splitlines():
        if ln.startswith("## variant"):
            var = "tag" if "tag" in ln else "raw"
            cols = None
        elif var and ln.startswith("dataset |"):
            cols = [c.strip() for c in ln.split("|")]
        elif var and cols and ln.count("|") == len(cols) - 1 and not ln.startswith("#"):
            f = [c.strip() for c in ln.split("|")]
            row = {}
            for c, v in zip(cols[2:], f[2:]):
                row[c] = (float(v), len(v.split(".")[1]) if "." in v else 0)
            out[(var, f[0], f[1])] = row
    return out


def compare_p1(result: dict, table: dict, same_host: bool = False) -> dict:
    """Regenerated tab:p1 (aggregated item result) vs the released P1_TABLE.md values. Cross-host tolerances (R3) + the half-unit rounding of the printed value.
    Returns {status, fails, rows (every comparison: for the generated cross-host comparison table), n_compared}."""
    idx = {(r["model"], r["dataset"], r["cell"]): r["metrics"] for r in result["rows"]}
    rows, fails = [], []
    for (var, ds, arm), cols in sorted(table.items()):
        mid = ("ramen_s45_tag" if var == "tag" else "ramen_s45") + "@" + arm
        for col, (ref, dec) in cols.items():
            if col not in P1_COLS:
                continue
            m, cell = P1_COLS[col]
            cv = (idx.get((mid, ds, cell)) or {}).get(m)
            if ref != ref and (cv is None or cv != cv):
                continue                                              # nan == nan
            if cv is None:
                rows.append({"variant": var, "dataset": ds, "arm": arm, "metric": m, "cell": cell, "cand": None, "released": ref, "dev": None, "ok": False})
                fails.append(rows[-1])
                continue
            ok, dev = within(m, cv, ref, same_host)
            ok = ok or abs(cv - ref) <= 0.5 * 10 ** (-dec) + 1e-12
            rows.append({"variant": var, "dataset": ds, "arm": arm, "metric": m, "cell": cell, "cand": cv, "released": ref, "dev": dev, "ok": bool(ok)})
            if not ok:
                fails.append(rows[-1])
    return {"status": "FAIL" if fails or not rows else "PASS", "fails": fails, "flags": [], "rows": rows, "n_compared": len(rows)}
