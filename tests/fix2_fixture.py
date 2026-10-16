"""A SYNTHETIC scratch batch for provenance/dn01_fix2/read_fix2.py (tests + the Auditor's 'reader output on a scratch batch'). NOT results: invented numbers shaped like the scorer's files, with designed profiles.
Stage 1 profiles (multipliers of released at TAG):
  a1h  (control)  T <= 1 BUT GUARD FAIL (night aligned, static flicker pooled, night TAE, KITTI st_llb3 / st_llvar flicker)
  x1   (anneal)   every guard at V2's level, T <= 1                  -> BEATS RELEASED with margin, FIXES
  x3   (static50) KITTI static flicker fixed, night TAE still fails   -> T <= 1 BUT GUARD FAIL, PARTIAL
  s46  (seed)     a1h with small moves                               -> T <= 1 BUT GUARD FAIL, NO FIX
Stage 2 (SET=stage2a, with S1_DIR = the stage-1 pull): cp50 with the a1h controls read from the stage-1 batch. A1HFIX batch (AH_DIR): f1 = BEATS RELEASED thin margin, f2 = PARTIAL, s1 = SCALE LOST, L1 steps.
Layout as the real pulls: <root>/<name>/RAMEN_eval_dn01_fx2_<SET>_abc1234/results/{bench,heldout,fnoise,ema}/, an audited REF dir, a METRICFIX pull (MF_PULL), run dirs (TRAIN_DIRS) with metrics.jsonl / nonfinite_skips.json / logs_<arm>."""
from __future__ import annotations

import json
from pathlib import Path

KITTI_MANIFEST = "7024ba29f3f0cb1e68e99bcc44ed5fc3cc1810085f4097dc7b427dd2c7f92e56"
VARS = ["none", "agf0.2t0.06"] + [f"ema{a}" for a in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]
TAILS = [("heldout", "nuscenes_night__real__all"), ("fnoise", "bvi_rlv_test__fnoise")] + [("bench", f"{d}__{c}__dev") for d in ("kitti", "bonn") for c in ("llb3", "st_llb3", "st_llvar")]
ADDED = [("heldout", "nyu__st_llb3__all"), ("heldout", "nyu__st_llvar__all"), ("heldout", "scannetpp__st_llb3__all"), ("heldout", "scannetpp__st_llvar__all"),
         ("bench", "vkitti__st_llb3__dev"), ("bench", "vkitti__st_llvar__dev")]
CLIPS = {"nyu": 20, "scannetpp": 10, "vkitti": 10}
SKY = ["2011_09_28_drive_0037_sync", "2011_09_26_drive_0005_sync", "2011_09_26_drive_0023_sync", "2011_10_03_drive_0047_sync", "2011_09_26_drive_0020_sync", "2011_09_26_drive_0113_sync", "2011_09_26_drive_0095_sync"]

REL = dict(n_absU=0.320, n_absA=0.205, n_tae=9.63, n_med=17.3, n_fit=1.22, k_absU=0.120, k_absA=0.1013, k_tae=2.26, k_fit=1.0, k_med=14.0, b_absU=0.200, b_absA=0.080, b_tae=1.17, b_fit=1.3, b_med=3.0,
           fnoise=1.0, k_stl=0.0006, k_stv=0.0007, b_stl=0.0012, b_stv=0.0015, nyu_stl=0.0010, nyu_stv=0.0011, sn_stl=0.0009, sn_stv=0.0010, vk_stl=0.0008, vk_stv=0.0009, sky=0.95, d47=0.125, skymed=440.0)
ONE = {k: 1.0 for k in REL}


def prof(**kw):
    return {**ONE, **kw}


GOOD = dict(n_absU=0.80, k_absU=0.80, b_absU=0.36, d47=0.80, n_absA=0.97, b_absA=0.97, n_tae=0.875, b_tae=0.97, k_stl=0.83, k_stv=0.86, b_stl=0.58, b_stv=0.60, nyu_stl=0.9, nyu_stv=0.9, sn_stl=0.9, sn_stv=0.9,
            sky=1.0, fnoise=0.95, skymed=0.36, b_fit=1.0)
A1H = dict(n_absU=0.894, k_absU=0.789, b_absU=0.341, d47=0.80, n_absA=0.97 * 1.046, b_absA=0.97 * 1.027, n_tae=0.875 * 1.082, b_tae=0.97 * 0.961, k_stl=0.83 * 1.39, k_stv=0.86 * 1.25, b_stl=0.58 * 1.01, b_stv=0.60 * 0.98,
           nyu_stl=0.9 * 1.05, nyu_stv=0.9 * 1.04, sn_stl=0.9, sn_stv=0.9, sky=0.95, fnoise=0.95 * 0.8, skymed=0.26, b_fit=0.98)
CLEAN = dict(n_absU=0.67, k_absU=0.80, b_absU=0.60, d47=0.8, n_tae=0.9, sky=0.95)
PROFILES = {
    "metric_vda_s": prof(),
    "v2_45": prof(**{**GOOD, "n_absU": 0.77, "k_absU": 0.82, "b_absU": 0.37, "d47": 0.82}),
    "dn01_vm_s1_pgt_n45": prof(n_absU=2.66, k_absU=1.75, b_absU=0.33, d47=1.75, n_tae=0.95, sky=0.9),
    "a1h": prof(**A1H),
    "a1hclean": prof(**CLEAN),
    "x1": prof(**GOOD),                                                                                             # BEATS RELEASED with margin: every guard at V2's level
    "x3": prof(**{**A1H, "n_absA": 0.97, "k_stl": 0.83, "k_stv": 0.86, "b_absA": 0.97}),              # KITTI static flicker + aligned fixed; night TAE still fails
    "s46": prof(**{**A1H, "n_absU": 0.90, "n_tae": 0.875 * 1.07}),
    "cp50": prof(**{**A1H, "n_absU": 0.80}),
    "f1": prof(**{**GOOD, "n_tae": 0.875 * 1.030}),                                                                 # BEATS RELEASED, thin margin (night TAE just under its threshold)
    "f2": prof(**{**A1H, "k_stl": 0.83, "k_stv": 0.86}),
    "s1": prof(**{**GOOD, "n_absU": 1.20}),
}
for _k in ("x1clean", "x3clean", "s46clean", "f1clean", "f2clean", "s1clean", "x2clean"):
    PROFILES[_k] = dict(PROFILES["a1hclean"])
PROFILES["x2"] = dict(PROFILES["x1"])
PROFILES["a1h@1000"] = {**PROFILES["a1h"], "n_absU": 0.95}
PROFILES["a1h@1500"] = dict(PROFILES["a1h"])
PROFILES["a1h@2000"] = dict(PROFILES["a1h"])
SEEDS = {"v2_46": 1.02, "v2_47": 0.985}

RUN = {"x1": "dn01_fx2_x1_anneal_n45", "x1clean": "dn01_fx2_x1_anneal_clean_n45", "x3": "dn01_fx2_x3_static50_n45", "x3clean": "dn01_fx2_x3_static50_clean_n45", "s46": "dn01_fx2_a1h_n46", "s46clean": "dn01_fx2_a1hclean_n46",
       "x2": "dn01_fx2_x2_ema_n45", "x2clean": "dn01_fx2_x2_ema_clean_n45", "cp50": "dn01_fx2_cp50_n45",
       "f1": "dn01_ah_f1_n45", "f2": "dn01_ah_f2_n45", "s1": "dn01_ah_s1_n45", "f1clean": "dn01_ah_f1clean_n45", "f2clean": "dn01_ah_f2clean_n45", "s1clean": "dn01_ah_s1clean_n45",
       "a1h": "dn01_mf_a1h_n45", "a1hclean": "dn01_mf_a1hclean_n45", "v2_45": "dn01_f1s_v2_f_n45", "v2_46": "dn01_f1s_v2_f_n46", "v2_47": "dn01_f1s_v2_f_n47", "dn01_vm_s1_pgt_n45": "dn01_vm_s1_pgt_n45"}
DEGRADED = {"x1", "x3", "s46", "x2", "cp50", "f1", "f2", "s1", "a1h", "a1h@1000", "a1h@1500", "a1h@2000", "v2_45", "metric_vda_s"}


def vfactor(var: str, key: str) -> float:
    """TAG lowers flicker and TAE for EVERY model (released included); raw = 1; ema in between."""
    f = key.endswith(("stl", "stv"))
    if var == "agf0.2t0.06":
        return 0.55 if f else 0.97 if key.endswith("tae") else 1.0
    if var == "none":
        return 1.0
    return 0.7 if f else 1.0


def values(model: str, var: str, ov: dict | None = None) -> dict:
    p = dict(PROFILES["v2_45"] if model in SEEDS else PROFILES[model])
    if ov:
        p.update(ov)
    out = {}
    for k, r in REL.items():
        x = r * p[k] * vfactor(var, k)
        if model in SEEDS and k not in ("sky", "skymed", "fnoise", "n_med", "n_fit", "k_fit", "k_med", "b_fit", "b_med", "k_absA", "k_tae"):
            x *= SEEDS[model]
        out[k] = x
    return out


def tail_json(sub: str, tail: str, v: dict, num_clips: int | None = None) -> dict:
    ds = tail.split("__")[0]
    if sub == "fnoise":
        return dict(macro=dict(f_noise=v["fnoise"]))
    if tail.startswith("nuscenes_night"):
        return dict(video_macro=dict(abs_rel_aligned=v["n_absA"], abs_rel_unaligned=v["n_absU"], median_depth=v["n_med"], fit_scale=v["n_fit"], tae=v["n_tae"], tae_gtvalid=v["n_tae"] * 0.8), num_clips=15)
    if ds in ("kitti", "bonn"):
        c, d = tail.split("__")[1], ds[0]
        if c == "llb3":
            vm = dict(abs_rel_aligned=v[f"{d}_absA"], abs_rel_unaligned=v[f"{d}_absU"], median_depth=v[f"{d}_med"], fit_scale=v[f"{d}_fit"], tae=v[f"{d}_tae"], tae_gtvalid=v[f"{d}_tae"] * 0.75)
            if d == "k":
                vm["sky_ge80"] = v["sky"]
        else:
            vm = dict(abs_rel_aligned=0.1, abs_rel_unaligned=0.2, median_depth=1.0, fit_scale=1.0, flicker=v[f"{d}_{'stl' if c == 'st_llb3' else 'stv'}"])
        out = dict(video_macro=vm, num_clips=7 if d == "k" else 13)
        if d == "k":
            out["manifest_sha256"] = KITTI_MANIFEST
            if c == "llb3":
                out["per_video"] = {k: dict(abs_rel_aligned=v["k_absA"], abs_rel_unaligned=v["d47"] * (3 if "0047" in k else 1), sky_med=v["skymed"] * (1.5 if "0047" in k else 1.0)) for k in SKY}
            else:
                out["per_video"] = {k: dict(flicker=vm["flicker"]) for k in SKY}
        elif c == "llb3":
            out["per_video"] = {"rgbd_bonn_static_close_far": dict(fit_scale=v["b_fit"] * 0.9), "rgbd_bonn_synchronous": dict(fit_scale=v["b_fit"]), "rgbd_bonn_balloon": dict(fit_scale=v["b_fit"] * 1.1)}
        return out
    if ds in ("nyu", "scannetpp"):
        key = ("nyu" if ds == "nyu" else "sn") + ("_stl" if "st_llb3" in tail else "_stv")
        return dict(video_macro=dict(abs_rel_aligned=0.1, abs_rel_unaligned=0.2, median_depth=1.0, fit_scale=1.0, flicker=v[key]), num_clips=num_clips if num_clips is not None else CLIPS[ds])
    if ds == "vkitti":
        return dict(video_macro=dict(flicker=v["vk_stl" if "st_llb3" in tail else "vk_stv"]), per_video={}, num_clips=num_clips if num_clips is not None else CLIPS["vkitti"])
    raise KeyError(tail)


def slug(model: str) -> str:
    if model == "metric_vda_s":
        return model
    run, _, st = model.partition("@")
    run = RUN.get(run, run)
    return f"ours-_projects_b5dh_RAMEN_train_{run}_runs_{run}_step{int(st) if st else 4000}"


def fname(model: str, var: str, tail: str) -> str:
    return f"{slug(model)}{'' if var == 'none' else '+' + var}__{tail}.json"


def write(path: Path, d) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d) if not isinstance(d, str) else d)


def batch(res: Path, models: list, drop=(), overrides=None, clips=None) -> None:
    overrides, clips = overrides or {}, clips or {}
    for m in models:
        for var in VARS:
            for sub, tail in TAILS + ADDED:
                if (sub, tail) in ADDED and m not in DEGRADED:
                    continue
                if tail.startswith("vkitti") and m.startswith("a1h@"):
                    continue
                n = fname(m, var, tail)
                if any(x in n for x in drop):
                    continue
                ov = next((o for (mm, ts), o in overrides.items() if mm == m and ts in tail), None)
                nc = next((c for (mm, ts), c in clips.items() if mm == m and ts in tail), None)
                write(res / sub / n, tail_json(sub, tail, values(m, var, ov), nc))
    write_probes(res, models)


PROBE_SHA = "226b6cc91245e570e04ae25ee5d343ee95f83b44fcd20ebeccd0a83e0216c2aa"
PROBE_B = 0.95                                                                                                         # default probe (b) of every probed model: SKY OK passes unless a test mutates a file


def probe_json(variant: str | None, b: float = PROBE_B, deg: float = 0.5) -> dict:
    """A scaleprobe__*.json as scripts/dn01_vm_scaleprobe.py writes it: stored variant, clip list sha, clean + degraded mode, vkitti n = 6."""
    return dict(model="x", variant=variant, clip_list_sha256=PROBE_SHA, modes={"clean": {"vkitti": dict(n=6, unsup_sky_frac_gt80=b)}, "degraded": {"vkitti": dict(n=6, unsup_sky_frac_gt80=deg)}})


def probe_name(m: str, variant: str) -> str:
    return f"scaleprobe__{slug(m)}__{variant}.json"


def write_probes(res: Path, models: list) -> None:
    """The label models (the DEGRADED set = the guard kind of the scoring) get exactly the two windowed probe runs, --variant none and --variant agf0.2t0.06; never a bare file."""
    for m in models:
        if m in DEGRADED:
            for var in ("none", "agf0.2t0.06"):
                write(res / "probe" / probe_name(m, var), probe_json(var))


def train_dirs(td: Path, runs: list, lr_last: float = 1e-9) -> None:
    for r in runs:
        write(td / r / "nonfinite_skips.json", dict(skipped_microbatches=0, optimizer_steps=4000))
        (td / r / "metrics.jsonl").write_text("\n".join(json.dumps(dict(event="outdoor_monitor", step=s, outdoor_scale_ratio=1.0, outdoor_scale_ratio_clean=1.0)) for s in (0, 100, 250, 500, 1000, 1500, 2000, 4000)) + "\n")
        last = lr_last if "x1_anneal" in r else 9.2e-5
        write(td / f"logs_{r}" / f"{r}_1.out", f"e0 step20/4000 loss=0.6 lr=2.0e-06 1.0it/s\ne0 step500/4000 loss=0.6 lr=1.00e-04 1.0it/s\ne0 step4000/4000 loss=0.6 lr={last:.2e} 1.0it/s\n")


def build(root: Path, set_: str = "stage1", drop=(), overrides=None, clips=None, reserve: bool = False, ah: bool = False, lr_last: float = 1e-9, ema_same: bool = False) -> dict:
    """Write a scratch pull (and the shared refs / METRICFIX pull) under ``root``; return the reader env. stage1 = x1 x3 s46 (+ x2 with ``reserve``) + controls; stage2a = cp50 + released / V2 (needs a stage-1 build in the same root)."""
    ref, pull, td = root / "ref", root / "pull" / "RAMEN_eval_dn01_mf_score_cc2b8a5" / "results", root / "train"
    if not ref.exists():
        for m in ("metric_vda_s", "v2_45", "v2_46", "v2_47", "dn01_vm_s1_pgt_n45"):
            for var in VARS:
                for sub, tail in TAILS + [a for a in ADDED if not a[1].startswith("vkitti")]:
                    if m == "dn01_vm_s1_pgt_n45" and (sub, tail) not in TAILS:
                        continue
                    write(ref / fname(m, var, tail), tail_json(sub, tail, values(m, var)))
        for m in ("a1h", "a1hclean"):
            for var in ("none", "agf0.2t0.06"):
                for sub, tail in TAILS:
                    write(pull / sub / fname(m, var, tail), tail_json(sub, tail, values(m, var)))
    env = {"MF_PULL": str(root / "pull"), "REF_DIRS": str(ref), "SET": set_}
    if set_ == "stage1":
        news = ["x1", "x1clean", "x3", "x3clean", "s46", "s46clean"] + (["x2", "x2clean"] if reserve else [])
        res = root / "s1" / "RAMEN_eval_dn01_fx2_stage1_abc1234" / "results"
        batch(res, news + ["a1h", "a1hclean", "v2_45", "dn01_vm_s1_pgt_n45", "metric_vda_s"], drop, overrides, clips)
        runs = [RUN[m] for m in news]
        env.update(FX_DIR=str(root / "s1"), RESERVE="1" if reserve else "0")
        if reserve:
            sha = []
            for m in ("x2", "x2clean"):
                h = f"{abs(hash(m)) % 16 ** 8:08x}" * 8
                e, l_ = ("e" * 64, "e" * 64) if ema_same else ("e" * 64, "f" * 64)
                write(res / "ema" / f"{RUN[m]}.json", dict(ckpt_sha256=h, ema_differs_from_live=not ema_same, scorer_loads="trainable_ema", scorer_loads_sha256=e, trainable_ema_sha256=e, trainable_sha256=l_, max_abs_ema_minus_live=0.0 if ema_same else 1e-3))
                sha.append(f"{h}  /projects/b5dh/RAMEN_train_{RUN[m]}/runs/{RUN[m]}/step4000.pt")
            write(res / "ckpt_sha256.txt", "\n".join(sha) + "\n")
    elif set_ == "stage2a":
        news = ["cp50"]
        res = root / "s2" / "RAMEN_eval_dn01_fx2_stage2a_abc1234" / "results"
        batch(res, news + ["v2_45", "metric_vda_s"], drop, overrides, clips)
        runs = [RUN["cp50"]]
        env.update(FX_DIR=str(root / "s2"), S1_DIR=str(root / "s1"))
    else:
        raise KeyError(set_)
    train_dirs(td, runs, lr_last)
    env["TRAIN_DIRS"] = " ".join([str(td / "dn01_fx2_*"), str(td / "logs_dn01_fx2_*")])
    if ah:
        resa = root / "ah" / "RAMEN_eval_dn01_ah_score_def5678" / "results"
        batch(resa, ["f1", "f2", "s1", "f1clean", "f2clean", "s1clean", "a1h", "a1hclean", "a1h@1000", "a1h@1500", "a1h@2000", "v2_45", "dn01_vm_s1_pgt_n45", "metric_vda_s"])
        env["AH_DIR"] = str(root / "ah")
    return env
