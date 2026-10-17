"""Freeze the per-env TS2 flags into ONE json table (ts2_env_table.json) read by data/tartanair_ts2.py.
Inputs (this directory): sky_ids_74.json, far_near_class_74.csv, ta_sky_followup_decisions.csv, unlabelled_decisions_extension.csv.
Usage: python make_ts2_env_table.py [out.json]   (default ts2_env_table.json next to this script); prints the sha256."""
import csv, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = ["sky_ids_74.json", "far_near_class_74.csv", "ta_sky_followup_decisions.csv", "unlabelled_decisions_extension.csv"]


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def _tf(s, field, env):
    s = s.strip()
    if s == "True": return True
    if s == "False": return False
    raise ValueError(f"{env}: {field} not True/False: {s!r}")


def build():
    ids = json.load(open(f"{HERE}/sky_ids_74.json"))["envs"]
    far = {r["env"]: r["cls"] for r in csv.DictReader(open(f"{HERE}/far_near_class_74.csv"))}
    dec = {r["env"]: r for r in csv.DictReader(open(f"{HERE}/ta_sky_followup_decisions.csv"))}
    unl = {r["env"]: r["extension_decision"].strip() for r in csv.DictReader(open(f"{HERE}/unlabelled_decisions_extension.csv"))}
    assert set(ids) == set(far) == set(dec), "env sets differ"
    envs = {}
    for env in sorted(ids):
        sky_ids = sorted(int(i) for i in ids[env]["sky_ids"])
        cls = far[env]
        assert cls in ("far_dome", "near_dome", "no_seg_sky"), (env, cls)
        ok = dec[env]["seg_sky_ok"].strip()
        seg_sky_ok = True if ok == "n/a" else _tf(ok, "seg_sky_ok", env)   # n/a = nothing to test (no seg-sky)
        sn = dec[env]["sat_nonsky_is_sky"].strip()
        sat_nonsky_is_sky = sn == "True"                                   # '', 'False', 'False (default ...)' -> False
        if sn not in ("", "True", "False") and not sn.startswith("False (default"):
            raise ValueError(f"{env}: unexpected sat_nonsky_is_sky {sn!r}")
        fb = False
        if env in unl:
            d = unl[env]
            assert d in ("fallback_sat_is_sky=True", "uncertain"), (env, d)
            fb = d == "fallback_sat_is_sky=True"
        envs[env] = dict(sky_ids=sky_ids, far_dome=(cls == "far_dome"), seg_sky_ok=seg_sky_ok,
                         sat_nonsky_is_sky=sat_nonsky_is_sky, fallback_sat_is_sky=fb)
    assert set(unl) <= set(envs), set(unl) - set(envs)
    return {"schema": "ts2_env_table/1", "rule": "outdoor_debug/parts/TA_sky/ts_v2_draft.py::ts_v2_guarded",
            "source_sha256": {f: _sha(f"{HERE}/{f}") for f in SRC}, "n_envs": len(envs), "envs": envs}


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else f"{HERE}/ts2_env_table.json"
    t = build()
    with open(out, "w") as fh:
        fh.write(json.dumps(t, indent=1, sort_keys=True) + "\n")
    print(out, "n_envs", t["n_envs"], "sha256", _sha(out))
