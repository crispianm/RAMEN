#!/bin/bash
# First-run stop check for Stage 1 (6 paired continuations; adapted from the screen's redone submit/check_first_run.sh + Auditor O39). READ-ONLY: greps logs / metrics / config yamls only.
# usage: check_first_run_stage1.sh [-r RUNS_DIR] [-c CONFIG_DIR] [-s] NAME=LOGFILE [NAME=LOGFILE ...]      NAME in S1_n{46,47,48}_{TS2,H300}; list S1_n46_TS2 first (reference for items 8/9)
#   -r RUNS_DIR: dir holding confirm1_<NAME>/metrics.jsonl (default $PROJECT_DIR/runs or ./runs).  -c CONFIG_DIR: dir with arm_<NAME>.yaml (default: stage1/configs next to this script).
#   -s: smoke logs (dirs confirm1_smoke50_<NAME>; total steps 12050; LR rows above 12050 are 'WAIT').
# Per-run expectations derived from NAME: start c3_nXX step12000 (resume line path), seed S = 50/51/52 (log '[data] ... seed=S*1000'; config data.seed AND training.resume_reseed == S -- resume_reseed is NOT printed by train.py,
# so it is asserted from the config yaml that the sbatch sha-checked, and the log's ' config : ... sha256' line must equal that yaml's sha), [ts2] x4 with env sha f19d0b83 in ALL six, hinge key only in H300.
# Exit 1 on any FAIL (=> scancel the job). WAIT = the log has not reached that step yet.
HERE="$(cd "$(dirname "$0")" && pwd)"
RUNS="${PROJECT_DIR:-.}/runs"; SMOKE=0; CFGD="$HERE/configs"
while getopts "r:c:s" o; do case $o in r) RUNS="$OPTARG";; c) CFGD="$OPTARG";; s) SMOKE=1;; *) exit 2;; esac; done; shift $((OPTIND-1))
[[ $# -ge 1 ]] || { sed -n 2,9p "$0"; exit 2; }
export RUNS SMOKE CFGD
exec "${PYTHON:-python3}" -I - "$@" <<'PY'
import json, os, re, sys, math, hashlib
RUNS, SMOKE, CFGD = os.environ["RUNS"], os.environ["SMOKE"] == "1", os.environ["CFGD"]
TOTAL = 12050 if SMOKE else 14000
SEED = {"46": 50, "47": 51, "48": 52}
TABLE = {12020: 2.00e-05, 12040: 4.00e-05, 12060: 6.00e-05, 12080: 8.00e-05, 12100: 1.00e-04, 12120: 1.00e-04, 12200: 9.93e-05, 12400: 9.40e-05, 12600: 8.39e-05, 13000: 5.42e-05, 13400: 2.27e-05, 13800: 2.74e-06}
SHA = "f19d0b838252c9a2409498b35f46738d99b3646afba984d6a82ffea3cd610b76"
fails = 0
def out(arm, item, status, msg=""):
    global fails
    if status == "FAIL": fails += 1
    print(f"[{arm}] item {item}: {status} {msg}")
def scale(g):   # independent closed form of the Add.45 restart schedule: warmup 100 from 0 to peak 1.0 at 12000, then cosine from 12100 to 0 at 14000 (printed lr at step G = 1e-4 * scale(G-1))
    s = g - 1
    if s < 12100: return (s - 12000 + 1) / 100.0
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (s - 12100) / (14000 - 12100))))
data0, means = None, {}
for spec in sys.argv[1:]:
    arm, _, log = spec.partition("=")
    m = re.fullmatch(r"S1_n(46|47|48)_(TS2|H300)", arm)
    if not m: print(f"unknown run name {arm}"); sys.exit(2)
    n, kind = m.group(1), m.group(2); S = SEED[n]; hinge = kind == "H300"
    start = f"/projects/b5dh/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n{n}/step12000.pt"
    txt = open(log, errors="replace").read().splitlines(); blob = "\n".join(txt)
    steps = {int(x.group(1)): (float(x.group(3)), float(x.group(4))) for l in txt if (x := re.search(r"step(\d+)/(\d+) loss=(\S+) lr=(\S+)", l))}
    tot = {int(x.group(2)) for l in txt if (x := re.search(r"step(\d+)/(\d+) loss=", l))}
    print(f"=== {arm}  {log}  (last printed step {max(steps) if steps else None}) ===")
    # 1 LR: restart table + every printed value vs the closed form; never 0 in [12000,14000)
    for s, ref in sorted(TABLE.items()):
        if s > TOTAL: continue
        if s not in steps: out(arm, 1, "WAIT", f"step {s} not printed yet"); continue
        out(arm, 1, "ok" if abs(steps[s][1] / ref - 1) <= 0.015 else "FAIL", f"table step {s} lr {steps[s][1]:.3e} expected {ref:.3e}")
    bad = [(s, lr) for s, (_, lr) in sorted(steps.items()) if 12000 < s <= TOTAL and abs(lr - 1e-4 * scale(s)) > max(0.015 * 1e-4 * scale(s), 2e-9)]
    out(arm, 1, "FAIL" if bad else "ok", f"{len(steps)} printed lr values vs restart closed form; mismatches {bad[:3]}")
    if [s for s, (_, lr) in steps.items() if 12000 <= s < 14000 and lr == 0.0]: out(arm, 1, "FAIL", "lr == 0 inside [12000,14000) (no-op restart bug)")
    # 2 resume from THIS pair's own start checkpoint; totals; snapshot list; params
    res = [l for l in txt if l.startswith("[resume]")]
    good = len(res) == 1 and res[0].split()[1] == start and re.search(r"step=12000\b", res[0])
    out(arm, 2, "ok" if good else "FAIL", f"resume line {res[0][:150] if res else 'MISSING'} (want exactly one, path {start}, step=12000)")
    out(arm, 2, "ok" if TOTAL in tot else ("WAIT" if not tot else "FAIL"), f"totals printed {sorted(tot)} (expect {TOTAL})")
    ck = [l for l in txt if l.startswith("[checkpoint] permanent")]
    out(arm, 2, "ok" if ck and "[12500, 13000, 13500, 14000]" in ck[0] else "FAIL", f"snapshot list: {ck[0] if ck else 'MISSING'}")
    pm = [l for l in txt if l.startswith("[params]")]
    out(arm, 2, "ok" if pm and "trainable=8.55M" in pm[0] and "unfrozen_backbone=False" in pm[0] else "FAIL", f"{pm[0] if pm else 'MISSING'}")
    # 3 hinge key
    mp = os.path.join(RUNS, f"confirm1_smoke50_{arm}" if SMOKE else f"confirm1_{arm}", "metrics.jsonl")
    if not os.path.exists(mp): out(arm, 3, "WAIT", f"{mp} missing")
    else:
        rows = [r for r in (json.loads(l) for l in open(mp) if '"loss_terms"' in l) if 12000 < r["step"] <= 12050]
        mism = [r["step"] for r in rows if ("sky_hinge_weighted" in r) != hinge]; vals = [r.get("sky_hinge_weighted") for r in rows]
        if len(rows) < 5: out(arm, 3, "WAIT", f"only {len(rows)} loss_terms rows (need 5)")
        elif mism: out(arm, 3, "FAIL", f"hinge key presence wrong (want {hinge}) at steps {mism}")
        elif hinge and any(not math.isfinite(v) or v < 0 for v in vals): out(arm, 3, "FAIL", f"non-finite/negative hinge {vals}")
        elif hinge and all(v == 0.0 for v in vals): out(arm, 3, "WARN", "hinge exactly 0 in all 5 rows: report 'hinge inert at this start point', not as evidence")
        else: out(arm, 3, "ok", f"{len(rows)} rows, hinge key present={hinge}, values {vals if hinge else '-'}")
    # 4 TS2 line x4 with the env sha, in ALL six
    n_ts = len(re.findall(r"\[ts2\] TartanAir TS2 sky/mask ON", blob)); n_sha = len(re.findall(r"env_table_sha256=" + SHA, blob))
    out(arm, 4, "ok" if (n_ts == 4 and n_sha == 4) else "FAIL", f"[ts2] occurrences={n_ts}, with env_table_sha256 {SHA[:8]}: {n_sha} (expect 4 and 4)")
    # 5 finite
    nf = [l for l in txt if "[warn] non-finite loss" in l and (x := re.search(r"step (\d+)", l)) and int(x.group(1)) <= 12050]
    out(arm, 5, "FAIL" if nf else "ok", f"{len(nf)} non-finite warnings in first 50 steps")
    # 6 vkitti bound 655
    b = [l for l in txt if l.startswith("[data] effective train depth upper bound")]
    if not b: out(arm, 6, "WAIT", "bound line not in log")
    else: out(arm, 6, "ok" if re.search(r"vkitti=655(,|$)", b[0]) else "FAIL", f"want vkitti=655; got: {b[0][-120:]}")
    # 7 seeds: log value, config data.seed and training.resume_reseed, config sha line
    seeds = sorted({int(x) for l in txt if l.startswith("[data]") for x in re.findall(r"seed=(\d+)", l)})
    out(arm, 7, "ok" if seeds == [S * 1000] else "FAIL", f"[data] seed values {seeds} (want [{S*1000}] = data.seed {S} x1000)")
    cp = os.path.join(CFGD, f"arm_{arm}.yaml")
    if not os.path.exists(cp): out(arm, 7, "WAIT", f"{cp} not found (config cross-check skipped)")
    else:
        y = open(cp).read(); ds = re.findall(r"^  seed: (\d+)$", y, re.M); rs = re.findall(r"^  resume_reseed: (\d+)$", y, re.M)
        out(arm, 7, "ok" if ds == [str(S)] and rs == [str(S)] else "FAIL", f"config data.seed {ds} and training.resume_reseed {rs} (want both [{S}])")
        csha = hashlib.sha256(open(cp, "rb").read()).hexdigest(); cl = [l for l in txt if re.match(r"\s*config : ", l)]
        if cl: out(arm, 7, "ok" if (f"arm_{arm}.yaml" in cl[0] and csha in cl[0]) else "FAIL", f"log config line {cl[0][:170]} vs sha256 of {cp} {csha[:12]}")
        else: out(arm, 7, "WAIT", "no ' config :' line in log (smoke logs and fulls have it from the sbatch preflight)")
    # 8 identical [data] lines across all runs (seed masked): clip counts identical within a pair and across all six
    dl = [re.sub(r"seed=\d+", "seed=<S>", l) for l in txt if l.startswith("[data]")]
    if data0 is None: data0 = dl; out(arm, 8, "ok", f"reference ({len(dl)} [data] lines)")
    else: out(arm, 8, "ok" if dl == data0 else "FAIL", "[data] lines identical to reference" if dl == data0 else f"differs from reference: {[x for x in dl if x not in data0][:3]}")
    first = [steps[s][0] for s in sorted(steps) if s <= 12100]
    if first: means[arm] = sum(first) / len(first); print(f"[{arm}] loss mean(steps<=12100, n={len(first)}) = {means[arm]:.4f}")
# 9 pair loss ratio H300/TS2 (same start, same seed) and cross-pair sanity
for n in SEED:
    a, h = f"S1_n{n}_TS2", f"S1_n{n}_H300"
    if a in means and h in means:
        r = means[h] / means[a]; print(f"[pair n{n}] loss mean H300/TS2 = {r:.2f} -> " + ("REPORT (>2x or <0.5x within a pair)" if (r > 2 or r < 0.5) else "ok"))
print("RESULT:", "FAIL (scancel)" if fails else "no FAIL (WAIT/WARN lines may remain)")
sys.exit(1 if fails else 0)
PY
