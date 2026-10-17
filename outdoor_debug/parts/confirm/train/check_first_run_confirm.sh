#!/bin/bash
# First-run stop check for the CONFIRMATORY FROM-INIT 12k arms (C_TS2 control, C_TS2_H300). READ-ONLY: greps logs / metrics only. NO resume, NO lr_restart: plain 500-step warmup then cosine to ~0 at 12000.
# usage: check_first_run_confirm.sh [-r RUNS_DIR] [-s] ARM=LOGFILE [ARM=LOGFILE ...]      (list C_TS2 first: the spike rule and item 7 compare with it)
#   ARM in C_TS2 C_TS2_H300.  -r RUNS_DIR: dir holding confirm_<ARM>/metrics.jsonl (default $PROJECT_DIR/runs or ./runs).  -s: smoke logs (max_steps 50, dirs confirm_smoke50_<ARM>, total printed 50).
# Run once the log shows step >= 100 (5 loss_terms rows exist from step 50). Exit 1 on any FAIL (=> scancel the job). WAIT = the log has not reached that step yet.
RUNS="${PROJECT_DIR:-.}/runs"; SMOKE=0
while getopts "r:s" o; do case $o in r) RUNS="$OPTARG";; s) SMOKE=1;; *) exit 2;; esac; done; shift $((OPTIND-1))
[[ $# -ge 1 ]] || { sed -n 2,6p "$0"; exit 2; }
export RUNS SMOKE
exec "${PYTHON:-python3}" -I - "$@" <<'PY'
import json, os, re, sys, math
RUNS, SMOKE = os.environ["RUNS"], os.environ["SMOKE"] == "1"
TOTAL = 50 if SMOKE else 12000
HINGE = {"C_TS2_H300"}; ARMS = {"C_TS2", "C_TS2_H300"}
SHA = "f19d0b838252c9a2409498b35f46738d99b3646afba984d6a82ffea3cd610b76"
SEED_LOG = 50000                       # data.seed 50 x seed_scheme x1000 (job 7226119 logged seed=46000 for data.seed 46)
DATA_LINE = "train clips=94581 val clips=3 world_size=4"   # as logged by job 7226119 (74-env tree; same sources)
LOSS_REF = 0.814                       # mean printed loss at steps 20..100 of job 7226119 (0.8702 0.8206 0.8016 0.7838 0.7944)
def scale(step, W=500, D=12000):       # independent closed form of the plain recipe (== train.make_lr_scale without restart keys; verified vs 600 real prints of job 7226119)
    if step < W: return (step + 1) / W
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - W) / (D - W))))
# printed lr at step G = 1e-4 * scale(G-1) (LR set before gstep += 1). Table from the REAL make_lr_scale (checks_confirm_results.json) for the headline steps:
TABLE = {20: 4.00e-06, 40: 8.00e-06, 100: 2.00e-05, 500: 1.00e-04, 520: 1.00e-04, 6000: 5.34e-05}
fails = 0
def out(arm, item, status, msg=""):
    global fails
    if status == "FAIL": fails += 1
    print(f"[{arm}] item {item}: {status} {msg}")
data0, loss0 = None, None
for spec in sys.argv[1:]:
    arm, _, log = spec.partition("=")
    if arm not in ARMS: print(f"unknown arm {arm}"); sys.exit(2)
    txt = open(log, errors="replace").read().splitlines(); blob = "\n".join(txt)
    steps = {int(m.group(1)): (float(m.group(3)), float(m.group(4))) for l in txt if (m := re.search(r"step(\d+)/(\d+) loss=(\S+) lr=(\S+)", l))}
    tot = {int(m.group(2)) for l in txt if (m := re.search(r"step(\d+)/(\d+) loss=", l))}
    print(f"=== {arm}  {log}  (last printed step {max(steps) if steps else None}) ===")
    # 1 LR: every printed value vs the closed form (rel 1.5%, abs floor 2e-10); headline table; never exactly 0 before the end
    bad = [(s, lr, 1e-4 * scale(s - 1)) for s, (_, lr) in sorted(steps.items()) if abs(lr - 1e-4 * scale(s - 1)) > max(0.015 * 1e-4 * scale(s - 1), 2e-10)]
    zero = [s for s, (_, lr) in steps.items() if lr == 0.0 and s < TOTAL]
    out(arm, 1, "FAIL" if (bad or zero or not steps) else "ok", f"{len(steps)} printed lr values vs 500-warmup/cosine-12000 closed form; mismatches {bad[:3]}; lr==0 at {zero[:3]}")
    for s, ref in sorted(TABLE.items()):
        if s > TOTAL: continue
        if s not in steps: out(arm, 1, "WAIT", f"step {s} not printed yet"); continue
        got = steps[s][1]; out(arm, 1, "ok" if abs(got / ref - 1) <= 0.015 else "FAIL", f"table step {s} lr {got:.3e} expected {ref:.3e}")
    # 2 from-init: NO resume line, total printed = 12000 (smoke 50), checkpoint snapshot list, params line
    res = [l for l in txt if l.startswith("[resume]")]
    out(arm, 2, "FAIL" if res else ("ok" if TOTAL in tot else "WAIT"), f"resume lines {len(res)} (must be 0: from the released init); totals printed {sorted(tot)} (expect {TOTAL})")
    ck = [l for l in txt if l.startswith("[checkpoint] permanent")]
    out(arm, 2, "ok" if ck and "[6000, 8000, 12000]" in ck[0] else "FAIL", f"snapshot list: {ck[0] if ck else 'MISSING'}")
    pm = [l for l in txt if l.startswith("[params]")]
    out(arm, 2, "ok" if pm and "trainable=8.55M" in pm[0] and "unfrozen_backbone=False" in pm[0] else "FAIL", f"{pm[0] if pm else 'MISSING'} (expect total=30.61M trainable=8.55M unfrozen_backbone=False as job 7226119)")
    # 3 hinge key in metrics.jsonl rows 0 < step <= 100
    mdir = f"confirm_smoke50_{arm}" if SMOKE else f"confirm_{arm}"
    mp = os.path.join(RUNS, mdir, "metrics.jsonl")
    if not os.path.exists(mp): out(arm, 3, "WAIT", f"{mp} missing")
    else:
        rows = [r for r in (json.loads(l) for l in open(mp) if '"loss_terms"' in l) if 0 < r["step"] <= 100]
        want = arm in HINGE; mism = [r["step"] for r in rows if ("sky_hinge_weighted" in r) != want]; vals = [r.get("sky_hinge_weighted") for r in rows]
        if len(rows) < 5: out(arm, 3, "WAIT", f"only {len(rows)} loss_terms rows (need 5)")
        elif mism: out(arm, 3, "FAIL", f"hinge key presence wrong (want {want}) at steps {mism}")
        elif want and any(not math.isfinite(v) or v < 0 for v in vals): out(arm, 3, "FAIL", f"non-finite/negative hinge {vals}")
        elif want and all(v == 0.0 for v in vals): out(arm, 3, "WARN", "hinge exactly 0 in all rows (report 'hinge inert at init', not evidence)")
        else: out(arm, 3, "ok", f"{len(rows)} rows, hinge key present={want}, values {[round(v,4) for v in vals] if want else '-'}")
    # 4 [ts2] startup: 4 ranks => 4 occurrences with the table sha (count OCCURRENCES: rank prints can interleave on one line), BOTH arms
    n_ts = len(re.findall(r"\[ts2\] TartanAir TS2 sky/mask ON", blob)); n_sha = len(re.findall(r"env_table_sha256=" + SHA, blob))
    out(arm, 4, "ok" if (n_ts == 4 and n_sha == 4) else "FAIL", f"[ts2] occurrences={n_ts}, with env_table_sha256 {SHA[:8]}: {n_sha} (expect 4 and 4)")
    # 5 finite
    nf = [l for l in txt if "[warn] non-finite loss" in l and (m := re.search(r"step (\d+)", l)) and int(m.group(1)) <= 100]
    out(arm, 5, "FAIL" if nf else "ok", f"{len(nf)} non-finite warnings in the first 100 steps")
    # 6 effective bound
    b = [l for l in txt if l.startswith("[data] effective train depth upper bound")]
    out(arm, 6, "WAIT" if not b else ("ok" if re.search(r"vkitti=655(,|$)", b[0]) and "tartanair=10000" in b[0] else "FAIL"), f"want vkitti=655, tartanair=10000; got: {b[0][-150:] if b else 'no line'}")
    # 7 seed asserted + [data] lines identical across arms (seed masked) + clip counts
    dl = [re.sub(r"seed=\d+", "seed=<S>", l) for l in txt if l.startswith("[data]")]
    seeds = sorted({int(x) for l in txt if l.startswith("[data]") for x in re.findall(r"seed=(\d+)", l)})
    out(arm, 7, "ok" if seeds == [SEED_LOG] else "FAIL", f"[data] seed values {seeds} (want [{SEED_LOG}])")
    out(arm, 7, "ok" if any(DATA_LINE in l for l in dl) else "FAIL", f"'{DATA_LINE}' in [data] lines")
    if data0 is None: data0 = dl; out(arm, 7, "ok", f"reference ({len(dl)} [data] lines)")
    else: out(arm, 7, "ok" if dl == data0 else "FAIL", "[data] lines identical to reference" if dl == data0 else f"differs from reference: {[x for x in dl if x not in data0][:3]}")
    # spike rule
    first = [steps[s][0] for s in sorted(steps) if s <= 100]
    if first:
        m = sum(first) / len(first)
        if loss0 is None: loss0 = m
        flag = m > 2 * LOSS_REF or m > 2 * loss0
        print(f"[{arm}] loss mean(steps<=100, n={len(first)}) = {m:.4f}; job 7226119 reference {LOSS_REF}; ratio to first arm {m/loss0:.2f} -> " + ("REPORT (>2x reference)" if flag else "ok"))
print("RESULT:", "FAIL (scancel)" if fails else "no FAIL (WAIT/WARN lines may remain)")
sys.exit(1 if fails else 0)
PY
