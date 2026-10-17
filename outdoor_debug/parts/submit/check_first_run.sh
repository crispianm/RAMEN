#!/bin/bash
# First-run stop check (outdoor_debug/parts/arm_checks/first_run_stop_check.md items 1-7 + the loss-spike rule). READ-ONLY: greps logs / metrics only.
# usage: check_first_run.sh [-r RUNS_DIR] [-s] ARM=LOGFILE [ARM=LOGFILE ...]      (list A0 first: items 7 and the spike rule compare with it)
#   ARM in A0 A0b TS2 TS2_H80 TS2_H300 TS2_VK655.  -r RUNS_DIR: dir holding outdoor_screen_<ARM>/metrics.jsonl (default $PROJECT_DIR/runs or ./runs).
#   -s: smoke logs (checkpoint dirs outdoor_screen_smoke50_<ARM>; total steps 12050; LR steps above 12050 are reported as 'not reached', not FAIL).
# Exit 1 on any FAIL (=> scancel the job). 'WAIT' = the log has not reached that step yet (re-run later); it is not a FAIL.
RUNS="${PROJECT_DIR:-.}/runs"; SMOKE=0
while getopts "r:s" o; do case $o in r) RUNS="$OPTARG";; s) SMOKE=1;; *) exit 2;; esac; done; shift $((OPTIND-1))
[[ $# -ge 1 ]] || { sed -n 2,7p "$0"; exit 2; }
export RUNS SMOKE
exec "${PYTHON:-python3}" -I - "$@" <<'PY'
import json, os, re, sys, math
RUNS, SMOKE = os.environ["RUNS"], os.environ["SMOKE"] == "1"
TOTAL = 12050 if SMOKE else 14000
TS2_ARMS = {"TS2", "TS2_H80", "TS2_H300", "TS2_VK655"}; HINGE = {"TS2_H80", "TS2_H300"}
TABLE = {12020: 2.00e-05, 12040: 4.00e-05, 12060: 6.00e-05, 12080: 8.00e-05, 12100: 1.00e-04, 12120: 1.00e-04, 12200: 9.93e-05}
SHA = "f19d0b838252c9a2409498b35f46738d99b3646afba984d6a82ffea3cd610b76"
fails = 0
def out(arm, item, status, msg=""):
    global fails
    if status == "FAIL": fails += 1
    print(f"[{arm}] item {item}: {status} {msg}")
data0, loss0 = None, None
for spec in sys.argv[1:]:
    arm, _, log = spec.partition("=")
    txt = open(log, errors="replace").read().splitlines()
    steps = {int(m.group(1)): (float(m.group(3)), float(m.group(4))) for l in txt if (m := re.search(r"step(\d+)/(\d+) loss=(\S+) lr=(\S+)", l))}
    tot = {int(m.group(2)) for l in txt if (m := re.search(r"step(\d+)/(\d+) loss=", l))}
    print(f"=== {arm}  {log}  (last printed step {max(steps) if steps else None}) ===")
    # 1 LR
    for s, ref in sorted(TABLE.items()):
        if s > TOTAL: continue
        if s not in steps: out(arm, 1, "WAIT", f"step {s} not printed yet"); continue
        got = steps[s][1]; ok = abs(got / ref - 1) <= 0.015
        out(arm, 1, "ok" if ok else "FAIL", f"step {s} lr {got:.3e} expected {ref:.3e}")
    bad = [s for s, (_, lr) in steps.items() if 12000 <= s < 14000 and lr == 0.0]
    if bad: out(arm, 1, "FAIL", f"lr == 0 at steps {bad[:5]} (no-op restart bug)")
    # 2 resume
    res = [l for l in txt if l.startswith("[resume]") and re.search(r"step=12000\b", l)]
    out(arm, 2, "ok" if res and TOTAL in tot else ("WAIT" if res and not tot else "FAIL"), f"resume line {'found' if res else 'MISSING'}; totals printed {sorted(tot)} (expect {TOTAL})")
    # 3 hinge key in metrics.jsonl rows 12000 < step <= 12050
    mdir = f"outdoor_screen_smoke50_{arm}" if SMOKE else f"outdoor_screen_{arm}"
    mp = os.path.join(RUNS, mdir, "metrics.jsonl")
    if not os.path.exists(mp): out(arm, 3, "WAIT", f"{mp} missing")
    else:
        rows = [r for r in (json.loads(l) for l in open(mp) if '"loss_terms"' in l) if 12000 < r["step"] <= 12050]
        want = arm in HINGE
        mism = [r["step"] for r in rows if ("sky_hinge_weighted" in r) != want]
        vals = [r.get("sky_hinge_weighted") for r in rows]
        if len(rows) < 5: out(arm, 3, "WAIT", f"only {len(rows)} loss_terms rows (need 5)")
        elif mism: out(arm, 3, "FAIL", f"hinge key presence wrong (want {want}) at steps {mism}")
        elif want and any(not math.isfinite(v) or v < 0 for v in vals): out(arm, 3, "FAIL", f"non-finite/negative hinge {vals}")
        elif want and all(v == 0.0 for v in vals): out(arm, 3, "WARN", "hinge exactly 0 in all 5 rows: report as 'hinge inert at this start point' (O18), not as evidence")
        else: out(arm, 3, "ok", f"{len(rows)} rows, hinge key present={want}, values {vals if want else '-'}")
    # 4 TS2 startup line
    ts = [l for l in txt if l.startswith("[ts2] TartanAir TS2 sky/mask ON")]
    if arm in TS2_ARMS: out(arm, 4, "ok" if ts and SHA in ts[0] else "FAIL", f"[ts2] line {'with env_table_sha256 ' + SHA[:8] if ts and SHA in ts[0] else 'missing or wrong sha'}")
    else: out(arm, 4, "FAIL" if ts else "ok", "no [ts2] line expected" + (" but found one" if ts else ""))
    # 5 finite
    nf = [l for l in txt if "[warn] non-finite loss" in l and (m := re.search(r"step (\d+)", l)) and int(m.group(1)) <= 12050]
    out(arm, 5, "FAIL" if nf else "ok", f"{len(nf)} non-finite warnings in first 50 steps")
    # 6 effective bound
    b = [l for l in txt if l.startswith("[data] effective train depth upper bound")]
    want = "vkitti=656" if arm == "TS2_VK655" else "vkitti=655"
    if not b: out(arm, 6, "WAIT", "bound line not in log")
    else: out(arm, 6, "ok" if re.search(re.escape(want) + r"(,|$)", b[0]) else "FAIL", f"want {want}; got: {b[0][-120:]}")
    # 7 identical source list / counts vs the first (A0) log
    dl = [re.sub(r"seed=\d+", "seed=<S>" if arm == "A0b" else "seed=<S>", re.sub(r"vkitti=65[56]", "vkitti=<B>", l)) for l in txt if l.startswith("[data]")]   # the data seed is a declared A0b key (49 vs 45): ignored in the compare (the seed itself is checked by the config diff)
    if data0 is None: data0 = dl; out(arm, 7, "ok", f"reference ({len(dl)} [data] lines)")
    else: out(arm, 7, "ok" if dl == data0 else "FAIL", "[data] lines identical to reference" if dl == data0 else f"differs from reference: {[x for x in dl if x not in data0][:3]}")
    # spike rule: mean printed loss over the first printed steps vs A0's
    first = [steps[s][0] for s in sorted(steps) if s <= 12100]
    if first:
        m = sum(first) / len(first)
        if loss0 is None: loss0 = m; print(f"[{arm}] loss mean(steps<=12100, n={len(first)}) = {m:.4f} (reference)")
        else: print(f"[{arm}] loss mean = {m:.4f}, ratio to A0 {m / loss0:.2f} -> " + ("REPORT (>2x A0; Adam-moment mismatch?)" if m > 2 * loss0 else "ok"))
print("RESULT:", "FAIL (scancel)" if fails else "no FAIL (WAIT/WARN lines may remain)")
sys.exit(1 if fails else 0)
PY
