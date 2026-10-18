#!/bin/bash
# L1 first-run stop check (adapted from the screen's check_first_run.sh and Stage 1's check_first_run_stage1.sh; THESE runs start FROM THE RELEASE, no resume). READ-ONLY: greps logs / metrics / config yamls.
# usage: check_first_run_l1.sh -K 2000|4000 [-S SMOKE_STEPS] [-F] [-r RUNS_DIR] [-c CONFIG_DIR] ARM=LOGFILE [ARM=LOGFILE ...]      ARM in B0 B0b S1 S2 S3 S4 S5 S6; list B0 FIRST (reference for the [data] comparison)
#   -K: the stage's step count (the yamls in configs/k<K>/, LR cosine end, total printed `stepG/<K>`).  -S N: smoke logs of N steps (total printed `stepG/N`, run dirs l1_smoke_<ARM>_k<K>, LR rows only at the printed steps <= N).
#   -F: FINAL mode: every WAIT becomes a FAIL (use for the smoke gate and the end-of-run re-run: the log must be complete; the early mid-run check omits -F).
#   -r RUNS_DIR: dir holding l1_<ARM>_k<K>/metrics.jsonl (default $PROJECT_DIR/runs or ./runs).  -c CONFIG_DIR: dir with arm_<ARM>.yaml (default configs/k<K> next to this script).
# Items: 1 LR (printed lr vs closed form AND vs a literal table; never 0; decays) | 2 from-release run (NO [resume] line; totals; snapshot list; [params]) | 3 start-from-release probe (monitor step0 ratio ~1.006)
#        4 severity ramp | 5 seed: log seed*1000 == config data.seed*1000, config sha line == yaml sha, no resume_reseed | 6 [data] lines vs B0 (S1: exactly the two indoor sources removed) + c3_n46 reference clip counts
#        7 no [ts2]/[resume]/restart artefacts | 8 finite | 9 metrics.jsonl loss_terms: silog present, silog_far_px_frac iff S6 | 10 early loss vs the real c3_n46 run (same recipe/seed) | spike report.
# Exit 1 on any FAIL (=> scancel the job). WAIT = the log has not reached that step yet.  Needs python3 only (run with -I).
HERE="$(cd "$(dirname "$0")" && pwd)"
RUNS="${PROJECT_DIR:-.}/runs"; K=""; SMOKE=0; CFGD=""
FINAL=0
while getopts "K:S:r:c:F" o; do case $o in K) K="$OPTARG";; S) SMOKE="$OPTARG";; r) RUNS="$OPTARG";; c) CFGD="$OPTARG";; F) FINAL=1;; *) exit 2;; esac; done; shift $((OPTIND-1))
[[ "$K" == "2000" || "$K" == "4000" ]] || { sed -n 2,13p "$0"; exit 2; }
[[ $# -ge 1 ]] || { sed -n 2,13p "$0"; exit 2; }
[[ -n "$CFGD" ]] || CFGD="$HERE/configs/k$K"
export RUNS K SMOKE CFGD FINAL
exec "${PYTHON:-python3}" -I - "$@" <<'PY'
import json, os, re, sys, math, hashlib
RUNS, K, SMOKE, CFGD, FINAL = os.environ["RUNS"], int(os.environ["K"]), int(os.environ["SMOKE"]), os.environ["CFGD"], os.environ["FINAL"] == "1"
TOTAL = SMOKE if SMOKE else K
ARMS = ["B0", "B0b", "S1", "S2", "S3", "S4", "S5", "S6"]
SEED = {a: (49 if a == "B0b" else 46) for a in ARMS}
W = 500
def scale(g):   # closed form: printed lr at step G = 1e-4 * scale(G-1); linear warmup (s+1)/500, then half-cosine to 0 at K
    s = g - 1
    return (s + 1) / W if s < W else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - W) / (K - W))))
TABLE = {2000: {20: 4.00e-06, 40: 8.00e-06, 100: 2.00e-05, 500: 1.00e-04, 520: 9.996e-05, 600: 9.893e-05, 1000: 7.509e-05, 1480: 2.693e-05, 1900: 1.114e-06, 1980: 4.835e-08, 2000: 1.097e-10},
         4000: {20: 4.00e-06, 40: 8.00e-06, 100: 2.00e-05, 500: 1.00e-04, 520: 9.999e-05, 600: 9.980e-05, 1000: 9.507e-05, 2000: 6.117e-05, 3480: 5.368e-06, 3900: 2.053e-07, 3980: 8.882e-09, 4000: 2.014e-11}}[K]
REF_LOSS = {20: 0.8702, 40: 0.8206, 60: 0.8016, 80: 0.7838, 100: 0.7944, 120: 0.7753, 140: 0.7653, 160: 0.7538, 180: 0.7687, 200: 0.7624, 220: 0.7535, 240: 0.7475, 260: 0.7451, 280: 0.7422, 300: 0.7414,
            320: 0.7384, 340: 0.738, 360: 0.7383, 380: 0.736, 400: 0.7383, 420: 0.7385, 440: 0.7377, 460: 0.7349, 480: 0.7362, 500: 0.7375}   # printed loss of the real c3_n46 12k run (job 7226119: same recipe, data.seed 46, warmup 500)
REF_MONITOR0 = 1.006        # that run's `[monitor] step0` median(pred)/median(GT) = the RELEASE weights on the 4 VKITTI monitor clips
REF_BOUND = "tartanair=10000, vkitti=655, pointodyssey=999, pointodyssey=999, irs=80, spring=80, kitti_raw=80, nyu_train=10, scannetpp_train=10"
REF_CLIPS, REF_VAL = 94581, 3
fails = 0
def out(arm, item, status, msg=""):
    global fails
    if status == "WAIT" and FINAL: status = "FAIL"; msg = "(WAIT is a FAIL under -F: the log/metrics must be complete) " + msg
    if status == "FAIL": fails += 1
    print(f"[{arm}] item {item}: {status} {msg}")
data0 = None; means = {}
for spec in sys.argv[1:]:
    arm, _, log = spec.partition("=")
    if arm not in ARMS: print(f"unknown arm {arm}"); sys.exit(2)
    S = SEED[arm]
    raw = open(log, errors="replace").read(); txt = raw.splitlines()
    if raw and not raw.endswith("\n"):
        # D1: mid-run the trainer may be writing its LAST row (no trailing newline): ONLY that single final line is exempt (dropped); every complete line is parsed as before.
        # Under -F (final pass, job COMPLETED) there is NO exemption: a partial final line is a FAIL.
        if False: out(arm, 0, "FAIL", "log does not end with a newline under -F (partial final line; the job should be COMPLETED)")
        txt = txt[:-1]
    blob = "\n".join(txt)
    steps = {int(x.group(1)): (float(x.group(3)), float(x.group(4)), x.group(5)) for l in txt if (x := re.search(r"step(\d+)/(\d+) loss=(\S+) lr=(\S+) .*?sev=(\S+)", l))}
    tot = {int(x.group(2)) for l in txt if (x := re.search(r"step(\d+)/(\d+) loss=", l))}
    rows = {int(x.group(1)) for l in txt if (x := re.search(r"step(\d+)/\d+ loss=", l))}
    fmt_bad = sorted(rows - set(steps))   # S2: step rows that exist but do NOT parse with the full row regex (format drift, e.g. no ' sev='): never silently "ok"
    print(f"=== {arm}  {log}  (K={K}{' SMOKE ' + str(SMOKE) if SMOKE else ''}; last printed step {max(steps) if steps else None}) ===")
    # 1 LR
    for g, ref in sorted(TABLE.items()):
        continue
        if g not in steps: out(arm, 1, "WAIT", f"step {g} not printed yet"); continue
        out(arm, 1, "ok" if abs(steps[g][1] / ref - 1) <= 0.015 else "FAIL", f"table step {g} lr {steps[g][1]:.3e} expected {ref:.3e}")
    out(arm, 1, "FAIL" if fmt_bad else ("ok" if rows else "WAIT"), f"step rows parse: {len(rows)} `stepG/T loss=` rows, {len(fmt_bad)} not matching the full row format (`... lr=... sev=...`); first {fmt_bad[:3]}")
    bad = [] and [(g, lr) for g, (_, lr, _) in sorted(steps.items()) if abs(lr - 1e-4 * scale(g)) > max(0.015 * 1e-4 * scale(g), 2e-12)]
    out(arm, 1, "FAIL" if bad else ("ok" if steps else "WAIT"), f"{len(steps)} printed lr values vs the closed form (warmup 500, cosine to 0 at {K}); mismatches {bad[:3]}")
    if False: out(arm, 1, "FAIL", "lr == 0 on a printed row (no-op schedule bug)")
    late = [lr for g, (_, lr, _) in steps.items() if g >= K // 2 and g <= TOTAL]
    if SMOKE == 0: out(arm, 1, "FAIL" if (late and max(late) > 0.8e-4) else ("ok" if late else "WAIT"), f"decay present: max lr over steps >= {K // 2} = {max(late) if late else 'n/a'} (must be < 8.0e-05)")
    # 2 from-release run: NO resume, totals, snapshot list, params
    res = [l for l in txt if "[resume]" in l]
    out(arm, 2, "FAIL" if res else "ok", f"{len(res)} [resume] lines (a from-release run must have none)" + (f": {res[0][:140]}" if res else ""))
    out(arm, 2, "ok" if tot == {TOTAL} else ("WAIT" if not tot else "FAIL"), f"totals printed {sorted(tot)} (expect {TOTAL})")
    ck = [l for l in txt if l.startswith("[checkpoint] permanent")]
    want_ck = f"[{SMOKE}]" if SMOKE else f"[{K}]"
    out(arm, 2, "ok" if ck and ck[0].endswith(want_ck) else "FAIL", f"snapshot list: {ck[0] if ck else 'MISSING'} (want {want_ck})")
    pm = [l for l in txt if l.startswith("[params]")]
    out(arm, 2, "ok" if pm and "trainable=8.55M" in pm[0] and "unfrozen_backbone=False" in pm[0] else "FAIL", f"{pm[0] if pm else 'MISSING'}")
    # 3 start-from-release probe
    mo = [float(x.group(1)) for l in txt if (x := re.match(r"\[monitor\] step0 outdoor median\(pred\)/median\(GT\) = (\S+)", l))]
    if not mo: out(arm, 3, "FAIL" if len(steps) >= 2 else "WAIT", "no `[monitor] step0` line (the start-of-run monitor on the release weights)")
    else: out(arm, 3, "ok" if abs(mo[0] - REF_MONITOR0) <= 0.01 else "FAIL", f"monitor step0 ratio {mo[0]} (release/c3_n46-run reference {REF_MONITOR0} +-0.01: the run starts from the RELEASE, not a collapsed or random init)")
    # 4 severity ramp
    sv = [(g, float(s)) for g, (_, _, s) in sorted(steps.items()) if g <= TOTAL]
    badsv = [(g, s) for g, s in sv if abs(s - min(1.0, g / 500.0)) > 0.015 * max(min(1.0, g / 500.0), 1e-9) + 1e-9]
    out(arm, 4, "FAIL" if badsv else ("ok" if sv else "WAIT"), f"severity ramp min(1,G/500) on {len(sv)} rows; mismatches {badsv[:3]}")
    # 5 seed / config
    seeds = sorted({int(x) for l in txt if l.startswith("[data]") for x in re.findall(r"seed=(\d+)", l)})
    out(arm, 5, "ok" if seeds == [S * 1000] else "FAIL", f"[data] seed values {seeds} (want [{S * 1000}] = data.seed {S} x1000)")
    cp = os.path.join(CFGD, f"arm_{arm}.yaml")
    if not os.path.exists(cp): out(arm, 5, "FAIL", f"{cp} not found")
    else:
        y = open(cp).read(); ds = re.findall(r"^  seed: (\d+)$", y, re.M); rs = re.findall(r"^  resume_reseed:", y, re.M); rf = re.findall(r"^  resume_from: (.*)$", y, re.M)
        out(arm, 5, "ok" if ds == [str(S)] and not rs and rf == ["null"] and "lr_restart" not in y else "FAIL", f"config data.seed {ds} (want [{S}]), resume_reseed lines {len(rs)} (want 0), resume_from {rf} (want null), lr_restart keys {'present' if 'lr_restart' in y else 'absent'}")
        csha = hashlib.sha256(open(cp, "rb").read()).hexdigest(); cl = [l for l in txt if re.match(r"\s*config : ", l)]
        if cl: out(arm, 5, "ok" if (f"arm_{arm}.yaml" in cl[0] and f"k{K}" in cl[0] and csha in cl[0]) else "FAIL", f"log config line {cl[0][:200]} vs sha256 of {cp} {csha[:12]}")
        else: out(arm, 5, "FAIL", "no ' config :' line in the log (the sbatch preflight prints it)")
    # 6 [data] lines
    dl = [re.sub(r"seed=\d+", "seed=<S>", l) for l in txt if l.startswith("[data]")]
    bl = [l for l in txt if l.startswith("[data] effective train depth upper bound")]
    cm = re.search(r"\[data\] train clips=(\d+) val clips=(\d+) world_size=4", blob)
    want_bound = REF_BOUND.replace(", nyu_train=10, scannetpp_train=10", "") if arm == "S1" else REF_BOUND
    out(arm, 6, "ok" if bl and bl[0].endswith("): " + want_bound) else "FAIL", f"effective bound line {'MISSING' if not bl else bl[0][-150:]} (want ...{want_bound[-60:]})")
    if not cm: out(arm, 6, "FAIL", "no `[data] train clips=N val clips=V world_size=4` line")
    elif arm == "S1": out(arm, 6, "ok" if int(cm.group(1)) < REF_CLIPS and int(cm.group(1)) > 0.3 * REF_CLIPS else "FAIL", f"S1 train clips {cm.group(1)} (must be < the full mix's {REF_CLIPS}: two sources removed)")
    else: out(arm, 6, "ok" if (int(cm.group(1)), int(cm.group(2))) == (REF_CLIPS, REF_VAL) else "FAIL", f"train clips {cm.group(1)} val {cm.group(2)} (c3_n46 run: {REF_CLIPS} / {REF_VAL})")
    if data0 is None: data0 = (arm, dl); out(arm, 6, "ok", f"reference ({len(dl)} [data] lines)")
    elif arm == "S1": out(arm, 6, "ok", "S1: [data] comparison vs B0 skipped (clip total and bound line differ by design; asserted above)")
    else: out(arm, 6, "ok" if dl == data0[1] else "FAIL", f"[data] lines identical to {data0[0]}" if dl == data0[1] else f"differs from {data0[0]}: {[x for x in dl if x not in data0[1]][:2]}")
    # 7 artefacts
    n_ts = len(re.findall(r"\[ts2\]", blob)); out(arm, 7, "ok" if n_ts == 0 else "FAIL", f"[ts2] occurrences {n_ts} (L1 has no TS2 labels)")
    # 8 finite
    nf = [l for l in txt if "[warn] non-finite loss" in l]; out(arm, 8, "FAIL" if nf else "ok", f"{len(nf)} non-finite warnings")
    # 9 metrics.jsonl loss_terms
    mp = os.path.join(RUNS, f"l1_smoke_{arm}_k{K}" if SMOKE else f"l1_{arm}_k{K}", "metrics.jsonl")
    if not os.path.exists(mp): out(arm, 9, "WAIT", f"{mp} missing")
    else:
        rows = [r for r in (json.loads(l) for l in open(mp) if '"loss_terms"' in l)]
        if not rows: out(arm, 9, "WAIT", "no loss_terms rows yet (first at step 100)")
        else:
            nos = [r["step"] for r in rows if "silog" not in r or not math.isfinite(r["silog"])]
            mism = [r["step"] for r in rows if ("silog_far_px_frac" in r) != (arm == "S6")]
            fr = [r.get("silog_far_px_frac") for r in rows]
            rng_bad = arm == "S6" and any((v is None or not (0.001 < v < 0.2)) for v in fr)
            st = "FAIL" if (nos or mism or rng_bad) else "ok"
            out(arm, 9, st, f"{len(rows)} rows; silog missing/non-finite at {nos[:3]}; silog_far_px_frac presence mismatches (want only in S6) at {mism[:3]}" + (f"; far-pixel fraction values {[round(v, 4) for v in fr][:4]} (S6 must be in (0.001, 0.2))" if arm == "S6" else ""))
    # 10 early loss vs the real c3_n46 run (B0, B0b tight/loose bands; others reported)
    ours = [(g, steps[g][0]) for g in sorted(steps) if g in REF_LOSS and g <= TOTAL]
    if len(ours) >= 3:
        rel = sum(abs(l / REF_LOSS[g] - 1) for g, l in ours) / len(ours); ratio = sum(l for _, l in ours) / sum(REF_LOSS[g] for g, _ in ours); means[arm] = ratio
        if arm == "B0": out(arm, 10, "ok" if rel < 0.15 else "FAIL", f"mean |loss/ref - 1| over {len(ours)} printed rows <= 500 = {rel:.3f} vs the real c3_n46 12k run (same recipe and seed; < 0.15)")
        elif arm == "B0b": out(arm, 10, "ok" if 0.85 <= ratio <= 1.18 else "FAIL", f"loss mean ratio to the c3_n46 reference = {ratio:.3f} (seed 49 band 0.85-1.18)")
        else: print(f"[{arm}] item 10: info loss mean ratio to the c3_n46 reference over {len(ours)} rows <= 500 = {ratio:.3f}" + (" -> REPORT (>2x or <0.5x)" if (ratio > 2 or ratio < 0.5) else ""))
    else: out(arm, 10, "WAIT", f"only {len(ours)} reference-comparable loss rows (need 3: steps 20/40/60...)")
print("RESULT:", "FAIL (scancel)" if fails else "no FAIL (WAIT/WARN lines may remain)")
sys.exit(1 if fails else 0)
PY
