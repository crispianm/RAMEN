"""Mutation test of check_first_run_l1.sh (second review S3/S2): each mutation is applied to a COPY of the checker; the package's own negative-control suite is re-run against it;
CAUGHT = the suite reports a WRONG line / non-zero exit. usage: python3 mutants_stop_l1.py [outdir]   (outdir default: ./mutants_stop_work, written under the job tmp dir by the caller)"""
import json, shutil, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
SRC = Path(__file__).resolve().parent; REAL = Path(sys.argv[2] if len(sys.argv) > 2 else "/home/wg19671/.claude/jobs/a0d391cb/tmp/scr_logs/dn01_c3rep_20261009_c3_n46_7226119.out")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "mutants_stop_work").resolve(); OUT.mkdir(parents=True, exist_ok=True)
CHK = (SRC / "check_first_run_l1.sh").read_text()
M = {
 "M01 LR tol 1.5%->50% (table+closed form)": [('<= 0.015 else "FAIL", f"table step', '<= 0.5 else "FAIL", f"table step'), ("max(0.015 * 1e-4 * scale(g), 2e-12)", "max(0.5 * 1e-4 * scale(g), 2e-12)")],
 "M01a table tol only ->50%": [('<= 0.015 else "FAIL", f"table step', '<= 0.5 else "FAIL", f"table step')],
 "M01b closed-form tol only ->50%": [("max(0.015 * 1e-4 * scale(g), 2e-12)", "max(0.5 * 1e-4 * scale(g), 2e-12)")],
 "M01c closed-form check removed": [("bad = [(g, lr) for", "bad = [] and [(g, lr) for")],
 "M01d table check removed": [("        if g > TOTAL: continue\n", "        continue\n")],
 "M01e closed-form AND table removed (decay check only)": [("bad = [(g, lr) for", "bad = [] and [(g, lr) for"), ("        if g > TOTAL: continue\n", "        continue\n")],
 "M02 decay check removed": [('if SMOKE == 0: out(arm, 1, "FAIL" if (late and max(late) > 0.8e-4)', 'if False: out(arm, 1, "FAIL" if (late and max(late) > 0.8e-4)')],
 "M02+M01c decay and closed form removed": [('if SMOKE == 0: out(arm, 1, "FAIL" if (late and max(late) > 0.8e-4)', 'if False: out(arm, 1, "FAIL" if (late and max(late) > 0.8e-4)'), ("bad = [(g, lr) for", "bad = [] and [(g, lr) for")],
 "M03 lr==0 check removed": [("if [g for g, (_, lr, _) in steps.items() if lr == 0.0]:", "if False:")],
 "M04 [resume] check skipped": [('res = [l for l in txt if "[resume]" in l]', "res = []")],
 "M05 totals accepted": [('"ok" if tot == {TOTAL} else', '"ok" if True else')],
 "M06 snapshot list accepted": [('"ok" if ck and ck[0].endswith(want_ck) else "FAIL"', '"ok" if True else "FAIL"')],
 "M07 [params] accepted": [('"ok" if pm and "trainable=8.55M" in pm[0] and "unfrozen_backbone=False" in pm[0] else "FAIL"', '"ok" if True else "FAIL"')],
 "M07b [params] unfrozen check dropped": [(' and "unfrozen_backbone=False" in pm[0] else "FAIL"', ' else "FAIL"')],
 "M08 missing step0 monitor accepted": [('out(arm, 3, "FAIL" if len(steps) >= 2 else "WAIT", "no', 'out(arm, 3, "ok", "no')],
 "M08b missing step0 monitor -> WAIT only": [('out(arm, 3, "FAIL" if len(steps) >= 2 else "WAIT", "no', 'out(arm, 3, "WAIT", "no')],
 "M08c monitor tolerance 0.01->0.5": [("abs(mo[0] - REF_MONITOR0) <= 0.01", "abs(mo[0] - REF_MONITOR0) <= 0.5")],
 "M09 severity ramp removed": [("badsv = [(g, s) for", "badsv = [] and [(g, s) for")],
 "M10 log seed: any seed accepted": [('"ok" if seeds == [S * 1000] else "FAIL"', '"ok" if seeds else "FAIL"')],
 "M10b log seed: 46000 or 49000 accepted": [('"ok" if seeds == [S * 1000] else "FAIL"', '"ok" if seeds in ([46000], [49000]) else "FAIL"')],
 "M11 yaml checks (seed/reseed/resume_from/lr_restart) removed": [('"ok" if ds == [str(S)] and not rs and rf == ["null"] and "lr_restart" not in y else "FAIL"', '"ok" if True else "FAIL"')],
 "M11a yaml resume_reseed dropped": [("ds == [str(S)] and not rs and", "ds == [str(S)] and")],
 "M11b yaml resume_from dropped": [('rf == ["null"] and "lr_restart"', '"lr_restart"')],
 "M11c yaml lr_restart dropped": [('rf == ["null"] and "lr_restart" not in y else', 'rf == ["null"] else')],
 "M11d yaml seed dropped": [('"ok" if ds == [str(S)] and not rs', '"ok" if not rs')],
 "M12 config line sha dropped": [("and csha in cl[0])", "and True)")],
 "M12a config line arm name dropped": [('(f"arm_{arm}.yaml" in cl[0] and', '(True and')],
 "M12b config line k dropped": [('and f"k{K}" in cl[0] and', 'and True and')],
 "M12c missing config line accepted": [('else: out(arm, 5, "FAIL", "no \' config :\' line', 'else: out(arm, 5, "ok", "no \' config :\' line')],
 "M13 bound line accepted": [('out(arm, 6, "ok" if bl and bl[0].endswith("): " + want_bound) else "FAIL"', 'out(arm, 6, "ok" if True else "FAIL"')],
 "M14 non-S1 clip counts accepted": [('"ok" if (int(cm.group(1)), int(cm.group(2))) == (REF_CLIPS, REF_VAL) else "FAIL"', '"ok" if True else "FAIL"')],
 "M14b S1 clip count accepted": [('"ok" if int(cm.group(1)) < REF_CLIPS and int(cm.group(1)) > 0.3 * REF_CLIPS else "FAIL"', '"ok" if True else "FAIL"')],
 "M14c world_size any": [(r"val clips=(\d+) world_size=4", r"val clips=(\d+) world_size=\d+")],
 "M15 [data] vs B0 comparison dropped": [('else: out(arm, 6, "ok" if dl == data0[1] else "FAIL"', 'else: out(arm, 6, "ok" if True else "FAIL"')],
 "M16 [ts2] dropped": [('out(arm, 7, "ok" if n_ts == 0 else "FAIL"', 'out(arm, 7, "ok" if True else "FAIL"')],
 "M17 non-finite dropped": [('out(arm, 8, "FAIL" if nf else "ok"', 'out(arm, 8, "ok"')],
 "M18 silog-missing dropped": [('nos = [r["step"] for r in rows if "silog" not in r or not math.isfinite(r["silog"])]', "nos = []")],
 "M18b far-frac presence dropped": [('mism = [r["step"] for r in rows if ("silog_far_px_frac" in r) != (arm == "S6")]', "mism = []")],
 "M18c far-frac range dropped": [('rng_bad = arm == "S6" and any(', 'rng_bad = False and any(')],
 "M18d missing metrics.jsonl -> ok": [('out(arm, 9, "WAIT", f"{mp} missing")', 'out(arm, 9, "ok", f"{mp} missing")')],
 "M18e no loss_terms rows -> ok": [('out(arm, 9, "WAIT", "no loss_terms rows yet', 'out(arm, 9, "ok", "no loss_terms rows yet')],
 "M19 B0 loss band 0.15->10": [("rel < 0.15", "rel < 10")],
 "M19b B0b loss band 0.85-1.18 -> 0.1-10": [("0.85 <= ratio <= 1.18", "0.1 <= ratio <= 10")],
 "M19c item 10 <3 rows -> ok": [('else: out(arm, 10, "WAIT", f"only', 'else: out(arm, 10, "ok", f"only')],
 "M20 -F does not turn WAIT into FAIL": [('if status == "WAIT" and FINAL:', 'if False:')],

 "S2a step-row format check removed": [('out(arm, 1, "FAIL" if fmt_bad else', 'out(arm, 1, "ok" if True else')],
 "S2b zero step rows -> ok (closed-form line)": [('("ok" if steps else "WAIT"), f"{len(steps)} printed lr values', '"ok", f"{len(steps)} printed lr values')],
 "S2c step-row format: WAIT instead of FAIL": [('out(arm, 1, "FAIL" if fmt_bad else', 'out(arm, 1, "WAIT" if fmt_bad else')],
}

def one(item):
    name, edits = item
    d = OUT / name.split()[0]
    if d.exists(): shutil.rmtree(d)
    d.mkdir(); shutil.copytree(SRC / "configs", d / "configs"); shutil.copy(SRC / "make_fixtures_and_negctl_l1.py", d)
    t = CHK
    for o, n in edits:
        assert t.count(o) == 1, f"{name}: pattern found {t.count(o)}x: {o!r}"
        t = t.replace(o, n)
    assert t != CHK
    (d / "check_first_run_l1.sh").write_text(t)
    r = subprocess.run([sys.executable, str(d / "make_fixtures_and_negctl_l1.py"), str(REAL)], capture_output=True, text=True, cwd=d)
    wrong = [l for l in r.stdout.splitlines() if "WRONG" in l]
    return name, (r.returncode != 0 or bool(wrong)), wrong
with ThreadPoolExecutor(6) as ex: res = list(ex.map(one, M.items()))
for name, caught, wrong in res: print(f"{'CAUGHT ' if caught else 'MISSED '} {name}  ({len(wrong)} WRONG)" + "".join("\n      " + w[:170] for w in wrong[:2]))
print("MISSED total:", sum(1 for r in res if not r[1]), "of", len(res))
