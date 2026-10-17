# L1 mutant table (second review fix round)

Method: each mutation is applied to a COPY of the shipped file and the package's own suite is re-run against it (`mutants_stop_l1.py`, `mutants_opt_l1.py`, `mutants_readout_l1.py` / `test_l1_readout.py` source mutants). CAUGHT = the suite reports a WRONG line (a NEG that returned rc 0, or a POSITIVE that failed) or a non-zero exit. The shipped files give 0 WRONG (control `real`).

## A. check_first_run_l1.sh (48 mutants; reviewer's M01-M20 = 20 SURVIVORS before, plus 3 new S2 mutants): MISSED now = 0

| mutant | before | now | first observed failing control (output of the mutated run) |
|---|---|---|---|
| M01 LR tol 1.5%->50% (table+closed form) | SURVIVED | CAUGHT | NEG[iso] (1) lr x1.05 everywhere (beyond the 1.5% tolerance): rc=0 expected FAIL -> WRONG |
| M01a table tol only ->50% | SURVIVED | CAUGHT | NEG[iso-sub-check] the table check ALONE catches: lr x0.97: rc=0 expected FAIL -> WRONG |
| M01b closed-form tol only ->50% | SURVIVED | CAUGHT | NEG[iso-sub-check] the closed-form check ALONE catches: lr x0.97: rc=0 expected FAIL -> WRONG |
| M01c closed-form check removed | SURVIVED | CAUGHT | WRONG isolation builder 'tb': sub-check text not found exactly once in check_first_run_l1.sh ('bad = [(g, lr) for') -> that sub-check was removed or altered |
| M01d table check removed | SURVIVED | CAUGHT | WRONG isolation builder 'cf': sub-check text not found exactly once in check_first_run_l1.sh ('        if g > TOTAL: continue\n') -> that sub-check was removed or alte |
| M02 decay check removed | SURVIVED | CAUGHT | WRONG isolation builder 'cf': sub-check text not found exactly once in check_first_run_l1.sh ('if SMOKE == 0: out(arm, 1, "FAIL" if (late and max(late) > 0') -> that s |
| M02+M01c decay and closed form removed | SURVIVED | CAUGHT | WRONG isolation builder 'cf': sub-check text not found exactly once in check_first_run_l1.sh ('if SMOKE == 0: out(arm, 1, "FAIL" if (late and max(late) > 0') -> that s |
| M03 lr==0 check removed | SURVIVED | CAUGHT | WRONG isolation builder 'cf': sub-check text not found exactly once in check_first_run_l1.sh ('if [g for g, (_, lr, _) in steps.items() if lr == 0.0]:') -> that sub-ch |
| M08b missing step0 monitor -> WAIT only | SURVIVED | CAUGHT | MID no monitor step0 line (item 3 FAIL once >= 2 step rows exist, not WAIT): rc=0 expected FAIL -> WRONG |
| M11 yaml checks (seed/reseed/resume_from/lr_restart) removed | SURVIVED | CAUGHT | NEG[iso] (5) resume_reseed in the yaml (log sha line updated to the edited yaml: only the content check can fail): rc=0 expected FAIL -> WRONG |
| M11a yaml resume_reseed dropped | SURVIVED | CAUGHT | NEG[iso] (5) resume_reseed in the yaml (log sha line updated to the edited yaml: only the content check can fail): rc=0 expected FAIL -> WRONG |
| M11b yaml resume_from dropped | SURVIVED | CAUGHT | NEG[iso] (5) resume_from set in the yaml (log sha line updated to the edited yaml: only the content check can fail): rc=0 expected FAIL -> WRONG |
| M11c yaml lr_restart dropped | SURVIVED | CAUGHT | NEG[iso] (5) lr_restart key in the yaml (log sha line updated to the edited yaml: only the content check can fail): rc=0 expected FAIL -> WRONG |
| M11d yaml seed dropped | SURVIVED | CAUGHT | NEG[iso] (5) data.seed 47 in the yaml (log sha line updated to the edited yaml: only the content check can fail): rc=0 expected FAIL -> WRONG |
| M14 non-S1 clip counts accepted | SURVIVED | CAUGHT | NEG[iso] (6) B0 (the reference) clip total 94580: rc=1 expected FAIL -> WRONG |
| M14c world_size any | SURVIVED | CAUGHT | NEG[iso] (6) world_size=2 on EVERY arm (no arm differs from the reference): rc=0 expected FAIL -> WRONG |
| M15 [data] vs B0 comparison dropped | SURVIVED | CAUGHT | NEG[iso] (6) S3 carries an extra [data] line that B0 lacks (clip counts and bound line equal): rc=0 expected FAIL -> WRONG |
| M18d missing metrics.jsonl -> ok | SURVIVED | CAUGHT | NEG[iso] (9) complete logs, NO metrics.jsonl at all, -F: rc=0 expected FAIL -> WRONG |
| M18e no loss_terms rows -> ok | SURVIVED | CAUGHT | NEG[iso] (9) complete logs, metrics.jsonl with no loss_terms rows, -F: rc=0 expected FAIL -> WRONG |
| M19c item 10 <3 rows -> ok | SURVIVED | CAUGHT | NEG[iso] (10) only 2 reference-comparable loss rows, -F: item 10 itself FAILs: rc=1 expected FAIL -> WRONG |
| S2a step-row format check removed | new | CAUGHT | NEG[iso] (S2) the same, MID-RUN (no -F): FAIL, not WAIT/ok: rc=0 expected FAIL -> WRONG |
| S2b zero step rows -> ok (closed-form line) | new | CAUGHT | POSITIVE-WAIT (S2): header-only log mid-run: rc=0, item 1 is WAIT and never 'ok' -> WRONG |
| S2c step-row format: WAIT instead of FAIL | new | CAUGHT | NEG[iso] (S2) the same, MID-RUN (no -F): FAIL, not WAIT/ok: rc=0 expected FAIL -> WRONG |

All other stop-check mutants (the 28 that already failed the old suite) remain CAUGHT: 48 of 48 CAUGHT in total.

## B. check_optimizer_lr.py (18 mutants + control `real`): MISSED = only the control

| mutant | now | first observed failing control |
|---|---|---|
| real | control (0 WRONG expected) | - |
| O1 cfg compare removed | CAUGHT | FORCED-FAIL (the second review's B1): the same correct smoke checkpoint WITHOUT --smoke is rejected (full-run yaml compared): rc=0 expected FAIL -> WRONG |
| O2 group sizes removed | CAUGHT | NEG wrong group sizes: rc=0 expected FAIL -> WRONG |
| O3 initial_lr check removed | CAUGHT | WRONG isolation builder 'ratio_only': check text not found exactly once in check_optimizer_lr.py ('len(pg) == 3 and all(abs(g.get("initial_lr", -1) /') |
| O4 saved-lr vs closed form removed | CAUGHT | WRONG isolation builder 'init_only': check text not found exactly once in check_optimizer_lr.py ('len(pg) == 3 and all(abs(g["lr"] - g["initial_lr"]') |
| O4b saved-lr tol 1e-6 -> 0.5 | CAUGHT | WRONG isolation builder 'init_only': check text not found exactly once in check_optimizer_lr.py ('len(pg) == 3 and all(abs(g["lr"] - g["initial_lr"]') |
| O5 ratio check removed | CAUGHT | WRONG isolation builder 'init_only': check text not found exactly once in check_optimizer_lr.py ('len(pg) == 3 and pg[0]["lr"] > 0 and abs(pg[1]["lr') |
| O3+O5 initial_lr and ratio removed | CAUGHT | WRONG isolation builder 'init_only': check text not found exactly once in check_optimizer_lr.py ('len(pg) == 3 and pg[0]["lr"] > 0 and abs(pg[1]["lr') |
| O6 --expect-step ignored | CAUGHT | NEG expected step mismatch (--expect-step 1000, ckpt at 500): rc=0 expected FAIL -> WRONG |
| O7 final-snapshot step check removed | CAUGHT | NEG -F: step2000.pt whose saved step is 1500: rc=0 expected FAIL -> WRONG |
| O8 0<step<=K removed | CAUGHT | NEG[iso] step-range check ALONE: step 2100 > K (lr consistent = 0): rc=0 expected FAIL -> WRONG |
| O9 -F does not turn unreadable into FAIL | CAUGHT | NEG -F: truncated file => FAIL: rc=0 expected FAIL -> WRONG |
| O10 S5 head accepted at either 1e-4 or 1e-5 | CAUGHT | WRONG isolation builder 'init_only': check text not found exactly once in check_optimizer_lr.py ('len(pg) == 3 and pg[0]["lr"] > 0 and abs(pg[1]["lr') |
| B1a --smoke: substitutions not applied (want = unmodified arm yaml) | CAUGHT | POSITIVE SMOKE B0 K=2000 (cfg = the sbatch sed of the arm yaml, step 120): rc=1 expected no FAIL -> WRONG |
| B1b --smoke: max_steps substitution removed | CAUGHT | NEG SMOKE[iso] substitution count: the sbatch sed would not match max_steps (cfg compare alone would PASS): rc=1 expected FAIL -> WRONG |
| B1c --smoke: checkpoint_dir substitution removed | CAUGHT | POSITIVE SMOKE B0 K=2000 (cfg = the sbatch sed of the arm yaml, step 120): rc=1 expected no FAIL -> WRONG |
| B1d --smoke: checkpoint_at_steps substitution removed | CAUGHT | POSITIVE SMOKE B0 K=2000 (cfg = the sbatch sed of the arm yaml, step 120): rc=1 expected no FAIL -> WRONG |
| B1e --smoke: substitution-count check removed | CAUGHT | NEG SMOKE[iso] substitution count: the sbatch sed would not match max_steps (cfg compare alone would PASS): rc=1 expected FAIL -> WRONG |
| B1f --smoke: cfg compare removed in smoke mode | CAUGHT | NEG SMOKE one UNDECLARED key changed (data.seed 47): rc=0 expected FAIL -> WRONG |

Mutants O3 (initial_lr), O5 (ratio), O8 (step range) and B1e (smoke substitution count) SURVIVED the first version of this round's suite (redundant checks); they are now killed by the `NEG[iso]` controls (each check alone, via probe variants with the other checks removed; the count check by a yaml whose max_steps line the sbatch sed would not match).

## C. l1_readout.py source mutants (`test_l1_readout.py`, printed by `mutants_readout_l1.py`): MISSED = 0

The reviewer's 6 survivors are: bin >= -> >, leg V > -> >=, leg D > -> >=, guard NF floor, guard abs, NF_d floor.

| mutant | status | first observed mismatch |
|---|---|---|
| G0 D <= -> < | CAUGHT | G0 boundary 0.93/0.75 fails |
| G0 V <= -> < | CAUGHT | G0 boundary 0.93/0.75 fails |
| N1 slack removed (float tie at the 0.01 floor) | CAUGHT | case ('floor tie 0.72->0.71', 'legD False', 'legD True') |
| NF_b 0.01 floor dropped | CAUGHT | case ('floor tie 0.72->0.71', 'legD False', 'legD True') |
| NF_d 0.01 floor dropped | CAUGHT | case ('NF_d floor .0099', 'legD True', 'False') |
| NF_p 1% relative floor dropped | CAUGHT | case ('NF_p 1% floor', 'legV True', 'False') |
| S1 exemption dropped | CAUGHT | case ('S1 nyu abs_rel worse: reported only', 'CAUSE-IMPLICATED (guard fails:', 'FIX-CANDIDATE') |
| S1 exemption widened to kitti | CAUGHT | case ('S1 kitti abs_rel worse: gated', 'FIX-CANDIDATE (= CAUSE-IMPLICA', 'CAUSE-IMPLICATED') |
| bin >= -> > (growth exactly NF_b is outside; with the slack also removed) | CAUGHT | case ('[80,120) grows by exactly NF_b=.03 (<= -> inside)', 'PRIMARY-ONLY', 'FIX-CANDIDATE') |
| dev() replaced by signed 1 - x | CAUGHT | case ('x_V overshoots too far (1.20: dev .20 > .15)', 'FIX-CANDIDATE (= CAUSE-IMPLICA', 'DDAD-ONLY') |
| guard NF 1% relative floor dropped | CAUGHT | case ('kitti guard worse by exactly NF (.0014) -> within', 'CAUSE-IMPLICATED (guard fails:', 'FIX-CANDIDATE') |
| guard NF |B0 - B0b| dropped (floor only) | CAUGHT | case ('leg V exactly = NF_p (strict >) -> no', 'FIX-CANDIDATE (= CAUSE-IMPLICA', 'DDAD-ONLY') |
| guard w > nf -> >= | CAUGHT | case ('exact guard tie', 'CAUSE-IMPLICATED (gu', 'FIX') |
| guard worse test uses abs() | CAUGHT | case ('guard better (lower) is fine', 'CAUSE-IMPLICATED (guard fails:', 'FIX-CANDIDATE') |
| leg D > -> >= | CAUGHT | case ('leg D exactly = NF_d (strict >) -> no', 'FIX-CANDIDATE (= CAUSE-IMPLICA', 'PRIMARY-ONLY') |
| leg V > -> >= | CAUGHT | case ('leg V exactly = NF_p (strict >) -> no', 'FIX-CANDIDATE (= CAUSE-IMPLICA', 'DDAD-ONLY') |
| S5 ckpt sha cross-file compare removed | CAUGHT | CLI expectations violated |
| S5 ddad file step not checked | CAUGHT | CLI expectations violated |
| S5 integrity failure returns 0 (continues) | CAUGHT | CLI expectations violated |
| S5 missing declared arm returns 0 | CAUGHT | CLI expectations violated |
| S5 sha presence check removed | CAUGHT | CLI expectations violated |
| S5 step == stage check removed (scores.json) | CAUGHT | CLI expectations violated |
| total | MISSED | 0 |

## D. train.py S6 log term

`(gt_si >= far_w[1])` mutated to `>` in a scratch edit of train.py (restored; train.py sha unchanged 475ffdbf): `tests/test_silog_far_weight.py::test_log_term_uses_ge_at_the_boundary` FAILED (`1 failed, 14 passed`), so the mutant is killed.
