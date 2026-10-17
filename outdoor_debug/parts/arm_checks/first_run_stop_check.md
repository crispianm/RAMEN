# First-run stop check (submit job; all six arms; resume from c3_n46 step12000, S3 LR peak 1.0)
Run it on the arm's stdout log (slurm out) and `<checkpoint_dir>/metrics.jsonl` once the log shows step >= 12050 (or after 5 `loss_terms` rows). STOP (scancel) on any FAIL. Nothing here changes the loss: `loss.log_terms: true` is already in all six YAMLs and `loss.log_terms_every: 10` (added to all six) only changes how often the already-accumulated term means are flushed to metrics.jsonl (train.py `term_flush`, default 100 -> 10 so the first 50 steps contain rows at 12010..12050).

## What the trainer actually logs (verified in train.py)
- stdout, every 20 optimizer steps (after `gstep += 1`): `e<epoch> step<G>/<total> loss=<..> lr=<%.2e> ...` where `lr` = `optimizer.param_groups[0]['lr']` (GROUP 0 ONLY) = initial_lr * lr_scale(G-1) (the LR was set before the increment). Total printed = 14000.
- metrics.jsonl, every `log_terms_every` steps: `{"event":"loss_terms","step":G,<term>:mean,...}`; the hinge term key is `sky_hinge_weighted`, added to the accumulator ONLY when `loss.sky_hinge_weight > 0` (train.py ~l.1235). So in A0/A0b/TS2/TS2_VK655 the key is ABSENT, never 0.
- Per-group LRs other than group 0 are NOT logged. The checkpoint has 3 groups (initial_lr 1e-4, 1e-4, 1e-3 = gate group, gate_lr_scale 10), all multiplied by the same lr_scale, so group-0 LR x10 = gate-group LR (not directly observable in the log; proven from the checkpoint + `make_lr_scale`, see README).

## Expected printed `lr=` (group 0; from the real `make_lr_scale` on arm_A0.yaml; tolerance 1.5% relative on the %.2e print)
| printed step G | 12020 | 12040 | 12060 | 12080 | 12100 | 12120 | 12200 | 12500 | 13000 | 13500 | 14000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lr (group 0) | 2.00e-05 | 4.00e-05 | 6.00e-05 | 8.00e-05 | 1.00e-04 | 9.998e-05 -> prints 1.00e-04 | 9.93e-05 | 8.95e-05 | 5.42e-05 | 1.62e-05 | 6.8e-11 |
(Gate group = 10x these. The first applied update, gstep 12000, uses scale 0.01 = 1.0e-06 and is not printed.)

## Checks
1. LR (FAIL if any): every `lr=` line with step in {12020,12040,12060,12080,12100,12120,12200} matches the table (relative error <= 1.5%). A line equal to `0.00e+00` at any step 12000..13999 = the no-op bug (lr_decay_steps=12000 without the restart keys): FAIL immediately.
2. Resume line: stdout contains `[resume] ... step=12000` (exactly) and `step12020/14000` appears (total 14000).
3. Hinge key (FAIL if any), on metrics.jsonl rows with 12000 < step <= 12050:
   - arms A0, A0b, TS2, TS2_VK655: no row contains `sky_hinge_weighted`.
   - arms TS2_H80, TS2_H300: every row contains `sky_hinge_weighted` (present, finite, >= 0). If ALL five values are exactly 0.0 print WARN (valid only if the model already predicts >= the floor, 80 / 300 m, on every TS2 sky pixel, or the sampled clips had no sky; the term is a one-sided hinge so 0 is not a bug by itself). A nonzero value is expected for H80/H300 at the start only when sky preds < floor; do not require it.
4. Startup lines (grep, FAIL if missing): `[ts2] TartanAir TS2 sky/mask ON ... env_table_sha256=f19d0b838252c9a2409498b35f46738d99b3646afba984d6a82ffea3cd610b76` in the four TS2 arms; absent in A0/A0b.
5. Finite: no `[warn] non-finite loss` in the first 50 steps.

```sh
LOG=slurm-<jobid>.out; M=<checkpoint_dir>/metrics.jsonl; ARM=<arm>
grep -E "^\[resume\]" $LOG | grep -q "step=12000" || echo FAIL resume
grep -E "step(12020|12040|12060|12080|12100|12120|12200)/14000" $LOG | sed -E 's/.*step([0-9]+)\/.* lr=([0-9.e+-]+).*/\1 \2/'    # compare with the table
grep -E "lr=0.00e\+00" $LOG && echo FAIL lr_zero
python - <<PY
import json,sys
rows=[json.loads(l) for l in open("$M") if '"loss_terms"' in l]; rows=[r for r in rows if 12000<r["step"]<=12050]
h=("$ARM" in ("TS2_H80","TS2_H300"))
bad=[r["step"] for r in rows if ("sky_hinge_weighted" in r)!=h]
print("rows",len(rows),"hinge-key mismatches",bad,"values",[r.get("sky_hinge_weighted") for r in rows])
sys.exit(1 if bad or len(rows)<5 else 0)
PY
```
