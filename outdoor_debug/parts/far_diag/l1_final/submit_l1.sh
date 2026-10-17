#!/bin/bash
# L1 (Track 2 light screen): THE sbatch COMMANDS. DRAFT -- NOT EXECUTED. Run by the parent/Manager after approval, BY HAND, block by block (a script only so that `bash -n` can check it).
set -euo pipefail
exit 99   # safety: refuses to run as a script

### Block 0 (login node, once): clone at the pinned commit. Sbatch copies with the pin substituted live OUTSIDE the clone (the clone must stay clean).
export SHA="<40-hex commit that contains outdoor_debug/parts/far_diag/l1_final/** and train.py with S6>"      # the Manager sets this to the final commit of branch l1-final (pushed/bundled by the user/parent)
export CL=/projects/b5dh/RAMEN_l1_${SHA:0:8}
git clone --branch <branch> https://github.com/crispianm/RAMEN.git $CL && cd $CL && git checkout --detach $SHA && test "$(git rev-parse HEAD)" = "$SHA" && test -z "$(git status --porcelain)"
#   (GitHub reachability from Isambard is unverified: use the outdoor screen's bundle option B if it fails.)  Never fetch/pull/reset any existing cluster clone.
mkdir -p logs ../l1_sbatch_${SHA:0:8} /projects/b5dh/RAMEN_l1_scores; S=$(cd ../l1_sbatch_${SHA:0:8} && pwd); D=outdoor_debug/parts/far_diag/l1_final
for f in $D/sbatch/TRAIN_*.sbatch $D/sbatch/SCORE_*.sbatch; do sed "s/UNSET_PIN_COMMIT/$SHA/" $f > $S/$(basename $f); done
grep -L "$SHA" $S/*.sbatch && echo "ERROR: a copy lacks the pin" && false       # must print nothing

### Block 1: two smokes (120 real steps, 4 GPU, 25 min limit each): S1 (mix change) and S5 (head param group / checkpoint probe)
SM1=$(sbatch --parsable --time=00:25:00 --job-name=l1_smoke_S1 --export=ALL,SMOKE=120,PROJECT_DIR=$CL $S/TRAIN_S1_k2000.sbatch)
SM5=$(sbatch --parsable --time=00:25:00 --job-name=l1_smoke_S5 --export=ALL,SMOKE=120,PROJECT_DIR=$CL $S/TRAIN_S5_k2000.sbatch)
### gate when both are COMPLETED 0:0 (sacct):   (python3 only; S5 listed first = the [data] reference)
$D/check_first_run_l1.sh -F -S 120 -K 2000 -r runs S5=logs/l1_smoke_S5_${SM5}.out S1=logs/l1_smoke_S1_${SM1}.out          # no FAIL
#   checkpoint-side probe (needs torch: run on a compute node, ~1 min):
srun --partition=workq --nodes=1 --ntasks=1 --time=00:10:00 --mem=16G bash -c "source /projects/b5dh/FocusMamba/.venv/bin/activate && cd $CL && python $D/check_optimizer_lr.py --ckpt runs/l1_smoke_S5_k2000/step120.pt --arm S5 -K 2000 --expect-step 120 -F && python $D/check_optimizer_lr.py --ckpt runs/l1_smoke_S1_k2000/step120.pt --arm S1 -K 2000 --expect-step 120 -F"

### Block 2: BASELINES FIRST (2000 steps), behind the smokes
B0=$(sbatch --parsable --dependency=afterok:$SM1:$SM5 --export=ALL,PROJECT_DIR=$CL $S/TRAIN_B0_k2000.sbatch)
B0B=$(sbatch --parsable --dependency=afterok:$SM1:$SM5 --export=ALL,PROJECT_DIR=$CL $S/TRAIN_B0b_k2000.sbatch)
### when BOTH logs show step >= 200 (the loss_terms row at 200 exists): first-run stop check (B0 first). ANY FAIL => scancel both jobs and report. No -F here (later rows are WAIT).
$D/check_first_run_l1.sh -K 2000 -r runs B0=logs/l1_B0_k2000_${B0}.out B0b=logs/l1_B0b_k2000_${B0B}.out
### after step 520 (last.pt is saved at step 500):  the checkpoint-side probe on last.pt (FAIL => scancel):  [same srun form]  python $D/check_optimizer_lr.py --ckpt runs/l1_B0_k2000/last.pt --arm B0 -K 2000   (and B0b)
### when COMPLETED: final pass, all rows must exist (-F):
#   $D/check_first_run_l1.sh -F -K 2000 -r runs B0=... B0b=... ; probe: --ckpt runs/l1_B0_k2000/step2000.pt --arm B0 -K 2000 --expect-step 2000 -F (and B0b)

### Block 3: score the baselines (single GH200 each, ~12 min), behind their training jobs
sbatch --dependency=afterok:$B0  --export=ALL,PROJECT_DIR=$CL $S/SCORE_B0_k2000.sbatch
sbatch --dependency=afterok:$B0B --export=ALL,PROJECT_DIR=$CL $S/SCORE_B0b_k2000.sbatch

### Block 4: read G0 (stdlib python; the ONLY thing read before the arms run)
python3 -I $D/l1_readout.py --scores-dir /projects/b5dh/RAMEN_l1_scores/k2000 --stage 2000 --ddad-gate A --arms B0,B0b ; echo rc=$?
#   rc 0  -> G0 PASS: Block 5 (arms at 2000).      rc 10 -> G0 FAIL at 2000: Block 6 (AUTOMATIC 4000-step retry of B0/B0b; the 6 arms are NOT run at 2000).      rc 11 -> never at this stage.   rc 3 -> baselines not scored.

### Block 5 (G0 passed at 2000): S1..S5 at 2000, one job each, all at once (no dependency); stop check on each log at step >= 200 (B0's log first for the [data] reference)
for A in S1 S2 S3 S4 S5; do eval "J_$A=\$(sbatch --parsable --export=ALL,PROJECT_DIR=$CL $S/TRAIN_${A}_k2000.sbatch)"; done
#   stop check:  $D/check_first_run_l1.sh -K 2000 -r runs B0=logs/l1_B0_k2000_${B0}.out S1=logs/l1_S1_k2000_${J_S1}.out S2=... S3=... S4=... S5=...        (FAIL => scancel that job)
for A in S1 S2 S3 S4 S5; do eval "sbatch --dependency=afterok:\$J_$A --export=ALL,PROJECT_DIR=$CL $S/SCORE_${A}_k2000.sbatch"; done
### 5c: S6 is a SEPARATE GATED JOB: only after the Auditor has checked S6_diff.patch and recorded the sha256 of the patch it checked:
export S6SHA=$(sha256sum $D/S6_diff.patch | cut -d' ' -f1)      # must equal the sha the Auditor cleared (the job refuses otherwise; it also refuses if train.py != base 8e1d9ae + that patch)
SM6=$(sbatch --parsable --time=00:25:00 --job-name=l1_smoke_S6 --export=ALL,SMOKE=120,PROJECT_DIR=$CL,S6_CLEARED_PATCH_SHA=$S6SHA $S/TRAIN_S6_k2000.sbatch)
#   gate: COMPLETED 0:0 and  $D/check_first_run_l1.sh -F -S 120 -K 2000 -r runs S6=logs/l1_smoke_S6_${SM6}.out   (item 9 needs the step-100 loss_terms row with silog_far_px_frac) + probe: --ckpt runs/l1_smoke_S6_k2000/step120.pt --arm S6 -K 2000 --expect-step 120 -F
J_S6=$(sbatch --parsable --dependency=afterok:$SM6 --export=ALL,PROJECT_DIR=$CL,S6_CLEARED_PATCH_SHA=$S6SHA $S/TRAIN_S6_k2000.sbatch)
sbatch --dependency=afterok:$J_S6 --export=ALL,PROJECT_DIR=$CL $S/SCORE_S6_k2000.sbatch
### final readout (all arms scored):  python3 -I $D/l1_readout.py --scores-dir /projects/b5dh/RAMEN_l1_scores/k2000 --stage 2000 --ddad-gate A

### Block 6 (G0 FAILED at 2000, rc 10): AUTOMATIC retry of the baselines at 4000 steps (new run dirs l1_*_k4000; cosine to 0 at 4000, same everything else)
B0_4=$(sbatch --parsable --export=ALL,PROJECT_DIR=$CL $S/TRAIN_B0_k4000.sbatch); B0B_4=$(sbatch --parsable --export=ALL,PROJECT_DIR=$CL $S/TRAIN_B0b_k4000.sbatch)
#   stop check at step >= 200: $D/check_first_run_l1.sh -K 4000 -r runs B0=logs/l1_B0_k4000_${B0_4}.out B0b=logs/l1_B0b_k4000_${B0B_4}.out ; probes with -K 4000 ; final pass with -F
sbatch --dependency=afterok:$B0_4  --export=ALL,PROJECT_DIR=$CL $S/SCORE_B0_k4000.sbatch ; sbatch --dependency=afterok:$B0B_4 --export=ALL,PROJECT_DIR=$CL $S/SCORE_B0b_k4000.sbatch
python3 -I $D/l1_readout.py --scores-dir /projects/b5dh/RAMEN_l1_scores/k4000 --stage 4000 --ddad-gate A --arms B0,B0b ; echo rc=$?
#   rc 0 -> G0 PASS at 4000: run S1..S5 (+ gated S6) exactly as Block 5 but with the *_k4000 sbatch files and -K 4000 everywhere.    rc 11 -> STOP and report (no arm is run, no verdict).
