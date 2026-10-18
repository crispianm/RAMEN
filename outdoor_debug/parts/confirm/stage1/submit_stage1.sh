#!/bin/bash
# Stage 1: THE sbatch COMMANDS. DRAFT -- NOT EXECUTED. Run by the parent/Manager after approval, by hand, block by block (a script only so that `bash -n` can check it).
set -euo pipefail
exit 99   # safety: refuses to run as a script

### Block 0 (login node, once): clone at the pinned commit; sbatch copies with the pin substituted live OUTSIDE the clone (the clone must stay clean)
export SHA="<40-hex commit holding every file in outdoor_debug/parts/confirm/stage1>"      # NOT YET EXISTING; Manager sets after the commit
export CL=/projects/b5dh/RAMEN_confirm1_${SHA:0:8}
git clone --branch outdoor-debug https://github.com/crispianm/RAMEN.git $CL && cd $CL && git checkout --detach $SHA && test "$(git rev-parse HEAD)" = "$SHA" && test -z "$(git status --porcelain)"
#   (GitHub reachability from Isambard unverified: use the screen's bundle option B if it fails)
mkdir -p logs ../confirm1_sbatch_${SHA:0:8}; S=../confirm1_sbatch_${SHA:0:8}; D=outdoor_debug/parts/confirm/stage1
for N in 46 47 48; do for A in TS2 H300; do sed "s/UNSET_PIN_COMMIT/$SHA/" $D/TRAIN_S1_n${N}_${A}.sbatch > $S/TRAIN_S1_n${N}_${A}.sbatch; done; done

### Block 1: two smokes (50 real steps, 4 GPU, 25 min limit): n47 TS2 (seed 51, control path) and n48 H300 (seed 52, hinge path)
SM1=$(sbatch --parsable --time=00:25:00 --job-name=confirm1_smoke_S1_n47_TS2  --export=ALL,SMOKE=1,PROJECT_DIR=$CL $S/TRAIN_S1_n47_TS2.sbatch)
SM2=$(sbatch --parsable --time=00:25:00 --job-name=confirm1_smoke_S1_n48_H300 --export=ALL,SMOKE=1,PROJECT_DIR=$CL $S/TRAIN_S1_n48_H300.sbatch)
### >>> GATE (explicit step, BY HAND; Block 2 is NOT submitted before it PASSES; NO sbatch --dependency from either smoke to any full job): both smokes COMPLETED 0:0 (sacct), then (python3 only):
$D/check_first_run_stage1.sh -s -r runs S1_n47_TS2=logs/confirm1_smoke_S1_n47_TS2_$SM1.out S1_n48_H300=logs/confirm1_smoke_S1_n48_H300_$SM2.out     # no FAIL; ANY FAIL => STOP and report, do not submit Block 2

### Block 2 (ONLY AFTER THE GATE PASSED; separate step): the six fulls (3 h limit each), NO --dependency on the smokes
for N in 46 47 48; do for A in TS2 H300; do
  sbatch --parsable --export=ALL,PROJECT_DIR=$CL $S/TRAIN_S1_n${N}_${A}.sbatch
done; done
### once every log shows step >= 12100 (about 5-10 min after start; items 3 needs the 5 loss_terms rows 12010..12050; LR rows 12020..12120):
# $D/check_first_run_stage1.sh -r runs S1_n46_TS2=logs/confirm1_S1_n46_TS2_<id>.out S1_n46_H300=logs/confirm1_S1_n46_H300_<id>.out S1_n47_TS2=... S1_n47_H300=... S1_n48_TS2=... S1_n48_H300=...   (any FAIL -> scancel that job and report)
### later (optionally) re-run the same line when logs reach 14000 to complete the LR decay rows.
### Block 3 (scoring): NOT drafted here, see SCORING_PLACEHOLDER.md. Checkpoints for scoring: runs/confirm1_<NAME>/step14000.pt (+ 12500/13000/13500 snapshots, last.pt).
