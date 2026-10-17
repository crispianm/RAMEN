#!/bin/bash
# outdoor-debug screen: THE sbatch COMMANDS. DRAFT -- NOT EXECUTED, to be run by the parent/Manager on approval, from the cluster clone dir
# (cd /projects/b5dh/RAMEN_outdoor_screen_2a72223 first). Every variable passed through sbatch is literal or exported (Auditor reminder).
# This file is a script only so that `bash -n` can check it; read it, then run the blocks by hand in order: 0 (clone), 1 (smokes), 2 (fulls), 3 (score).
set -euo pipefail
exit 99   # safety: refuses to run as a script. Run the blocks by hand after the smoke gate.

### Block 0: clone (to-be-run-on-approval, login node, once). Previous jobs used cluster-local clones whose origin was another cluster clone
### (e.g. /projects/b5dh/RAMEN_train_dn01_v3_bfix_C3_B), never GitHub. GitHub reachability from the cluster is UNVERIFIED here. Option A (if outbound https works):
export SHA=2a72223c068d5ba0bde869a7c27d0e90d21abb5d
git clone --branch outdoor-debug https://github.com/crispianm/RAMEN.git /projects/b5dh/RAMEN_outdoor_screen_2a72223
cd /projects/b5dh/RAMEN_outdoor_screen_2a72223 && git checkout --detach "$SHA" && test "$(git rev-parse HEAD)" = "$SHA" && test -z "$(git status --porcelain)"
### Option B (no outbound): on a machine with the repo: `git bundle create outdoor_debug_2a72223.bundle outdoor-debug`, copy it to /projects/b5dh/ via the usual relay,
### sha256sum it both sides, then: git clone -b outdoor-debug /projects/b5dh/outdoor_debug_2a72223.bundle /projects/b5dh/RAMEN_outdoor_screen_2a72223  (then the same checkout/clean lines).
### Never fetch/pull/reset any existing cluster clone. runs/ and logs/ are in .gitignore, so they do not dirty the tree.
mkdir -p logs /projects/b5dh/RAMEN_outdoor_screen_2a72223_scores
S=outdoor_debug/parts/submit

### Block 1: smokes (50 steps each, 4 GPU, 25 min limit): A0 and TS2_VK655 (the two arms whose data path differs most)
SM_A0=$(sbatch --parsable --time=00:25:00 --job-name=outdoor_smoke_A0 --export=ALL,SMOKE=1,PROJECT_DIR=/projects/b5dh/RAMEN_outdoor_screen_2a72223 $S/TRAIN_A0.sbatch)
SM_VK=$(sbatch --parsable --time=00:25:00 --job-name=outdoor_smoke_TS2_VK655 --export=ALL,SMOKE=1,PROJECT_DIR=/projects/b5dh/RAMEN_outdoor_screen_2a72223 $S/TRAIN_TS2_VK655.sbatch)
echo "smokes: $SM_A0 $SM_VK"
### when both are COMPLETED 0:0 run:  $S/check_first_run.sh -s -r runs A0=logs/outdoor_smoke_A0_$SM_A0.out TS2_VK655=logs/outdoor_smoke_TS2_VK655_$SM_VK.out

### Block 2: the six full runs, all at once, held behind the smokes (earlier practice: 6 fulls queued afterok on 2 smokes; limits 256 running / 512 submitted, no GrpTRES cap)
for A in A0 A0b TS2 TS2_H80 TS2_H300 TS2_VK655; do
  sbatch --parsable --dependency=afterok:$SM_A0:$SM_VK --export=ALL,PROJECT_DIR=/projects/b5dh/RAMEN_outdoor_screen_2a72223 $S/TRAIN_$A.sbatch
done
### once a full run reaches step>=12050: $S/check_first_run.sh -r runs A0=logs/outdoor_screen_A0_<id>.out <ARM>=logs/outdoor_screen_<ARM>_<id>.out ...   (scancel on FAIL)

### Block 3: scoring, one single-GPU job per arm, afterok on that arm's full job id
# sbatch --dependency=afterok:$TRAIN_ID_A0 --export=ALL,PROJECT_DIR=/projects/b5dh/RAMEN_outdoor_screen_2a72223 $S/SCORE_A0.sbatch   (repeat per arm)
