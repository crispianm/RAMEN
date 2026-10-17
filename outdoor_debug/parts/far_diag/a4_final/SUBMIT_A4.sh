#!/bin/bash
# A4 (released Metric-VDA-S at 518, report-only): THE commands. NOT EXECUTED. Run block by block by hand on Manager/Auditor go, after the Auditor scorer diff check.
set -euo pipefail
# (exit-99 guard removed after Auditor O44 diff check PASS + Manager GO; SUBMITTED by hand, see SUBMITTED.txt)
### Pins to substitute: A4_COMMIT = the a4-final commit (or its merge) holding ddad_scorer.py sha 8e71511b1192a1a15797a71482c40b547bac7d905641b7d2f145ad1da079a503; it must be on the cluster clone /projects/b5dh/RAMEN_outdoor_screen_ddad
###   (the A3 job used that clone; update it the same way as for 7505639 -- no fetch/pull/reset of any other clone). Never touch AV2 data.
export A4_COMMIT=81887d15b20a10aff933551fa5bd783873f8c51b   # actual: a NEW clean clone from a git bundle at /projects/b5dh/RAMEN_outdoor_screen_ddad_a4 (no fetch/pull of any existing clone); sbatch filled outside the tree, submitted with --export=ALL,DDAD_PROJECT_DIR=...
S=outdoor_debug/parts/submit/score
### Block 1 (cluster clone): confirm commit + scorer sha + clean tree, and that the 378 reference row exists (the checker reads it)
cd /projects/b5dh/RAMEN_outdoor_screen_ddad && test "$(git rev-parse HEAD)" = "$A4_COMMIT" && test -z "$(git status --porcelain)"
sha256sum outdoor_debug/parts/LR_DEV/ddad_scorer.py | grep -q 8e71511b1192a1a15797a71482c40b547bac7d905641b7d2f145ad1da079a503
test -f /projects/b5dh/RAMEN_outdoor_screen_scores/DDAD_RELEASE_MetricVDA_S/ddad_scores.json
test ! -e /projects/b5dh/RAMEN_outdoor_screen_scores/DDAD_RELEASE_MetricVDA_S_518
### Block 2 (local/cluster copy of the sbatch with the commit literal filled; checked-in file holds UNSET and aborts)
sed "s/^export DDAD_COMMIT=\"UNSET\"/export DDAD_COMMIT=\"$A4_COMMIT\"/" $S/DDAD_RELEASE_MVDA_518.sbatch > /tmp/DDAD_RELEASE_MVDA_518.filled.sbatch
grep -q "DDAD_COMMIT=\"$A4_COMMIT\"" /tmp/DDAD_RELEASE_MVDA_518.filled.sbatch
### Block 3: submit (one GH200, 1 h limit) -- first-518-run stop check: the job itself aborts if the log lacks ', 3, 518, 826) -> depth (' or if check_a4_518.py has a FAIL
mkdir -p logs && sbatch --parsable /tmp/DDAD_RELEASE_MVDA_518.filled.sbatch
### Block 4 (after COMPLETED 0:0, pull via the usual Mac rsync, then locally):
#   python -I outdoor_debug/parts/far_diag/a4_final/check_a4_518.py <pulled>/ddad_scores.json --log <pulled>/ddad_scorer.log --scorer-sha 8e71511b1192a1a15797a71482c40b547bac7d905641b7d2f145ad1da079a503 --ref378 outdoor_debug/parts/LR_DEV/ddad_release/ddad_scores.json
#   (must print RESULT: ALL PASS; the Auditor re-runs it)
