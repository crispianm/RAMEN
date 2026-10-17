#!/bin/bash
# Regenerates SCORE_REF_step12000.sbatch and SCORE_<ARM>.sbatch x6 from score_template.sbatch.
# Usage: bash gen_score.sh <SCORE_COMMIT 40-hex | UNSET>     (UNSET writes the placeholder; such a job aborts at start)
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
SC="${1:?usage: gen_score.sh <40-hex SCORE_COMMIT|UNSET>}"
[[ "$SC" =~ ^[0-9a-f]{40}$ || "$SC" == "UNSET" ]] || { echo "bad SCORE_COMMIT '$SC'"; exit 1; }
TRAIN=/projects/b5dh/RAMEN_outdoor_screen_2a72223
REF_CK=/projects/b5dh/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n46/step12000.pt
REF_SHA=f5c7c8f98fe0c9f97bcac3586e577d04e914fbbe1e2ac41dbdbd8ce0501db1af
gen() {  # label ckpt step ckptsha
  sed -e "s#@LABEL@#$1#g" -e "s#@CKPT@#$2#g" -e "s#@EXP_STEP@#$3#g" -e "s#@EXP_CKPT_SHA@#$4#g" -e "s#@SCORE_COMMIT@#$SC#g" "$D/score_template.sbatch" > "$D/SCORE_$1.sbatch"
  echo "SCORE_$1.sbatch"
}
gen REF_step12000 "$REF_CK" 12000 "$REF_SHA"
for A in A0 A0b TS2 TS2_H80 TS2_H300 TS2_VK655; do gen "$A" "$TRAIN/runs/outdoor_screen_$A/step14000.pt" 14000 ""; done
