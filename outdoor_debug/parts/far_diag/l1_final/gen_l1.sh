#!/bin/bash
# Regenerates sbatch/TRAIN_<ARM>_k<K>.sbatch and sbatch/SCORE_<ARM>_k<K>.sbatch (8 arms x 2 stages) from the templates with the sha256 of the committed arm yamls. Run from anywhere; regenerate after ANY yaml/template change.
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"; mkdir -p "$D/sbatch"
for K in 2000 4000; do
  TLIM=$([[ $K == 2000 ]] && echo 03:00:00 || echo 05:30:00)      # expected 1.45 h / 2.9 h; limits ~2x / 1.9x
  for A in B0 B0b S1 S2 S3 S4 S5 S6; do
    Y="$(sha256sum "$D/configs/k$K/arm_$A.yaml" | cut -d' ' -f1)"; SEED=$([[ $A == B0b ]] && echo 49 || echo 46)
    sed -e "s/@ARM@/$A/g" -e "s/@K@/$K/g" -e "s/@YSHA@/$Y/g" -e "s/@SEED@/$SEED/g" -e "s/@TLIM@/$TLIM/g" "$D/train_template_l1.sbatch" > "$D/sbatch/TRAIN_${A}_k$K.sbatch"
    sed -e "s/@ARM@/$A/g" -e "s/@K@/$K/g" "$D/score_template_l1.sbatch" > "$D/sbatch/SCORE_${A}_k$K.sbatch"
    echo "$A k$K cfg=$Y"
  done
done
