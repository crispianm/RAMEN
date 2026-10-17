#!/bin/bash
# Regenerates TRAIN_<ARM>.sbatch from train_template.sbatch using the sha256 of the committed arm YAMLs. Run from anywhere.
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"; C="$D/../arm_checks/configs"
for A in A0 A0b TS2 TS2_H80 TS2_H300 TS2_VK655; do
  S="$(sha256sum "$C/arm_$A.yaml" | cut -d' ' -f1)"
  sed -e "s/@ARM@/$A/g" -e "s/@YSHA@/$S/g" "$D/train_template.sbatch" > "$D/TRAIN_$A.sbatch"
  sed -e "s/@ARM@/$A/g" "$D/score_template.sbatch" > "$D/SCORE_$A.sbatch"
  echo "$A $S"
done
