#!/bin/bash
# Regenerates TRAIN_C_<ARM>.sbatch from train_template_confirm.sbatch using the sha256 of the committed arm YAMLs. Run from anywhere.
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"; C="$D/configs"
for A in C_TS2 C_TS2_H300; do
  S="$(sha256sum "$C/arm_$A.yaml" | cut -d' ' -f1)"
  sed -e "s/@ARM@/$A/g" -e "s/@YSHA@/$S/g" "$D/train_template_confirm.sbatch" > "$D/TRAIN_$A.sbatch"
  echo "$A $S"
done
