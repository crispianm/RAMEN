#!/bin/bash
# Regenerates TRAIN_S1_n<N>_<ARM>.sbatch from train_template_stage1.sbatch (sha256 of the committed yamls/bases/start checkpoints substituted). Run from anywhere.
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
declare -A SEED=([46]=50 [47]=51 [48]=52)
declare -A SSHA=([46]=f5c7c8f98fe0c9f97bcac3586e577d04e914fbbe1e2ac41dbdbd8ce0501db1af [47]=2ee98b7ee25bd6d5f044ea1f6bb7854d1a0c978488be52d4c87d305a5df6a54c [48]=bf053bd38fdb97d55228fceb737fc2d6353cca26d9ab13338c6d9dd3df95da12)
for N in 46 47 48; do for A in TS2 H300; do
  NAME="S1_n${N}_${A}"
  Y="$(sha256sum "$D/configs/arm_$NAME.yaml" | cut -d' ' -f1)"; B="$(sha256sum "$D/base/effective_ramen_n$N.yaml" | cut -d' ' -f1)"
  sed -e "s/@NAME@/$NAME/g" -e "s/@N@/$N/g" -e "s/@ARM@/$A/g" -e "s/@SEED@/${SEED[$N]}/g" -e "s/@YSHA@/$Y/g" -e "s/@BSHA@/$B/g" -e "s/@SSHA@/${SSHA[$N]}/g" "$D/train_template_stage1.sbatch" > "$D/TRAIN_$NAME.sbatch"
  echo "$NAME cfg=$Y base=$B"
done; done
