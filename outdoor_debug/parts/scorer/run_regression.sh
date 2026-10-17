#!/bin/bash
# Regression of the by-range scorer on the START checkpoint (c3_n46 step12000). RUN BY THE LEAD ON THE GPU UNDER THE GPU_LOCK; this file is not run by the author.
# Usage: GPU_LOCK held; bash run_regression.sh [OUT_DIR]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$HERE/../../.." && pwd)"
CK=/mnt/DATA/DegradedDepth/checkpoints_v3/eval_20261012_c3rep/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n46/step12000.pt
CK_SHA=f5c7c8f98fe0c9f97bcac3586e577d04e914fbbe1e2ac41dbdbd8ce0501db1af
OUT="${1:-$REPO/outdoor_debug/parts/scorer/regression_c3_n46_step12000_$(date -u +%Y%m%dT%H%M%SZ)}"
[[ ! -e "$OUT" ]] || { echo "ERROR: $OUT exists"; exit 1; }
[[ "$(sha256sum "$CK" | cut -d' ' -f1)" == "$CK_SHA" ]] || { echo "ERROR: checkpoint sha mismatch"; exit 1; }
[[ "$(sha256sum "$REPO/bench_joint.py" | cut -d' ' -f1)" == 5d3d048ce7b26fb53e1e6a5aadf8265cb92c19dafc7340bc2e04cc6bcdf6e7ef ]] || { echo "ERROR: bench_joint sha"; exit 1; }
mkdir -p "$OUT"
cd "$REPO"
# 4090 numbers were made with fp16 autocast and NYUv2 truncated to 160 frames: same flags here (the arms use the SAME flags).
~/.venvs/ramen/bin/python -I "$HERE/scorer.py" --ckpt "$CK" --out "$OUT/scores.json" --datasets kitti,nyuv2,scannetpp,vkitti,horse \
    --device cuda --precision fp16 --nyuv2-frames 160 2>&1 | tee "$OUT/scorer.log"
~/.venvs/ramen/bin/python -I "$HERE/compare_regression.py" "$OUT/scores.json" | tee "$OUT/compare.txt"
sha256sum "$OUT"/* > "$OUT/SHA256SUMS"
