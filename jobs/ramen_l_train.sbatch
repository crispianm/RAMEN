#!/bin/bash
# RAMEN-L (experimental ViT-L variant) training job: 1 node, 4 GPUs, same launch shape as the Tables 1/2 runs (4 x batch 1 x accum 4).
# Submit with the cluster-specific values in the environment (never in this file) and the wall limit on the command line, e.g.
#   RAMEN_CLONE=... RAMEN_VENV=... RAMEN_WEIGHTS=... RAMEN_PIN=<40-hex commit> RAMEN_CONFIG=configs/experiments/ramen_l_smoke.yaml RAMEN_SMOKE=1 \
#     sbatch --export=ALL --time=01:00:00 jobs/ramen_l_train.sbatch
#SBATCH --job-name=ramen_l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --mem=400G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
: "${RAMEN_CLONE:?checkout of the pushed RAMEN-L commit}" "${RAMEN_VENV:?python venv (>=3.10)}" "${RAMEN_WEIGHTS:?dir holding metric_video_depth_anything_vitl.pth}"
: "${RAMEN_PIN:?40-hex commit this job must run}" "${RAMEN_CONFIG:?config relative to the clone}"
RAMEN_SMOKE="${RAMEN_SMOKE:-0}"
export RAMEN_WEIGHTS
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" PYTHONFAULTHANDLER=1 PYTHONDONTWRITEBYTECODE=1 PYTORCH_ALLOC_CONF=expandable_segments:True

cd "$RAMEN_CLONE"
source "$RAMEN_VENV/bin/activate"        # BEFORE any python: the system python on some clusters is too old for the preflight
OUT_DIR="$(python3 -I -c 'import sys,yaml;print(next(l.split(":",1)[1].strip() for l in open(sys.argv[1]) if l.strip().startswith("checkpoint_dir:")))' "$RAMEN_CONFIG")"
SMOKE_FLAG=""; [[ "$RAMEN_SMOKE" == "1" ]] && SMOKE_FLAG="--smoke"
python3 -I scripts/ramen_l_preflight.py --config "$RAMEN_CONFIG" $SMOKE_FLAG --pin "$RAMEN_PIN" --out-dir "$OUT_DIR" --build \
  || { echo "ERROR: preflight REFUSED"; exit 1; }
export VAL01_EXPECTED_COMMIT="$(git rev-parse HEAD)"
echo " config : $RAMEN_CONFIG sha256 $(sha256sum "$RAMEN_CONFIG" | cut -d' ' -f1) OK"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
torchrun --standalone --nproc_per_node=4 train.py --config "$RAMEN_CONFIG"
{ echo "HEAD $(git rev-parse HEAD)"; echo "PIN  $RAMEN_PIN"; echo "porcelain:"; git status --porcelain; } > "$OUT_DIR/post_clean.txt"
[[ "$(git rev-parse HEAD)" == "$RAMEN_PIN" && -z "$(git status --porcelain)" ]] || { echo "ERROR: clone not clean / not at the pin"; exit 1; }
echo "done $(date -u +%FT%TZ)"
