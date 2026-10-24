#!/bin/bash
# RAMEN-L (experimental ViT-L variant) training job: 1 node, 4 GPUs, same launch shape as the Tables 1/2 runs (4 x batch 1 x accum 4).
# Submit with the cluster-specific values in the environment (never in this file) and the wall limit on the command line, e.g.
#   RAMEN_CLONE=... RAMEN_VENV=... RAMEN_WEIGHTS=... RAMEN_PIN=<40-hex commit> RAMEN_CONFIG=configs/experiments/ramen_l_smoke.yaml RAMEN_SMOKE=1 \
#     sbatch --export=ALL --time=01:00:00 jobs/ramen_l_train.sbatch
# Depth-only VDA-L fine-tune: RAMEN_PROFILE=ft RAMEN_CONFIG=configs/experiments/vdal_ft_n46.yaml (resume: vdal_ft_n46_resume.yaml + RAMEN_RESUME=1).
# Second segment of a run that hit the wall limit: RAMEN_RESUME=1 RAMEN_CONFIG=configs/experiments/ramen_l_c3_n46_resume.yaml (the out dir must exist and hold last.pt).
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
RAMEN_SMOKE="${RAMEN_SMOKE:-0}"; RAMEN_RESUME="${RAMEN_RESUME:-0}"; RAMEN_PROFILE="${RAMEN_PROFILE:-}"   # RAMEN_PROFILE=ft: depth-only VDA-L FT (configs/experiments/vdal_ft_n46.yaml)
[[ -z "$RAMEN_PROFILE" || "$RAMEN_PROFILE" == "ft" ]] || { echo "ERROR: RAMEN_PROFILE must be empty or ft"; exit 1; }
[[ "$RAMEN_SMOKE" == "1" && "$RAMEN_RESUME" == "1" ]] && { echo "ERROR: RAMEN_SMOKE and RAMEN_RESUME are exclusive"; exit 1; }
export RAMEN_WEIGHTS
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" PYTHONFAULTHANDLER=1 PYTHONDONTWRITEBYTECODE=1 PYTORCH_ALLOC_CONF=expandable_segments:True

cd "$RAMEN_CLONE"
source "$RAMEN_VENV/bin/activate"        # BEFORE any python: the system python on some clusters is too old for the preflight
OUT_DIR="$(python3 -I -c 'import sys,pathlib;sys.path.insert(0,".");import train;print(train.load_config(pathlib.Path(sys.argv[1]))["training"]["checkpoint_dir"])' "$RAMEN_CONFIG")"
FLAG=""; SUFFIX=""; [[ "$RAMEN_SMOKE" == "1" ]] && FLAG="--smoke"; [[ "$RAMEN_RESUME" == "1" ]] && { FLAG="--resume"; SUFFIX="_resume"; }
python3 -I scripts/ramen_l_preflight.py --config "$RAMEN_CONFIG" $FLAG ${RAMEN_PROFILE:+--profile $RAMEN_PROFILE} --pin "$RAMEN_PIN" --out-dir "$OUT_DIR" --build \
  || { echo "ERROR: preflight REFUSED"; exit 1; }
export VAL01_EXPECTED_COMMIT="$(git rev-parse HEAD)"
echo " config : $RAMEN_CONFIG sha256 $(sha256sum "$RAMEN_CONFIG" | cut -d' ' -f1) OK"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
torchrun --standalone --nproc_per_node=4 train.py --config "$RAMEN_CONFIG"
{ echo "HEAD $(git rev-parse HEAD)"; echo "PIN  $RAMEN_PIN"; echo "porcelain:"; git status --porcelain; } > "$OUT_DIR/post_clean${SUFFIX}.txt"
[[ "$(git rev-parse HEAD)" == "$RAMEN_PIN" && -z "$(git status --porcelain)" ]] || { echo "ERROR: clone not clean / not at the pin"; exit 1; }
echo "done $(date -u +%FT%TZ)"
