#!/bin/bash
# DDAD LR_DEV scoring (exploratory DEV, report-only) of ONE checkpoint on ONE GH200 with ddad_scorer.py (Auditor-checked O26). First use: the REFERENCE checkpoint c3_n46 step12000 only.
# Code: clean clone DDAD_PROJECT_DIR at DDAD_COMMIT (40-hex literal written at submit time by sed; the checked-in file holds UNSET and aborts).
# Data: DDAD_LR_ROOT on the cluster = /projects/b5dh/scratch_ddad_20261017/DEV_LR (ddad_val -> ../val_subset, ddad_val_pairs.tsv); frame list = the repo's ddad_val_frames.tsv (sha asserted by the scorer).
#SBATCH --job-name=outdoor_ddad_TS2_H80
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --mem=110G
#SBATCH --cpus-per-task=18
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
export LABEL="DDAD_TS2_H80_step14000"
export DDAD_COMMIT="UNSET"
export DDAD_PROJECT_DIR="${DDAD_PROJECT_DIR:-/projects/b5dh/RAMEN_outdoor_screen_ddad}"
export SCORE_ROOT="${SCORE_ROOT:-/projects/b5dh/RAMEN_outdoor_screen_scores}"
export CKPT="/projects/b5dh/RAMEN_outdoor_screen_2a72223/runs/outdoor_screen_TS2_H80/step14000.pt"
export EXP_CKPT_SHA=""   # arm ckpt: sha recorded, not asserted at submit time (computed independently after training)
export DDAD_SCORER_SHA="80916c4f3b2ab0b8999b07e142847f8df157db5da756192f02f6df5dea1ed09f"
export BJ_SHA="5d3d048ce7b26fb53e1e6a5aadf8265cb92c19dafc7340bc2e04cc6bcdf6e7ef"
export DDAD_LR_ROOT="/projects/b5dh/scratch_ddad_20261017/DEV_LR"
export VENV="/projects/b5dh/FocusMamba/.venv"
export OUT="$SCORE_ROOT/$LABEL"
echo "======================================== outdoor_ddad_${LABEL}  job ${SLURM_JOB_ID:-local} node $(hostname) $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} PYTHONFAULTHANDLER=1
CACHE_ROOT="/projects/b5dh/.torch_cache"; mkdir -p "$CACHE_ROOT/tmp"; export TMPDIR="$CACHE_ROOT/tmp"
cd "$DDAD_PROJECT_DIR" || { echo "ERROR: cannot cd $DDAD_PROJECT_DIR"; exit 1; }
[[ "$DDAD_COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "ERROR: DDAD_COMMIT not filled"; exit 1; }
HEAD_SHA="$(git rev-parse HEAD)"; echo " Commit: $HEAD_SHA"
[[ "$HEAD_SHA" == "$DDAD_COMMIT" ]] || { echo "ERROR: HEAD != DDAD_COMMIT"; exit 1; }
[[ -z "$(git status --porcelain)" ]] || { echo "ERROR: dirty tree"; exit 1; }
S=outdoor_debug/parts/LR_DEV/ddad_scorer.py
[[ "$(sha256sum "$S" | cut -d' ' -f1)" == "$DDAD_SCORER_SHA" ]] || { echo "ERROR: ddad_scorer.py sha mismatch"; exit 1; }
[[ "$(sha256sum bench_joint.py | cut -d' ' -f1)" == "$BJ_SHA" ]] || { echo "ERROR: bench_joint sha mismatch"; exit 1; }
[[ "$(sha256sum "$DDAD_LR_ROOT/ddad_val_pairs.tsv" | cut -d' ' -f1)" == 962d499612c40bca7d314f8ff16ce1d6c7b03962ae78938f472a101766150245 ]] || { echo "ERROR: pairs sha"; exit 1; }
[[ -d "$DDAD_LR_ROOT/ddad_val/ddad_train_val/000150" ]] || { echo "ERROR: DDAD scenes missing under $DDAD_LR_ROOT"; exit 1; }
[[ -f "$CKPT" ]] || { echo "ERROR: ckpt missing"; exit 1; }
CK_SHA="$(sha256sum "$CKPT" | cut -d' ' -f1)"; true
[[ ! -e "$OUT" ]] || { echo "ERROR: $OUT exists"; exit 1; }
mkdir -p "$SCORE_ROOT" "$OUT"
trap 'rc=$?; if [[ $rc -ne 0 ]]; then mv "$OUT" "$OUT.FAILED_${SLURM_JOB_ID:-local}" 2>/dev/null || true; echo "FAILED rc=$rc"; fi' EXIT
echo "$HEAD_SHA" > "$OUT/HEAD.txt"; echo "$CK_SHA  $CKPT" > "$OUT/checkpoint_sha256.txt"
source "$VENV/bin/activate"
python -I "$S" --ckpt "$CKPT" --out "$OUT/ddad_scores.json" --device cuda 2>&1 | tee "$OUT/ddad_scorer.log"
( cd "$OUT" && sha256sum HEAD.txt checkpoint_sha256.txt ddad_scores.json ddad_scorer.log > SHA256SUMS )
echo "======================================== done $(date -u +%FT%TZ) -> $OUT"
