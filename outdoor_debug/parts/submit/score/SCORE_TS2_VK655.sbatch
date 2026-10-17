#!/bin/bash
# =============================================================================
# outdoor-debug screen DEV scoring with the by-range scorer, row TS2_VK655: ONE GH200, ONE checkpoint. DRAFT, NOT SUBMITTED.
# Generated from score_template.sbatch by gen_score.sh -- edit the template, not this file. Supersedes ../score_template.sbatch / ../SCORE_*.sbatch (bench_joint only).
# Runs outdoor_debug/parts/scorer/scorer.py (README there) with EXACTLY the flags of scorer/run_regression.sh:
#   --datasets kitti,nyuv2,scannetpp,vkitti --device cuda --precision fp16 --nyuv2-frames 160     (no horse: not on the cluster)
# Pins: scorer.py sha256 a8b0731e7270b911f92d56eade55b413338b12fd4373aac2b93ae1a7da8171df, bench_joint.py sha256 5d3d048ce7b26fb53e1e6a5aadf8265cb92c19dafc7340bc2e04cc6bcdf6e7ef
#   (both re-checked here AND inside the scorer; the scorer additionally asserts the DEV clip-list shas). check_scores.py validates the result and fails the job.
# Code: a SECOND clean clone (SCORE_PROJECT_DIR) of crispianm/RAMEN branch outdoor-debug at SCORE_COMMIT (the commit that contains scorer.py + this submit/score dir).
#   The training clone (..._2a72223) does NOT contain scorer.py and is only READ for the checkpoint.
# bench_joint PATHS: no override needed; bench_joint.py picks the "isambard" host table because /projects/b5dh exists (kitti_rgbfix, dn01_eval_data/{kitti_raw_val_drives,kitti_calib},
#   data/test/benchmarks/{nyuv2_video,scannetpp_eval}, data/train/vkitti), same as every earlier Isambard bench_joint job.
# Time limit 06:00:00 is a PLACEHOLDER; per Manager rule tighten to ~1.5x the first measured run (4090 took 755 s for the same scorer) before submitting rows 2..7.
# Output: $SCORE_ROOT/TS2_VK655/ (refuses to overwrite; on failure it is renamed *.FAILED_<jobid>). Submit from SCORE_PROJECT_DIR with logs/ present (see SCORE_SPEC.md).
# =============================================================================
#SBATCH --job-name=outdoor_score_TS2_VK655
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=110G
#SBATCH --cpus-per-task=18
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

export LABEL="TS2_VK655"
export SCORE_COMMIT="UNSET"          # literal sha written by gen_score.sh; the job aborts if it is still the placeholder or HEAD differs
export SCORE_PROJECT_DIR="${SCORE_PROJECT_DIR:-/projects/b5dh/RAMEN_outdoor_screen_scorer}"
export SCORE_ROOT="${SCORE_ROOT:-/projects/b5dh/RAMEN_outdoor_screen_scores}"
export CKPT="/projects/b5dh/RAMEN_outdoor_screen_2a72223/runs/outdoor_screen_TS2_VK655/step14000.pt"
export EXP_STEP="14000"
export EXP_CKPT_SHA=""          # known only for the reference row; empty for arms (sha is recorded, not asserted)
export SCORER_SHA="a8b0731e7270b911f92d56eade55b413338b12fd4373aac2b93ae1a7da8171df"
export BJ_SHA="5d3d048ce7b26fb53e1e6a5aadf8265cb92c19dafc7340bc2e04cc6bcdf6e7ef"
export VENV="/projects/b5dh/FocusMamba/.venv"
export OUT="$SCORE_ROOT/$LABEL"

echo "======================================== outdoor_score_${LABEL}"
echo " Job ID : ${SLURM_JOB_ID:-local}   Node: $(hostname)   Started: $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONFAULTHANDLER=1
CACHE_ROOT="/projects/b5dh/.torch_cache"
mkdir -p "$CACHE_ROOT/tmp"
export TMPDIR="$CACHE_ROOT/tmp"
touch "$TMPDIR/.probe_${SLURM_JOB_ID:-local}" || { echo "ERROR: TMPDIR not writable"; exit 1; }
rm -f "$TMPDIR/.probe_${SLURM_JOB_ID:-local}"

cd "$SCORE_PROJECT_DIR" || { echo "ERROR: cannot cd into $SCORE_PROJECT_DIR"; exit 1; }

# --- pre-flight checks (pure bash; abort on any mismatch) ------------------------------------
[[ "$SCORE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "ERROR: SCORE_COMMIT '$SCORE_COMMIT' is not a 40-hex sha (placeholder not filled by gen_score.sh)"; exit 1; }
HEAD_SHA="$(git rev-parse HEAD)"
echo " Commit : $HEAD_SHA   branch: $(git rev-parse --abbrev-ref HEAD)"
[[ "$HEAD_SHA" == "$SCORE_COMMIT" ]] || { echo "ERROR: HEAD $HEAD_SHA != SCORE_COMMIT $SCORE_COMMIT"; exit 1; }
[[ -z "$(git status --porcelain)" ]] || { echo "ERROR: working tree dirty:"; git status --porcelain; exit 1; }
S=outdoor_debug/parts/scorer/scorer.py
[[ "$(sha256sum "$S" | cut -d' ' -f1)" == "$SCORER_SHA" ]] || { echo "ERROR: scorer.py sha mismatch"; exit 1; }
[[ "$(sha256sum bench_joint.py | cut -d' ' -f1)" == "$BJ_SHA" ]] || { echo "ERROR: bench_joint.py sha mismatch"; exit 1; }
[[ -f outdoor_debug/parts/submit/score/check_scores.py ]] || { echo "ERROR: check_scores.py missing in clone"; exit 1; }
[[ -f "$CKPT" ]] || { echo "ERROR: missing checkpoint $CKPT"; exit 1; }
CK_SHA="$(sha256sum "$CKPT" | cut -d' ' -f1)"
if [[ -n "$EXP_CKPT_SHA" ]]; then [[ "$CK_SHA" == "$EXP_CKPT_SHA" ]] || { echo "ERROR: ckpt sha $CK_SHA != $EXP_CKPT_SHA"; exit 1; }; fi
echo " ckpt   : $CKPT sha256 $CK_SHA (expected step $EXP_STEP)"
[[ ! -e "$OUT" ]] || { echo "ERROR: $OUT exists (never overwrite a score dir)"; exit 1; }
mkdir -p "$SCORE_ROOT"
mkdir "$OUT"
trap 'rc=$?; if [[ $rc -ne 0 ]]; then mv "$OUT" "$OUT.FAILED_${SLURM_JOB_ID:-local}" 2>/dev/null || true; echo "FAILED rc=$rc, moved to $OUT.FAILED_${SLURM_JOB_ID:-local}"; fi' EXIT
echo "$HEAD_SHA" > "$OUT/HEAD.txt"; echo "$CK_SHA  $CKPT" > "$OUT/checkpoint_sha256.txt"

source "$VENV/bin/activate"
echo " python : $(which python)  $(python -c 'import torch;print(torch.__version__)')"

# --- the scoring run: same flags as scorer/run_regression.sh, minus horse ----------------------
python -I "$S" --ckpt "$CKPT" --out "$OUT/scores.json" --datasets kitti,nyuv2,scannetpp,vkitti \
    --device cuda --precision fp16 --nyuv2-frames 160 2>&1 | tee "$OUT/scorer.log"

# --- validate; a failed check fails the job (the dir is then renamed *.FAILED_*) ----------------
python -I outdoor_debug/parts/submit/score/check_scores.py "$OUT/scores.json" --expect-step "$EXP_STEP" \
    ${EXP_CKPT_SHA:+--expect-ckpt-sha "$EXP_CKPT_SHA"} --expect-host-contains "GH200" 2>&1 | tee "$OUT/check_scores.txt"

cp "logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "$OUT/slurm_stdout_partial.log" 2>/dev/null || true   # best effort (job still running)
( cd "$OUT" && sha256sum HEAD.txt checkpoint_sha256.txt scores.json scorer.log check_scores.txt > SHA256SUMS )
echo "======================================== done $(date -u +%FT%TZ)  -> $OUT"
