#!/bin/bash
# =============================================================================
# L1 scoring of ONE trained run (B0, 2000 steps) on ONE GH200: by-range scorer (VKITTI/KITTI/NYUv2/ScanNet++ DEV, 'exploratory, unaudited') THEN the DDAD val scorer ('exposed DEV gate'). DRAFT, NOT SUBMITTED.
# Generated from score_template_l1.sbatch by gen_l1.sh. Same clone/commit/host type for all runs. DEV only (no --allow-test). The checkpoint's sha256 is the one the TRAIN job recorded next to it
# (step<K>.pt.sha256) and must equal the sha computed here (a file swapped in between aborts); the Auditor recomputes it independently later.
# Pins: scorer.py a8b0731e.., bench_joint.py 5d3d048c.., ddad_scorer.py 80916c4f.., DDAD frame list b8d9664f.., pairs 962d4996.. (all re-checked by the scorers themselves).
# Output: $SCORE_ROOT/k2000/B0/ {scores.json, check_scores.txt, ddad_scores.json, check_ddad.txt, ...}; refuses to overwrite; a failed check renames the dir *.FAILED_<jobid>. Readout: l1_readout.py --scores-dir $SCORE_ROOT/k2000 --stage 2000 --ddad-gate A.
# =============================================================================
#SBATCH --job-name=l1score_B0_k2000
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=110G
#SBATCH --cpus-per-task=18
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
export ARM="B0"
export K="2000"
export PIN_COMMIT="UNSET_PIN_COMMIT"
export PROJECT_DIR="${PROJECT_DIR:-/projects/b5dh/RAMEN_l1_UNSET}"
export SCORE_ROOT="${SCORE_ROOT:-/projects/b5dh/RAMEN_l1_scores}"
export CKPT="$PROJECT_DIR/runs/l1_${ARM}_k${K}/step${K}.pt"
export SCORER_SHA="a8b0731e7270b911f92d56eade55b413338b12fd4373aac2b93ae1a7da8171df"
export BJ_SHA="5d3d048ce7b26fb53e1e6a5aadf8265cb92c19dafc7340bc2e04cc6bcdf6e7ef"
export DDAD_SCORER_SHA="80916c4f3b2ab0b8999b07e142847f8df157db5da756192f02f6df5dea1ed09f"
export DDAD_PAIRS_SHA="962d499612c40bca7d314f8ff16ce1d6c7b03962ae78938f472a101766150245"
export DDAD_LR_ROOT="/projects/b5dh/scratch_ddad_20261017/DEV_LR"
export VENV="/projects/b5dh/FocusMamba/.venv"
export OUT="$SCORE_ROOT/k${K}/$ARM"

echo "======================================== l1score_${ARM}_k${K}  job ${SLURM_JOB_ID:-local} node $(hostname) $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} PYTHONFAULTHANDLER=1
CACHE_ROOT="/projects/b5dh/.torch_cache"; mkdir -p "$CACHE_ROOT/tmp"; export TMPDIR="$CACHE_ROOT/tmp"
cd "$PROJECT_DIR" || { echo "ERROR: cannot cd $PROJECT_DIR"; exit 1; }
[[ "$PIN_COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "ERROR: PIN_COMMIT not substituted"; exit 1; }
HEAD_SHA="$(git rev-parse HEAD)"; echo " Commit : $HEAD_SHA"
[[ "$HEAD_SHA" == "$PIN_COMMIT" ]] || { echo "ERROR: HEAD != PIN_COMMIT"; exit 1; }
[[ -z "$(git status --porcelain)" ]] || { echo "ERROR: dirty tree"; exit 1; }
S=outdoor_debug/parts/scorer/scorer.py; DS=outdoor_debug/parts/LR_DEV/ddad_scorer.py
[[ "$(sha256sum "$S" | cut -d' ' -f1)" == "$SCORER_SHA" ]] || { echo "ERROR: scorer.py sha mismatch"; exit 1; }
[[ "$(sha256sum bench_joint.py | cut -d' ' -f1)" == "$BJ_SHA" ]] || { echo "ERROR: bench_joint.py sha mismatch"; exit 1; }
[[ "$(sha256sum "$DS" | cut -d' ' -f1)" == "$DDAD_SCORER_SHA" ]] || { echo "ERROR: ddad_scorer.py sha mismatch"; exit 1; }
[[ "$(sha256sum "$DDAD_LR_ROOT/ddad_val_pairs.tsv" | cut -d' ' -f1)" == "$DDAD_PAIRS_SHA" ]] || { echo "ERROR: DDAD pairs sha"; exit 1; }
[[ -d "$DDAD_LR_ROOT/ddad_val/ddad_train_val/000150" ]] || { echo "ERROR: DDAD scenes missing under $DDAD_LR_ROOT"; exit 1; }
[[ -f "$CKPT" && -f "$CKPT.sha256" ]] || { echo "ERROR: missing $CKPT or its recorded sha"; exit 1; }
CK_SHA="$(sha256sum "$CKPT" | cut -d' ' -f1)"; REC_SHA="$(cut -d' ' -f1 "$CKPT.sha256")"
[[ "$CK_SHA" == "$REC_SHA" ]] || { echo "ERROR: ckpt sha $CK_SHA != the sha the train job recorded $REC_SHA"; exit 1; }
echo " ckpt   : $CKPT sha256 $CK_SHA (== recorded) expected step $K"
[[ ! -e "$OUT" && ! -e "$OUT.FAILED_${SLURM_JOB_ID:-x}" ]] || { echo "ERROR: $OUT exists (never overwrite a score dir)"; exit 1; }
mkdir -p "$SCORE_ROOT/k${K}"; mkdir "$OUT"
trap 'rc=$?; if [[ $rc -ne 0 ]]; then mv "$OUT" "$OUT.FAILED_${SLURM_JOB_ID:-local}" 2>/dev/null || true; echo "FAILED rc=$rc, moved to $OUT.FAILED_${SLURM_JOB_ID:-local}"; fi' EXIT
echo "$HEAD_SHA" > "$OUT/HEAD.txt"; echo "$CK_SHA  $CKPT" > "$OUT/checkpoint_sha256.txt"
source "$VENV/bin/activate"

# 1) by-range scorer: the flags of scorer/run_regression.sh minus horse (same as the outdoor screen)
python -I "$S" --ckpt "$CKPT" --out "$OUT/scores.json" --datasets kitti,nyuv2,scannetpp,vkitti --device cuda --precision fp16 --nyuv2-frames 160 2>&1 | tee "$OUT/scorer.log"
python -I outdoor_debug/parts/submit/score/check_scores.py "$OUT/scores.json" --expect-step "$K" --expect-ckpt-sha "$CK_SHA" --expect-host-contains GH200 2>&1 | tee "$OUT/check_scores.txt"
# 2) DDAD val (exposed DEV gate), 378 input, report in the readout's DDAD leg
python -I "$DS" --ckpt "$CKPT" --out "$OUT/ddad_scores.json" --device cuda 2>&1 | tee "$OUT/ddad_scorer.log"
python -I outdoor_debug/parts/LR_DEV/check_ddad_output.py "$OUT/ddad_scores.json" --expect-ckpt-sha "$CK_SHA" --reference outdoor_debug/parts/LR_DEV/ddad_ref2/ddad_scores.json --log "$OUT/ddad_scorer.log" 2>&1 | tee "$OUT/check_ddad.txt"
( cd "$OUT" && sha256sum HEAD.txt checkpoint_sha256.txt scores.json scorer.log check_scores.txt ddad_scores.json ddad_scorer.log check_ddad.txt > SHA256SUMS )
echo "======================================== done $(date -u +%FT%TZ) -> $OUT"
