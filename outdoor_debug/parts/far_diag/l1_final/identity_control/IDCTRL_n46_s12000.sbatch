#!/bin/bash
# =============================================================================
# L1 scoring of ONE trained run (IDCTRL_n46_s12000, 12000 steps) on ONE GH200: by-range scorer (VKITTI/KITTI/NYUv2/ScanNet++ DEV, 'exploratory, unaudited') THEN the DDAD val scorer ('exposed DEV gate'). DRAFT, NOT SUBMITTED.
# Generated from score_template_l1.sbatch by gen_l1.sh. Same clone/commit/host type for all runs. DEV only (no --allow-test). The checkpoint's sha256 is the one the TRAIN job recorded next to it
# (step<K>.pt.sha256) and must equal the sha computed here (a file swapped in between aborts); the Auditor recomputes it independently later.
# Pins: scorer.py a8b0731e.., bench_joint.py 5d3d048c.., ddad_scorer.py 8e71511b.., DDAD frame list b8d9664f.., pairs 962d4996.. (all re-checked by the scorers themselves).
# Output: $SCORE_ROOT/k12000/IDCTRL_n46_s12000/ {scores.json, check_scores.txt, ddad_scores.json, check_ddad.txt, ...}; refuses to overwrite; a failed check renames the dir *.FAILED_<jobid>. Readout: l1_readout.py --scores-dir $SCORE_ROOT/k12000 --stage 12000 --ddad-gate A.
# =============================================================================
#SBATCH --job-name=l1idctrl_IDCTRL_n46_s12000_k12000
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --mem=110G
#SBATCH --cpus-per-task=18
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
export ARM="IDCTRL_n46_s12000"
export K="12000"
export PIN_COMMIT="114df74abe3e38a3fc70bd8ee99798a15e220b56"
export PROJECT_DIR="${PROJECT_DIR:-/projects/b5dh/RAMEN_l1_UNSET}"
export SCORE_ROOT="${SCORE_ROOT:-/projects/b5dh/RAMEN_l1_scores}"
export CKPT="/projects/b5dh/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n46/step12000.pt"   # identity control: the c3_n46 start checkpoint
export SCORER_SHA="a8b0731e7270b911f92d56eade55b413338b12fd4373aac2b93ae1a7da8171df"
export BJ_SHA="5d3d048ce7b26fb53e1e6a5aadf8265cb92c19dafc7340bc2e04cc6bcdf6e7ef"
export DDAD_SCORER_SHA="8e71511b1192a1a15797a71482c40b547bac7d905641b7d2f145ad1da079a503"   # A4 scorer (post cdb86a0): identical to 80916c4f at the default 378 path (Auditor-checked); the reference row ddad_ref2 was produced by 80916c4f
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
[[ -f "$CKPT" ]] || { echo "ERROR: missing $CKPT"; exit 1; }
CK_SHA="$(sha256sum "$CKPT" | cut -d' ' -f1)"; REC_SHA="f5c7c8f98fe0c9f97bcac3586e577d04e914fbbe1e2ac41dbdbd8ce0501db1af"   # registered sha of c3_n46 step12000 (c3rep SHA256SUMS_local)
[[ "$CK_SHA" == "$REC_SHA" ]] || { echo "ERROR: ckpt sha $CK_SHA != the sha the train job recorded $REC_SHA"; exit 1; }
echo " ckpt   : $CKPT sha256 $CK_SHA (== recorded) expected step $K"
[[ ! -e "$OUT" && ! -e "$OUT.FAILED_${SLURM_JOB_ID:-x}" ]] || { echo "ERROR: $OUT exists (never overwrite a score dir)"; exit 1; }
mkdir -p "$SCORE_ROOT/k${K}"; mkdir "$OUT"
trap 'rc=$?; if [[ $rc -ne 0 ]]; then mv "$OUT" "$OUT.FAILED_${SLURM_JOB_ID:-local}" 2>/dev/null || true; echo "FAILED rc=$rc, moved to $OUT.FAILED_${SLURM_JOB_ID:-local}"; fi' EXIT
echo "$HEAD_SHA" > "$OUT/HEAD.txt"; echo "$CK_SHA  $CKPT" > "$OUT/checkpoint_sha256.txt"
source "$VENV/bin/activate"

# 1) by-range scorer: the flags of scorer/run_regression.sh minus horse (same as the outdoor screen)
# 2) DDAD val (exposed DEV gate), 378 input, report in the readout's DDAD leg
python -I "$DS" --ckpt "$CKPT" --out "$OUT/ddad_scores.json" --device cuda 2>&1 | tee "$OUT/ddad_scorer.log"
python -I outdoor_debug/parts/LR_DEV/check_ddad_output.py "$OUT/ddad_scores.json" --expect-ckpt-sha "$CK_SHA" --scorer-sha "$DDAD_SCORER_SHA" --reference outdoor_debug/parts/LR_DEV/ddad_ref2/ddad_scores.json --log "$OUT/ddad_scorer.log" 2>&1 | tee "$OUT/check_ddad.txt"
( cd "$OUT" && sha256sum HEAD.txt checkpoint_sha256.txt ddad_scores.json ddad_scorer.log check_ddad.txt > SHA256SUMS )
echo "======================================== done $(date -u +%FT%TZ) -> $OUT"
