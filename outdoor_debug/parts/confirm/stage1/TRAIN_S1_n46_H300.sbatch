#!/bin/bash
# =============================================================================
# outdoor-debug STAGE 1 (Add.48 / Auditor O39), run S1_n46_H300: 2000-step PAIRED CONTINUATION (steps 12000 -> 14000) of c3_n46 step12000, data.seed = resume_reseed = 50,
# arm H300 (TS2 control | H300 = TS2 + sky_hinge_weight 0.1, sky_hinge_min_depth 300.0). 4 GH200, 1 node. Tests recovery, not prevention. SINGLE SEED PER PAIR (sign/consistency over 3 pairs).
# DRAFT, NOT SUBMITTED. Generated from train_template_stage1.sbatch by gen_train_stage1.sh -- edit the template, not this file.
# Config (committed at PIN_COMMIT): outdoor_debug/parts/confirm/stage1/configs/arm_S1_n46_H300.yaml  sha256 809b9e640247b9494b1a9b7bc3c7a35a77993f52df8d0121eb1f3fbc8f431a83
# Launch conventions copied from the outdoor screen (workq, 1 node, 1 task, gres gpu:4, 72 cpus, 400G, FocusMamba venv, torchrun --standalone, VAL01_EXPECTED_COMMIT = HEAD).
# SMOKE=1 (via --export) trains a 50-step copy (max_steps 12050, smoke checkpoint_dir; LR keys untouched so the 50 steps are the real restart warmup).
# =============================================================================
#SBATCH --job-name=confirm1_S1_n46_H300
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --mem=400G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

export NAME="S1_n46_H300"
export ARM="H300"
export SEED="50"
export PIN_COMMIT="UNSET_PIN_COMMIT"     # sed at submit (SUBMIT.md block 0) to the full 40-hex sha, into a copy OUTSIDE the clone; the script refuses UNSET
export PROJECT_DIR="${PROJECT_DIR:-/projects/b5dh/RAMEN_confirm1_UNSET}"
export SMOKE="${SMOKE:-0}"
export CFG_REL="outdoor_debug/parts/confirm/stage1/configs/arm_${NAME}.yaml"
export CFG_SHA="809b9e640247b9494b1a9b7bc3c7a35a77993f52df8d0121eb1f3fbc8f431a83"
export BASE_REL="outdoor_debug/parts/confirm/stage1/base/effective_ramen_n46.yaml"
export BASE_SHA="d286b62cef5f5e603ca909675b7daf22c4e120b0b99dfe3d5ec69bcbebd911a1"                                   # resolved config of the start's own c3rep job
export START_CKPT="/projects/b5dh/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n46/step12000.pt"
export START_SHA="f5c7c8f98fe0c9f97bcac3586e577d04e914fbbe1e2ac41dbdbd8ce0501db1af"
export TS2_SEG_ROOT="/projects/b5dh/data/train/tartanair_seg/"
export TS2_TABLE_SHA="f19d0b838252c9a2409498b35f46738d99b3646afba984d6a82ffea3cd610b76"
export VENV="/projects/b5dh/FocusMamba/.venv"
export NGPU=4

echo "======================================== confirm1_${NAME} smoke=${SMOKE}"
echo " Job ID : ${SLURM_JOB_ID:-local}   Node: $(hostname)   Started: $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONFAULTHANDLER=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
CACHE_ROOT="/projects/b5dh/.torch_cache"
mkdir -p "$CACHE_ROOT/torchinductor" "$CACHE_ROOT/triton" "$CACHE_ROOT/tmp"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TMPDIR="$CACHE_ROOT/tmp"
touch "$TMPDIR/.probe_${SLURM_JOB_ID:-local}" || { echo "ERROR: TMPDIR not writable"; exit 1; }
rm -f "$TMPDIR/.probe_${SLURM_JOB_ID:-local}"

cd "$PROJECT_DIR" || { echo "ERROR: cannot cd into $PROJECT_DIR"; exit 1; }

# --- pre-flight checks (pure bash; abort on any mismatch) ------------------------------------
[[ "$PIN_COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "ERROR: PIN_COMMIT '$PIN_COMMIT' is not a 40-hex sha (placeholder not substituted)"; exit 1; }
HEAD_SHA="$(git rev-parse HEAD)"
echo " Commit : $HEAD_SHA   branch: $(git rev-parse --abbrev-ref HEAD)"
[[ "$HEAD_SHA" == "$PIN_COMMIT" ]] || { echo "ERROR: HEAD $HEAD_SHA != PIN_COMMIT $PIN_COMMIT"; exit 1; }
[[ -z "$(git status --porcelain)" ]] || { echo "ERROR: working tree dirty:"; git status --porcelain; exit 1; }
[[ -f "$CFG_REL" ]] || { echo "ERROR: missing $CFG_REL"; exit 1; }
chk() { local got; got="$(sha256sum "$2" | cut -d' ' -f1)"; [[ "$got" == "$3" ]] || { echo "ERROR: $1 sha $got != $3 ($2)"; exit 1; }; echo " $1 : $2 sha256 $got OK"; }
chk config  "$CFG_REL"    "$CFG_SHA"
chk base    "$BASE_REL"   "$BASE_SHA"
chk resume  "$START_CKPT" "$START_SHA"
chk ts2table "outdoor_debug/parts/TA_sky/ts2_env_table.json" "$TS2_TABLE_SHA"
grep -q "^  resume_from: $START_CKPT\$" "$CFG_REL" || { echo "ERROR: resume_from in config != START_CKPT"; exit 1; }
grep -q "^  seed: $SEED\$" "$CFG_REL"           || { echo "ERROR: data.seed $SEED not in config"; exit 1; }
grep -q "^  resume_reseed: $SEED\$" "$CFG_REL"  || { echo "ERROR: training.resume_reseed $SEED not in config"; exit 1; }
for kv in "lr_restart_step: 12000" "lr_restart_warmup: 100" "lr_restart_peak: 1.0" "lr_decay_steps: 14000" "max_steps: 14000"; do
  grep -q "^  $kv\$" "$CFG_REL" || { echo "ERROR: config lacks '$kv'"; exit 1; }
done
grep -q "ts2_seg_root: $TS2_SEG_ROOT\$" "$CFG_REL" || { echo "ERROR: config ts2_seg_root != $TS2_SEG_ROOT"; exit 1; }
grep -q "^    ts2_sky: true\$" "$CFG_REL" || { echo "ERROR: ts2_sky not true"; exit 1; }
[[ -d "$TS2_SEG_ROOT" ]] || { echo "ERROR: ts2_seg_root $TS2_SEG_ROOT missing"; exit 1; }
echo " ts2 seg root entries: $(ls "$TS2_SEG_ROOT" | wc -l) (75 seen on 2026-10-17)"
case "$ARM" in
  H300) grep -q "^  sky_hinge_weight: 0.1\$" "$CFG_REL" && grep -q "^  sky_hinge_min_depth: 300.0\$" "$CFG_REL" || { echo "ERROR: H300 hinge keys wrong"; exit 1; } ;;
  TS2)  grep -q "^  sky_hinge_weight: 0.0\$" "$CFG_REL" && ! grep -q "sky_hinge_min_depth" "$CFG_REL" || { echo "ERROR: control carries hinge keys"; exit 1; } ;;
  *) echo "ERROR: unknown ARM $ARM"; exit 1 ;;
esac

RUN_CFG="$CFG_REL"
if [[ "$SMOKE" == "1" ]]; then
    RUN_CFG="$TMPDIR/arm_${NAME}_smoke50_${SLURM_JOB_ID:-local}.yaml"
    sed -e 's/^  max_steps: 14000$/  max_steps: 12050/' \
        -e "s#^  checkpoint_dir: runs/confirm1_${NAME}\$#  checkpoint_dir: runs/confirm1_smoke50_${NAME}#" "$CFG_REL" > "$RUN_CFG"
    [[ "$(diff "$CFG_REL" "$RUN_CFG" | grep -c '^>')" == "2" ]] || { echo "ERROR: smoke config diff is not exactly 2 lines"; diff "$CFG_REL" "$RUN_CFG" || true; exit 1; }
    echo " SMOKE config $RUN_CFG (sha256 $(sha256sum "$RUN_CFG" | cut -d' ' -f1)); diff vs arm yaml:"; diff "$CFG_REL" "$RUN_CFG" || true
    OUT_DIR="runs/confirm1_smoke50_${NAME}"
else
    OUT_DIR="runs/confirm1_${NAME}"
fi
if [[ -e "$OUT_DIR" ]]; then echo "ERROR: $OUT_DIR already exists (never reuse an output dir)"; exit 1; fi
echo " ckpt dir (only write location for checkpoints): $PROJECT_DIR/$OUT_DIR"

export VAL01_EXPECTED_COMMIT="$HEAD_SHA"
source "$VENV/bin/activate"
python -c "import train; print('train.py imports OK')"
echo " launching: torchrun --standalone --nproc_per_node=$NGPU train.py --config $RUN_CFG"
torchrun --standalone --nproc_per_node="$NGPU" train.py --config "$RUN_CFG"
echo "======================================== done $(date -u +%FT%TZ)"
