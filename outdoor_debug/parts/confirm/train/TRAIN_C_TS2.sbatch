#!/bin/bash
# =============================================================================
# CONFIRMATORY from-init 12k run, arm C_TS2 (seed 50), 4 GH200, 1 node. NEW RECIPE vs released C3 (carries the TS2 loader; H300 also the sky hinge). SINGLE SEED: sign only, not a variance estimate.
# DRAFT, NOT SUBMITTED. Generated from train_template_confirm.sbatch by gen_train_confirm.sh -- edit the template, not this file.
# Config (committed in the repo at PIN_COMMIT): outdoor_debug/parts/confirm/train/configs/arm_C_TS2.yaml  sha256 3fc800bf153c7184e0535d31893a358690e3e9fa2b784c60e6964689422faad6
# Recipe = job 7226119 (c3rep n46, 08:36:42 on 4 GH200): released Metric-VDA-S init, NO resume_from, 500-step warmup + cosine over 12000, AdamW 1e-4 (gates 1e-3).
# Launch conventions copied from 7226119 / the outdoor screen: workq, 1 node, 1 task, gres gpu:4, 72 cpus, 400G, FocusMamba venv, torchrun --standalone, VAL01_EXPECTED_COMMIT = HEAD.
# SMOKE=1 (via --export) writes a 50-step copy of the config to $TMPDIR (max_steps 50, smoke checkpoint dir; lr_decay_steps stays 12000 so the 50 steps are the real warmup) and trains that.
# =============================================================================
#SBATCH --job-name=confirm_C_TS2
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

export ARM="C_TS2"
export PIN_COMMIT="UNSET_PIN_COMMIT"     # sed at submit (SUBMIT.md block 0) to the full 40-hex sha; the script refuses UNSET
export PROJECT_DIR="${PROJECT_DIR:-/projects/b5dh/RAMEN_confirm_UNSET}"
export SMOKE="${SMOKE:-0}"
export CFG_REL="outdoor_debug/parts/confirm/train/configs/arm_${ARM}.yaml"
export CFG_SHA="3fc800bf153c7184e0535d31893a358690e3e9fa2b784c60e6964689422faad6"
export BASE_REL="outdoor_debug/parts/C2_config_train/effective_ramen_n46.yaml"
export BASE_SHA="d286b62cef5f5e603ca909675b7daf22c4e120b0b99dfe3d5ec69bcbebd911a1"                      # resolved config of job 7226119; the arm yaml differs only by the declared keys (config_diffs_confirm.md)
export INIT_CKPT="/projects/b5dh/benchmark/weights/metric_video_depth_anything_vits.pth"                 # released Metric-VDA-S (model.vda_checkpoint_path, vda_init released)
export INIT_SHA="3c28432b4e1f0d7bb31cad5151b6313b49457db5aa58d82e85bfb0f8b1311b33"
export DAV2_CKPT="/projects/b5dh/DaBiT/weights/depth_anything_v2_vits.pth"                               # model.da_v2_checkpoint_path (preflight 'foundation_checkpoint_sha256' of 7226119)
export DAV2_SHA="715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378"
export TEMP2_CKPT="/projects/b5dh/RAMEN_train_dn01_st_recal/runs/recal_full/temp2_distilled.pt"         # model.st_temp2_init
export TEMP2_SHA="96d35f08ecd5c988387e502bbd0ad85875539cafd9e329448ad5c76804916c4b"
export TS2_SEG_ROOT="/projects/b5dh/data/train/tartanair_seg/"
export TS2_TABLE_SHA="f19d0b838252c9a2409498b35f46738d99b3646afba984d6a82ffea3cd610b76"
export VENV="/projects/b5dh/FocusMamba/.venv"
export NGPU=4

echo "======================================== confirm_${ARM} smoke=${SMOKE}"
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
chk config  "$CFG_REL"   "$CFG_SHA"
chk base    "$BASE_REL"  "$BASE_SHA"
chk init    "$INIT_CKPT" "$INIT_SHA"
chk dav2    "$DAV2_CKPT" "$DAV2_SHA"
chk temp2   "$TEMP2_CKPT" "$TEMP2_SHA"
grep -q "^  resume_from: null\$" "$CFG_REL" || { echo "ERROR: resume_from is not null in config (from-init run!)"; exit 1; }
if grep -q "lr_restart\|resume_reseed" "$CFG_REL"; then echo "ERROR: lr_restart/resume_reseed key in config"; exit 1; fi
grep -q "^  seed: 50\$" "$CFG_REL" || { echo "ERROR: data.seed 50 not in config"; exit 1; }
grep -q "vda_checkpoint_path: $INIT_CKPT\$" "$CFG_REL" || { echo "ERROR: model.vda_checkpoint_path != INIT_CKPT"; exit 1; }
grep -q "ts2_seg_root: $TS2_SEG_ROOT\$" "$CFG_REL" || { echo "ERROR: config ts2_seg_root != $TS2_SEG_ROOT"; exit 1; }
[[ -d "$TS2_SEG_ROOT" ]] || { echo "ERROR: ts2_seg_root $TS2_SEG_ROOT missing"; exit 1; }
echo " ts2 seg root entries: $(ls "$TS2_SEG_ROOT" | wc -l) (75 seen on 2026-10-17; 74 env zips + extras)"
chk ts2table "outdoor_debug/parts/TA_sky/ts2_env_table.json" "$TS2_TABLE_SHA"
case "$ARM" in
  C_TS2_H300) grep -q "^  sky_hinge_weight: 0.1\$" "$CFG_REL" && grep -q "^  sky_hinge_min_depth: 300.0\$" "$CFG_REL" || { echo "ERROR: H300 hinge keys wrong"; exit 1; } ;;
  C_TS2)      grep -q "^  sky_hinge_weight: 0.0\$" "$CFG_REL" && ! grep -q "sky_hinge_min_depth" "$CFG_REL" || { echo "ERROR: control carries hinge keys"; exit 1; } ;;
  *) echo "ERROR: unknown ARM $ARM"; exit 1 ;;
esac

RUN_CFG="$CFG_REL"
if [[ "$SMOKE" == "1" ]]; then
    RUN_CFG="$TMPDIR/arm_${ARM}_smoke50_${SLURM_JOB_ID:-local}.yaml"
    sed -e 's/^  max_steps: 12000$/  max_steps: 50/' \
        -e "s#^  checkpoint_dir: runs/confirm_${ARM}\$#  checkpoint_dir: runs/confirm_smoke50_${ARM}#" "$CFG_REL" > "$RUN_CFG"
    [[ "$(diff "$CFG_REL" "$RUN_CFG" | grep -c '^>')" == "2" ]] || { echo "ERROR: smoke config diff is not exactly 2 lines"; diff "$CFG_REL" "$RUN_CFG" || true; exit 1; }
    echo " SMOKE config $RUN_CFG (sha256 $(sha256sum "$RUN_CFG" | cut -d' ' -f1)); diff vs arm yaml:"; diff "$CFG_REL" "$RUN_CFG" || true
    OUT_DIR="runs/confirm_smoke50_${ARM}"
else
    OUT_DIR="runs/confirm_${ARM}"
fi
if [[ -e "$OUT_DIR" ]]; then echo "ERROR: $OUT_DIR already exists (never reuse an output dir)"; exit 1; fi
echo " ckpt dir (only write location for checkpoints): $PROJECT_DIR/$OUT_DIR"

export VAL01_EXPECTED_COMMIT="$HEAD_SHA"
source "$VENV/bin/activate"
python -c "import train; print('train.py imports OK')"
echo " launching: torchrun --standalone --nproc_per_node=$NGPU train.py --config $RUN_CFG"
torchrun --standalone --nproc_per_node="$NGPU" train.py --config "$RUN_CFG"
echo "======================================== done $(date -u +%FT%TZ)"
