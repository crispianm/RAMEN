#!/bin/bash
# =============================================================================
# outdoor-debug 6-arm screen, arm TS2: 2000-step continuation of c3_n46 step12000 (steps 12000 -> 14000), 4 GH200, 1 node.
# DRAFT, NOT SUBMITTED. Generated from train_template.sbatch by gen_train.sh -- edit the template, not this file.
# Config (committed in the repo): outdoor_debug/parts/arm_checks/configs/arm_TS2.yaml  sha256 0e04c9c61f0a115a1c7ddfb225300500d54ab4f6f77c30c4bd0b976191567ead
# Start checkpoint sha256 f5c7c8f98fe0c9f97bcac3586e577d04e914fbbe1e2ac41dbdbd8ce0501db1af (c3_n46 step12000)
# Code: clean clone of crispianm/RAMEN branch outdoor-debug at 2a72223c.. (or its code-identical parent 0034ac71..).
# Launch conventions copied from the c3rep n46 12k job 7226119 / dn01_jt_ctrl_cont8to12 7243692 (jobs/run_ramen_v2.sh, NGPU=4):
#   workq, 1 node, 1 task, gres gpu:4, 72 cpus, 400G, FocusMamba venv, torchrun --standalone, same cache dirs, VAL01_EXPECTED_COMMIT = HEAD.
# Submit from the clone dir with logs/ present (submit.sh does this).
# SMOKE=1 (via --export) writes a 50-step copy of the config to $TMPDIR (max_steps 12050, smoke checkpoint_dir) and trains that instead.
# No account line / module loads: the earlier jobs used none (only the default brics modules and the venv).
# =============================================================================
#SBATCH --job-name=outdoor_screen_TS2
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

export ARM="TS2"
export PROJECT_DIR="${PROJECT_DIR:-/projects/b5dh/RAMEN_outdoor_screen_2a72223}"
export SMOKE="${SMOKE:-0}"
export CFG_REL="outdoor_debug/parts/arm_checks/configs/arm_${ARM}.yaml"
export CFG_SHA="0e04c9c61f0a115a1c7ddfb225300500d54ab4f6f77c30c4bd0b976191567ead"
export START_CKPT="/projects/b5dh/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n46/step12000.pt"
export START_SHA="f5c7c8f98fe0c9f97bcac3586e577d04e914fbbe1e2ac41dbdbd8ce0501db1af"
export OK_COMMITS="2a72223c068d5ba0bde869a7c27d0e90d21abb5d 0034ac711866d704a1cabe0b8ea38ed1d0b022ac"
export TS2_SEG_ROOT="/projects/b5dh/data/train/tartanair_seg/"
export TS2_TABLE_SHA="f19d0b838252c9a2409498b35f46738d99b3646afba984d6a82ffea3cd610b76"
export VENV="/projects/b5dh/FocusMamba/.venv"
export NGPU=4

echo "======================================== outdoor_screen_${ARM} smoke=${SMOKE}"
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
HEAD_SHA="$(git rev-parse HEAD)"
echo " Commit : $HEAD_SHA   branch: $(git rev-parse --abbrev-ref HEAD)"
case " $OK_COMMITS " in *" $HEAD_SHA "*) ;; *) echo "ERROR: HEAD $HEAD_SHA not in allowed set: $OK_COMMITS"; exit 1;; esac
[[ -z "$(git status --porcelain)" ]] || { echo "ERROR: working tree dirty:"; git status --porcelain; exit 1; }
[[ -f "$CFG_REL" ]] || { echo "ERROR: missing $CFG_REL"; exit 1; }
GOT="$(sha256sum "$CFG_REL" | cut -d' ' -f1)"
[[ "$GOT" == "$CFG_SHA" ]] || { echo "ERROR: config sha $GOT != $CFG_SHA"; exit 1; }
echo " config : $CFG_REL sha256 $GOT OK"
GOT="$(sha256sum "$START_CKPT" | cut -d' ' -f1)"
[[ "$GOT" == "$START_SHA" ]] || { echo "ERROR: start checkpoint sha $GOT != $START_SHA"; exit 1; }
echo " resume : $START_CKPT sha256 $GOT OK"
grep -q "resume_from: $START_CKPT\$" "$CFG_REL" || { echo "ERROR: resume_from in config != START_CKPT"; exit 1; }

RUN_CFG="$CFG_REL"
if [[ "$SMOKE" == "1" ]]; then
    RUN_CFG="$TMPDIR/arm_${ARM}_smoke50_${SLURM_JOB_ID:-local}.yaml"
    sed -e 's/^  max_steps: 14000$/  max_steps: 12050/' \
        -e "s#^  checkpoint_dir: runs/outdoor_screen_${ARM}\$#  checkpoint_dir: runs/outdoor_screen_smoke50_${ARM}#" "$CFG_REL" > "$RUN_CFG"
    [[ "$(diff "$CFG_REL" "$RUN_CFG" | grep -c '^>')" == "2" ]] || { echo "ERROR: smoke config diff is not exactly 2 lines"; diff "$CFG_REL" "$RUN_CFG" || true; exit 1; }
    echo " SMOKE config $RUN_CFG (sha256 $(sha256sum "$RUN_CFG" | cut -d' ' -f1)); diff vs arm yaml:"; diff "$CFG_REL" "$RUN_CFG" || true
    OUT_DIR="runs/outdoor_screen_smoke50_${ARM}"
else
    OUT_DIR="runs/outdoor_screen_${ARM}"
fi
if [[ -e "$OUT_DIR" ]]; then echo "ERROR: $OUT_DIR already exists (never reuse an output dir)"; exit 1; fi
echo " ckpt dir (only write location for checkpoints): $PROJECT_DIR/$OUT_DIR"

case "$ARM" in
  A0|A0b) ;;   # no TS2 labels in these two arms
  *)
    [[ -d "$TS2_SEG_ROOT" ]] || { echo "ERROR: ts2_seg_root $TS2_SEG_ROOT missing"; exit 1; }
    grep -q "ts2_seg_root: $TS2_SEG_ROOT\$" "$CFG_REL" || { echo "ERROR: config ts2_seg_root != $TS2_SEG_ROOT"; exit 1; }
    echo " ts2 seg root entries: $(ls "$TS2_SEG_ROOT" | wc -l) (74 env zips + extras expected; 75 seen at drafting)"
    TBL="outdoor_debug/parts/TA_sky/ts2_env_table.json"
    echo " ts2 committed env table: $(sha256sum "$TBL" | cut -d' ' -f1)  (expected $TS2_TABLE_SHA; the train log prints env_table_sha256 too)"
    [[ "$(sha256sum "$TBL" | cut -d' ' -f1)" == "$TS2_TABLE_SHA" ]] || { echo "ERROR: ts2 env table sha mismatch"; exit 1; } ;;
esac

export VAL01_EXPECTED_COMMIT="$HEAD_SHA"
source "$VENV/bin/activate"
python -c "import train; print('train.py imports OK')"
echo " launching: torchrun --standalone --nproc_per_node=$NGPU train.py --config $RUN_CFG"
torchrun --standalone --nproc_per_node="$NGPU" train.py --config "$RUN_CFG"
echo "======================================== done $(date -u +%FT%TZ)"
