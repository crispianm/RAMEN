#!/bin/bash
# =============================================================================
# L1 (Track 2 light screen), run S4 at 4000 steps: a FROM-RELEASE training run (no resume; released Metric-VDA-S init, C3 recipe, cosine to 0 at 4000, warmup 500), 4 GH200, 1 node.
# DRAFT, NOT SUBMITTED. Generated from train_template_l1.sbatch by gen_l1.sh -- edit the template, not this file.
# Config (committed at PIN_COMMIT): outdoor_debug/parts/far_diag/l1_final/configs/k4000/arm_S4.yaml  sha256 d673f7829e1d5329e5677f454f4de359c4de565301cb383a112e5e30576bc234
# Launch conventions copied from the c3rep n46 job 7226119 and the outdoor screen (workq, 1 node, 1 task, gres gpu:4, 72 cpus, 400G, FocusMamba venv, torchrun --standalone, VAL01_EXPECTED_COMMIT = HEAD).
# SMOKE=<N> (via --export, N = 120) trains an N-step copy: max_steps N, smoke checkpoint_dir, checkpoint_at_steps [N] (exactly 3 changed lines, asserted); LR keys untouched so the N steps are the real warmup.
# S6 only runs after the Auditor's diff check: it needs S6_CLEARED_PATCH_SHA=<sha256 of S6_diff.patch the Auditor checked> in --export and refuses otherwise.
# =============================================================================
#SBATCH --job-name=l1_S4_k4000
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --mem=400G
#SBATCH --time=05:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

export ARM="S4"
export K="4000"
export PIN_COMMIT="UNSET_PIN_COMMIT"     # sed at submit (SUBMIT.md block 0) to the full 40-hex sha, into a copy OUTSIDE the clone; the script refuses UNSET
export PROJECT_DIR="${PROJECT_DIR:-/projects/b5dh/RAMEN_l1_UNSET}"
export SMOKE="${SMOKE:-0}"
export CFG_REL="outdoor_debug/parts/far_diag/l1_final/configs/k${K}/arm_${ARM}.yaml"
export CFG_SHA="d673f7829e1d5329e5677f454f4de359c4de565301cb383a112e5e30576bc234"
export BASE_REL="outdoor_debug/parts/confirm/stage1/base/effective_ramen_n46.yaml"
export BASE_SHA="d286b62cef5f5e603ca909675b7daf22c4e120b0b99dfe3d5ec69bcbebd911a1"      # the c3_n46 job's resolved config (the L1 base)
export REL_WEIGHTS="/projects/b5dh/benchmark/weights/metric_video_depth_anything_vits.pth"
export REL_SHA="3c28432b4e1f0d7bb31cad5151b6313b49457db5aa58d82e85bfb0f8b1311b33"       # the RELEASED Metric-VDA-S (model.vda_init released; same file the DDAD release row used)
export SEED="46"
export TRAIN_PY_SHA="475ffdbfdf79e3d6dc10c30dad5bd05bd9948b4b1f0153004300122682eb8822"   # sha256 of the S6-containing train.py the Auditor checked (O48)
export TRAIN_BASE_COMMIT="8e1d9ae"      # S6 gate: train.py may differ from this base ONLY by S6_diff.patch
export VENV="/projects/b5dh/FocusMamba/.venv"
export NGPU=4

echo "======================================== l1_${ARM}_k${K} smoke=${SMOKE}"
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
chk() { local got; got="$(sha256sum "$2" | cut -d' ' -f1)"; [[ "$got" == "$3" ]] || { echo "ERROR: $1 sha $got != $3 ($2)"; exit 1; }; echo " $1 : $2 sha256 $got OK"; }
chk config  "$CFG_REL"      "$CFG_SHA"
chk base    "$BASE_REL"     "$BASE_SHA"
chk release "$REL_WEIGHTS"  "$REL_SHA"
grep -q "^  resume_from: null\$" "$CFG_REL"       || { echo "ERROR: resume_from is not null (L1 runs start FROM THE RELEASE)"; exit 1; }
! grep -q "resume_reseed\|lr_restart" "$CFG_REL"  || { echo "ERROR: config carries resume_reseed / lr_restart keys"; exit 1; }
grep -q "^  seed: $SEED\$" "$CFG_REL"              || { echo "ERROR: data.seed $SEED not in config"; exit 1; }
for kv in "max_steps: $K" "lr_decay_steps: $K" "warmup_steps: 500" "severity_warmup_steps: 500"; do
  grep -q "^  $kv\$" "$CFG_REL" || { echo "ERROR: config lacks '$kv'"; exit 1; }
done
grep -q "^  vda_init: released\$" "$CFG_REL"      || { echo "ERROR: model.vda_init is not 'released'"; exit 1; }
if [[ "$ARM" == "S6" ]]; then
    PATCH="outdoor_debug/parts/far_diag/l1_final/S6_diff.patch"
    [[ -n "${S6_CLEARED_PATCH_SHA:-}" ]] || { echo "ERROR: S6 needs S6_CLEARED_PATCH_SHA (the Auditor's diff check has not been recorded)"; exit 1; }
    [[ "$(sha256sum "$PATCH" | cut -d' ' -f1)" == "$S6_CLEARED_PATCH_SHA" ]] || { echo "ERROR: S6_diff.patch sha != the cleared sha"; exit 1; }
    # host-independent diff text: --full-index (no abbreviation, so core.abbrev cannot change the hash), --no-color, --no-ext-diff (Auditor O48)
    [[ "$(git diff --full-index --no-color --no-ext-diff "$TRAIN_BASE_COMMIT" HEAD -- train.py | sha256sum | cut -d' ' -f1)" == "$S6_CLEARED_PATCH_SHA" ]] || { echo "ERROR: train.py at HEAD differs from base+cleared patch (git diff --full-index $TRAIN_BASE_COMMIT HEAD -- train.py != the cleared patch)"; exit 1; }
    [[ "$(sha256sum train.py | cut -d' ' -f1)" == "$TRAIN_PY_SHA" ]] || { echo "ERROR: train.py sha != TRAIN_PY_SHA (S6 gate)"; exit 1; }
    echo " S6 gate OK: train.py == $TRAIN_BASE_COMMIT + the cleared patch ($S6_CLEARED_PATCH_SHA), train.py sha $TRAIN_PY_SHA"
fi

RUN_CFG="$CFG_REL"
if [[ "$SMOKE" != "0" ]]; then
    N="$SMOKE"; [[ "$N" =~ ^[0-9]+$ ]] || { echo "ERROR: SMOKE must be a step count"; exit 1; }
    RUN_CFG="$TMPDIR/arm_${ARM}_k${K}_smoke${N}_${SLURM_JOB_ID:-local}.yaml"
    sed -e "s/^  max_steps: ${K}\$/  max_steps: ${N}/" \
        -e "s#^  checkpoint_dir: runs/l1_${ARM}_k${K}\$#  checkpoint_dir: runs/l1_smoke_${ARM}_k${K}#" \
        -e "/^  checkpoint_at_steps:\$/{n;s/^  - ${K}\$/  - ${N}/}" "$CFG_REL" > "$RUN_CFG"
    [[ "$(diff "$CFG_REL" "$RUN_CFG" | grep -c '^>')" == "3" ]] || { echo "ERROR: smoke config diff is not exactly 3 lines"; diff "$CFG_REL" "$RUN_CFG" || true; exit 1; }
    echo " SMOKE config $RUN_CFG (sha256 $(sha256sum "$RUN_CFG" | cut -d' ' -f1)); diff vs arm yaml:"; diff "$CFG_REL" "$RUN_CFG" || true
    OUT_DIR="runs/l1_smoke_${ARM}_k${K}"
else
    OUT_DIR="runs/l1_${ARM}_k${K}"
fi
if [[ -e "$OUT_DIR" ]]; then echo "ERROR: $OUT_DIR already exists (never reuse an output dir)"; exit 1; fi
echo " ckpt dir (only write location for checkpoints): $PROJECT_DIR/$OUT_DIR"

export VAL01_EXPECTED_COMMIT="$HEAD_SHA"
source "$VENV/bin/activate"
python -c "import train; print('train.py imports OK')"
# the stop check greps the config line below (arm yaml path + sha); keep the format
echo " config : $CFG_REL sha256 $(sha256sum "$CFG_REL" | cut -d' ' -f1) OK"
echo " launching: torchrun --standalone --nproc_per_node=$NGPU train.py --config $RUN_CFG"
torchrun --standalone --nproc_per_node="$NGPU" train.py --config "$RUN_CFG"
# record the independent checksum of the final snapshot (the scorer asserts it)
FINAL="$OUT_DIR/step$([[ "$SMOKE" == "0" ]] && echo "$K" || echo "$SMOKE").pt"
[[ -f "$FINAL" ]] && sha256sum "$FINAL" | tee "$FINAL.sha256" || { echo "ERROR: final snapshot $FINAL missing"; exit 1; }
echo "======================================== done $(date -u +%FT%TZ)"
