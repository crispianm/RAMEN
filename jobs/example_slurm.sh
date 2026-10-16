#!/bin/bash
# Example Slurm launcher for a RAMEN training run (adapt partition / account / modules to your cluster).
#
#   sbatch --job-name=ramen_n46 --time=08:00:00 jobs/example_slurm.sh configs/experiments/dn01_c3rep_20261009_c3_n46.yaml
#
# The paper runs used 4 GPUs x batch 1 x grad_accum 4 (effective batch 16): keep NGPU * training.grad_accum = 16.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=18
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
CONFIG="${1:?usage: sbatch jobs/example_slurm.sh <config.yaml>}"
NGPU="${NGPU:-4}"
cd "${PROJECT_DIR:-$PWD}"
mkdir -p logs

# train.py refuses to run under Slurm unless the checked-out commit is the one you declare
# (and the working tree is clean), so that every checkpoint records the exact code that made it.
export VAL01_EXPECTED_COMMIT="$(git rev-parse HEAD)"

# Edit the model repo/checkpoint paths and data roots in the config first (see README, "Training").
torchrun --standalone --nproc_per_node="$NGPU" train.py --config "$CONFIG"
