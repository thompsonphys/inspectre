#!/bin/bash
#SBATCH --job-name=scantest
#SBATCH --partition=full
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=logs/scan_%A_%a.out
#SBATCH --error=logs/scan_%A_%a.err
# One array task per config file. The --array range is supplied by
# submit_scans.sh at submit time, sized to the number of configs.

set -euo pipefail

REPO_ROOT="$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# bin/scansweep has an RPATH into the conda env, so GSL/FFTW resolve without
# activating the env. Working dir must be the repo root: scansweep reads the
# config and writes its `outfile` via relative paths.

CFG_DIR="${CFG_DIR:-test/scan_configs}"
mapfile -t CONFIGS < <(ls -1 "$CFG_DIR"/*.cfg | sort)
CFG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

echo "[task $SLURM_ARRAY_TASK_ID] host=$(hostname) cfg=$CFG start=$(date -Is)"
# Per-method timings land in the CSV (time_s column). The bash `time` keyword
# adds total job wall/user/sys to the .err log. GNU /usr/bin/time isn't installed.
time bin/scansweep "$CFG"
echo "[task $SLURM_ARRAY_TASK_ID] done=$(date -Is)"
