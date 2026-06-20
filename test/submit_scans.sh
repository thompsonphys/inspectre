#!/bin/bash
# Submit one scantest timing job per config in CFG_DIR (Slurm array).
# Usage: bash test/submit_scans.sh [config-dir]   (default: test/scan_configs)
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
CFG_DIR="${1:-test/scan_configs}"

# 1. Ensure binary is current (no-op if up to date).
make -C test scansweep

# 2. Collect configs; fail loudly if none.
mapfile -t CONFIGS < <(ls -1 "$CFG_DIR"/*.cfg 2>/dev/null | sort)
[ "${#CONFIGS[@]}" -gt 0 ] || { echo "no .cfg files in $CFG_DIR" >&2; exit 1; }

# 3. Abort on duplicate outfile= across configs (concurrent jobs would clobber
#    each other's CSV). Compare the trimmed value after the first '='.
declare -A SEEN
for cfg in "${CONFIGS[@]}"; do
    out="$(sed -n 's/^[[:space:]]*outfile[[:space:]]*=[[:space:]]*//p' "$cfg" | tail -n1)"
    [ -n "$out" ] || { echo "$cfg: no outfile= set" >&2; exit 1; }
    if [ -n "${SEEN[$out]:-}" ]; then
        echo "duplicate outfile '$out' in $cfg and ${SEEN[$out]}" >&2
        exit 1
    fi
    SEEN[$out]="$cfg"
done

# 4. Logs dir + submit array 0..n-1.
mkdir -p logs
N=${#CONFIGS[@]}
echo "submitting $N timing jobs from $CFG_DIR"
sbatch --array=0-$((N-1)) --export=ALL,CFG_DIR="$CFG_DIR" test/scan_timing.sbatch
