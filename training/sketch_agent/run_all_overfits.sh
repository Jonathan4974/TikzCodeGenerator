#!/bin/bash
# Wipes output_overfit/ then submits one overfit.sbatch job per
# conditioning_mode
#
# Output is namespaced by mode under output_overfit/<mode>/ (see overfit_run.py's --conditioning-mode)
#
# Usage:
#   bash training/sketch_agent/run_all_overfits.sh                          # all 4 modes
#   bash training/sketch_agent/run_all_overfits.sh lineart anime_lineart    # specific modes
set -euo pipefail

cd "$(dirname "$0")/../.."

MODES=("$@")
if [ ${#MODES[@]} -eq 0 ]; then
  MODES=(canny scribble lineart anime_lineart)
fi

echo "Wiping training/sketch_agent/output_overfit/ ..."
rm -rf training/sketch_agent/output_overfit
echo "Cleared."

for mode in "${MODES[@]}"; do
  run_name="overfit_${mode}_$(date +%Y%m%d-%H%M%S)"
  echo "Submitting overfit.sbatch for conditioning_mode=$mode (run-name=$run_name) ..."
  sbatch --job-name="sketch_agent_overfit_${mode}" training/sketch_agent/overfit.sbatch \
    --conditioning-mode "$mode" --run-name "$run_name"
done
