#!/bin/bash
# Submits one train.sbatch job per <conditioning_mode>:<sketchfig_only|mixed> combo, each with a
# distinct --job-name (sketch_agent_train_<mode>_<data-tag>).
# Output is namespaced under output/experiments/<mode>_<data-tag>/ (see train.py), so different
# combos never collideç
#
# Usage:
#   bash training/sketch_agent/run_train_experiments.sh                    # canny+lineart, SketchFig-only
#   bash training/sketch_agent/run_train_experiments.sh canny:mixed lineart:sketchfig_only
#   MAX_STEPS=500 TIME_BUDGET=01:00:00 bash training/sketch_agent/run_train_experiments.sh
set -euo pipefail

cd "$(dirname "$0")/../.."

MAX_STEPS="${MAX_STEPS:-1000}"
TIME_BUDGET="${TIME_BUDGET:-08:00:00}"
MIXED_SKETCHFIG_FRACTION="${MIXED_SKETCHFIG_FRACTION:-0.25}"

COMBOS=("$@")
if [ ${#COMBOS[@]} -eq 0 ]; then
  COMBOS=(canny:sketchfig_only lineart:sketchfig_only)
fi

for combo in "${COMBOS[@]}"; do
  mode="${combo%%:*}"
  data_tag="${combo##*:}"
  extra_args=()
  case "$data_tag" in
    sketchfig_only) extra_args+=(--no-use-synthetic-data) ;;
    mixed) extra_args+=(--use-synthetic-data --sketchfig-train-fraction "$MIXED_SKETCHFIG_FRACTION") ;;
    *) echo "unknown data config '$data_tag' in '$combo' - expected sketchfig_only or mixed" >&2; exit 1 ;;
  esac

  run_name="train_${mode}_${data_tag}_$(date +%Y%m%d-%H%M%S)"
  echo "Submitting train.sbatch for conditioning_mode=$mode data=$data_tag (run-name=$run_name, max-steps=$MAX_STEPS, time=$TIME_BUDGET, extra=${extra_args[*]}) ..."
  sbatch --job-name="sketch_agent_train_${mode}_${data_tag}" --time="$TIME_BUDGET" training/sketch_agent/train.sbatch \
    --conditioning-mode "$mode" "${extra_args[@]}" --max-steps "$MAX_STEPS" --run-name "$run_name"
done
