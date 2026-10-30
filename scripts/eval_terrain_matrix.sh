#!/usr/bin/env bash
# Run scripts/terrain_eval.py for one checkpoint on every test terrain and seed (one process per terrain and seed).
#
# usage (from the repo root, env `pbhc` active):
#   scripts/eval_terrain_matrix.sh CKPT [key=value ...]
#   e.g. the released checkpoint (predates the terrain options):
#   scripts/eval_terrain_matrix.sh example/pretrained_horse_stance_pose/model_50000.pt \
#       env.config.terrain_aware.recenter_motion_xy=True env.config.terrain_aware.ground_relative_ref=True
# Summarise with: python scripts/summarize_terrain_eval.py CKPT [CKPT ...]
set -euo pipefail

CKPT=$1; shift
SEEDS=${SEEDS:-"0 1"}
ONLY=${ONLY:-}  # e.g. ONLY=bumps_d10: run only these terrains
SUFFIX=${SUFFIX:-}  # appended to the result names, e.g. SUFFIX=__point05 -> step_d10__point05_seed0.json
EPISODES=${EPISODES:-200}

# name | terrain overrides
TERRAINS=(
  "flat|terrain.terrain_types=[flat] terrain.terrain_proportions=[1.0]"
  "rough_d09|terrain.terrain_types=[rough] terrain.terrain_proportions=[1.0] terrain.fixed_difficulty=0.9"
  "bumps_d10|terrain.terrain_types=[bumps] terrain.terrain_proportions=[1.0] terrain.fixed_difficulty=1.0"
  "tilt_d05|terrain.terrain_types=[tilt] terrain.terrain_proportions=[1.0] terrain.fixed_difficulty=0.5"
  "tilt_d10|terrain.terrain_types=[tilt] terrain.terrain_proportions=[1.0] terrain.fixed_difficulty=1.0"
  "step_d05|terrain.terrain_types=[step] terrain.terrain_proportions=[1.0] terrain.fixed_difficulty=0.5"
  "step_d10|terrain.terrain_types=[step] terrain.terrain_proportions=[1.0] terrain.fixed_difficulty=1.0"
)

for entry in "${TERRAINS[@]}"; do
    IFS='|' read -r name overrides <<< "$entry"
    [ -n "$ONLY" ] && [[ " $ONLY " != *" $name "* ]] && continue
    read -r -a ov <<< "$overrides"
    for seed in $SEEDS; do
        python scripts/terrain_eval.py "$CKPT" --name "$name$SUFFIX" --seed "$seed" --episodes "$EPISODES" "${ov[@]}" "$@" \
            2>&1 | grep -aE "^RESULT|Traceback|Error"
    done
done
