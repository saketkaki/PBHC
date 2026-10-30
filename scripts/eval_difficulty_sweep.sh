#!/usr/bin/env bash
# How far a checkpoint holds up beyond the training difficulty: scripts/terrain_eval.py on step, tilt and bumps at
# difficulties up to 2x the curriculum maximum (difficulty 1 = step 10 cm, tilt 12 deg, bumps 4 cm; all three scale
# linearly with difficulty), plus flat and the held-out rough terrain. One process per terrain, difficulty and seed.
#
# usage (from the repo root, env `pbhc` active):
#   [SEEDS="0 1 2"] [TYPES="step tilt bumps"] [DIFFS="0.5 1.0 1.25 1.5 2.0"] [EPISODES=200] \
#       scripts/eval_difficulty_sweep.sh CKPT [key=value ...]
# Results: CKPT_DIR/metrics/terrain_eval_v2/<ckpt name>/<type>_d<100 x difficulty>_seed<seed>.json (e.g. step_d125);
# summarise with: python scripts/summarize_difficulty_sweep.py LABEL=CKPT [LABEL=CKPT ...]
set -euo pipefail

CKPT=$1; shift
SEEDS=${SEEDS:-"0 1 2"}
TYPES=${TYPES:-"step tilt bumps"}
DIFFS=${DIFFS:-"0.5 1.0 1.25 1.5 2.0"}
EPISODES=${EPISODES:-200}

run() {  # name, overrides...
    local name=$1 seed; shift
    for seed in $SEEDS; do
        python scripts/terrain_eval.py "$CKPT" --name "$name" --seed "$seed" --episodes "$EPISODES" "$@" \
            2>&1 | grep -aE "^RESULT|Traceback|Error"
    done
}

run flat "terrain.terrain_types=[flat]" "terrain.terrain_proportions=[1.0]" "$@"
run rough_d090 "terrain.terrain_types=[rough]" "terrain.terrain_proportions=[1.0]" terrain.fixed_difficulty=0.9 "$@"
for type in $TYPES; do
    for diff in $DIFFS; do
        run "${type}_d$(printf '%03d' "$(echo "$diff * 100 / 1" | bc)")" "terrain.terrain_types=[$type]" \
            "terrain.terrain_proportions=[1.0]" terrain.fixed_difficulty="$diff" "$@"
    done
done
