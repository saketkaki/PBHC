#!/usr/bin/env bash
# Robustness of a height-scan policy to perception errors: scripts/eval_terrain_matrix.sh on a few terrains under each
# height-scan noise condition (env.config.terrain_aware.height_scan_noise; one error at a time, then all combined).
# The `height_scan` term itself is perturbed, so policies trained on the exact scan can be evaluated.
#
# usage (from the repo root, env `pbhc` active):
#   [SEEDS="0"] [TERRAINS="flat tilt_d05 tilt_d10 step_d05 bumps_d10"] [CONDITIONS="exact point05 ..."] \
#       scripts/eval_scan_noise.sh CKPT [key=value ...]
#   e.g. scripts/eval_scan_noise.sh logs/MotionTracking/<run>/model_53500.pt env.config.terrain_aware.ground_relative_ref=True
# Results: CKPT_DIR/metrics/terrain_eval_v2/<ckpt name>/<terrain>__<condition>_seed<seed>.json;
# summarise with: python scripts/summarize_scan_noise.py CKPT [CKPT ...]
set -euo pipefail

CKPT=$1; shift
export SEEDS=${SEEDS:-"0"}
TERRAINS=${TERRAINS:-"flat tilt_d05 tilt_d10 step_d05 bumps_d10"}
N=env.config.terrain_aware.height_scan_noise

# name | overrides (delay in control steps of 20 ms)
ALL_CONDITIONS=(
  "exact|"
  "point02|$N.point_std=0.02"
  "point05|$N.point_std=0.05"
  "bias02|$N.episode_bias_std=0.02"
  "bias05|$N.episode_bias_std=0.05"
  "drift05|$N.episode_drift_xy_std=0.05"
  "drift10|$N.episode_drift_xy_std=0.1"
  "drop30|$N.dropout=0.3"
  "drop60|$N.dropout=0.6"
  "delay2|$N.delay_steps=[2,2]"
  "delay5|$N.delay_steps=[5,5]"
  "blind_flat|$N.blind=flat"
  "blind_zero|$N.blind=zero"
  "combined|$N.point_std=0.02 $N.episode_bias_std=0.02 $N.episode_drift_xy_std=0.03 $N.dropout=0.1 $N.delay_steps=[0,2]"
)
CONDITIONS=${CONDITIONS:-}

for entry in "${ALL_CONDITIONS[@]}"; do
    IFS='|' read -r cond overrides <<< "$entry"
    [ -n "$CONDITIONS" ] && [[ " $CONDITIONS " != *" $cond "* ]] && continue
    read -r -a ov <<< "$overrides"
    echo "== $cond"
    ONLY="$TERRAINS" SUFFIX="__$cond" scripts/eval_terrain_matrix.sh "$CKPT" $N.perturb_height_scan=True "${ov[@]}" "$@"
done
