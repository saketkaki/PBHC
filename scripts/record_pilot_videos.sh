#!/usr/bin/env bash
# Record comparison videos of the released checkpoint (A) and the latest checkpoints of two terrain runs
# (default: pilot B and pilot C) on each test terrain, put them side by side, and attach them to the W&B runs.
#
# usage (from the repo root, env `pbhc` active, a display available):
#   scripts/record_pilot_videos.sh B_RUN_DIR|B_CKPT C_RUN_DIR|C_CKPT [SECONDS] [OUT_DIR]
# (a run directory means its latest checkpoint; pass checkpoint files to compare at equal iterations)
set -euo pipefail

run_dir() { if [ -f "$1" ]; then dirname "$1"; else echo "$1"; fi; }
B_ARG=$1; C_ARG=$2
B_DIR=$(run_dir "$B_ARG"); C_DIR=$(run_dir "$C_ARG")
SECONDS_PER_CLIP=${3:-10}
OUT=${4:-logs/MotionTracking/videos_$(date +%Y%m%d_%H%M%S)}
A_CKPT=example/pretrained_horse_stance_pose/model_50000.pt
A_FLAGS=(--merge humanoidverse/config/terrain/terrain_kungfu.yaml
         env.config.terrain_aware.recenter_motion_xy=True env.config.terrain_aware.ground_relative_ref=True)
mkdir -p "$OUT"

latest_ckpt() { ls "$1"/model_*.pt | sed 's/.*model_\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2; }
wandb_run() {  # ENTITY/PROJECT/RUN_ID from the run's local wandb folder
    local id
    id=$(ls -d "$1"/.wandb/wandb/run-* 2>/dev/null | tail -1 | sed 's/.*-//')
    [ -n "$id" ] && echo "${WANDB_ENTITY}/MotionTracking/${id}"
}

ckpt() { if [ -f "$1" ]; then echo "$1"; else latest_ckpt "$1"; fi; }
B_CKPT=$(ckpt "$B_ARG"); C_CKPT=$(ckpt "$C_ARG")
B_IT=$(basename "$B_CKPT" .pt | cut -d_ -f2); C_IT=$(basename "$C_CKPT" .pt | cut -d_ -f2)
B_WANDB=$(wandb_run "$B_DIR" || true); C_WANDB=$(wandb_run "$C_DIR" || true)
echo "A: $A_CKPT | B: $B_CKPT ($B_WANDB) | C: $C_CKPT ($C_WANDB) -> $OUT"

# name | terrain overrides | label
TERRAINS=(
  "step_d10|terrain.terrain_types=[step] terrain.terrain_proportions=[1.0] terrain.fixed_difficulty=1.0|step 10 cm"
  "step_d05|terrain.terrain_types=[step] terrain.terrain_proportions=[1.0] terrain.fixed_difficulty=0.5|step 6.5 cm"
  "tilt_d10|terrain.terrain_types=[tilt] terrain.terrain_proportions=[1.0] terrain.fixed_difficulty=1.0|tilt 12 deg"
  "rough_d09|terrain.terrain_types=[rough] terrain.terrain_proportions=[1.0] terrain.fixed_difficulty=0.9|rough"
  "flat|terrain.terrain_types=[flat] terrain.terrain_proportions=[1.0]|flat"
)

for entry in "${TERRAINS[@]}"; do
    IFS='|' read -r name overrides label <<< "$entry"
    read -r -a ov <<< "$overrides"
    python scripts/record_video.py record "$A_CKPT" --out "$OUT/A_$name.mp4" --seconds "$SECONDS_PER_CLIP" \
        --label "A: released (flat-trained), $label" "${A_FLAGS[@]}" "${ov[@]}"
    python scripts/record_video.py record "$B_CKPT" --out "$OUT/B_$name.mp4" --seconds "$SECONDS_PER_CLIP" \
        --label "B: naive ref, it $B_IT, $label" "${ov[@]}" \
        ${B_WANDB:+--wandb-run "$B_WANDB" --wandb-key "videos/$name"}
    python scripts/record_video.py record "$C_CKPT" --out "$OUT/C_$name.mp4" --seconds "$SECONDS_PER_CLIP" \
        --label "C: ground-relative ref, it $C_IT, $label" "${ov[@]}" \
        ${C_WANDB:+--wandb-run "$C_WANDB" --wandb-key "videos/$name"}
    python scripts/record_video.py stack "$OUT/compare_$name.mp4" "$OUT/A_$name.mp4" "$OUT/B_$name.mp4" "$OUT/C_$name.mp4"
done

# attach the side-by-side comparisons to both runs
python - "$OUT" "$B_WANDB" "$C_WANDB" <<'EOF'
import sys
from pathlib import Path
import wandb
out, runs = Path(sys.argv[1]), [r for r in sys.argv[2:] if r]
for path in runs:
    entity, project, run_id = path.split('/')
    run = wandb.init(entity=entity, project=project, id=run_id, resume='allow')
    run.log({f'videos/compare_{p.stem[len("compare_"):]}': wandb.Video(str(p), format='mp4', caption='A released | B naive | C ground-relative')
             for p in sorted(out.glob('compare_*.mp4'))})
    run.finish()
    print(f'uploaded comparisons to {path}')
EOF
echo "videos in $OUT"
