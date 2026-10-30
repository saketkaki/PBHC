#!/usr/bin/env bash
# Record comparison videos of the released checkpoint (A) and the checkpoints of terrain runs on each test
# terrain, put them side by side, and attach them to the W&B runs.
#
# usage (from the repo root, env `pbhc` active, a display available):
#   [SECONDS_PER_CLIP=10] [OUT=DIR] [CAM_OFFSET="2.2 -1.2 -0.3"] [NO_WANDB="TAG ..."] \
#       scripts/record_pilot_videos.sh TAG=RUN_DIR|CKPT [TAG=RUN_DIR|CKPT ...]
#   e.g. scripts/record_pilot_videos.sh B=logs/.../model_53500.pt C=logs/.../model_53500.pt E=logs/.../model_53500.pt
# (a run directory means its latest checkpoint; pass checkpoint files to compare at equal iterations)
# Known tags get a description in the caption (B naive ref, C ground-relative ref, E height scan, E1/E2/E2pf for
# rounds 1 and 2 of E, E2c for the continuation of E2, E2c2 and E2ik for its plain and leg-IK continuations, F2s1/F2s2 for the fall-penalty runs with the 0.7 m drift cutoff, F3s1/F3s2 for perception v2 from F2s2). Tags listed in NO_WANDB are recorded but nothing is uploaded to their W&B runs.
# The camera is low and to the side (CAM_OFFSET, relative to the start root) so steps and bumps show, and the
# height-scan points are drawn on the ground (blue) in every clip to show the terrain shape.
set -euo pipefail

SECONDS_PER_CLIP=${SECONDS_PER_CLIP:-10}
read -r -a CAM <<< "${CAM_OFFSET:-2.2 -1.2 -0.3}"
# every clip uses terrain_kungfu without the curriculum (as terrain_eval.py does): a run trained with
# terrain.curriculum=True would otherwise build curriculum rows and ignore fixed_difficulty
VIEW=(--cam-offset "${CAM[@]}" env.config.terrain_aware.debug_draw_height_scan=True
      --merge humanoidverse/config/terrain/terrain_kungfu.yaml)
OUT=${OUT:-logs/MotionTracking/videos_$(date +%Y%m%d_%H%M%S)}
A_CKPT=example/pretrained_horse_stance_pose/model_50000.pt
A_FLAGS=(env.config.terrain_aware.recenter_motion_xy=True env.config.terrain_aware.ground_relative_ref=True)
declare -A DESCRIPTION=([B]="naive ref" [C]="ground-relative ref" [E]="height scan + ground-relative ref"
                        [E1]="height scan, round 1" [E2]="height scan, gentle fine-tune"
                        [E2pf]="height scan + per-foot ref, gentle fine-tune"
                        [E2c]="height scan, gentle fine-tune continued"
                        [E2c2]="height scan, E2c continued" [E2ik]="height scan + leg-IK ref, from E2c"
                        [F2s1]="height scan, fall penalty + 0.7 m cutoff, seed 1"
                        [F2s2]="height scan, fall penalty + 0.7 m cutoff, seed 2"
                        [F3s1]="perception v2 (wide + foot scans, exact height), seed 1"
                        [F3s2]="perception v2 (wide + foot scans, exact height), seed 2")
read -r -a NO_WANDB <<< "${NO_WANDB:-}"
mkdir -p "$OUT"

latest_ckpt() { ls "$1"/model_*.pt | sed 's/.*model_\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2; }
record() {  # record_video.py record with retries: the viewer can segfault on start while another renderer uses the GPU
    local attempt
    for attempt in 1 2 3 4 5; do
        python scripts/record_video.py record "$@" && return 0
        echo "record attempt $attempt failed (exit $?), retrying in 2 min"; sleep 120
    done
    return 1
}
wandb_run() {  # ENTITY/PROJECT/RUN_ID from the run's local wandb folder
    local id
    id=$(ls -d "$1"/.wandb/wandb/run-* 2>/dev/null | tail -1 | sed 's/.*-//')
    [ -n "$id" ] && echo "${WANDB_ENTITY}/MotionTracking/${id}"
}

TAGS=(); declare -A CKPT LABEL WANDB_RUN
for arg in "$@"; do
    tag=${arg%%=*}; path=${arg#*=}
    if [ -f "$path" ]; then dir=$(dirname "$path"); ckpt=$path; else dir=$path; ckpt=$(latest_ckpt "$path"); fi
    TAGS+=("$tag"); CKPT[$tag]=$ckpt
    LABEL[$tag]="$tag: ${DESCRIPTION[$tag]:-$tag}, it $(basename "$ckpt" .pt | cut -d_ -f2)"
    WANDB_RUN[$tag]=$(wandb_run "$dir" || true)
    [[ " ${NO_WANDB[*]} " == *" $tag "* ]] && WANDB_RUN[$tag]=""
    echo "$tag: $ckpt (${WANDB_RUN[$tag]:-no W&B run})"
done
echo "A: $A_CKPT -> $OUT"

# name | terrain overrides | label
TERRAINS=(
  "step_d10|terrain.terrain_types=[step] terrain.terrain_proportions=[1.0] terrain.fixed_difficulty=1.0|step 10 cm"
  "step_d05|terrain.terrain_types=[step] terrain.terrain_proportions=[1.0] terrain.fixed_difficulty=0.5|step 6.5 cm"
  "tilt_d10|terrain.terrain_types=[tilt] terrain.terrain_proportions=[1.0] terrain.fixed_difficulty=1.0|tilt 12 deg"
  "tilt_d05|terrain.terrain_types=[tilt] terrain.terrain_proportions=[1.0] terrain.fixed_difficulty=0.5|tilt 6 deg"
  "bumps_d10|terrain.terrain_types=[bumps] terrain.terrain_proportions=[1.0] terrain.fixed_difficulty=1.0|bumps 4 cm"
  "flat|terrain.terrain_types=[flat] terrain.terrain_proportions=[1.0]|flat"
)

for entry in "${TERRAINS[@]}"; do
    IFS='|' read -r name overrides label <<< "$entry"
    read -r -a ov <<< "$overrides"
    record "$A_CKPT" --out "$OUT/A_$name.mp4" --seconds "$SECONDS_PER_CLIP" \
        --label "A: released (flat-trained), $label" "${VIEW[@]}" "${A_FLAGS[@]}" "${ov[@]}"
    clips=("$OUT/A_$name.mp4")
    for tag in "${TAGS[@]}"; do
        run=${WANDB_RUN[$tag]}
        record "${CKPT[$tag]}" --out "$OUT/${tag}_$name.mp4" --seconds "$SECONDS_PER_CLIP" \
            --label "${LABEL[$tag]}, $label" "${VIEW[@]}" "${ov[@]}" ${run:+--wandb-run "$run" --wandb-key "videos/$name"}
        clips+=("$OUT/${tag}_$name.mp4")
    done
    python scripts/record_video.py stack "$OUT/compare_$name.mp4" "${clips[@]}"
done

# attach the side-by-side comparisons to every run
runs=(); for tag in "${TAGS[@]}"; do [ -n "${WANDB_RUN[$tag]}" ] && runs+=("${WANDB_RUN[$tag]}"); done
python - "$OUT" "A | ${TAGS[*]}" "${runs[@]}" <<'EOF'
import sys
from pathlib import Path
import wandb
out, caption, runs = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
for path in runs:
    entity, project, run_id = path.split('/')
    try:
        run = wandb.init(entity=entity, project=project, id=run_id, resume='allow',
                         settings=wandb.Settings(init_timeout=300))
        run.log({f'videos/compare_{p.stem[len("compare_"):]}': wandb.Video(str(p), format='mp4', caption=caption)
                 for p in sorted(out.glob('compare_*.mp4'))})
        run.finish()
        print(f'uploaded comparisons to {path}')
    except Exception as e:
        print(f'WARNING: upload of comparisons to {path} failed ({e})')
EOF
echo "videos in $OUT"
