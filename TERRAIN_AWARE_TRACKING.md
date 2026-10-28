# Terrain-Aware Motion Tracking for PBHC

Final-project plan: extend PBHC (KungfuBot, NeurIPS 2025) so a humanoid can perform highly-dynamic motions on non-flat ground.

## 1. Motivation

The PBHC paper states its own limitation (Section 6):

> (i) It lacks environment awareness, such as terrain perception and obstacle avoidance, which restricts deployment in unstructured real-world settings.

PBHC trains and evaluates only on a flat plane. Its tracking objective also assumes flat ground: reference motions are placed at one fixed height per environment, and the rewards compare body positions in world coordinates. On a slope or a step, the reference floats above the ground on one side and sinks into it on the other, so the policy is rewarded for the wrong behaviour.

**Goal:** make PBHC's motion tracking terrain-aware, with three parts:

1. **Ground-relative reference adaptation** (core contribution): track the reference motion relative to the local ground, not in world coordinates.
2. **Height-scan perception**: give the policy a local terrain height map.
3. **Terrain curriculum**: progress from flat to rough, tilted and stepped ground.

## 2. What the existing code gives us

All paths are relative to the repo root.

| Fact | Where | Consequence |
|---|---|---|
| The example motions travel only 0.2–1.0 m horizontally (measured on all 7 motions in `example/motion_data/`). | — | The task is performing kung-fu/dance on **uneven footing**, not walking across terrain. Terrain features must sit under the robot's feet. |
| The reference motion is offset by `env_origins`, a constant per environment. | `humanoidverse/envs/motion_tracking/motion_tracking.py` (`kick_motion_res`, ~L548) | One fixed height per environment, so the reference is wrong on a slope or step. |
| Each origin's z is the **maximum** terrain height in the central 1×1 m of its tile. | `humanoidverse/envs/env_utils/terrain.py` (`add_terrain_to_map`, ~L110) | Must not be double-counted once a per-frame height shift is added. |
| Every reference state (observations, rewards, termination, reset) flows through `kick_motion_res()` / `kick_motion_res_multiplestep()`. | `motion_tracking.py` ~L548–L580 | A single place to implement the reference adaptation. |
| Terrain generation (heightfield/trimesh, curriculum layout) already exists; the Isaac Gym backend stores `self.simulator.height_samples`. | `terrain.py`, `humanoidverse/simulator/isaacgym/isaacgym.py` ~L132 | The infrastructure is there, but nothing samples heights. |
| `measure_heights` is only a config flag ("keep it False"); `_update_terrain_curriculum` is only mentioned in a docstring. | `config/terrain/*.yaml`, `legged_robot_base.py` ~L494 | Height scan and terrain curriculum must be written. |
| The contact-mask reward compares measured contact forces with the reference mask. | `motion_tracking.py` `_reward_teleop_contact_mask` ~L1154 | Already terrain-agnostic; **no change needed**. |
| Body-position, VR 3-point and feet rewards, and the `motion_far` termination, use global positions (`dif_global_body_pos`). | `motion_tracking.py` ~L1177–L1220, `_update_reset_buf` ~L330 | These break on terrain and are fixed by the reference adaptation. |
| The adaptive tracking factor σ is a single EMA averaged over all environments. | `motion_tracking.py` `_update_adaptive_sigma` ~L1030 | Mixed terrain difficulty may distort σ; monitor it. |
| Observations are discovered by name (`_get_obs_<name>`) and declared in YAML. | `config/obs/motion_tracking/main.yaml` | Adding a height-scan observation is a getter plus config entries. |
| The motion-tracking experiment uses `mh_ppo` (vectorised critic). Teacher–student/DAgger exists only in `ppo_mimic` (used by `general_tracking`). | `config/exp/motion_tracking.yaml`, `agents/ppo/ppo_mimic.py` | Distillation is not plug-and-play here; treat it as a stretch goal. |

## 3. Implementation plan

### Step 0: Terrain infrastructure (~1 day)

**Height lookup helper** (in `MotionTracking`, or in `LeggedRobotBase` for reuse), following the legged_gym convention:

```python
def _terrain_height_at(self, xy):            # xy: [..., 2] world coords
    if self.config.terrain.mesh_type == 'plane':
        return torch.zeros(xy.shape[:-1], device=self.device)
    hs = self.simulator.height_samples
    s, b = self.config.terrain.horizontal_scale, self.config.terrain.border_size
    px = ((xy[..., 0] + b) / s).long().clamp(0, hs.shape[0] - 2)
    py = ((xy[..., 1] + b) / s).long().clamp(0, hs.shape[1] - 2)
    h = torch.min(torch.min(hs[px, py], hs[px + 1, py]), hs[px, py + 1])
    return h.float() * self.config.terrain.vertical_scale
```

Verify it visually by drawing spheres at sampled heights with `self.simulator.draw_sphere` (as `_draw_debug_vis` does for markers), with the viewer enabled.

**New terrain types** in `terrain.py`. The robot stands at the **centre** of each tile. legged_gym's pyramid stairs and slopes have a flat platform there, so they would leave the robot on flat ground. Features must pass through the centre, scaled by `difficulty`:

| Function | Terrain |
|---|---|
| `tilt_terrain_func` | Planar ramp through the centre, height = tan(θ)·x, θ up to ~10–15°. |
| `rough_terrain_func` | Already exists (random uniform noise); cap amplitude at ~3–5 cm. |
| `step_terrain_func` | Half the tile raised by 3–10 cm, edge near the centre, so the feet are at different heights. |
| `stones_terrain_func` | Stretch goal: small raised blocks. |

**Config:** create `humanoidverse/config/terrain/terrain_kungfu.yaml` from `terrain_locomotion.yaml` with `mesh_type: trimesh`, the new types and proportions, fewer rows/columns (e.g. 5×8), and smaller tiles (`terrain_length`/`terrain_width` ≈ 3 m), since each robot only uses about 1 m².

**Spawn clearance:** add 1–2 cm to the root height at reset (there is a commented-out line for this at `motion_tracking.py` ~L509).

### Step 1: Motivating experiment (~½ day, no training)

Evaluate the released checkpoint `example/pretrained_horse_stance_pose/model_50000.pt` zero-shot on each terrain type and difficulty:

```bash
python humanoidverse/ratio_eps.py +device=cuda:0 \
  +checkpoint=example/pretrained_horse_stance_pose/model_50000.pt \
  +opt=record +num_envs=100 +num_episodes=100 +eps_eval_name=terrain_zeroshot \
  +terrain=terrain_kungfu
```

On flat ground this checkpoint reaches a 98.4% episode-length ratio (measured on our setup). The expected drop on slopes and steps is the motivating figure for the report.

### Step 2: Ground-relative reference adaptation (core contribution, ~2–3 days)

Implement inside `kick_motion_res()` and `kick_motion_res_multiplestep()`:

1. **Remove the fixed origin height.** Pass the origin's x, y as the offset with z = 0.
2. **Per-frame support height.** Look up terrain heights under the reference feet (`rg_pos_t[:, feet_ids, :2]`). *(Implemented differently from first planned; see Section 8.)* A rigid vertical shift cannot put two feet on ground of different heights, and averaging the two heights left the foot on higher ground up to 3.5 cm *inside* the terrain. The implementation uses a soft maximum over the supporting feet: `h_sup = Σ wᵢ·hᵢ`, `w = softmax((hᵢ − liftᵢ)/τ)`, where `liftᵢ` is how far foot *i* is raised in the reference and τ = 0.02 m. It is continuous in time, needs no state, and is exactly 0 on flat ground.
3. **Shift the reference.** Add `h_sup` to the z of `root_pos`, `rg_pos` and `rg_pos_t`. Rotations are unchanged; optionally add d(`h_sup`)/dt to the vertical velocities.
4. **Ground-relative feet reward.** In `_reward_teleop_body_position_feet` (~L1210), compare each foot's height above the terrain beneath it, `foot_z − h(foot_xy)`, with the reference foot's height above flat ground; keep x, y as before.
   The idea to present in the report: **track the motion in a ground-relative frame instead of the world frame.**
5. **Leg-joint conflict.** On a slope, ankle and knee angles must deviate from the reference. Add a `teleop_joint_pos_lower_weight` mask in `_reward_teleop_joint_position` (~L1265); ablate 1.0 vs ~0.3.
6. **Regression test (required).** With `mesh_type: plane` all heights are zero, so the modified code must reproduce PBHC exactly. Run `sample_eps.py` on the pretrained checkpoint before and after the change; the metrics must be identical.

**Evaluation scripts:** `sample_eps.py` / `ratio_eps.py` compute metrics against the reference loaded from file (flat ground; see `root_trans_offset` handling at `motion_tracking.py` ~L872). Save the **adapted** reference together with the robot trajectory so terrain metrics are computed fairly.

### Step 3: Height-scan perception (~1–2 days)

1. **Scan grid:** e.g. 11 × 7 points at 10 cm spacing (1.0 m × 0.6 m), **centred under the feet** (the motions barely travel, so no look-ahead is needed). Precompute in the robot's heading frame and rotate by heading each step (`calc_heading_quat` is already imported).
2. **Getter:** add `_get_obs_height_scan` to `MotionTracking`, returning `clip(root_z − h(points) − c, −1, 1)` with `c` ≈ 0.6–0.7. Kung-fu squats move root height from ~0.55 to ~0.8 m, so an alternative is heights relative to the lower foot, which is invariant to squatting. Justify the choice in the report.
3. **Config:** copy `config/obs/motion_tracking/main.yaml` to `main_terrain.yaml`; add `height_scan` to `critic_obs` (and `actor_obs` for the perceptive variant), `obs_dims` (`height_scan: 77`), `obs_scales` (1.0) and `noise_scales` (~0.02–0.05 for the actor).

| Variant | Actor sees scan | Critic sees scan |
|---|---|---|
| Blind | no | no |
| Privileged critic | no | yes |
| Perceptive | yes | yes |

Stretch goal: teacher–student distillation (perceptive teacher → student with noisy/partial scan), which requires porting the DAgger logic from `ppo_mimic` to `mh_ppo`.

### Step 4: Terrain curriculum (~1 day)

- Set `terrain.curriculum: True`; `humanoidverse/envs/base_task/base_task.py` (~L117–L129) already initialises `terrain_levels`, `terrain_types` and `terrain_origins`.
- Implement `_update_terrain_curriculum(env_ids)` and call it from `_reset_tasks_callback` (`motion_tracking.py` ~L265) for resetting environments:
  - level up if the episode reached the motion end (`reset_buf_terminate_by["motion_end"]`);
  - level down if the episode-length ratio was below ~0.5;
  - then `env_origins[env_ids] = terrain_origins[levels, types]`.
- Log the mean terrain level in `log_dict`.
- **Monitor adaptive σ.** Since σ is one EMA over all environments, easy tiles may pull σ down and over-penalise hard tiles. If that happens, a per-terrain-level σ is a cheap, well-motivated extension.

## 4. Experiments

**Motions:** 2–3, e.g. Horse-stance pose (easy), Horse-stance punch (easy), Roundhouse kick (hard).
**Test terrains:** flat, rough, tilt, step, each at 3 difficulty levels.

| Model (rows) / test terrain (columns) | flat | rough | tilt | step |
|---|---|---|---|---|
| A. PBHC as released (pretrained, zero-shot) | | | | |
| B. PBHC trained on terrain, world-frame reference (naive) | | | | |
| C. B + ground-relative reference (Step 2) | | | | |
| C′. C with lower-body joint weight 0.3 (ablation) | | | | |
| D. C + privileged-critic height scan | | | | |
| E. C + actor height scan (full method) | | | | |

**Metrics:** PBHC's own (`E_gmpbpe`, `E_mpbpe`, `E_mpjpe`, velocity/acceleration errors), episode-length ratio, plus foot penetration and stumble counts (`_reward_penalty_stumble` already detects stumbling).

**What each comparison shows:**
- A → B: training on terrain alone is not enough.
- B → C: effect of the ground-relative reference (core contribution).
- C → D → E: effect of perception.
- C vs C′: how strictly the legs should follow reference joint angles on uneven ground.

**Fairness:** same iteration budget and seed set for every row.

## 5. Budget and schedule

The paper trains each model for 27 h on an RTX 4090; our RTX 5090 is shared, so plan **fixed-budget runs of ~6–10 h**, 2 seeds where possible, prioritising rows B, C and E. Develop with `num_envs` 128–512 and the viewer on; scale up only for final runs.

| Order | Work | Output |
|---|---|---|
| 1 | Step 0 + Step 1 | Terrain in sim; zero-shot failure plot (motivation) |
| 2 | Step 2 + regression test | Ground-relative tracking, verified identical to PBHC on flat ground |
| 3 | Train row C (and B) | First terrain results |
| 4 | Step 3; train rows D, E | Perception results |
| 5 | Step 4 (if time) | Curriculum |
| 6 | Evaluation, ablations, report | Final tables and figures |

## 6. Risks

| Risk | Mitigation |
|---|---|
| Reference jumps when contact switches feet | EMA-smooth `h_sup`; fall back to the previous value when airborne. |
| Robot spawns inside the terrain | 1–2 cm spawn clearance; start from flat/low difficulty. |
| Bugs silently change flat-ground behaviour | Mandatory flat-ground regression test (Step 2.6). |
| Adaptive σ distorted by mixed difficulty | Log σ per term; per-level σ if needed. |
| Not enough compute for all rows/seeds | Prioritise B, C, E; fixed budgets; 2–3 motions only. |

## 7. Positioning against prior work

Survey terrain-conditioned humanoid motion imitation from 2025 before claiming novelty (e.g. VideoMimic, which we believe does terrain-aware imitation from video; verify). Frame our contribution as: **adding terrain awareness to PBHC's adaptive motion-tracking framework for highly-dynamic skills, via ground-relative reference adaptation and height-scan perception**, and state clearly how it differs.

## 8. Implementation status (Steps 0–3 done, 2026-10-27)

Branch: `terrain-aware-tracking`. Everything is **off by default**, so existing configs and checkpoints behave exactly as before.

### What was implemented

| File | Change |
|---|---|
| `humanoidverse/envs/env_utils/terrain.py` | New terrain types `tilt` (ramp through the tile centre, up to `tilt_max_deg`) and `step` (half the tile raised by `step_height_range`, edge within ±15 cm of the centre). Row-based difficulty for both; optional `fixed_difficulty` for evaluation. |
| `humanoidverse/config/terrain/terrain_kungfu.yaml` | New terrain group: trimesh, 4×4 m tiles, 5 difficulty rows × 8 columns, types flat/rough/tilt/step. |
| `humanoidverse/envs/motion_tracking/motion_tracking.py` | `_terrain_height_at` (height lookup); `_ref_offset` + `_apply_ground_relative_ref` (ground-relative reference, applied in `kick_motion_res` and `kick_motion_res_multiplestep`); motion re-centring on the tile; ground-relative feet reward; leg-joint weighting in the joint-position reward; spawn clearance; `height_scan` observation; terrain-consistent trajectory saving for evaluation; terrain diagnostics in the training log; optional viewer drawing of the scan points. |
| `humanoidverse/config/env/motion_tracking.yaml` | New `env.config.terrain_aware` block (all options, defaults = original PBHC). |
| `humanoidverse/config/obs/motion_tracking/main_terrain.yaml` | `main.yaml` + 77-point `height_scan` for actor and critic (actor 457, critic 707 dims). |
| `humanoidverse/config/obs/motion_tracking/main_terrain_priv.yaml` | `main.yaml` + `height_scan` for the critic only (actor 380 = PBHC, critic 707). |
| `scripts/terrain_aware_check.py` | Deterministic regression check (flat ground) and reference-clearance check (terrain). |

### `env.config.terrain_aware` options

| Option | Default | Meaning |
|---|---|---|
| `ground_relative_ref` | `False` | Shift the reference onto the local terrain (Step 2); also enables the ground-relative feet reward. |
| `recenter_motion_xy` | `False` | Start each motion at its tile centre, where the terrain features are. **Use for every terrain run**, including the naive baseline, so all rows stand on the same features. |
| `support_height_tau` | `0.02` | Softness [m] of the soft maximum over supporting feet. |
| `per_foot_ref` | `False` | Row C+: also shift each leg towards the ground under its own foot (ankle 1, knee 0.5, hip and above 0); the support height becomes a lift-weighted mean of the ground under the feet. |
| `support_lift_tau` | `0.03` | `per_foot_ref` only: softness [m] of the stance-foot weighting by reference foot lift. |
| `ref_velocity_correction` | `True` | Add the rate of change of the terrain offsets (finite difference over one control step) to the reference z velocities. |
| `ref_velocity_correction_max` | `1.0` | Clamp [m/s] of that correction: offsets jump when a foot crosses a step edge. |
| `feet_ground_footprint` | `0.04` | Half-length [m] of the footprint over which the ground under a foot is averaged (reference adaptation and feet reward); `0` = point lookup. |
| `spawn_clearance` | `0.0` | Extra root height [m] at reset (0.01 recommended on terrain). |
| `leg_joint_pos_weight` | `1.0` | Weight of hip/knee/ankle joints in the joint-position reward (ablation C′: 0.3). |
| `height_scan.*` | 11×7 grid, 0.1 m | Scan grid in the heading frame; `obs_dims.height_scan` must equal the number of points. |
| `debug_draw_height_scan` | `False` | Draw env 0's scan points in the viewer. |

### Verification

**Flat-ground regression (exact).** `python scripts/terrain_aware_check.py regress example/pretrained_horse_stance_pose/model_50000.pt num_envs=16`: over 60 steps × 16 envs with random actions, every reference tensor, the feet reward and the height scan are **identical (difference 0)** between the original and the terrain-aware code paths.

Closed-loop `sample_eps.py` cannot serve as a regression test: three runs of the *same* code on flat ground gave E_gmpbpe of 33.5, 36.1 and 43.2 mm (29.5 mm before any change). Evaluation is not deterministic, so **report means over many episodes and several seeds**.

**Reference foot clearance on terrain** (`terrain_aware_check.py terrain`, 64 envs × 120 steps; lower reference foot's ankle-link height above the terrain under it; ~0.03–0.05 m on flat ground):

| Terrain | Naive (original offset) mean / floating >9 cm | Ground-relative mean / below ground / floating |
|---|---|---|
| Tilt | 0.111 m / 66% of frames | 0.035 m / 0.0% / 0.1% |
| Step | 0.063 m / 23% | 0.037 m / 0.0% / 0.0% |
| Rough | 0.059 m / 0.1% | 0.034 m / 0.0% / 0.0% |

**Fixes after the viewer check (per-foot reference, velocity correction, footprint smoothing).** The flat-ground regression still gives 19/19 identical with the new defaults, and also with `per_foot_ref=True`. Step terrain at difficulty 1.0, 64 envs × 120 steps; error = foot clearance above the ground under it minus the clearance authored on flat ground:

| Variant | mean \|error\| | feet > 3 cm too high | lower foot below ground | \|v_z − Δz/Δt\| | worst offset jump per step |
|---|---|---|---|---|---|
| Naive | 0.063 m | 63% | 0% | 0.032 m/s | 0 |
| Rigid shift, no velocity correction | 0.022 m | 22% | 3.3% | 0.056 m/s | 0.056 m |
| Rigid shift (row C) | 0.022 m | 22% | 3.3% | 0.038 m/s | 0.056 m |
| Per-foot (row C+) | 0.004 m | 4.9% | 4.0% | 0.035 m/s | 0.044 m |

On tilt (12°) per-foot cuts the mean error from 0.039 to 0.005 m and the share of feet more than 3 cm too high from 40% to 0%. The velocity correction brings the velocity–position mismatch back to the level of the motion data itself (naive). Footprint trade-off, rigid shift on step: point lookup (`0`) gives 0% below ground but 0.100 m offset jumps in a single step; `0.04` gives 0.056 m jumps and the ankle target up to ~4 cm below ground at edges in ~3% of frames; `0.08` gives 0.055 m jumps but 10% of frames below ground. Run-to-run variation of these numbers is a few percentage points (zero-action rollouts reset at random times).

**Training smoke tests** (256 and 128 envs, a few minutes each): the full method (`main_terrain`) and the privileged variant (`main_terrain_priv`) both train without errors; the logged reference foot clearance stays at ~0.036 m.

### Step 1 result: the released policy fails on uneven footing

`ratio_eps.py`, `example/pretrained_horse_stance_pose/model_50000.pt` (Horse-stance pose), 100 envs × 100 episodes, motions re-centred. Values are episode-length ratios.

| Terrain | Difficulty | Naive reference | Ground-relative reference |
|---|---|---|---|
| Flat (trimesh) | – | 98.9% | 99.4% |
| Rough | 0.9 | 22.5% | 22.5% |
| Tilt | 0.5 (6°) | 16.4% | 16.2% |
| Tilt | 1.0 (12°) | 10.9% | 10.6% |
| Step | 0.5 (6.5 cm) | 14.0% | 19.0% |
| Step | 1.0 (10 cm) | 23.5% | 33.6% |

The two reference variants give similar numbers because the policy is blind: only the reference used by the `motion_far` termination changes. So these are genuine falls, not artefacts of the floating reference. Step difficulty 1.0 scoring above 0.5 likely reflects the random edge direction and offset; use more episodes and seeds for final numbers. JSON results: `example/pretrained_horse_stance_pose/metrics/ckpt_50000/zs*_ratio.json` (`zs_` = naive without re-centring, `zsn_` = naive re-centred, `zsg_` = ground-relative).

### How to run

Shared arguments for all training rows (Horse-stance pose shown; swap the motion file for other motions):

```bash
conda activate pbhc && cd ~/ws/PBHC
COMMON="+simulator=isaacgym +exp=motion_tracking +terrain=terrain_kungfu project_name=MotionTracking +opt=wandb \
  +robot=g1/g1_23dof_lock_wrist +domain_rand=main +rewards=motion_tracking/main \
  robot.motion.motion_file=example/motion_data/Horse-stance_pose.pkl seed=1 +device=cuda:0 headless=True \
  num_envs=1024 env.config.terrain_aware.recenter_motion_xy=True env.config.terrain_aware.spawn_clearance=0.01"
GR="env.config.terrain_aware.ground_relative_ref=True"
```

| Row | Command |
|---|---|
| B. PBHC on terrain, naive reference | `python humanoidverse/train_agent.py $COMMON +obs=motion_tracking/main experiment_name=B_naive` |
| C. + ground-relative reference | `python humanoidverse/train_agent.py $COMMON $GR +obs=motion_tracking/main experiment_name=C_ground_rel` |
| C+. C with per-foot reference | `... $GR env.config.terrain_aware.per_foot_ref=True +obs=motion_tracking/main experiment_name=Cplus_per_foot` |
| C′. C with leg weight 0.3 | `... $GR +obs=motion_tracking/main env.config.terrain_aware.leg_joint_pos_weight=0.3 experiment_name=C2_legw03` |
| D. + privileged-critic scan | `python humanoidverse/train_agent.py $COMMON $GR +obs=motion_tracking/main_terrain_priv experiment_name=D_priv_scan` |
| E. + actor scan (full method) | `python humanoidverse/train_agent.py $COMMON $GR +obs=motion_tracking/main_terrain experiment_name=E_full` |

`+opt=wandb` logs to W&B: the `pbhc` conda env sets `WANDB_API_KEY` and `WANDB_ENTITY` (`saketkaki-the-university-of-texas-at-austin`) in `etc/conda/activate.d/wandb.sh`, and `config/opt/wandb.yaml` reads the entity from `WANDB_ENTITY`; drop `+opt=wandb` to train without W&B.

`num_envs` depends on free GPU memory: each run at 2048 envs uses ~7.3 GB and runs at ~2.5–2.8 s per iteration (~17–20k steps/s) on the RTX 5090, so a 2 h pilot is ~2,500–2,900 iterations (the released checkpoint has 50,000). Pilots B and C were started on 2026-10-27 with `num_envs=2048 algo.config.save_interval=250` under `timeout --signal=INT 2h`. Evaluate a trained checkpoint on one terrain type and difficulty (checkpoints trained with the flags above already carry them in their `config.yaml`):

```bash
python humanoidverse/ratio_eps.py +device=cuda:0 +checkpoint=<run>/model_<N>.pt +opt=record \
  +num_envs=100 +num_episodes=100 +eps_eval_name=tilt_d10 +terrain=terrain_kungfu \
  "terrain.terrain_types=[tilt]" "terrain.terrain_proportions=[1.0]" terrain.fixed_difficulty=1.0
```

To evaluate the released checkpoint (which predates these options), add `+env.config.terrain_aware.recenter_motion_xy=True` and, for the ground-relative variant, `+env.config.terrain_aware.ground_relative_ref=True`.

**Videos.** `scripts/record_video.py record CKPT --out clip.mp4 [--seconds 10] [--label ...] [--wandb-run ENTITY/PROJECT/RUN_ID] [overrides]` renders one frame per control step (real-time 50 fps, unlike the viewer's own R-key recorder, which drops frames), uses a fixed camera on the start position, prints when and why each episode ended, and burns the fall count into the video. Terrain and tile assignment are seeded (`--seed`, default 0), so different checkpoints are filmed on the same tile and step edge. `record_video.py stack OUT.mp4 IN1.mp4 IN2.mp4 ...` puts clips side by side. `scripts/record_pilot_videos.sh B_RUN_DIR C_RUN_DIR` records A (released), B and C (latest checkpoints) on step 1.0/0.5, tilt 1.0, rough 0.9 and flat, attaches each clip to its W&B run (`videos/<terrain>`) and the A | B | C comparisons to both runs (`videos/compare_<terrain>`). Videos need a display (the viewer window opens).

### Pilot results (B and C, 2 h each, 2026-10-27)

Horse-stance pose, 2048 envs, seed 1, trained from scratch for ~3,000 iterations (the released checkpoint has 50,000). W&B: `saketkaki-the-university-of-texas-at-austin/MotionTracking`, runs `2zsv2h25` (B) and `vn3zl3ac` (C); videos under `videos/*`. Episode-length ratio at iteration 3000 (`ratio_eps.py`, 100 envs × 100 episodes; A = released checkpoint zero-shot, ground-relative):

| Terrain | A released | B (naive) | C (ground-relative) |
|---|---|---|---|
| Flat | 99.4% | 31.0% | 33.4% |
| Rough 0.9 | 22.5% | 31.4% | 30.5% |
| Tilt 6° | 16.2% | 32.7% | 30.1% |
| Tilt 12° | 10.6% | 30.5% | 24.2% |
| Step 6.5 cm | 19.0% | 28.3% | 25.0% |
| Step 10 cm | 33.6% | 22.6% | 22.5% |

**The pilots are too short to compare B and C.** Both score ~30% *even on flat ground*. `ratio_eps.py` starts every episode at the motion start and ends it when any body is more than 0.7 m from its reference (falls are not checked). From the start, the 3,000-iteration policies lose track at ~1.5–2.5 s (median 1.8 s; pelvis xy error ~0.3 m, the head target passes 0.7 m), on every terrain. Training starts episodes at random motion phases, which hides this in the training curves (`end_time_ratio` ~0.82). Separately, the training motion-far threshold curriculum had only reached 0.41 m (it goes down to 0.3 m), and falls during training had dropped to ~0.

**Next:** fine-tune B and C from the released checkpoint (`+checkpoint=example/pretrained_horse_stance_pose/model_50000.pt`; same observation sizes), so that both start from a policy that already tracks the whole motion on flat ground and only terrain adaptation is compared; or train from scratch for much longer (≥ 6–10 h per row).

### Fine-tuning from the released checkpoint (B, C, E; started 2026-10-27 21:56)

The pilots showed that 3,000 iterations from scratch do not learn the motion itself, so rows are now fine-tuned from the released checkpoint (50,000 iterations on flat ground) and only terrain adaptation is compared.

New code:

| Where | What |
|---|---|
| `motion_tracking.py` `_update_terrain_curriculum` | Step 4. With `terrain.curriculum=True`: a robot that reaches the motion end (after covering at least half the motion) moves one difficulty row up; one whose episode ends before half the remaining motion moves one row down; solving the top row sends it to a random row. Logged as `terrain_level`. |
| `mh_ppo.py` `actor_warmup_iterations` | Critic-only updates for the first N iterations of a run (the flat-ground value function adapts to terrain before it drives the actor). The adaptive learning-rate schedule is skipped during the warm-up: with a frozen actor the KL is 0 and would push both learning rates to 1e-2. |
| `scripts/expand_checkpoint.py` | Warm start for a larger observation set: rebuilds the first actor and critic layer for the new layout (terms are sorted by name, so `height_scan` lands in the middle: actor columns 72–148, critic 229–305) and gives new terms zero weights. Verified: with the same seed, the actor observations of the two layouts agree exactly outside the new columns. |
| `scripts/terrain_eval.py` | Evaluation from the motion start with three outcomes: completed, fall (gravity termination), drift (any body > 0.7 m from its reference). Reports rates, episode-length ratio, and mean body error relative to the root (like E_mpbpe), global body error and joint error. A falling robot usually crosses 0.7 m before the gravity check, so "drift" includes many falls. |
| `scripts/eval_terrain_matrix.sh`, `scripts/summarize_terrain_eval.py` | All six test terrains × seeds 0, 1 (200 episodes each), and a mean ± std markdown table. |

Fine-tuning arguments (added to the training command; `+opt=wandb` included):

```bash
FT="... num_envs=1536 env.config.terrain_aware.recenter_motion_xy=True env.config.terrain_aware.spawn_clearance=0.01 \
  algo.config.load_optimizer=False algo.config.actor_learning_rate=3e-4 algo.config.critic_learning_rate=3e-4 \
  +algo.config.actor_warmup_iterations=150 env.config.termination_curriculum.terminate_when_motion_far_initial_threshold=0.7 \
  terrain.curriculum=True terrain.max_init_terrain_level=1 algo.config.save_interval=250"
# B: +obs=motion_tracking/main checkpoint=example/pretrained_horse_stance_pose/model_50000.pt
# C: B + env.config.terrain_aware.ground_relative_ref=True
# E: +obs=motion_tracking/main_terrain env.config.terrain_aware.ground_relative_ref=True checkpoint=logs/warmstart/main_terrain_from_50000.pt
#    (made with: python scripts/expand_checkpoint.py example/pretrained_horse_stance_pose/model_50000.pt \
#                humanoidverse/config/obs/motion_tracking/main_terrain.yaml logs/warmstart/main_terrain_from_50000.pt)
```

Baseline A (released checkpoint, zero-shot, ground-relative reference; `eval_terrain_matrix.sh`, seeds 0 and 1, 200 episodes each):

| | flat | rough 0.9 | tilt 6° | tilt 12° | step 6.5 cm | step 10 cm |
|---|---|---|---|---|---|---|
| Completed | 99.0% | 0% | 0% | 0% | 0% | 0% |
| Fall (gravity) | 0.8% | 20.7% | 18.2% | 11.2% | 5.5% | 7.0% |
| Episode-length ratio | 99.5% | 20.2% | 16.1% | 11.2% | 12.3% | 12.8% |
| Root-relative body error | 20.4 mm | 51 mm | 57 mm | 61 mm | 58 mm | 54 mm |

Runs (3 h each, all three in parallel, ~6.7 GB GPU memory and ~2.7–3.3 s per iteration each; iteration numbers continue from 50,000): W&B `ft_B_naive` = `pdle14pn`, `ft_C_ground_rel` = `6k6a8guf`, `ft_E_full` = `zts2025b`.

### Not done yet

- Rows C′, C+ and D; final evaluation matrix of the fine-tuned B, C and E.
- Teacher–student distillation (stretch goal).
