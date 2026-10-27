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
2. **Per-frame support height.** Look up terrain heights under the reference feet (`rg_pos_t[:, feet_ids, :2]`). Weight them by the reference contact mask: `h_sup = Σ cᵢ·hᵢ / Σ cᵢ`, falling back to the previous value or the mean when both feet are airborne. Smooth `h_sup` with a short EMA so the reference doesn't jump when contact switches feet.
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
