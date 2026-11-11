# Terrain-Aware Motion Tracking for PBHC

Final project: extend PBHC (KungfuBot, NeurIPS 2025) so a humanoid can perform highly dynamic motions on non-flat ground.

**Status (2026-11-10):**
- The Horse-stance pose policy now handles tilts, bumps, held-out rough ground and steps, while matching the released policy on flat ground.
- Steps are the weakest terrain, but falls are rare: the usual failure is losing the motion in the opening step.
- A second motion (Horse-stance punch, one seed) now matches the pose on every terrain tested (10 cm steps 69%, 18° tilt 89–91%, 8 cm bumps 83–90%, falls below 1%). It needed three more changes: a reference that clears step edges, a curriculum measured from the motion start, and far more varied step layouts in training.

Every run, with its settings, results and problems, is in [LOG.md](LOG.md). The original plan from 2026-10-27 is kept in the appendix.

## 1. Motivation

The PBHC paper states its own limitation (Section 6):

> (i) It lacks environment awareness, such as terrain perception and obstacle avoidance, which restricts deployment in unstructured real-world settings.

PBHC trains and evaluates only on a flat plane. Its tracking objective also assumes flat ground: reference motions are placed at one fixed height per environment, and the rewards compare body positions in world coordinates. On a slope or a step, the reference floats above the ground on one side and sinks into it on the other, so the policy is rewarded for the wrong behaviour.

The example motions travel only 0.2–1.0 m, so the task is **performing a motion on uneven footing**, not walking across terrain: the terrain feature has to sit under the robot's feet.

## 2. Contributions compared with the original PBHC

### The original PBHC (as released)

- **Reference placement:** the reference motion sits at one fixed height per environment, and rewards compare body positions in world coordinates.
- **Actor inputs:** proprioception with a short history, plus the motion phase. The actor is blind to the terrain and to where the reference is; only the critic sees the reference body positions.
- **Training:**
  - adaptive tracking factor σ, from a running average of the tracking error;
  - a multi-head critic, one head per reward term;
  - a drift-termination threshold that shrinks towards 0.3 m;
  - one policy per motion, trained from scratch on flat ground.
- **Evaluation:** tracking error on flat ground.

Everything below is off by default. With it off, the code reproduces the original exactly on flat ground (checked tensor by tensor, Section 6.4).

### C1. Terrain suite and curriculum

- **New terrain types:** tilts, single steps through the tile centre, and smooth bumps, alongside legged_gym's rough ground. They are graded by difficulty row, with a fixed-difficulty mode for evaluation that extrapolates past the training range (steps to 17 cm, tilts to 24°, bumps to 8 cm).
- **Placement:** each motion is recentred on its tile, so the feature is under the feet.
- **Curriculum:**
  - Robots move up a row after completing the motion and down after an early failure.
  - Graduates stay in the hardest rows.
  - The top row is set per terrain type (steps go to 17 cm, tilt to 12°, bumps to 4 cm).
  - Each robot's level is saved in the checkpoint and restored when training continues.

### C2. Ground-relative reference

The reference is shifted onto the local ground every frame:
- The shift is a soft maximum of the ground height under the supporting feet.
- Ground heights are averaged over each foot's footprint, so the shift changes smoothly at step edges.
- The reference's vertical velocity is corrected for the changing shift.
- The feet reward compares heights above the ground under each foot.

This makes the tracking objective correct on terrain. On its own it mostly turns falls into drift terminations (round 1: falls 23–36% → 1–7%), because a blind policy still cannot place its feet.

Three refinements were built and tested but not adopted: per-foot leg shifts, leg inverse kinematics, and lifting a swinging foot over a nearby riser (Section 4).

### C3. Terrain perception

- **Actor:** an 11 × 11 height scan out to ±0.5 m in the heading frame, and a 2 cm scan under each foot that moves with the foot.
- **Critic only:** an exact 15 × 15 scan.
- **Exact terrain height:** legged_gym's lookup takes the minimum of three neighbouring height-field cells. It reads 1–3 cm low on tilts and misplaces step edges by up to 10 cm. It was replaced by a lookup in the rasterised trimesh, used for the scans and the reference.
- **Noise model:** per-point noise, per-episode bias and map shift, dropout and delay, for robustness tests.
- **Evidence that the policy uses the scan:** with the scan replaced by flat ground, the same weights track 18% of the motion on the 12° tilt instead of 61% (blind-scan ablation).

### C4. Reference preview: the biggest gain

A new actor and critic input gives the terrain-shifted reference pelvis and feet at 0, 0.1, …, 0.5 s ahead, relative to the robot's own root in its heading frame, plus the reference heading relative to the robot's (66 numbers).

- **Gains against the same training continued without it:**

  | Terrain | Without preview | With preview |
  |---|---|---|
  | 10 cm steps | 9% | 46% |
  | Rough | 18% | 52% |
  | 8 cm bumps | 18% | 35% |
  | Flat error | 26 mm | 22 mm |

  These held across three training seeds.
- **Why it works:** the original actor cannot see that it has drifted from the reference, and on terrain that drift is what ends episodes. The bottleneck was this, not terrain perception.
- **Negative results:**
  - Measuring the preview relative to the reference's own pose (no localisation needed) gives no gain with one motion, because it repeats the motion phase.
  - Extending the preview to 1.0 s made steps and tilts worse in all three runs that used it.

### C5. Fine-tuning recipe for terrain

- **Warm starts:** new observation inputs start at zero weight, and the Adam state is expanded to match (`scripts/expand_checkpoint.py`), so a run starts exactly from the released behaviour.
- **Gentle fine-tuning:**
  - critic-only warm-up, then a ramped, fixed actor learning rate of 2e-5;
  - critic learning rate 1e-4;
  - noise std frozen at the released value, and no entropy bonus.

  The first recipe (3e-4, fresh optimizer, entropy on) took flat-ground completion from 99% to 0% within 100 updates.
- **Falls cost more than drifting:** a drift termination costs a quarter of a fall.
- **Fixed drift cutoff:** the training cutoff stays at 0.7 m, the evaluation value. With PBHC's shrinking cutoff (0.3 m) episodes end before any fall can happen, so the fall penalty never applies.
- **σ frozen:** a single running average is distorted by mixed terrain difficulty.

### C6. Evaluation

- **Difficulty sweep:** 17 settings × 3 seeds × 200 episodes, from the motion start, every env contributing equally. The earlier counting over-weighted early failures and made blind policies look worse than they were.
- **Outcomes:** completed, fall or drift, plus tracking errors.
- **Step diagnostics:**
  - where the raised side is;
  - distance from the feet to the edge;
  - which feet start on the step;
  - when failures happen;
  - shin contact;
  - whether the robot's feet end on the same level as the reference's.
- **Comparison videos** with fixed cameras and scan markers, uploaded to W&B.

## 3. Results

Completion rate (%), from the motion start, 3 evaluation seeds × 200 episodes; multi-seed rows are means over training seeds.

| Model | Flat | Rough (held out) | Tilt 12° | Tilt 18° | Bumps 4 cm | Bumps 8 cm | Step 6.5 cm | Step 10 cm | Step 13.5 cm | Step 17 cm | Flat error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Released PBHC (zero-shot) | 98.8 | 0.0 | 0.0 | 0.0 | 0.5 | 0.0 | 7.3 | 21.0 | 23.0 | 23.2 | 20.2 mm |
| Ground-relative reference only (C, round 1) | 64.7 | 23.3 | 1.0 | 0.0 | 32.0 | 4.7 | 14.3 | 18.8 | 13.8 | 17.0 | 45.4 mm |
| + height scan (E1, round 1) | 91.5 | 21.8 | 28.3 | 2.2 | 33.7 | 2.3 | 27.2 | 11.7 | 5.7 | 3.0 | 45.0 mm |
| + gentle fine-tuning, continued (E2c) | 96.8 | 21.7 | 56.8 | 8.3 | 54.3 | 5.7 | 26.0 | 3.8 | 0.8 | 0.3 | 31.5 mm |
| + fall penalty, 0.7 m cutoff (F2, 2 seeds) | 95.3 | 20.2 | 63.6 | 10.5 | 58.4 | 8.2 | 24.6 | 8.9 | 2.8 | 1.7 | 32.2 mm |
| + perception v2 (F3, 2 seeds) | 97.3 | 18.9 | 82.7 | 31.8 | 71.9 | 15.1 | 34.2 | 9.3 | 4.4 | 1.6 | 28.2 mm |
| + reference preview (3 seeds) | 99.2 | 51.1 | 85.7 | 36.2 | 88.6 | 35.8 | 54.9 | 38.1 | 23.2 | 13.4 | 22.2 mm |
| + 3,250 more iterations (2 seeds) | **99.9** | 55.5 | 91.7 | 37.2 | 90.6 | 36.2 | 59.7 | 45.1 | 29.2 | 16.8 | **20.4 mm** |
| + step curriculum to 17 cm (2 seeds) | 99.6 | 55.0 | 89.6 | 44.8 | 92.7 | 42.3 | 62.3 | 50.7 | 36.2 | 25.6 | 20.6 mm |
| + tilt curriculum to 21°, bumps to 8 cm (2 seeds, 1,000 iterations) | 99.7 | 71.6 | **96.2** | 73.3 | **95.9** | **67.6** | 70.2 | 58.1 | 44.8 | 32.7 | 22.0 mm |
| + varied layouts: 24 terrain columns, placement jitter (seed 2, 1,000 iterations) | 98.8 | 75.3 | 94.0 | 80.3 | 95.3 | 67.5 | 74.5 | 60.3 | 53.5 | 43.0 | 22.9 mm |
| **+ curriculum level from early starts only (seed 2, 1,000 iterations, current best recipe)** | 99.7 | **80.5** | **98.3** | **85.8** | **96.7** | **82.0** | **79.7** | **64.5** | **56.5** | **49.8** | 23.1 mm |

Fall rates of the current best: 0% on flat, rough, bumps up to 8 cm and tilts up to 18°. On steps it falls 0.2, 0.8, 1.5 and 3.5% at 6.5, 10, 13.5 and 17 cm. The last two rows are one training seed (seed 2); the two-seed row above them reads 99.7 / 64.2 / 96.3 / 71.7 / 96.0 / 65.3 / 68.0 / 59.8 / 48.0 / 35.3 for that seed. A further 1,500 iterations of the same recipe (`ft33`, quick eval only) read 69.0% on 10 cm steps, 55.2% on 17 cm steps, 90.7% on 18° tilt, 82.5% on 8 cm bumps and 21.5 mm flat error.

- **Flat error** is the root-relative body position error on flat ground, as PBHC's E_mpbpe. The last three rows are 2–3 mm worse than the released policy; the rows before them match it.
- **The B and C rows** come from the first fine-tuning recipe, which damaged the released policy (Section 4). They show what the reference change does on its own, not a tuned baseline.
- **Released policy on steps:** its completion rises with step height, which a blind policy shouldn't do. The cause hasn't been checked, so treat that row with care.
- **Training seeds** differ by up to about 15 points on 10 cm steps, so differences of a few points between rows are not meaningful.

**How far the terrain can go** (current best, at least 50% of motions completed):

| Terrain | Holds | Breaks |
|---|---|---|
| Tilt | 18° (85.8%) | 24° (43.3%) |
| Bumps | 8 cm (82.0%), the hardest tested | – |
| Steps | 13.5 cm (56.5%) | 17 cm (49.8%; 55.2% after 1,500 more iterations) |

**Second motion (Horse-stance punch).** Completion rate (%) / falls (%), one training seed. Both checkpoints are evaluated with the motion recentred on its tile; the last column has only the quick eval (5 settings):

| Terrain | Released punch | First recipe (`ft18`, 6,500 it) | Current best (`ft32`, 18,250 it in total) |
|---|---|---|---|
| Flat | 99.2 / 0.0 | 99.3 / 0.0 | 100.0 / 0.0 |
| Tilt 18° | 0.0 / 5.2 | 12.8 / 4.5 | **88.8** / 0.0 |
| Bumps 8 cm | 0.0 / 9.2 | 37.2 / 1.3 | **90.3** / 0.0 |
| Steps 10 cm | 18.8 / 14.7 | 9.8 / 6.2 | **69.2** / 0.3 |
| Steps 17 cm | 20.3 / 11.5 | 5.8 / 15.3 | **53.8** / 0.8 |
| Flat error | **17.3 mm** | 25.4 mm | 25.2 mm |

What the punch needed beyond the pose recipe (details in LOG.md, 2026-11-10):
- **A reference that clears step edges.** The punch's opening side-step lifts the foot only 6 cm (the pose: 23 cm), so on a taller step the reference foot passed through the riser, and the swinging foot was taken as the supporting foot mid-step. `swing_clearance=True` and `support_lift_gain=4` fix both.
- **A curriculum measured from the motion start.** Training starts episodes anywhere in the motion, and only the opening is hard on steps. `early_start` (70% of episodes on terrain tiles start in the first 0.8 s) with `levels_only=True` (only those episodes move the level).
- **Varied layouts.** A step tile has one edge, and a run had 2–3 step tiles per level, each always met at the same place. `terrain.num_cols=24` and `placement_jitter=0.15` took 10 cm steps from 24 to 46% in 750 iterations.
- **A trap with this motion:** the motion file starts 1.17 m from the origin. Without `recenter_motion_xy` the robot stands away from the step edge, and the released punch then seems to complete about 80% of steps.
- **Open:** flat error is 8 mm worse than the released punch.

## 4. What works and what does not

| | Change | Evidence (details in LOG.md) |
|---|---|---|
| **Works** | Height scan (C3) | Only the scan policies complete motions on tilts; blinding the scan cuts the share of the 12° tilt motion tracked from 61 to 18% |
| | Ground-relative reference (C2) | Required for a correct objective; on its own it removes falls but not drift |
| | Gentle fine-tuning (C5) | Fixed the round-1 collapse; flat error 45 → 31 mm, then to 20 mm with later changes |
| | Fall penalty + 0.7 m training cutoff | Falls at most 7% for F2s1 everywhere, held-out rough 33 → 1% |
| | Perception v2: wide scan, foot scans, exact lookup | Tilt 12° 64 → 83%, tilt 18° 11 → 32%, bumps better. A control with the same training but old observations got worse on tilt and bumps |
| | Robot-frame reference preview (C4) | The largest gain on every terrain, across 3 seeds |
| | Step curriculum to 17 cm | Two-seed mean +6 to +9 points at every step height ≥ 10 cm (seed 4 +8 to +14; seed 2 mainly at 17 cm, +9.5); bumps 8 cm and tilt 18° up 6–8 points |
| | Tilt curriculum to 21°, bumps to 8 cm | Better on all 16 terrain settings for both seeds after 1,000 iterations: tilt 18° 45 → 73%, bumps 8 cm 42 → 68%, rough 55 → 72%, steps +6 to +9 points; flat error +1.4 mm |
| | Varied layouts: 24 terrain columns + placement jitter | Punch: 10 cm steps 24 → 46% in 750 iterations. Pose: rough +11, tilt 18° +9, steps ≥ 13.5 cm +5 to +8, mid steps unchanged |
| | Curriculum level from early starts only | The level stops being inflated by episodes that skip the opening. Punch: 10 cm steps 19.5 → 24% in 750 iterations. Pose: best or level on all 17 settings after 1,000 iterations (bumps 8 cm +14.5, tilt 24° +13, 17 cm steps +7) |
| | Swing clearance + support-lift gain (punch) | Needed for a motion whose step lifts the foot less than the step height; tilt 18° 35 → 51%, steps only with the two rows above |
| | Fair evaluation | Fixed over-counting of early failures (A on 10 cm steps 13% → 37% episode length) |
| **Does not work** | Higher learning rates (actor 1e-4, critic 3e-4; punch) | Worse on every terrain than the gentle rates from the same checkpoint (10 cm steps 11.8 against 19.5%) |
| | Per-foot reference | Fewer falls, but much worse on tilts (57% vs 82% on 6°) |
| | Leg-IK reference | Swapping it in or out at evaluation changes results by 1–4 points; its run's tilt gain came from training variance |
| | Swing clearance | No better than the plain continuation; worse on tilt |
| | min3 lookup for the reference | Worse everywhere |
| | Fall penalty with the 0.3 m cutoff | Never fired: training had no falls |
| | Preview in the reference frame | No gain with a single motion |
| | Preview to 1.0 s | Steps and tilt worse in all 3 runs |
| | Early starts on step tiles, with all starts moving the level | Even without curriculum demotion, no gain from the motion start (they work once only early starts move the level, see above) |
| | Shin-contact penalty (idea) | Shin contact is as common in completed episodes as in failed ones; dropped before building |
| | Training longer | Helps a run whose curriculum is on hard tiles (s2: 10 cm steps 46 → 54%), not one stuck on easy tiles (s4) |
| | Scan noise | Policies tolerate noise, dropout, map shift and delay; a 5 cm height offset costs 8–15 points |

## 5. Open problems and next steps

**Steps are the weakest terrain.**
- **Training saw too few layouts** (found 2026-11-10): one edge per step tile, 2–3 step tiles per level, every episode at the tile centre. This is the likely cause of each training seed having its own weak step direction; the direction numbers below are from before the varied-layout runs.
- **Where they fail (current best, seed 2):** in the 1–2.5 s weight shift and big step of the horse stance, and almost always by losing the motion, not by falling (10 cm steps: 34.7% drift, 0.8% falls).
  - Worst with the raised side on the robot's left (48% not completed at 10 cm, against 24% on the right) and when one reference foot starts on the upper level (48%, against 17% with none).
  - Before the varied-layout runs the main failure was a fall with the step up directly ahead (10–25% falls at 10 cm); that is now 3%.
- **Starting layout:** about half the evaluation starts straddle the edge, and those complete far less often.
- **Small steps are ramps:** the trimesh conversion turns height changes under 7.5 cm per 10 cm cell into ramps (`slope_treshold: 0.75`), so the 6.5 cm step and the lowest curriculum rows are not walls.

**Next steps, in order:**
1. **Foothold scan:** a small height patch around each reference foot 0.3–0.5 s ahead. The big step lands about 50 cm out, at the edge of the body scan, and the foot scans only cover where the feet are now.
2. **Move reference landing points away from step edges** horizontally (not only in height, which every reference variant so far changed).
3. **Report start layouts separately** (clean starts vs straddling the edge), and decide whether training should include straddle starts.
4. **Make small steps real walls** (`slope_treshold` ≈ 0.5). (The tilt and bump ceilings were raised on 2026-11-10.)
5. **Preview robustness:** add noise and drift to the robot position estimate the preview uses, since deployment needs localisation.
6. **More motions:**
   - the motions without a released checkpoint, starting with the Hooks punch (feet barely move), then the side and roundhouse kicks (long single-leg phases), the Charleston dance (20 s) and the Bruce Lee pose (deep crouch, 1 m of travel);
   - one terrain-aware policy for several motions, where the reference-frame preview could carry the motion content without localisation.

Seeds: with the old terrain (2–3 step layouts per level) training seeds differed by up to about 15 points on steps. Since 2026-11-10 changes are tested on seed 2 only; confirm the final recipe on a second seed before quoting small differences.

**Limitations compared with the paper:**
- The preview needs an estimate of the robot's position relative to the motion start, and the height scan needs an elevation map. The original policy needs neither.
- Everything is in simulation. The punch is the only second motion so far, and the current best pose and punch are one training seed each.

## 6. Implementation reference

Branch `terrain-aware-tracking`. Every option defaults to the original PBHC behaviour, so existing configs and checkpoints behave as before.

### 6.1 Files

| File | Change |
|---|---|
| `humanoidverse/envs/env_utils/terrain.py` | Terrain types `tilt`, `step`, `bumps` through the tile centre; row difficulty with a per-type ceiling; `fixed_difficulty` for evaluation |
| `humanoidverse/config/terrain/terrain_kungfu.yaml` | Terrain group: trimesh, 4 × 4 m tiles, 5 rows × 8 columns; flat / bumps / tilt / step = 0.1 / 0.3 / 0.3 / 0.3 |
| `humanoidverse/envs/motion_tracking/motion_tracking.py` | All of the following: <br>• height lookups (min3 and exact raster) and motion recentring; <br>• the ground-relative reference (rigid, per-foot, leg IK, swing clearance); <br>• the observations `height_scan`, `height_scan_noisy`, `foot_scan`, `privileged_height_scan` and `ref_preview`, with the scan noise model; <br>• terrain curriculum, early starts, the fall/drift penalty split, and terrain diagnostics in the log; <br>• terrain-consistent trajectory saving |
| `humanoidverse/agents/mh_ppo/mh_ppo.py` | Critic-only warm-up, actor learning-rate ramp, fixed noise std, configured learning rates over checkpointed ones; terrain levels saved in and restored from checkpoints |
| `humanoidverse/config/env/motion_tracking.yaml` | The `env.config.terrain_aware` block |
| `humanoidverse/config/obs/motion_tracking/main_terrain*.yaml` | Observation sets (Section 6.3) |
| `scripts/expand_checkpoint.py` | Warm start for a larger observation set: new inputs at zero weight, Adam state expanded, scan grid and preview times remapped |
| `scripts/terrain_eval.py`, `eval_difficulty_sweep.sh`, `summarize_difficulty_sweep.py` | Evaluation (completed / fall / drift, errors, step and drift diagnostics, `--start-time`), the 17-setting sweep and its summary tables |
| `scripts/eval_terrain_matrix.sh`, `summarize_terrain_eval.py` | The earlier 6–7-terrain evaluation |
| `scripts/eval_scan_noise.sh`, `summarize_scan_noise.py` | Scan-noise and blind-scan sweeps |
| `scripts/record_video.py`, `record_pilot_videos.sh` | Real-time clips judged by the evaluation's termination rules, side-by-side comparisons, W&B upload |
| `scripts/terrain_aware_check.py` | Flat-ground regression, reference placement, scan noise, lookup and swing checks |

### 6.2 Options

`env.config.terrain_aware`:

| Option | Default | Meaning |
|---|---|---|
| `ground_relative_ref` | `False` | Shift the reference onto the local terrain; also enables the ground-relative feet reward |
| `recenter_motion_xy` | `False` | Start each motion at its tile centre. **Use for every terrain run** |
| `support_height_tau` | `0.02` | Softness [m] of the soft maximum over supporting feet |
| `per_foot_ref`, `support_lift_tau` | `False`, `0.03` | Also shift each leg towards the ground under its own foot (ankle 1, knee 0.5) |
| `leg_ik_ref`, `leg_ik_support_ratio`, `leg_ik_iterations` | `False`, `0.0`, `6` | Re-solve the reference legs by IK so each foot stands on its own ground |
| `swing_clearance`, `swing_clearance_radius`, `swing_clearance_lift` | `False`, `0.15`, `0.02` | Raise a lifted reference foot over higher ground within the radius |
| `ref_velocity_correction`, `ref_velocity_correction_max` | `True`, `1.0` | Add the rate of change of the terrain offsets to the reference z velocities, clamped [m/s] |
| `feet_ground_footprint` | `0.04` | Half-length [m] of the footprint the ground under a foot is averaged over |
| `spawn_clearance` | `0.0` | Extra root height [m] at reset (0.01 on terrain) |
| `leg_joint_pos_weight` | `1.0` | Weight of hip/knee/ankle joints in the joint-position reward |
| `height_lookup`, `ref_height_lookup`, `exact_height_resolution` | `min3`, `null`, `0.01` | Terrain height for the scans and for the reference: `min3` (legged_gym) or `exact` (rasterised trimesh). **Use `exact` for both** |
| `height_scan.*` | x ±0.5, y ±0.3, 0.1 m | Body scan grid in the heading frame (use y ±0.5 with `main_terrain_v2*`) |
| `foot_scan.*` | 13 × 7 at 0.02 m | Grid under each foot |
| `privileged_height_scan.*` | 15 × 15 at 0.1 m | Critic-only scan (exact, no noise) |
| `ref_preview.times`, `ref_preview.frame` | 0–0.5 s, `robot` | Preview times; positions relative to the robot (`robot`) or to the current reference (`reference`) |
| `early_start.*` | `prob 0` | Start step-tile episodes in `[0, max_time]` with probability `prob`; `keep_level_on_failure` skips demotion |
| `height_scan_noise.*` | all 0 | Per-point noise, episode bias, xy drift, dropout, delay, `blind` ablations, `perturb_height_scan` |
| `debug_draw_height_scan` | `False` | Draw env 0's scan points in the viewer |

Other options:

| Option | Default | Meaning |
|---|---|---|
| `terrain.fixed_difficulty` | `null` | Evaluate a single difficulty (> 1 extrapolates) |
| `terrain.curriculum_max_difficulty` | `1.0` | Difficulty of the top curriculum row (rows span max/num_rows … max) |
| `terrain.curriculum_max_difficulty_by_type` | `null` | Per-type override, e.g. `{step: 2.0}` (rows 5.8–17 cm) |
| `terrain.curriculum_graduate_rows` | `null` | Robots that solve the top row go to one of this many hardest rows |
| `terrain.tilt_max_deg`, `step_height_range`, `bumps_max_height`, `bumps_spacing` | 12°, [0.03, 0.10] m, 0.04 m, 0.3 m | Terrain scales at difficulty 0 and 1 |
| `env.config.termination_scales.termination_drift_penalty_scale` | `1.0` | Termination penalty of a drift ending relative to a fall |
| `algo.config.actor_warmup_iterations` | `0` | Critic-only updates at the start of a run |
| `algo.config.actor_lr_ramp_iterations` | `0` | Linear actor learning-rate ramp after the warm-up |
| `algo.config.fix_noise_std` | `False` | Freeze the action noise std at the checkpoint's value |
| `algo.config.load_optimizer_lr` | `True` | `False` uses the configured learning rates instead of the checkpoint's |
| `algo.config.restore_terrain_levels` | `True` | Restore the terrain levels saved in the checkpoint |

### 6.3 Observation sets

| Config | Actor adds | Critic adds | Actor / critic inputs |
|---|---|---|---|
| `main` (PBHC) | – | – | 380 / 630 |
| `main_terrain_priv` | – | `height_scan` | 380 / 707 |
| `main_terrain` | `height_scan` (77) | `height_scan` | 457 / 707 |
| `main_terrain_noisy` | `height_scan_noisy` (77) | `height_scan` | 457 / 707 |
| `main_terrain_v2` | `height_scan` (121), `foot_scan` (182) | + `privileged_height_scan` (225) | 683 / 1,158 |
| `main_terrain_v2_preview` | + `ref_preview` (66) | + `ref_preview` | **749 / 1,224** |
| `main_terrain_v2_preview_long` | + `ref_preview` (88, to 1.0 s) | + `ref_preview` | 771 / 1,246 |

Terms are concatenated in sorted name order, which `expand_checkpoint.py` accounts for.

### 6.4 Verification

**Flat-ground regression (exact).** `python scripts/terrain_aware_check.py regress example/pretrained_horse_stance_pose/model_50000.pt num_envs=16`: over 60 steps × 16 envs with random actions, every reference tensor, the feet reward and the height scan are **identical (difference 0)** between the original and the terrain-aware code paths. This was re-checked after the per-foot reference, the leg IK and the scan noise were added.

Closed-loop evaluation is not deterministic (three runs of the same code on flat ground gave E_gmpbpe of 33.5, 36.1 and 43.2 mm), so results are reported as means over many episodes and seeds.

**Reference foot clearance on terrain** (`terrain_aware_check.py terrain`, 64 envs × 120 steps; lower reference foot's ankle-link height above the terrain under it; about 0.03–0.05 m on flat ground):

| Terrain | Naive (original offset): mean / floating > 9 cm | Ground-relative: mean / below ground / floating |
|---|---|---|
| Tilt | 0.111 m / 66% of frames | 0.035 m / 0.0% / 0.1% |
| Step | 0.063 m / 23% | 0.037 m / 0.0% / 0.0% |
| Rough | 0.059 m / 0.1% | 0.034 m / 0.0% / 0.0% |

Step terrain at difficulty 1.0; error = foot clearance above the ground under it minus the clearance authored on flat ground:

| Variant | Mean \|error\| | Feet > 3 cm too high | Lower foot below ground | \|v_z − Δz/Δt\| | Worst offset jump per step |
|---|---|---|---|---|---|
| Naive | 0.063 m | 63% | 0% | 0.032 m/s | 0 |
| Rigid shift, no velocity correction | 0.022 m | 22% | 3.3% | 0.056 m/s | 0.056 m |
| Rigid shift (default) | 0.022 m | 22% | 3.3% | 0.038 m/s | 0.056 m |
| Per-foot | 0.004 m | 4.9% | 4.0% | 0.035 m/s | 0.044 m |
| Leg IK | joint angles and foot positions agree to 0.01 cm | | | | |

**Exact height lookup** (`terrain_aware_check.py lookup`): the share of weight-bearing feet more than 2 cm from the looked-up ground drops from 20% to 5% on the 12° tilt, and from 16% to 3% on bumps.

## 7. How to run

```bash
conda activate pbhc && cd ~/ws/PBHC
```

**Current best run as a preset.** `humanoidverse/config/exp/terrain_tracking.yaml` holds every setting of the current best pose run (`ft31_E_honest`), so the command is:

```bash
python humanoidverse/train_agent.py +simulator=isaacgym +exp=terrain_tracking +opt=wandb +device=cuda:0 \
  robot.motion.motion_file=example/motion_data/Horse-stance_pose.pkl seed=2 \
  checkpoint=<checkpoint or warm start> experiment_name=<name>
```

It trains 1,000 iterations by default (`algo.config.num_learning_iterations`), and anything in it can be overridden on the command line.

**The same recipe written out** (STEP17; the preset adds `tilt:1.75,bumps:2.0` to the curriculum ceilings, `terrain.num_cols=24`, `placement_jitter=0.15` and the early starts with `levels_only=True`). The tracking σ values are frozen at the ones the pose run had adapted to:

```bash
SIGMA="rewards.reward_tracking_sigma.teleop_max_joint_pos=0.2952 rewards.reward_tracking_sigma.teleop_upper_body_pos=0.0021 \
  rewards.reward_tracking_sigma.teleop_lower_body_pos=0.0015 rewards.reward_tracking_sigma.teleop_vr_3point_pos=0.0022 \
  rewards.reward_tracking_sigma.teleop_feet_pos=0.0013 rewards.reward_tracking_sigma.teleop_body_rot=0.0998 \
  rewards.reward_tracking_sigma.teleop_body_vel=0.0713 rewards.reward_tracking_sigma.teleop_body_ang_vel=1.9885 \
  rewards.reward_tracking_sigma.teleop_joint_pos=0.0215 rewards.reward_tracking_sigma.teleop_joint_vel=3.3087"
```

```bash
python humanoidverse/train_agent.py +simulator=isaacgym +exp=motion_tracking +terrain=terrain_kungfu \
  project_name=MotionTracking +opt=wandb +robot=g1/g1_23dof_lock_wrist +domain_rand=main +rewards=motion_tracking/main \
  robot.motion.motion_file=example/motion_data/Horse-stance_pose.pkl seed=2 +device=cuda:0 headless=True num_envs=1536 \
  +obs=motion_tracking/main_terrain_v2_preview checkpoint=<checkpoint or warm start> experiment_name=<name> \
  env.config.terrain_aware.recenter_motion_xy=True env.config.terrain_aware.spawn_clearance=0.01 \
  env.config.terrain_aware.ground_relative_ref=True "env.config.terrain_aware.height_scan.y_range=[-0.5,0.5]" \
  env.config.terrain_aware.height_lookup=exact env.config.terrain_aware.ref_height_lookup=exact \
  algo.config.load_optimizer=True +algo.config.load_optimizer_lr=False algo.config.schedule=fixed \
  algo.config.actor_learning_rate=2e-5 algo.config.critic_learning_rate=1e-4 algo.config.entropy_coef=0 \
  +algo.config.fix_noise_std=True +algo.config.actor_lr_ramp_iterations=100 +algo.config.actor_warmup_iterations=100 \
  terrain.curriculum=True terrain.max_init_terrain_level=1 terrain.curriculum_graduate_rows=2 \
  "terrain.curriculum_max_difficulty_by_type={step:2.0}" \
  env.config.termination_curriculum.terminate_when_motion_far_curriculum=False \
  env.config.termination_scales.termination_motion_far_threshold=0.7 \
  env.config.termination_scales.termination_drift_penalty_scale=0.25 rewards.adaptive_tracking_sigma.enable=False \
  algo.config.save_interval=250 algo.config.num_learning_iterations=3250 $SIGMA
```

**Warm start from a released checkpoint** (new inputs at zero weight, optimizer state expanded):

```bash
python scripts/expand_checkpoint.py example/pretrained_horse_stance_pose/model_50000.pt \
  humanoidverse/config/obs/motion_tracking/main_terrain_v2_preview.yaml logs/warmstart/<name>.pt \
  "env.config.terrain_aware.height_scan.y_range=[-0.5,0.5]"
```

**Evaluation:**
- **One setting:**

  ```bash
  python scripts/terrain_eval.py <run>/model_<N>.pt --name step_d100 --seed 0 --episodes 200 \
    "terrain.terrain_types=[step]" "terrain.terrain_proportions=[1.0]" terrain.fixed_difficulty=1.0 \
    env.config.terrain_aware.ground_relative_ref=True
  ```

  Add `--start-time T` to start every episode at motion time T.
- **Full sweep:** `scripts/eval_difficulty_sweep.sh`, then `python scripts/summarize_difficulty_sweep.py --step-edges --drift LABEL=CKPT ...`.
- **The released checkpoint:** it predates these options, so add `+env.config.terrain_aware.recenter_motion_xy=True`.

**Videos.**
- **Command:** `scripts/record_pilot_videos.sh TAG=CKPT ...` records the released policy and each checkpoint on 17, 13.5, 10 and 6.5 cm steps, 12° and 6° tilts, 4 cm bumps and flat.
- **Comparisons:** side-by-side `compare_<terrain>.mp4` files are uploaded to each run's W&B page.
- **Options:** `OUT=`, `NO_WANDB="TAG ..."` and `SECONDS_PER_CLIP=`.
- **Requirements:** a display. Each clip is one episode, so use the sweep to compare models.

**W&B:** `+opt=wandb` logs to `saketkaki-the-university-of-texas-at-austin/MotionTracking`; the `pbhc` conda env sets `WANDB_ENTITY` and the API key.

**Checkpoint naming:** the final policy is saved under the name of the last periodic save (for example `model_68250.pt` for a run ending at 68,500). Copy it to `model_<final>.pt`.

---

## Appendix: the original plan (2026-10-27)

### How the plan turned out

| Plan step | Outcome |
|---|---|
| Step 0: terrain infrastructure | Done. `bumps` was added after the rough tile turned out to be about 1 cm noise in a pit. The exact lookup replaced the min3 helper below |
| Step 1: motivating experiment | Done. The released policy completes 0% on tilt, bumps and rough |
| Step 2: ground-relative reference | Done as a soft-max rigid shift (C2). Per-foot and leg-IK variants were built and not adopted. The leg-joint weight ablation (C′) was not run |
| Step 3: height scan | Done, then widened with foot scans and an exact lookup (C3). Teacher–student distillation was not done |
| Step 4: terrain curriculum | Done, with graduation rows, per-type ceilings and levels saved in checkpoints |
| Rows B, C, E | Run in round 1 with the first fine-tuning recipe. Later work continued only E. Row D (privileged critic only) was not run |
| Not in the plan | The reference preview (C4), the fine-tuning recipe (C5) and the evaluation work (C6) |

### A.1 What the existing code gives us

All paths are relative to the repo root. Line numbers are from the original code.

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

### A.2 Implementation plan

#### Step 0: Terrain infrastructure (~1 day)

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

#### Step 1: Motivating experiment (~½ day, no training)

Evaluate the released checkpoint `example/pretrained_horse_stance_pose/model_50000.pt` zero-shot on each terrain type and difficulty. On flat ground this checkpoint reaches a 98.4% episode-length ratio (measured on our setup). The expected drop on slopes and steps is the motivating figure for the report.

#### Step 2: Ground-relative reference adaptation (core contribution, ~2–3 days)

Implement inside `kick_motion_res()` and `kick_motion_res_multiplestep()`:

1. **Remove the fixed origin height.** Pass the origin's x, y as the offset with z = 0.
2. **Per-frame support height.** Look up terrain heights under the reference feet (`rg_pos_t[:, feet_ids, :2]`). *(Implemented as a soft maximum over the supporting feet instead: averaging the two heights left the foot on higher ground up to 3.5 cm inside the terrain.)*
3. **Shift the reference.** Add `h_sup` to the z of `root_pos`, `rg_pos` and `rg_pos_t`. Rotations are unchanged; optionally add d(`h_sup`)/dt to the vertical velocities.
4. **Ground-relative feet reward.** In `_reward_teleop_body_position_feet` (~L1210), compare each foot's height above the terrain beneath it, `foot_z − h(foot_xy)`, with the reference foot's height above flat ground; keep x, y as before.
5. **Leg-joint conflict.** On a slope, ankle and knee angles must deviate from the reference. Add a lower-body weight mask in `_reward_teleop_joint_position` (~L1265); ablate 1.0 vs ~0.3.
6. **Regression test (required).** With `mesh_type: plane` all heights are zero, so the modified code must reproduce PBHC exactly.

#### Step 3: Height-scan perception (~1–2 days)

1. **Scan grid:** e.g. 11 × 7 points at 10 cm spacing (1.0 m × 0.6 m), centred under the feet. Precompute in the robot's heading frame and rotate by heading each step.
2. **Getter:** add `_get_obs_height_scan` to `MotionTracking`, returning `clip(root_z − h(points) − c, −1, 1)` with `c` ≈ 0.6–0.7.
3. **Config:** copy `config/obs/motion_tracking/main.yaml` to `main_terrain.yaml` and add `height_scan` to the critic (and actor for the perceptive variant).

| Variant | Actor sees scan | Critic sees scan |
|---|---|---|
| Blind | no | no |
| Privileged critic | no | yes |
| Perceptive | yes | yes |

Stretch goal: teacher–student distillation, which requires porting the DAgger logic from `ppo_mimic` to `mh_ppo`.

#### Step 4: Terrain curriculum (~1 day)

- Set `terrain.curriculum: True`; `base_task.py` already initialises `terrain_levels`, `terrain_types` and `terrain_origins`.
- Implement `_update_terrain_curriculum(env_ids)`: level up after reaching the motion end, down after an early failure, then move the origin.
- Log the mean terrain level, and monitor adaptive σ.

### A.3 Experiments as planned

**Motions:** 2–3, e.g. Horse-stance pose (easy), Horse-stance punch (easy), Roundhouse kick (hard).
**Test terrains:** flat, rough, tilt, step, each at 3 difficulty levels.

| Row | Model |
|---|---|
| A | PBHC as released (pretrained, zero-shot) |
| B | PBHC trained on terrain, world-frame reference (naive) |
| C | B + ground-relative reference |
| C′ | C with lower-body joint weight 0.3 (ablation) |
| D | C + privileged-critic height scan |
| E | C + actor height scan (full method) |

**What each comparison shows:** A → B, training on terrain alone; B → C, the ground-relative reference; C → D → E, perception; C vs C′, how strictly the legs should follow the reference joint angles.

**Fairness:** same iteration budget and seed set for every row.

### A.4 Budget

The paper trains each model for 27 h on an RTX 4090. Our RTX 5090 is shared, so the plan was fixed-budget runs of about 6–10 h, with 2 seeds where possible, prioritising rows B, C and E. In practice every comparison became a 3,250-iteration fine-tune (about 1.7 h on a free GPU), and from 2026-10-31 two training seeds per variant.

### A.5 Risks

| Risk | Mitigation |
|---|---|
| Reference jumps when contact switches feet | Smooth the support height. *(Done with the soft maximum, footprint averaging and the velocity correction.)* |
| Robot spawns inside the terrain | 1–2 cm spawn clearance; start from flat or low difficulty. |
| Bugs silently change flat-ground behaviour | Mandatory flat-ground regression test. |
| Adaptive σ distorted by mixed difficulty | Log σ per term. *(σ frozen from Phase 1 on.)* |
| Not enough compute for all rows and seeds | Prioritise B, C, E; fixed budgets; 2–3 motions only. |

### A.6 Positioning against prior work

Survey terrain-conditioned humanoid motion imitation from 2025 before claiming novelty (for example VideoMimic, which we believe does terrain-aware imitation from video; verify). Frame the contribution as **adding terrain awareness to PBHC's adaptive motion-tracking framework for highly dynamic skills, via ground-relative reference adaptation, height-scan perception and a robot-relative reference preview**, and state clearly how it differs.
