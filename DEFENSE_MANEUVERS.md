# Defense Maneuvers for PBHC

A defender G1 learns to respond to attacks thrown by a second, offensive Unitree G1. The policy is conditioned on what
it sees of the attacker, in the same spirit as the terrain-aware work ([TERRAIN_AWARE_TRACKING.md](TERRAIN_AWARE_TRACKING.md)):
- The terrain policy got its perception (height scan, reference preview) as extra actor inputs.
- Here the attacker's observed motion is **embedded by a learned encoder**.
- A **belief head** turns that embedding into a posterior over which attack is coming and when it lands.
- The policy acts on the embedding and the belief.

**Status (2026-11-11):**
- The env, the attack-conditioned policy and the configs are implemented and verified (Section 6).
- First full flat-ground run: 0 % hits after 3,000 iterations, against 54 % for a held guard (Section 9).
- Defense on uneven terrain is implemented and in training (Section 9).

## 1. Setup

| | Choice |
|---|---|
| Attacks | The G1 attack dataset in `g1_attack_dataset/PBHC_attacks` (see its `attack_gen/README.md`). It has 8 classes × 60 clips: `jab`, `cross`, `lead_hook`, `rear_hook`, `lead_uppercut`, `rear_uppercut`, `body_shot` and `calf_kick`, all from an orthodox guard |
| Defender objective | Track the static guard pose (`defender_guard.pkl`) as a style prior, and get rewards for not being hit. The dataset has no defensive reference motions, so slips, covers and kick checks must emerge from RL |
| Attacker | A **kinematic ghost**: clips are replayed from forward kinematics. It is not a simulated actor and never collides |
| Scoring | Geometric, from the striker's path between control steps |
| Actor's view of the attacker | 10 keypoints relative to the defender's root in its heading frame, over a short history. Deployable from vision or mocap |
| Conditioning | 1D-conv encoder → latent z → belief head (class, phase, time to impact), supervised by privileged labels. The policy gets [proprioception, z, belief] |

## 2. Attacker and outcomes

**Playback (`envs/defense/attack_bank.py`):**
- Every clip goes through the motion library's forward kinematics (`Humanoid_Batch`) once at start-up, so the run-time attacker is a lookup of body positions.
- Clips play back to back.
- Each clip is placed so that its canonical defender lands on the real defender's root and heading at the clip start, then stays fixed. Moving the target away therefore works, while drifting away between attacks does not.
- The pose eases from one clip into the next over 0.25 s, inside the clip's opening guard.
- **Standoff** (`standoff`, 0.25 m): in the clips the two guards are only 1.5–8 cm apart, so a jab's fist is already within `block_radius` of the defender's forearm at its onset and is "blocked" on its first frame. The attacker therefore stands 0.25 m farther back in guard. It steps in during the strike (smoothstep from the onset to the impact frame), so the striker lands exactly where the clip aimed it, then steps back out by the recovered frame. The ghost's feet slide during the step.

**Outcomes (`envs/defense/defense.py`).** Each attack resolves once, between its onset and recovered frames:

| Outcome | Rule |
|---|---|
| **blocked** | The striker's swept path passes within `block_radius` (7 cm) of a defender forearm axis (elbow to hand). Checked first |
| **hit** | The path passes within `hit_radius` (8 cm) of the target point, `target_link` + `target_local_offset` on the defender |
| **evaded** | Neither, by the recovered frame |

**Do-nothing baseline.** This is a canonical defender holding its guard perfectly, scored by the same rules (`scripts/defense_check.py bank`). "Gap" is the smallest distance between the two guards before the onset: any attacker body to any defender body or hand, and the striker to a forearm.

| | jab | cross | lead hook | rear hook | lead upper | rear upper | body shot | calf kick | all |
|---|---|---|---|---|---|---|---|---|---|
| hit, standoff 0 | 5 % | 100 % | 8 % | 100 % | 10 % | 0 % | 0 % | 100 % | 40 % |
| blocked, standoff 0 | 95 % | 0 % | 92 % | 0 % | 90 % | 100 % | 100 % | 0 % | 60 % |
| gap, standoff 0 | 1.5 cm | 3.6 cm | 3.2 cm | 6.3 cm | 4.7 cm | 7.8 cm | 5.7 cm | 1.3 cm | |
| hit, standoff 0.25 m | 0 % | 100 % | 38 % | 100 % | 97 % | 0 % | 0 % | 100 % | 54 % |
| blocked, standoff 0.25 m | 100 % | 0 % | 62 % | 0 % | 3 % | 100 % | 100 % | 0 % | 46 % |
| gap, standoff 0.25 m | 15.7 cm | 11.7 cm | 9.8 cm | 9.6 cm | 11.0 cm | 9.4 cm | 12.0 cm | 21.7 cm | |

No attack is evaded by a static guard either way, so the step-in keeps every strike on target. With the standoff, a held guard blocks only strikes whose path crosses a forearm mid-flight (the jab, rear uppercut and body shot). It no longer blocks strikes that start against the forearm. At 0.2 m, 8 of the 480 clips (hooks and uppercuts, whose guards overlap sideways) still start within `block_radius`. With `block_radius=0` every attack must be evaded.

## 3. Policy: attack embedding and belief

`agents/modules/attack_modules.py`, built by `mh_ppo` when `algo.config.module_dict.actor.attack_encoder` is set:

```
attacker_keypoints (10 frames x 10 keypoints x 3) --Conv1D encoder--> z (32)
z --belief head--> class logits (8), phase logits (4), time to impact (1)
[rest of actor_obs, z, softmax(class), softmax(phase), time to impact] --MLP 512-256-128--> action mean
```

- **Belief loss:** cross-entropy on class and phase plus smooth-L1 on time to impact, against `attack_privileged` in the critic observation. It is added to the actor loss with weight `belief_loss_coef`.
- **Posterior interpretation:** a cross-entropy classifier on the observed history estimates p(attack | motion so far). Before the onset it should stay near uniform; as the tells and the strike appear it sharpens. `Learn/Belief_class_acc_strike` tracks accuracy during strike and retract.
- **Export:** everything is one module on the flat `actor_obs`, so `act_inference` and the ONNX export work unchanged.

## 4. Observations and rewards

| Config | Actor | Critic | Inputs |
|---|---|---|---|
| `obs/defense/main` | PBHC proprioception and history, `attacker_keypoints` (300) | + PBHC privileged terms, `attacker_keypoints`, `attack_privileged` (17) | 675 / 942 |

`ref_motion_phase` is dropped because the guard is static and held past its 2 s clip.

**Rewards (`rewards/defense/main`).** These are PBHC's tracking rewards plus the terms below. Scales are multiplied by dt = 0.02.

| Term | Scale | Effective | Notes |
|---|---|---|---|
| `defense_hit` | −500 | −10 per hit | |
| `defense_block` | +100 | +2 | |
| `defense_evade` | +100 | +2 | |
| `defense_striker_proximity` | −10 | up to −0.2 per step | exp(−(d/0.1)²) of the striker-to-target distance while a strike is live |

Other reward settings:
- `feet_air_time` is off.
- Adaptive σ is off, because a static reference would shrink σ until any defensive move loses all tracking reward.
- In the threat window (0.1 s before onset to recovered), the `teleop_*` tracking rewards are scaled by 1 − `tracking_relax` (0.5).

**Episodes:** 20 s, about 10 attacks. They end on falls or when any body is more than 0.6 m from the guard reference.

## 5. Files and options

| File | Content |
|---|---|
| `humanoidverse/envs/defense/attack_bank.py` | Clip loading (with a NumPy-2 pickle shim), FK, balanced sampling, placement, swept-distance helpers |
| `humanoidverse/envs/defense/defense.py` | `LeggedRobotDefense`: playback, outcomes, observations, rewards, logging, viewer drawing (attacker as a G1, striker, target) |
| `humanoidverse/simulator/isaacgym/isaacgym.py` | `add_visual_link_actors`: visual-only actors per env; env ids are mapped to robot actor indices (identity with one actor per env) |
| `humanoidverse/agents/modules/attack_modules.py` | `AttackEncoder`, `AttackConditionedPolicy`, `AttackAwareActor` (with `belief_loss`) |
| `humanoidverse/agents/mh_ppo/mh_ppo.py` | Builds the actor when `attack_encoder` is configured; `belief_loss_coef` (default 0, other runs unchanged) |
| `humanoidverse/config/{env,exp}/defense.yaml`, `obs/defense/main.yaml`, `rewards/defense/main.yaml` | Configs |
| `scripts/defense_check.py` | Checks (Section 6) |

`env.config.defense` (defaults in `config/env/defense.yaml`):

| Option | Default | Meaning |
|---|---|---|
| `attack_file` | `.../attacks_all.pkl` | File or directory of clips |
| `attacks` | `null` | Subset of classes, e.g. `[jab,cross]` (for a curriculum) |
| `keypoints` | 10 bodies | Attacker bodies in `attacker_keypoints` |
| `keypoint_history.length`, `.stride` | 10, 2 | Frames, and control steps between them (covers 0.36 s) |
| `placement_noise.yaw_deg`, `.range` | 0, 0 | Per-clip randomisation of the attacker's angle and distance |
| `standoff` | 0.25 m | The attacker stands this much farther back in guard and steps in during the strike; 0 = the clips' spacing (checkpoints before this option) |
| `blend_time` | 0.25 s | Pose blending between clips |
| `hit_radius`, `block_radius` | 0.08, 0.07 m | Outcome rules; `block_radius=0` disables blocks |
| `threat_lead_time`, `tracking_relax` | 0.1 s, 0.5 | Tracking relaxation during attacks |
| `proximity_sigma`, `tti_clip` | 0.1 m, 1 s | Shaping reward width; time-to-impact clip |
| `terminate_on_hit` | `False` | End the episode on a hit |
| `debug_draw` | `True` | Draw env 0's striker, target, aim point and outcome markers in the viewer (and the attacker skeleton without `attacker_actor`) |
| `outcome_marker_time` | 1 s | How long the viewer shows where the last attack ended |
| `blind_actor` | `none` | Evaluation only: `freeze` holds the actor's `attacker_keypoints` at the last guard frame before each strike, `zero` zeroes them. The critic and scoring are unchanged |
| `attacker_actor` | `auto` | Show the attacker as a G1: on with a viewer, off headless (training and evals) |

`algo.config.belief_loss_coef` (1.0) and `algo.config.module_dict.actor.attack_encoder` (`type: Conv1D|MLP`, `latent_dim`, `condition_on_belief`) are set in `config/exp/defense.yaml`.

## 6. Verification

`python scripts/defense_check.py MODE`:
- **`bank`:** the FK striker at each clip's impact frame matches the dataset's `impact_point` (0.00 mm mean and max over 480 clips). It also prints the do-nothing baseline above.
- **`policy`:** the belief gradient reaches the encoder, the action depends on the attacker slice, and the ONNX export matches PyTorch (1e-7).
- **`env`:** Isaac Gym with zero actions. Observation shapes match the configs, every clip start places the canonical defender on the real defender (< 0.1 mm), and outcome events and rates are logged.

## 7. How to run

```bash
conda activate pbhc && cd ~/Documents/Schoolwork/PBHC
python humanoidverse/train_agent.py +simulator=isaacgym +exp=defense +terrain=terrain_locomotion_plane \
  project_name=Defense +opt=wandb num_envs=4096 +obs=defense/main +robot=g1/g1_23dof_lock_wrist +domain_rand=main \
  +rewards=defense/main robot.motion.motion_file=g1_attack_dataset/PBHC_attacks/motion_data/attacks/defender_guard.pkl \
  experiment_name=defense_v0 seed=1 +device=cuda:0 headless=True
```

**What to watch:**
- `Env/defense_{hit,block,evade}_rate` and `Env/defense_hit_rate_<attack>`, against the baseline in Section 2.
- `Learn/Belief_class_acc_strike`, `Learn/Belief_tti`.
- `Env/terminate_by_motion_far`, which is high early in training as in PBHC.

**Evaluation (`scripts/defense_eval.py`):**
- Runs fixed 20 s episodes with the training terminations.
- Reports hit, block and evade rates per attack class, attacks interrupted by a fall or drift, survival, guard tracking error and the belief head's accuracy by attack phase.
- Takes `key=value` overrides for harder settings.
- Writes JSON to `<run>/metrics/defense_eval/`.
- `--blind freeze` hides the attacker from the actor: its keypoints are held at the last pre-strike guard frame, so the policy sees an attacker that never strikes. `--blind zero` feeds zeros instead, which the policy has never seen. Either way the scoring is unchanged. The flag sets the env option `blind_actor`, which also works in the viewer.

```bash
python scripts/defense_eval.py logs/Defense/<run>/model_20000.pt                       # 256 episodes, ~25 s
python scripts/defense_eval.py logs/Defense/<run>/model_20000.pt --name no_block env.config.defense.block_radius=0
python scripts/defense_eval.py logs/Defense/<run>/model_20000.pt --name blind_freeze --blind freeze
```

**Original PBHC baseline.** This is the unmodified PBHC tracker, trained only to hold the guard: the motion-tracking env, observations, rewards and actor, with no attacker and no defense rewards. The README command is changed in two ways: 20 s episodes that do not end with the 2 s clip, as in the defense env, and the same 4096 envs and 20000 iterations as the defense run. `ref_motion_phase` stays in its observations; it keeps growing past 1 because the clip is held, the same in training and in the eval. `--env-from` scores it in a defense run's env: attacks, standoff and terminations.

```bash
python humanoidverse/train_agent.py +simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
  project_name=Defense +opt=wandb num_envs=4096 +obs=motion_tracking/main +robot=g1/g1_23dof_lock_wrist +domain_rand=main \
  +rewards=motion_tracking/main robot.motion.motion_file=g1_attack_dataset/PBHC_attacks/motion_data/attacks/defender_guard.pkl \
  env.config.termination.terminate_when_motion_end=False env.config.max_episode_length_s=20 \
  experiment_name=pbhc_guard_tracker seed=1 +device=cuda:0 headless=True algo.config.num_learning_iterations=20000
python scripts/defense_eval.py logs/Defense/<tracker run>/model_<last>.pt --env-from logs/Defense/<defense run>
```

**Viewer:** `python humanoidverse/eval_agent.py +checkpoint=logs/Defense/<run>/model_20000.pt`, or `scripts/record_video.py record` for a video. Add `+env.config.defense.blind_actor=freeze` (`env.config.defense.blind_actor=freeze` for `record_video.py`) to watch the blinded policy. A purple stick figure then shows the attacker as the actor sees it. For a policy trained without the defense env, such as the PBHC baseline, add `+env_from=logs/Defense/<defense run>` so the viewer uses that run's env and attacker.
- The attacker is drawn as a red G1: one actor per body (24) carrying that link's meshes, with no joints, collision shapes or gravity, posed every step from the clip's forward kinematics.
- It cannot touch the defender and does not change the physics (eval results with and without it agree).
- Its link offsets match the simulated G1 exactly, also while blending between clips.

**Outcome markers (env 0).** Because the attacker passes through the defender, the viewer shows what was scored:

| Marker | Meaning |
|---|---|
| Striker sphere and trail | Orange while the attack is live, then the outcome colour: red hit, blue blocked, green evaded |
| Green sphere | The target on the defender's body |
| Yellow sphere | Where the attack is aimed (the striker at the clip's impact frame), shown from the threat window until it resolves |
| Large sphere in the outcome colour, held for `outcome_marker_time` | Where the attack ended: the contact point on the blocking forearm, the hit point, or the closest approach of an evaded attack |
| White line | From that point to the target (at the same moment), i.e. how far from the target the attack was stopped or missed |
| Thick blue line | The forearm that blocked |

The console prints one line per attack in env 0, e.g. `cross: BLOCKED by the left forearm, 14 cm from its target`.

## 8. Open problems and next steps

1. **First full training run**, then reward tuning. The trade-off between guard tracking (and `tracking_relax`) and the hit penalty decides whether the defender moves at all.
2. **A defense evaluation script** (per-class hit, block and evade rates over fixed episodes, like `terrain_eval.py`), plus ablations:
   - no belief loss;
   - no belief conditioning;
   - an MLP encoder;
   - plain concatenation as in the terrain work;
   - an oracle-label actor as the upper bound.
3. **Physical contact:** the striking link colliding with the defender, or an attacker driven by a tracking policy, so that blocks are physical rather than geometric.
4. **Robustness:**
   - placement noise;
   - keypoint noise, delay and dropout;
   - mirrored (southpaw) attacks;
   - real clips (AMASS retargeted with `smpl_retarget`), since the generated attacks are clean and stereotyped.
5. **Dataset:**
   - The clips are pickled with NumPy 2 (`numpy._core`), which `AttackBank` handles but PBHC's motion libraries cannot load under this env's NumPy 1.24.
   - The 90 MB `g1_attack_dataset/` is not committed.

## 9. Results, and defense on uneven terrain

Run details are in [LOG.md](LOG.md) (2026-11-11). All numbers: `scripts/defense_eval.py`, 256 episodes of 20 s, one training seed, one eval seed.

**Dataset note.** `generate_attacks.py` rejects every body shot under the `pbhc` env's MuJoCo 3.2.3 (it reports more self-penetration than newer versions). Generate the clips in a separate env with a recent MuJoCo (`conda create -n attackgen python=3.12`, `pip install mujoco "numpy<2" scipy joblib`); 469 of the 480 clips then equal the committed `index.json`.

**Flat ground (`def01_flat_s2`, the command of Section 7 with seed 2).**

| | Do-nothing guard | 2,000 it | 3,000 it |
|---|---|---|---|
| Hit / blocked / evaded | 54 / 46 / 0 % | 0.1 / 85.7 / 14.2 % | 0.0 / 88.6 / 11.4 % |
| Survived 20 s | – | 99.2 % | 100.0 % |
| Guard body error | – | 85.8 mm | 56.2 mm |
| Hit with the attacker hidden (`--blind freeze`) | – | 15.3 % | 32.7 % |

- The policy stops being hit within 2,000 iterations (about 45 minutes) and does not fall or drift away.
- It relies on seeing the attacker more as training goes on (hidden attacker: 15 % → 33 % hits), most of all for the calf kick.

**Terrain: what changed.**

| Piece | Change |
|---|---|
| Attacker height (`defense.py`) | With `terrain_aware.ground_relative_ref`, the clips stand on the support height of the defender's guard reference (the terrain under its feet), so strikes stay aimed at the guard on steps, tilts and bumps. The attacker is still a ghost: its own feet are not matched to the terrain under them |
| Terrain curriculum (`defense.py`) | The guard is held, so there is no motion to complete: up one difficulty row after a survived episode, down after an episode that ended before half time |
| Observations (`obs/defense/main_terrain`) | + height scan (11 × 11), foot scans; + the wide exact scan for the critic. Actor 978, critic 1,470 inputs |
| Preset (`exp/defense_terrain`) | Defense env on `terrain_kungfu` with the ground-relative, re-centred guard, 0.15 m placement jitter, 24 columns, top rows of 17 cm steps, 21° tilt and 8 cm bumps |
| Warm start (`scripts/expand_checkpoint.py`) | Handles the attack-conditioned actor: a flat defense checkpoint keeps its behaviour, the scans start with zero weights |
| Evaluation (`scripts/defense_eval.py`) | `--terrain {flat,step,tilt,bumps} --difficulty D` |

```bash
python scripts/expand_checkpoint.py logs/Defense/<flat run>/model_2000.pt humanoidverse/config/obs/defense/main_terrain.yaml \
    logs/warmstart/defense_terrain.pt "env.config.terrain_aware.height_scan.y_range=[-0.5,0.5]"
python humanoidverse/train_agent.py +simulator=isaacgym +exp=defense_terrain +opt=wandb +device=cuda:0 headless=True \
    seed=2 checkpoint=logs/warmstart/defense_terrain.pt experiment_name=<name>
python scripts/defense_eval.py logs/Defense/<terrain run>/model_<it>.pt --name step_d1.0 --terrain step --difficulty 1.0
```

**Terrain (`def02_terrain_s2`, in training).** Survived 20 s / hit rate of resolved attacks:

| Terrain | Flat policy, no terrain training | 500 it | 1,000 it |
|---|---|---|---|
| Flat | 98.0 / 0.1 % | 96.9 / 0.3 % | 98.8 / 0.2 % |
| Steps 10 cm | 26.2 / 0.1 % | 55.1 / 1.0 % | 79.7 / 0.7 % |
| Steps 17 cm | 28.9 / 0.4 % | 29.7 / 1.6 % | 63.7 / 1.8 % |
| Tilt 18° | 0.0 / 28.6 % | 26.6 / 1.4 % | 42.2 / 2.6 % |
| Bumps 8 cm | 7.4 / 1.0 % | 74.2 / 0.7 % | 83.2 / 0.1 % |

- The flat policy cannot hold its position on terrain; episodes end because it moves more than 0.6 m from the guard.
- With terrain training survival rises quickly while hits stay below 3 %. The guard is held loosely on terrain (150–210 mm body error).

