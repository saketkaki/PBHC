# Defense Maneuvers for PBHC

A defender G1 learns to respond to attacks thrown by a second, offensive Unitree G1. The policy is conditioned on what
it sees of the attacker, in the same spirit as the terrain-aware work ([TERRAIN_AWARE_TRACKING.md](TERRAIN_AWARE_TRACKING.md)):
- The terrain policy got its perception (height scan, reference preview) as extra actor inputs.
- Here the attacker's observed motion is **embedded by a learned encoder**.
- A **belief head** turns that embedding into a posterior over which attack is coming and when it lands.
- The policy acts on the embedding and the belief.

**Status (2026-11-04):**
- The env, the attack-conditioned policy and the configs are implemented and verified (Section 6).
- A 60-iteration smoke run trains without errors, and the belief head starts learning: attack-class accuracy goes from 12 % (chance) to 35 % in 40 iterations.
- No full training run yet.

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

**Outcomes (`envs/defense/defense.py`).** Each attack resolves once, between its onset and recovered frames:

| Outcome | Rule |
|---|---|
| **blocked** | The striker's swept path passes within `block_radius` (7 cm) of a defender forearm axis (elbow to hand). Checked first |
| **hit** | The path passes within `hit_radius` (8 cm) of the target point, `target_link` + `target_local_offset` on the defender |
| **evaded** | Neither, by the recovered frame |

**Do-nothing baseline.** This is a canonical defender holding its guard perfectly, scored by the same rules (`scripts/defense_check.py bank`):

| | jab | cross | lead hook | rear hook | lead upper | rear upper | body shot | calf kick | all |
|---|---|---|---|---|---|---|---|---|---|
| hit | 5 % | 100 % | 8 % | 100 % | 10 % | 0 % | 0 % | 100 % | 40 % |
| blocked | 95 % | 0 % | 92 % | 0 % | 90 % | 100 % | 100 % | 0 % | 60 % |

The generator aimed each strike at the canonical defender without regard to its guard. Five classes therefore pass within 1.5–6 cm of the guard forearms, and a held guard "blocks" them. That is realistic (a high guard does stop a jab), but the cross, rear hook and calf kick are where active defense is needed. With `block_radius=0` every attack must be evaded.

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
| `blend_time` | 0.25 s | Pose blending between clips |
| `hit_radius`, `block_radius` | 0.08, 0.07 m | Outcome rules; `block_radius=0` disables blocks |
| `threat_lead_time`, `tracking_relax` | 0.1 s, 0.5 | Tracking relaxation during attacks |
| `proximity_sigma`, `tti_clip` | 0.1 m, 1 s | Shaping reward width; time-to-impact clip |
| `terminate_on_hit` | `False` | End the episode on a hit |
| `debug_draw` | `True` | Draw env 0's striker and target in the viewer (and the attacker skeleton without `attacker_actor`) |
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

```bash
python scripts/defense_eval.py logs/Defense/<run>/model_20000.pt                       # 256 episodes, ~25 s
python scripts/defense_eval.py logs/Defense/<run>/model_20000.pt --name no_block env.config.defense.block_radius=0
```

**Viewer:** `python humanoidverse/eval_agent.py +checkpoint=logs/Defense/<run>/model_20000.pt`, or `scripts/record_video.py record` for a video.
- The attacker is drawn as a red G1: one actor per body (24) carrying that link's meshes, with no joints, collision shapes or gravity, posed every step from the clip's forward kinematics.
- It cannot touch the defender and does not change the physics (eval results with and without it agree).
- Its link offsets match the simulated G1 exactly, also while blending between clips.

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
