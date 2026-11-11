# G1 attack dataset for defender training (PBHC format)

Procedurally generated, labelled attack motions for a 23-DoF Unitree G1. They are written in the
exact motion format of this PBHC repo, so `humanoidverse` can load them directly. The attacker is
the same robot as the defender. Each clip is one discrete attack thrown from an orthodox
(left-foot-forward) guard. There are no combinations.

| label | attack | striking point | target on the defender |
|---|---|---|---|
| 0 | `jab` | left fist | face |
| 1 | `cross` | right fist | face |
| 2 | `lead_hook` | left fist | right temple |
| 3 | `rear_hook` | right fist | left temple |
| 4 | `lead_uppercut` | left fist | chin |
| 5 | `rear_uppercut` | right fist | chin |
| 6 | `body_shot` | left fist (lead hook to the body, with a level change) | liver (right side) |
| 7 | `calf_kick` | lower right shin | outside of the lead (left) calf |

Fist = PBHC's `left_hand_link` / `right_hand_link` (elbow link + 0.25 m). Shin = 75 % of the way
from the right knee to the ankle.

## Layout

```
motion_data/attacks/
  <attack>/<attack>_0000.pkl    one clip per file: joblib {stem: motion}
  attacks_all.pkl               every clip in one joblib {stem: motion}
  defender_guard.pkl            the canonical defender guard pose, as a static 2 s motion
  index.json / index.csv        labels, timings, targets and quality metrics for every clip
  previews/                     GIFs (attacker orange, canonical defender blue, target green)
attack_gen/                     generator, validator, preview renderer, AttackLibrary
```

## Motion format

Each motion dict has PBHC's keys, with the same shapes and conventions as `example/motion_data/*.pkl`:

| key | shape | notes |
|---|---|---|
| `root_trans_offset` | (T, 3) | pelvis position, world frame, z up |
| `root_rot` | (T, 4) | pelvis quaternion, **xyzw** |
| `dof` | (T, 23) | joint angles in `robot.dof_names` order |
| `pose_aa` | (T, 27, 3) | root rotvec, `dof_axis * dof`, 3 zero rows (extended bodies) |
| `smpl_joints` | (T, 27, 3) | zeros, as in the examples |
| `fps` | int | **50** (PBHC's policy rate; the loader reads `fps` per file) |
| `contact_mask` | (T, 2) | left/right foot contact, 1.0 or 0.0 |
| `attack_meta` | dict | extra key, ignored by PBHC (see below) |
| `phase` | (T,) int8 | extra key: 0 guard, 1 strike (onset to impact), 2 retract, 3 guard |

### The clip frame

In each clip the attacker starts near the origin facing +x. The canonical defender, a G1 in the same
guard, faces it along -x. `attack_meta` holds:

- `attack`, `label`, `striker`, `target_region`
- Target as a body point: `target_link` plus `target_local_offset`, in that link's frame. Use these
  for hit detection on your real, moving defender.
- `target_point`: the same point on the canonical defender, in the clip frame.
- Canonical defender placement: `defender_root_pos`, `defender_root_rot` (xyzw),
  `defender_root_yaw`, `defender_dof`, and `distance` (D).
- Timing: `onset_frame`, `impact_frame`, `full_extension_frame`, `recovered_frame` (and `*_time`).
  - The impact is when the striking point crosses the target.
  - Strikes are aimed a few cm through the target, so the fist is still moving at impact. Use
    `impact_velocity` and `impact_speed`.
- `miss_distance` (striking point to target at impact), `peak_striker_speed`, `speed_scale`
- `quality`: joint velocity, limit margin, foot slip, CoM and ZMP support margins, and self-penetration.

Each clip is: guard (random 0.35–0.9 s), then the attack (randomized speed ×0.85–1.2), then retract,
then guard again. Every clip also randomizes:

- stance length, width and blade angle
- crouch height and torso lean
- guard hand positions
- idle sway
- aim (±1.2 cm)
- range D

So the onset can't be predicted from the clip start, and two clips of the same class never look the same.

## Using it with PBHC

**Train the attacker to track one clip** (WJX library, one motion per run, as in the paper):

```bash
python humanoidverse/train_agent.py +simulator=isaacgym +exp=motion_tracking \
  +terrain=terrain_locomotion_plane project_name=Attacks num_envs=4096 \
  +obs=motion_tracking/main +robot=g1/g1_23dof_lock_wrist +domain_rand=main \
  +rewards=motion_tracking/main experiment_name=jab_0000 \
  robot.motion.motion_file="motion_data/attacks/jab/jab_0000.pkl" seed=1 +device=cuda:0
```

**Train one tracker on many clips** (KungfuBot2 general tracking, `origin` motion library). Point
`motion_file` at either:

- a class directory, e.g. `motion_data/attacks/jab`, or
- `motion_data/attacks/attacks_all.pkl`.

```bash
python humanoidverse/train_agent.py +simulator=isaacgym +exp=general_tracking \
  +terrain=terrain_locomotion_plane project_name=Attacks num_envs=4096 \
  +obs=motion_tracking/obs_ppo_teacher +robot=g1/g1_23dof_general +domain_rand=main \
  +rewards=motion_tracking/general_main experiment_name=attacks_teacher \
  robot.motion.motion_file="motion_data/attacks/attacks_all.pkl" seed=1 +device=cuda:0
```

I loaded all three modes (WJX single file, `origin` file, `origin` directory) through PBHC's own
`MotionLibRobotWJX` / `MotionLibRobot`. Their forward kinematics match MuJoCo's to 0.00 mm on every
body. Rerun that check with `python attack_gen/validate_attacks.py --data motion_data/attacks --pbhc`.

Note: this repo trains in **Isaac Gym** (`+simulator=isaacgym`). Its `isaacsim` backend targets
Isaac Lab 1.x and lacks features motion tracking needs (see `SETUP_RTX5090.md`). The data itself is
simulator-agnostic.

## Training the defender: replaying the attacker as a ghost

You don't need a tracking policy to start. `attack_gen/attack_library.py` replays clips
kinematically in batches. Load it with:

```python
from attack_gen.attack_library import AttackLibrary
lib = AttackLibrary("motion_data/attacks/attacks_all.pkl", device="cuda")
```

**On reset**, for each env:

- `ids = lib.sample(n)`: draws an attack class uniformly, then a clip.
- Put the defender in the canonical guard: `lib.def_dof[ids]` with the root pose of your choice.
- Keep `t = 0`.

**Every step:**

1. `s = lib.state(ids, t)` gives interpolated `dof_pos`/`dof_vel`, root state, `phase`, `time_to_impact`.
2. `pos, rot, lin_vel, ang_vel = lib.place_relative_to_defender(ids, s, def_xy, def_yaw)` maps the clip so
   its canonical defender lands on your defender's reset pose.
3. Write that into the attacker actor (root state + DoF state, e.g. with IsaacGym's
   `set_actor_root_state_tensor_indexed` and `set_dof_state_tensor_indexed`).
4. Then `t += dt`.

**Hit detection:**

- Striker position: `left_elbow_link`/`right_elbow_link` + `R @ [0.25, 0, 0]` for fists, or
  `right_knee_link` + `R @ (0.75 * p)` for the shin, where `p` is the `right_ankle_pitch_link`
  offset in the knee frame.
- Target position: the defender's `target_link` pose applied to `target_local_offset`.
- A hit is a striker within a few cm of the target, between `onset_frame` and `recovered_frame`.

**Notes:**

- **Labels:** `lib.label[ids]` is the attack class and `s["time_to_impact"]` the time left. Use them
  as privileged critic inputs, or as targets for an auxiliary "read the attack" head. Keep them out
  of the deployed actor.
- **Collisions with a kinematic ghost:** a kinematically driven actor pushes with unbounded force.
  The usual setup is one of these:
  - keep collision only between the attacker's striking link and the defender, and score blocks
    from that contact;
  - drive the attacker with a PD tracking policy trained as above, so contacts are physical.
- **Range:** the G1's arms are short (the shoulder-to-fist reach is at most about 0.45 m). Striking
  range is therefore close: D ≈ 0.36–0.50 m pelvis to pelvis for punches, 0.48–0.55 m for the kick.
  At that range the two guards nearly touch.

## Regenerating or extending

```bash
python attack_gen/generate_attacks.py --out motion_data/attacks --per_attack 60 --seed 0
python attack_gen/generate_attacks.py --out motion_data/attacks_big --per_attack 500 --workers 16
python attack_gen/generate_attacks.py --attacks jab,cross --per_attack 200 --out motion_data/straights
MUJOCO_GL=egl python attack_gen/render_previews.py --data motion_data/attacks   # or xvfb-run + glfw
python attack_gen/validate_attacks.py --data motion_data/attacks
```

Run these from the PBHC root (they read `description/robots/g1/`). They need `mujoco`, `numpy`,
`scipy` and `joblib`, all already in the PBHC env (tested on Python 3.8 + mujoco 3.2.3 and
Python 3.13 + mujoco 3.17). Each attack has its own seed stream, so clip `jab_0007` is the same
whichever other attacks you generate. Generation runs at about 0.5 s per clip per core.

## How the clips are made

1. **Choreography** (`g1_attacks/attacks.py`). Each attack is a set of task-space keyframe tracks:
   - fists and elbows
   - pelvis height, yaw and roll
   - torso yaw, pitch and roll
   - centre of mass
   - feet (and the shin for the kick)

   Tracks use minimum-jerk timing. Curved paths (hooks, uppercuts, kick) are splines through via
   points, so the strike has one smooth speed profile.
2. **Whole-body IK** (`g1_attacks/ik.py`). Damped least squares on the 23-DoF model in MuJoCo, each
   frame warm-started from the last. It enforces:
   - pinned, flat feet (the ankle sits 3.54 cm above the sole)
   - the CoM held inside the support polygon (and over the lead foot during the kick)
   - joint limits from the PBHC config, via an active set
   - limb self-collision avoidance
   - a small temporal-smoothness term
3. **Checks and rejection** (`g1_attacks/generate.py`). A clip is kept only if all of these hold:
   - the striker comes within 3 cm of the target (4.5 cm for the kick)
   - it is still moving at ≥ 1 m/s when it gets there
   - feet slide < 2 cm
   - self-penetration < 1.5 cm
   - the CoM stays inside the support polygon

   If a strike falls short, the defender is moved closer, never below the minimum range, and the
   clip is re-solved.

## Validation (the dataset shipped here: 60 clips per attack, seed 0)

| attack        | n  | D [m]     | onset->impact ms | len s | miss cm avg/max | v_imp m/s | v_peak | max qd rad/s | foot slip mm | min CoM margin cm | median min ZMP margin cm | self-pen mm |
|---------------|----|-----------|------------------|-------|-----------------|-----------|--------|--------------|--------------|-------------------|--------------------------|-------------|
| jab           | 60 | 0.42-0.50 | 166 (140-200)    | 1.59  | 1.1 / 1.8       | 1.5       | 3.5    | 19.5         | 1            | 6.1               | -8                       | 2           |
| cross         | 60 | 0.37-0.48 | 206 (180-240)    | 1.69  | 1.1 / 1.7       | 1.5       | 4.7    | 16.5         | 1            | 6.2               | -4                       | 12          |
| lead_hook     | 60 | 0.37-0.46 | 271 (220-320)    | 1.75  | 1.8 / 3.0       | 2.8       | 4.9    | 20.4         | 2            | 6.2               | -0                       | 4           |
| rear_hook     | 60 | 0.36-0.44 | 277 (220-340)    | 1.74  | 1.5 / 2.7       | 1.7       | 4.2    | 26.2         | 2            | 5.7               | -13                      | 12          |
| lead_uppercut | 60 | 0.36-0.44 | 295 (240-340)    | 1.80  | 2.7 / 3.0       | 2.1       | 3.9    | 13.0         | 0            | 6.1               | 1                        | 1           |
| rear_uppercut | 60 | 0.36-0.43 | 309 (260-380)    | 1.77  | 2.1 / 2.9       | 1.5       | 4.9    | 19.7         | 1            | 6.1               | -6                       | 10          |
| body_shot     | 60 | 0.38-0.43 | 341 (280-420)    | 1.85  | 1.6 / 3.0       | 2.6       | 4.6    | 18.9         | 2            | 6.3               | 1                        | 13          |
| calf_kick     | 60 | 0.48-0.55 | 701 (600-840)    | 2.59  | 1.7 / 3.3       | 3.2       | 4.7    | 15.3         | 20           | 0.6               | -65                      | 14          |

How to read the table:

- **CoM margin**: distance of the CoM inside the support polygon on its worst frame. It is
  positive for every clip.
- **ZMP margin**: computed from CoM acceleration only.
- **Joint speeds**: the fastest joint in the whole set reaches 76 % of PBHC's per-joint velocity
  limit. The median of each clip's peak speed is 15 rad/s.
- **Rejections**: about a third of the first attempts were rejected, mostly because a strike fell
  short or was too slow at the target. Rejected attempts were re-solved or resampled.

For reference, the PBHC example clips that the paper trains on score worse on the same checks:

| PBHC example | ZMP margin (min) | foot slip under the contact mask | max joint velocity |
|---|---|---|---|
| `Hooks_punch` | −8 cm | 31 cm | 9.2 rad/s |
| `Side_kick` | −30 cm | 35 cm | 12.9 rad/s |
| `Roundhouse_kick` | −53 cm | 28 cm | 7.1 rad/s |

So the generated clips are at least as trackable.

## Limitations

- **Procedural, not mocap.** The motion is clean and stereotyped. It has the big tells (hip load,
  level change, weight shift before the kick) but none of a real fighter's subtle ones. Before
  trusting a defender, test it against a few real clips. AMASS has punching clips, and PBHC's
  `smpl_retarget` can bring them onto the G1.
- **Feet stay planted** except for the calf kick: no stepping jab, no pivot on the rear heel.
- **ZMP excursions.** The CoM is inside the support polygon on every frame, but the fastest
  strikes briefly push the ZMP estimate (which ignores angular momentum) outside it. That's
  realistic for whipping motions and less severe than in PBHC's own example clips.
- **Orthodox stance only.** Mirroring y and swapping left/right joints gives southpaw.
- **One canonical target pose.** The defender used for aiming is a G1 standing in guard. When your
  defender moves, re-evaluate the target from `target_link` / `target_local_offset` as above.
