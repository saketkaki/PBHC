# Run log: terrain-aware motion tracking

One entry per training run and per evaluation-only experiment on branch `terrain-aware-tracking`, in the order they were run (2026-10-27 to 2026-11-04). For each run, the entry gives what changed, the result, what went wrong, and the verdict. The plan, the contributions and the current summary are in [TERRAIN_AWARE_TRACKING.md](TERRAIN_AWARE_TRACKING.md).

Motion: Horse-stance pose (`example/motion_data/Horse-stance_pose.pkl`) unless stated. Robot: G1 23-DoF, wrists locked. GPU: one RTX 5090, shared with other jobs (HCRL) for much of the time.

## How to read this log

### Evaluation protocols

All evaluations start every episode at the motion start, use the ground-relative reference, and end an episode on one of three outcomes:
- **completed:** the motion reached its end;
- **fall:** gravity termination;
- **drift:** any body more than 0.7 m from its reference.

On steps a drift ending is usually a fall in progress (see 2026-10-30, step failure timing). "a / b" in the tables below means completion % / fall %.

| Protocol | Folder (`<run>/metrics/...`) | Settings | Notes |
|---|---|---|---|
| v1 | `terrain_eval/` | 6–7 terrains × 2 seeds × 200 episodes (`eval_terrain_matrix.sh`) | Counted the first 200 episodes to finish, so early failures were over-counted. Reported mostly as episode-length ratio. Used until 2026-10-29. |
| v2 | `terrain_eval_v2/` | 17 settings × 3 seeds; every env contributes 2 episodes (`eval_difficulty_sweep.sh`) | Fair counting, plus step-edge diagnostics. Introduced in the 2026-10-30 Phase 0 sweep. |
| v3 | `terrain_eval_v3/` | Same 17 settings × 3 seeds × 200 episodes | Adds the drift breakdown and the raised-side direction. v2 and v3 agree where both exist (F3s2: flat 97.0 / 97.0, tilt 12° 79.7 / 79.5). |
| v3t | `terrain_eval_v3t/` | 10 cm steps × 3 seeds × 200 episodes | Per-episode end times, start layout and shin contact, summarised by a session script (not in the repo). |

The 17 settings are flat, rough 0.9 (held out, never trained on) and five difficulties each of step, tilt and bumps:

| Difficulty | 0.5 | 1.0 | 1.25 | 1.5 | 2.0 |
|---|---|---|---|---|---|
| Step height | 6.5 cm | 10 cm | 11.8 cm | 13.5 cm | 17 cm |
| Tilt | 6° | 12° | 15° | 18° | 24° |
| Bumps (amplitude) | 2 cm | 4 cm | 5 cm | 6 cm | 8 cm |

"Flat error" is the root-relative body position error on flat ground (like PBHC's E_mpbpe), in mm. The released checkpoint scores 20.2 mm.

### Names

| Name | Run (`logs/MotionTracking/*-<run>-*`) | One-line description |
|---|---|---|
| A | `example/pretrained_horse_stance_pose/model_50000.pt` | Released PBHC, zero-shot |
| B / C / E1 | `ft_B_naive(_resume)`, `ft_C_ground_rel(_resume)`, `ft_E_full(_resume)` | Round 1: naive reference / ground-relative reference / + height scan |
| E2, E2pf | `ft2_E_full`, `ft2_Eplus_per_foot` | Round 2: gentle fine-tuning; + per-foot reference |
| E2c, E2c2 | `ft2_E_full_cont`, `ft2_E_full_cont2` | E2 continued once, then twice |
| E2ik | `ft3_E_legik` | E2c + leg-IK reference |
| F1s1/2 | `ft4_E_fallpen_s1/2` | Phase 1: fall penalty, 0.3 m training cutoff |
| F2s1/2 | `ft5_E_far07_s1/2` | Phase 1b: fall penalty, 0.7 m training cutoff |
| F3s1/2 | `ft6_E_scan2_s1/2` | Phase 2: perception v2 |
| CTRL, SWING | `ft7_E_ctrl_s2`, `ft8_E_swing_s2` | Control for Phase 2; + swing clearance |
| REFLU, CONT | `ft9_E_reflookup_s2`, `ft11_E_cont_s2` | min3 reference lookup; plain continuation |
| PREVIEW_s2/3/4 | `ft10_E_preview_s2/3/4` | + reference preview (robot frame, 0.5 s) |
| PREVREF, EARLY, PREVLONG | `ft12_E_prevref_s2`, `ft13_E_early_s2`, `ft14_E_prevlong_s2` | Preview variants |
| EARLYLONG_s2/3 | `ft15_E_earlylong_s2/3` | 1.0 s preview + early starts without demotion |
| PREVMORE_s2/4 | `ft16_E_prevmore_s2/4` | PREVIEW trained 3,250 more iterations |
| STEP17_s2/4 | `ft17_E_step17_s2/4` | PREVIEW + step curriculum to 17 cm |
| PUNCH_released, PUNCH_s2 | `example/pretrained_horse_stance_punch/model_33000.pt`, `ft18_P_preview_s2` | Horse-stance punch: released, and with the full recipe |

### Lineage

```
released pose model_50000 (A)
├── pilot B, pilot C (from scratch, 3,000 it)
├── round 1: B, C, E1 ──────────────────────────────── → 53,500
└── round 2: E2, E2pf ──────────────────────────────── → 53,500
    └── E2 → E2c ───────────────────────────────────── → 56,500
        ├── E2c2, E2ik, F1s1, F1s2, F2s1, F2s2 ──────── → 59,500
        └── F2s2 → F3s1, F3s2, CTRL ────────────────── → 62,500
                   F3s2 → SWING, REFLU, CONT, PREVIEW_s2/s3/s4, PREVREF,
                          EARLY, PREVLONG, EARLYLONG_s2/s3 → 65,500
                          PREVIEW_s2/s4 → PREVMORE_s2/s4, STEP17_s2/s4 → 68,500
released punch model_33000 → PUNCH_s2 ──────────────── → 39,500 (running)
```

### Checkpoint naming quirk

`mh_ppo.learn` saves the final policy under the name of the last periodic save (for example `model_68250.pt` in a run that ends at 68,500). It overwrites that snapshot, so the real 68,250 snapshot is lost. Every run's final file was copied to `model_<final iteration>.pt`, and the evaluations use that name.

---

## 2026-10-27

### Setup: RTX 5090 environment

PBHC needs Isaac Gym Preview 4, which ships for older CUDA. PyTorch 2.4.1 was built from source for sm_120 in conda env `pbhc` (see [SETUP_RTX5090.md](SETUP_RTX5090.md)). Training and the released checkpoints run.

### Zero-shot baseline A on terrain (`ratio_eps.py`, eval only)

The released Horse-stance pose checkpoint on the new terrains, 100 envs × 100 episodes, as episode-length ratio:

| Reference | Flat | Rough 0.9 | Tilt 6° | Tilt 12° | Step 6.5 cm | Step 10 cm |
|---|---|---|---|---|---|---|
| Naive | 98.9% | 22.5% | 16.4% | 10.9% | 14.0% | 23.5% |
| Ground-relative | 99.4% | 22.5% | 16.2% | 10.6% | 19.0% | 33.6% |

The policy is blind, so the reference only changes the drift check: these are real falls. Later fair-counting sweeps (v2) put A at 98.8 / 0.5 on flat, 0% completion on rough, tilt and 4 cm bumps, and 7–23% on steps.

**Odd result:** A's step completion rises with step height (v2: 6.5 cm 7.3%, 10 cm 21.0%, 13.5 cm 23.0%, 17 cm 23.2%), which a blind policy shouldn't do. The cause was never checked.

### Pilots B and C from scratch (`pilot_B_naive`, `pilot_C_ground_rel`)

- **W&B:** `2zsv2h25` (B), `vn3zl3ac` (C).
- **Setup:** 2048 envs, seed 1, 2 h each (about 3,000 iterations); the released checkpoint has 50,000.
- **What changed:** B uses the naive reference; C adds the ground-relative reference.
- **Result:** episode-length ratio is about 30% on every terrain, flat included. From the motion start, both lose track at 1.5–2.5 s.
- **Problems:**
  - Far too short to learn the motion itself.
  - Training starts at random motion phases, which hid this in the training curves (`end_time_ratio` about 0.82).
  - The first launch at 19:16 was restarted at 19:27; its folders are empty.
- **Verdict:** compare methods by fine-tuning from the released checkpoint, not by training from scratch.

### Round 1: B, C and E1 fine-tuned from the released checkpoint

- **Runs and W&B:**
  - B: `ft_B_naive` `pdle14pn`, then `_resume` `8v6l43xv`
  - C: `ft_C_ground_rel` `6k6a8guf`, then `_resume` `ub9hc8tw`
  - E1: `ft_E_full` `zts2025b`, then `_resume` `e60m9jmo`
- **Iterations:** 50,000 → 53,500.
- **Settings:**
  - 1,536 envs; actor and critic learning rate 3e-4 with the adaptive (KL) schedule; fresh optimizer.
  - 150-iteration critic-only warm-up; drift threshold starting at 0.7 m and shrinking towards 0.3 m.
  - Terrain curriculum on (flat, rough, tilt, step).
  - E1 adds the 77-point height scan (`main_terrain`; actor 457, critic 707 inputs), warm-started with zero weights.

**Results** at 53,500 (v1; episode-length ratio):

| Row | Flat | Rough 0.9 | Tilt 6° | Tilt 12° | Step 6.5 cm | Step 10 cm |
|---|---|---|---|---|---|---|
| A | 99.5% | 20.2% | 16.1% | 11.2% | 12.3% | 12.8% |
| B | 61.1% | 37.4% | 31.7% | 18.2% | 19.1% | 15.4% |
| C | 74.0% | 39.0% | 36.8% | 17.0% | 18.0% | 15.8% |
| E1 | 95.2% | 38.5% | 81.0% | 42.0% | 28.4% | 20.1% |

**Falls:** B 23–36%, C 1–7%, E1 2–36%. **Flat error:** B 49.8 mm, C 45.8 mm, E1 45.1 mm (A 20.4 mm).

The same checkpoints under v2 (completion / falls):

| Row | Flat | Tilt 12° | Bumps 4 cm | Step 10 cm | Step 17 cm |
|---|---|---|---|---|---|
| B | 57.3 / 8.3 | 3.2 / 24.2 | 31.0 / 14.8 | 20.0 / 24.0 | 13.5 / 24.3 |
| C | 64.7 / 1.8 | 1.0 / 1.8 | 32.0 / 2.3 | 18.8 / 4.7 | 17.0 / 6.8 |
| E1 | 91.5 / 2.3 | 28.3 / 13.8 | 33.7 / 21.0 | 11.7 / 19.5 | 3.0 / 27.2 |

**Findings:**
- **The height scan (E1) is what makes terrain tracking work:** it is the only row that completes whole motions on tilts.
- **The ground-relative reference alone (C vs B)** barely changes how much of the motion is tracked, but almost removes falls: episodes end by drifting instead.

**Problems:**
- **Disk full.** All three runs died at about 23:10 on 2026-10-27, when the shared root volume filled up (exit 120).
  - They were resumed on 2026-10-28 from their last checkpoints (51,500 / 51,500 / 51,250) with the optimizer state.
  - The drift threshold was set to 0.3 m, where its curriculum had got to, and σ was passed in as the last logged values.
  - Terrain levels were not saved in checkpoints, so they restarted at 0.
- **Fine-tuning collapse.** On flat ground, B went from 99% completed (released weights) to 0% at 50,250, after only 100 actor updates, and recovered only to 39%.
  - Action noise std rose from 0.21 to about 0.59, and the action-rate penalty from −0.22 to −1.4.
  - Causes: a fresh Adam state, a learning rate of 3e-4, and the entropy bonus.
- **Curriculum rows never reached the test difficulty.** Rows were i/5 (difficulty 0–0.8), so the hardest training tile was an 8.6 cm step or a 10.4° tilt.
- **The rough tile is not a useful test.** It is 1–1.5 cm noise on the 10 cm grid, and it sits in a pit 2.5–7 cm below the surrounding ground. B, C and E1 all score about 38% there.
- **Round-1 videos are wrong for B, C and E.** They were filmed on easy curriculum tiles, not the test terrain (found and fixed on 2026-10-29). A's clips are correct.

**Verdict:** E is the right base. The fine-tuning recipe has to be gentler, and the curriculum has to cover the test difficulties.

---

## 2026-10-28 / 29

### Round 2: E2 and E2pf, gentle fine-tuning (`ft2_E_full`, `ft2_Eplus_per_foot`)

- **W&B:** `1jhr6keb` (E2), `gkqdf6z9` (E2pf).
- **Iterations:** 50,000 → 53,500, 1,536 envs, about 5 s per iteration (shared GPU).
- **Changes from round 1:**
  - Fixed learning-rate schedule: actor 2e-5 ramped in over 100 iterations after the 150-iteration warm-up; critic 1e-4.
  - Entropy coefficient 0, and noise std frozen at the released 0.212 (`fix_noise_std`).
  - The released optimizer state, expanded for the new inputs (`logs/warmstart/main_terrain_from_50000_opt.pt`).
  - Curriculum rows span difficulty 0.2–1.0.
  - The new smooth `bumps` terrain replaces rough in the mix.
  - Videos: low side camera, scan points drawn.
  - E2pf adds `per_foot_ref=True`.

**Results** (v1, episode-length ratio):

| Model | Flat | Rough | Bumps 4 cm | Tilt 6° | Tilt 12° | Step 6.5 cm | Step 10 cm |
|---|---|---|---|---|---|---|---|
| E1 | 95.2% | 38.5% | 35.8% | 81.0% | 42.0% | 28.4% | 20.1% |
| E2 | 91.7% | 27.0% | 41.0% | 82.2% | 42.7% | 22.1% | 18.0% |
| E2pf | 92.0% | 28.7% | 41.2% | 56.9% | 18.8% | 17.2% | 20.0% |

- **E2:** more precise than E1 (flat error 35.5 vs 45.1 mm), but it falls more (flat 9% vs 2%, 6.5 cm steps 44% vs 36%).
- **E2pf:** far fewer falls (2–20%), but much worse on tilts, mostly by drifting off the reference. The likely cause: the per-foot reference moves the ankle and knee positions while the joint reward still asks for flat-ground angles.

**Checkpoint trend** (E2 checkpoints from 50,500 to 53,500, eval only):
- Flat error rose from 28 to 35 mm within the first 500 iterations and then held.
- The tilts only took off after 52,000 (6° tilt: 42.8 → 82.2% over the last 1,500 iterations).
- So E2 was under-trained, not badly configured.

**Problems:**
- **Intermittent PhysX crash.** An illegal GPU memory access while building the mixed terrain, in smoke tests only. It wasn't reproducible (the same setup then passed six times) and no cause was found in our code.
- **Timeout restart.** The first launch had a 5 h timeout, too short at the shared-GPU speed, and was restarted with 8 h. The two aborted W&B runs show as crashed.
- **Video fixes**, made in `record_video.py` and `record_pilot_videos.sh`:
  - Clips used each run's own curriculum terrain; they now use the fixed test terrain.
  - The drift threshold differed per model; it is now 0.7 m for all.
  - The viewer segfaulted while the HCRL `video_logger.py` was running; clips now wait for it and retry.
  - A W&B upload timeout crashed the script; uploads now wait longer and failures only warn.

**Verdict:** continue E2. Set the per-foot reference aside.

### E2c: E2 continued (`ft2_E_full_cont`)

- **W&B:** `x334k2bz`.
- **Iterations:** 53,500 → 56,500, with E2's settings and optimizer state.
- **Carried over:** drift threshold 0.3 m and σ. No warm-up.

**Results** (v2 completion / falls):

| Terrain | E2c |
|---|---|
| Flat | 96.8 / 1.2 |
| Rough | 21.7 / 32.5 |
| Tilt 6° | 88.8 / 5.0 |
| Tilt 12° | 56.8 / 12.0 |
| Bumps 4 cm | 54.3 / 14.7 |
| Step 6.5 cm | 26.0 / 19.3 |
| Step 10 cm | 3.8 / 26.5 |

- **Flat error:** 31.5 mm.
- **Compared with E2:** better on every terrain except steps. For example, v1 bumps went from 41 to 68% and tilt 12° from 43 to 61%.
- **Steps:** unchanged at 20–30% of the motion (v1).

**Verdict:** E2c was the best policy at the time. Steps are the open problem.

### Scan-noise robustness sweep (eval only, E1 and E2)

New `env.config.terrain_aware.height_scan_noise` options:
- per-point noise;
- per-episode height bias and horizontal map shift;
- dropout;
- delay.

These come with the `height_scan_noisy` observation, `eval_scan_noise.sh` and `summarize_scan_noise.py`.

**Results** (E2, episode-length ratio):

| Condition | Flat | Tilt 6° | Bumps |
|---|---|---|---|
| Exact scan | 92.3 | 83.0 | 41.9 |
| 5 cm height offset | 83.9 | 73.1 | 35.9 |

Only the 5 cm height offset clearly hurts. Per-point noise up to 5 cm, a 10 cm map shift, 60% dropout and a 100 ms delay all stay within about 3 points.

**Problem:** I first read this as "the policy only uses body height from the scan". The blind test below showed that was wrong: tilts and bumps are smooth, so noise leaves their shape intact.

### Blind-scan ablation (eval only, E2c and E1)

Two blind modes:
- `blind=flat`: the scan shows flat ground at the height under the robot;
- `blind=zero`: no terrain information at all.

**Results** (E2c, episode-length ratio):

| Scan | Flat | Bumps | Tilt 6° | Tilt 12° | Step 10 cm |
|---|---|---|---|---|---|
| Exact | 98.7 | 67.6 | 94.0 | 61.0 | 19.0 |
| `blind=flat` | 98.1 | 46.1 | 48.5 | 17.8 | 14.6 |
| `blind=zero` | 44.8 | 34.0 | 36.1 | 17.9 | 19.9 |

- **The policy does use the terrain shape:** without it the 12° tilt drops from 61 to 18%.
- **On steps, blinding reduces falls** (10 cm: 30 → 7%). The policy sees the edge and reacts badly.

**Verdict:** this is the clean perception ablation (same weights, with and without the scan).

---

## 2026-10-29 / 30

### E2c2 and E2ik (`ft2_E_full_cont2`, `ft3_E_legik`)

- **W&B:** `jdxprtzn` (E2c2), `5bqj1f7o` (E2ik).
- **Iterations:** both start from E2c at 56,500 and run to 59,500.
- **What changed:**
  - **E2c2:** nothing (continuation). It is also the first run to log curriculum levels per terrain type: steps 1.96 against about 2.6 for the others.
  - **E2ik:** `leg_ik_ref=True`, which re-solves the reference legs by damped least-squares IK so each foot stands on its own ground. It has a 100-iteration warm-up.
- **IK reference checks:**
  - Joint angles and foot positions agree to 0.01 cm on 10 cm steps; the per-foot reference was off by 2 cm on average.
  - Flat ground is identical to the original reference.
  - Ankle roll hits its ±15° limit in about 13% of wide-stance cases.

**Results** (v2 completion / falls):

| Terrain | E2c2 | E2ik |
|---|---|---|
| Flat | 96.7 / 1.5 | 89.5 / 8.2 |
| Tilt 12° | 52.8 / 17.8 | **70.5** / 18.3 |
| Bumps 4 cm | 64.2 / 14.2 | 56.5 / 24.2 |
| Step 6.5 cm | 33.0 / 26.5 | 31.3 / 32.0 |
| Step 10 cm | 12.8 / 32.0 | 6.2 / 37.5 |

- **E2c2** tracks more closely than E2c but falls more on hard terrain: diminishing returns.
- **E2ik** is better on the steepest tilt and worse on flat and bumps.
- **Cross-check:** each policy was evaluated with the other reference. Swapping the reference changed results by only 1–4 points, so E2ik's differences come from what it learned, not from the reference.

**Verdict:**
- Leg IK stays in the code but off by default: a minor point.
- Two runs from the same checkpoint differ by 10–15 points, so training seeds are necessary.

### Phase 0: fair-counting difficulty sweep (eval only)

- **Models:** A, B, C, E1, E2c, E2c2 and E2ik.
- **Protocol:** v2, the 17 settings above.

**Results** (episode-length ratio):

| Terrain | A | B | C | E1 | E2c | E2c2 | E2ik |
|---|---|---|---|---|---|---|---|
| Flat | 99.5 | 71.7 | 76.3 | 95.0 | **98.4** | 97.9 | 93.3 |
| Tilt 12° | 11.3 | 23.7 | 20.9 | 54.6 | 74.7 | 73.6 | **82.2** |
| Step 10 cm | **37.0** | **37.1** | **36.7** | 31.7 | 24.9 | 33.3 | 28.2 |

- **The old counting was unfair to A, B and C.** A on 10 cm steps went from 13% to 37% with fair counting.
- **E is best everywhere except steps of 10 cm and up.**

**Step-edge diagnostic:** E2c falls 51% of the time on 10 cm steps when the edge is *not* between the feet, against 9% when it is. A, B and C show no such gap.

**How far E2c goes** (at least 50% of the motion completed):

| Terrain | Holds up to |
|---|---|
| Bumps | 5 cm |
| Tilt | 15° |
| Steps | 6.5 cm only |

### Phase 1: fall penalty (`ft4_E_fallpen_s1/s2`, F1)

- **W&B:** `xkvpw2p2` (s1), `sfciu1ij` (s2).
- **Iterations:** E2c 56,500 → 59,500.
- **What changed:**
  - A drift termination costs 0.25 of a fall (`termination_drift_penalty_scale=0.25`).
  - σ frozen (`adaptive_tracking_sigma.enable=False`).
  - Robots that pass the top row stay in the top 2 rows (`curriculum_graduate_rows=2`).

**Result:** no effect. Two-seed v2 means:

| Terrain | Completion | Falls |
|---|---|---|
| Flat | 93.8% | 3.4% |
| Tilt 12° | 53.6% | 24.1% |
| Step 10 cm | 11.2% | 25.5% |

**Problem:** the training drift cutoff had shrunk to 0.3 m while the evaluation allows 0.7 m. Episodes ended long before a fall could happen, so the penalty never fired: training falls were about 0%.

**Verdict:** the training cutoff must match the evaluation's.

### Phase 1b: fall penalty with a 0.7 m training cutoff (`ft5_E_far07_s1/s2`, F2)

- **W&B:** `h2qk5ajo` (s1), `los8bs4k` (s2).
- **Setup:** as Phase 1, plus `terminate_when_motion_far_curriculum=False` and the drift threshold fixed at 0.7 m.

**Results** (v2 completion / falls):

| Terrain | F2s1 | F2s2 |
|---|---|---|
| Flat | 94.5 / 0.0 | 96.2 / 0.0 |
| Rough | 20.2 / 0.8 | 20.3 / 1.7 |
| Tilt 12° | 57.3 / 0.7 | 69.8 / 0.8 |
| Tilt 24° | 0.0 / 4.3 | 0.2 / 15.0 |
| Bumps 8 cm | 5.2 / 0.7 | 11.3 / 4.3 |
| Step 10 cm | 10.2 / 2.2 | 7.7 / 18.7 |
| Step 17 cm | 3.0 / 3.7 | 0.3 / 22.8 |

- **Falls became rare:** F2s1 falls at most 7% everywhere, including 17 cm steps and 24° tilts.
- **Held-out rough:** falls went from 33% to 1–2%.
- **But completion didn't rise:** episodes that used to end in a fall now end in drift.
- **The seeds settled differently:** s1 is safest but tracks worse; s2 tracks as well as E2c2 and is best on tilt.

**Direction diagnostic** (10 cm steps; falls / not completed, by where the raised side is at the start):

| Policy | Ahead | Left | Behind | Right |
|---|---|---|---|---|
| E2c2 | **61** / 87% | 24 / 89% | 11 / 87% | 21 / 89% |
| F2s1 | 9 / 76% | 0 / 99% | 1 / 88% | 0 / 100% |
| F2s2 | **47** / 92% | 3 / 97% | 9 / 90% | 6 / 93% |

Falls come from a step up ahead; not completing happens in every direction.

**Problem: small steps are ramps.** The trimesh conversion turns any height change under 7.5 cm per 10 cm cell into a slope (`slope_treshold: 0.75`). So the 6.5 cm evaluation step, and every curriculum row below 7.5 cm, is a ramp across one 10 cm cell, not a wall. This was left unchanged so comparisons stay clean.

**Verdict:** keep the fall penalty and the 0.7 m cutoff in every later run. Start Phase 2 from F2s2.

### Phase 2: perception v2 (`ft6_E_scan2_s1/s2`, F3)

- **W&B:** `aai2iw4c` (s1), `nmjb30n1` (s2).
- **Iterations:** F2s2 59,500 → 62,500, warm start `logs/warmstart/v2_from_F2s2_59500.pt`.
- **What changed** (obs config `main_terrain_v2`):
  - Height scan widened to ±0.5 m (11 × 11 = 121 points).
  - A 2 cm foot scan under each foot (2 × 91 points).
  - Exact height lookup from the rasterised trimesh, for both the scans and the reference.
  - A critic-only 15 × 15 privileged scan.
  - Inputs: actor 457 → 683, critic 707 → 1,158. The expanded actor matches the original to 5e-6.
- **Exact lookup check:** feet more than 2 cm off their resting height drop from 20% to 5% on the 12° tilt, and from 16% to 3% on bumps.

**Results** (v2 completion / falls):

| Terrain | F2s2 | F3s1 | F3s2 |
|---|---|---|---|
| Flat | 96.2 / 0.0 | 97.7 / 0.0 | 97.0 / 0.0 |
| Tilt 12° | 69.8 / 0.8 | **85.8** / 0.2 | 79.7 / 0.2 |
| Tilt 18° | 13.5 / 8.8 | **34.7** / 6.2 | 28.8 / 5.8 |
| Bumps 4 cm | 62.3 / 0.8 | **73.2** / 0.0 | 70.7 / 0.8 |
| Step 10 cm | 7.7 / 18.7 | 6.0 / 19.5 | 12.7 / 12.2 |
| Rough | 20.3 / 1.7 | 17.7 / 2.7 | 20.2 / 6.2 |

- **Flat error:** 26.2 / 30.2 mm, against 32.5 for F2s2.
- **Tilt and bumps improved a lot; steps didn't.**
- **Step up ahead at 10 cm:** falls 55% (s1), 36% (s2), against 47% for F2s2.
- **Shin contact** now shows up in step falls: 15–31% against 3–9% in other episodes.

**Verdict:** the best policy so far except on steps. It needs a control at the same training budget.

### CTRL: control for Phase 2 (`ft7_E_ctrl_s2`)

- **W&B:** `vxgyovzc`.
- **Setup:** F2s2 59,500 → 62,500 with Phase 2's training settings but the old observations and the min3 lookup.

**Results** (v2 completion / falls):

| Terrain | CTRL | F3s2 |
|---|---|---|
| Tilt 12° | 51.5 / 0.3 | 79.7 / 0.2 |
| Bumps 4 cm | 43.0 / 1.2 | 70.7 / 0.8 |
| Step 10 cm | **26.8 / 5.8** | 12.7 / 12.2 |
| Step 17 cm | **9.0** | 1.2 |

- Phase 2's tilt and bumps gains come from the new scan: the same extra training without it made those terrains worse.
- On steps the control was best: 4% falls on a step up ahead, against 36% for F3s2.

**Verdict:** the step gap suggested the exact reference lookup as the cause, which REFLU then tested.

### SWING: swing clearance (`ft8_E_swing_s2`)

- **W&B:** `oho5rasc`.
- **Iterations:** F3s2 62,500 → 65,500.
- **What changed:** `swing_clearance=True` raises a lifted reference foot by how much the highest ground within 15 cm exceeds its own ground (the knee gets half).
- **Reference check:** swinging feet more than 1 cm into the ground on 10 cm steps go from 9.6% to 4.8%. Flat ground is unchanged.

**Results** (v3, compared with CONT, the plain continuation over the same iterations):

| Terrain | SWING | CONT |
|---|---|---|
| Step 10 cm | 13.0 / 10.5 | 9.0 / 9.5 |
| Tilt 12° | 54.3 / 0.7 | 71.0 / 0.5 |
| Bumps 4 cm | 74.2 / 0.0 | 81.2 / 0.0 |

**Verdict:** dropped. It is no better than CONT and worse on tilt.

---

## 2026-10-30 / 31: reference preview

Five runs from F3s2 at 62,500 to 65,500, trained together: REFLU, PREVIEW_s2, CONT, and SWING above, then the variants below.

### Step failure timing (eval only, F3s2, 10 cm steps)

New `terrain_eval.py --start-time` option. Completion by start point in the motion:

| Start point | Completed |
|---|---|
| 0 s | 3–6% |
| 0.3 s | 6–7% |
| 5.0 s | 88% |

- **Every failure from the motion start happens 1–2 s in:** the weight shift onto the left foot, then the right foot lifting up to 28 cm and swinging about 50 cm sideways and back while the pelvis drops from 0.80 to 0.57 m.
- **Drift endings there are falls in progress:**
  - 95% happen within 0.5 s;
  - the head is the body furthest off;
  - the pelvis is tilted about 30°, where the fall check needs about 50°.
- **Training hides this:** episodes start at random motion times, mostly in the easy later parts. That is why the training step level (about 3.5) looked good while the evaluation failed.
- **Unresolved oddity:** splitting uniform random-start evaluations by start time gave 19–33% early-start completion, which the controlled runs don't reproduce. Don't use those splits.

### REFLU: min3 lookup for the reference only (`ft9_E_reflookup_s2`)

- **W&B:** `tsf676xc`.
- **What changed:** the scans keep the exact lookup; the reference uses min3, like CTRL. On 10 cm steps, min3 differs from exact by more than 2 cm 7.8% of the time.
- **Result:** worse everywhere, flat included (93.2%; 10 cm steps 11.0 / 7.3).
- **Verdict:** the lookup doesn't explain CTRL's step advantage. Dropped.

### PREVIEW_s2: reference preview, robot frame (`ft10_E_preview_s2`)

- **W&B:** `frvy9ym3`.
- **Warm start:** `logs/warmstart/preview_from_F3s2_62500.pt` (obs `main_terrain_v2_preview`; actor 749, critic 1,224 inputs).
- **What changed:** the new `ref_preview` observation, for actor and critic. It contains the terrain-shifted reference root and both feet relative to the robot root in its heading frame, plus the reference heading relative to the robot's, at 0, 0.1, …, 0.5 s ahead: 11 × 6 = 66 numbers.
- **Check:** at time 0 the preview matches the tracked reference exactly.

**Results** (v3 completion, with falls in brackets):

| Terrain | F3s2 | CONT | REFLU | PREVIEW_s2 | SWING | CTRL |
|---|---|---|---|---|---|---|
| Flat | 97.0 | 98.3 | 93.2 | **99.5** | 98.2 | 95.7 |
| Rough | 18.2 | 17.7 | 12.2 | **51.8** | 20.0 | 13.0 |
| Tilt 12° | 79.5 | 71.0 | 59.2 | **82.8** | 54.3 | 52.0 |
| Bumps 8 cm | – | 17.8 | 10.5 | **35.2** | 13.5 | – |
| Step 6.5 cm | 34.2 | 40.0 | 29.8 | **60.2** | 32.5 | 34.2 |
| Step 10 cm | 14.0 (12.2) | 9.0 (9.5) | 11.0 (7.3) | **45.7 (2.8)** | 13.0 (10.5) | 25.8 (5.8) |
| Step 17 cm | 2.3 | 1.7 | 3.2 | **16.5** | 2.5 | 8.8 |

- **Best on every terrain.**
- **Flat error:** 21.9 mm, against 26.0 for CONT.
- **Step up ahead at 10 cm:** falls drop from 20% to 6%.

**Problems:**
- The sweep ran out of GPU memory partway through the bumps evaluations for F3s2 and CTRL, when the HCRL jobs started. Those two models are covered on bumps by v2.
- The videos also hit out-of-memory errors and retried until memory freed.
- A duplicate recorder briefly wrote into the same folder; it was restarted clean. Some CTRL/SWING clips may appear twice in W&B.

**Verdict:** this was the big win. The bottleneck was that the actor could not see where the reference is relative to itself (its tracking error), not terrain perception. The preview needs a robot pose estimate at deployment.

**What it meant for the three perception ideas:**
- **Longer history:** dropped. Failures are sudden, not slow drift: even with the preview, 85% of drift endings come within 0.5 s, and the error goes from half to the full threshold in about 0.3 s.
- **Denser scan ahead:** dropped for now. The preview fixed most step-up-ahead falls.
- **Foothold scan:** maybe later, on top of the preview.

### PREVREF: preview in the reference frame (`ft12_E_prevref_s2`)

- **W&B:** `pc5es9dp`.
- **What changed:** the same preview, measured relative to the current reference root, so no localisation is needed.

**Results** (v3 completion / falls):

| Terrain | PREVREF |
|---|---|
| Flat | 93.8 |
| Step 10 cm | 12.5 / 4.8 |
| Step 17 cm | 4.3 |
| Tilt 12° | 68.5 |
| Bumps 4 cm | 70.2 |
| Rough | 18.3 |

About the same as CONT, and worse on flat and bumps.

**Why:** with a single motion, a reference-frame preview carries the same information as the motion phase. The gain needs the robot-relative frame.

**Problem:** the sweep waited about 2 h because an HCRL `video_logger` took the freed GPU memory. The watcher was replaced with one that waits for 4.5 GB free and skips finished evaluations.

**Verdict:** ruled out for single-motion policies (but see the multi-motion idea in the plan).

### EARLY: preview + early starts on step tiles (`ft13_E_early_s2`)

- **W&B:** `q9l9qn7l`.
- **What changed:** robots on step tiles start the motion in the first 1.5 s with probability 0.5 (`early_start`).

**Results** (v3 completion / falls):

| Terrain | EARLY |
|---|---|
| Step 10 cm | 40.3 / 12.3 |
| Step 13.5 cm | 21.7 / 18.0 |
| Bumps 8 cm | **49.5** |
| Tilt 12° | 86.0 |

- **Problem:** failed early starts demoted step robots in the curriculum. Their step level was about 2.8 against 3.6 for PREVIEW, so the policy trained on lower steps.
- **Verdict:** inconclusive. It needs early starts excluded from demotion (added later as `keep_level_on_failure`).

### PREVLONG: preview to 1.0 s (`ft14_E_prevlong_s2`)

- **W&B:** `c6tga7bs`.
- **Warm start:** `preview_long_from_F3s2_62500.pt`.
- **What changed:** 8 preview times (0, 0.1, …, 0.5, 0.7, 1.0 s): 88 numbers.

**Results** (v3 completion / falls):

| Terrain | PREVLONG |
|---|---|
| Step 10 cm | 31.3 / 11.7 |
| Step 17 cm | 6.7 |
| Tilt 12° | 78.8 |
| Bumps 8 cm | **47.3** |

Like EARLY: worse on steps, better on bumps. Its step training level was also lower (about 3.25).

**Verdict:** two different changes gave the same pattern, which pointed to training-seed variance. Seed replicates were run next.

### PREVIEW seed replicates (`ft10_E_preview_s3/s4`)

- **W&B:** `a3e4qcod` (s3), `63n7zc3m` (s4).
- **Setup:** same as PREVIEW_s2, with training seeds 3 and 4.

**Results** (v3 completion, falls in brackets):

| Terrain | No preview (CONT) | PREVIEW_s2 | _s3 | _s4 | EARLY | PREVLONG |
|---|---|---|---|---|---|---|
| Step 6.5 cm | 40.0 | 60.2 | 47.2 | 57.5 | 66.0 | 59.0 |
| Step 10 cm | 9.0 (9.5) | 45.7 (2.8) | 31.3 (14.3) | 37.2 (6.8) | 40.3 (12.3) | 31.3 (11.7) |
| Step 17 cm | 1.7 | 16.5 | 8.0 | 15.7 | 11.5 | 6.7 |
| Tilt 12° | 71.0 | 82.8 | 84.0 | 90.2 | 86.0 | 78.8 |
| Bumps 8 cm | 17.8 | 35.2 | 35.3 | 37.0 | **49.5** | **47.3** |
| Rough | 17.7 | 51.8 | 48.2 | 53.3 | 51.8 | 52.7 |

- **The preview gain holds across seeds.**
- **Steps vary by about 15 points between training seeds** (31–46% at 10 cm), so EARLY and PREVLONG are within noise on steps. On bumps both beat all three seeds.

**Step timing** (v3t), across models:
- About 90% of step failures end at 1.0–3.5 s of the motion.
- The seed-dependent part is falls with the step up directly ahead and its edge 0–10 cm in front: 39–69% in weak runs, 8–15% in s2 and s4.
- About half of those falls had shin contact at first.

**Verdict:**
- Two training seeds per variant from now on.
- The 0.5 s robot-frame preview is the base.

---

## 2026-10-31 / 11-01

### EARLYLONG: 1.0 s preview + early starts without demotion (`ft15_E_earlylong_s2/s3`)

- **W&B:** `zk2756wr` (s2), `3im1398m` (s3).
- **What changed:** PREVLONG's 1.0 s preview, plus early starts with `keep_level_on_failure=True`. About 10% of episodes were early starts on step tiles.

**Results** (v3 completion / falls):

| Terrain | EARLYLONG_s2 | EARLYLONG_s3 |
|---|---|---|
| Step 10 cm | 30.7 / 11.7 | 31.7 / 13.3 |
| Tilt 12° | 65.7 | 63.3 |
| Bumps 8 cm | 36.5 | 45.8 |
| Flat error | 20.4 mm | 21.7 mm |

- **The demotion fix worked:** step levels 3.56 and 3.40, against 2.8 for EARLY.
- **Steps didn't improve.** All three runs with the 1.0 s preview fall 12–13% on 10 cm steps.
- **Tilt 12° got worse** (63–66% against 83–90% for PREVIEW).

**Step timing** (v3t):
- The falls are starts with both feet low and the step up 0–10 cm ahead, at about 1.1–1.3 s, during the weight shift. Weak runs fall there 30–53% of the time, PREVIEW s2/s4 8–15%.
- Straddle starts (one reference foot already on the step, about half of all episodes) complete only 23–37% in every run, mostly by drifting.
- Shin contact no longer separates failures from completions, so the shin-penalty idea was dropped.

**Problem:** seed 3 waited for 8 GB of free GPU memory while the HCRL jobs held about 24 GB. The chain was reordered so the seed 2 sweep ran first.

**Verdict:** the 1.0 s preview hurts. Early-start training on the hard segment doesn't carry over to episodes that start at the beginning of the motion.

### PREVMORE: PREVIEW trained 3,250 more iterations (`ft16_E_prevmore_s2/s4`)

- **W&B:** `e9d3en9x` (s2), `i4udxq7c` (s4).
- **Iterations:** PREVIEW_s2/s4 continued from 65,250 to 68,500.

**Results** (v3 completion / falls):

| Terrain | s2 before → after | s4 before → after |
|---|---|---|
| Step 10 cm | 45.7 / 2.8 → **54.3** / 6.3 | 37.2 / 6.8 → 35.8 / 5.8 |
| Step 13.5 cm | 29.5 → 36.0 | 23.0 → 22.3 |
| Step 17 cm | 16.5 → 19.0 | 15.7 → 14.7 |
| Tilt 12° | 82.8 → **92.5** | 90.2 → 90.8 |
| Rough | 51.8 → **57.5** | 53.3 → 53.5 |
| Bumps 8 cm | 35.2 → 34.3 | 37.0 → 38.2 |

- **Flat error:** 21.0 mm (s2), 19.8 mm (s4); A is 20.2 mm.
- **s2 timing:** the gain came from steps beside and behind the robot. On a step up ahead, failures went from 77 to 83 of 160 and falls from 13 to 22.

**Problems:**
- **Seed 4 barely improved.** Its step curriculum level only climbed back to about 2.3 (s2: 3.5), because terrain levels were not saved in checkpoints and both continuations restarted from level 1. Most of its extra training was on easy steps.
- **Seed 4 waited about 3.8 h** for 8 GB of free GPU memory (the HCRL jobs held 24.1 GB). Seed 2's timing evaluation ran meanwhile.

**Verdict:** mixed. The two-seed average on 10 cm steps went from about 41% to 45%, smaller than the seed spread.

---

## 2026-11-04

### Check: training never saw the evaluated step heights

The step curriculum's top row was difficulty 1.0, so the five rows were 4.4, 5.8, 7.2, 8.6 and 10 cm. A step level of about 3.5 means training happened mostly at 8.6–10 cm, while the sweep goes to 17 cm.

### Bug found: every run started one curriculum level down

`learn()` calls `env.reset_all()` after the environment's first step, when `common_step_counter` is already 1. That passed the curriculum's guard and counted as a failed episode for every robot, so all previous runs started every robot at level 0.

Fixed in `motion_tracking.py` with an `_in_reset_all` flag. A checkpoint with every level at 3 now stays at 3.0 after the first iteration (before the fix it went to 2).

### STEP17: step curriculum to 17 cm (`ft17_E_step17_s2/s4`)

- **W&B:** `0rior7qw` (s2), `ing955si` (s4).
- **Iterations:** PREVIEW_s2/s4 from 65,250 to 68,500, the same as PREVMORE (the control).
- **What changed:**
  - `terrain.curriculum_max_difficulty_by_type={step: 2.0}`, so the step rows are 5.8, 8.6, 11.4, 14.2 and 17 cm. Tilt and bumps are unchanged.
  - Terrain levels are now saved in checkpoints (`terrain_levels`, `terrain_types`) and restored on continuation. They weren't in PREVIEW's checkpoints, so these runs still start at levels 0–1, but without the reset_all demotion.
- **Training:** the final step level was 3.39 (s2) and 3.29 (s4). On the new rows, that means most step robots trained on 14–17 cm steps. Tilt, bumps and flat levels were 3.6–3.7, the same as the control.

**Results** (v3 completion / falls):

| Terrain | PREVMORE_s2 | STEP17_s2 | PREVMORE_s4 | STEP17_s4 |
|---|---|---|---|---|
| Step 6.5 cm | 62.7 / 2.5 | 61.3 / 1.8 | 56.7 / 0.0 | 63.2 / 1.3 |
| Step 10 cm | 54.3 / 6.3 | 51.7 / 7.8 | 35.8 / 5.8 | **49.7** / 4.8 |
| Step 11.8 cm | 46.3 / 7.5 | 49.8 / 9.5 | 29.2 / 6.7 | **43.5** / 7.5 |
| Step 13.5 cm | 36.0 / 14.7 | 39.2 / 13.7 | 22.3 / 9.2 | **33.2** / 13.0 |
| Step 17 cm | 19.0 / 21.5 | **28.5** / 17.7 | 14.7 / 9.2 | **22.7** / 14.5 |
| Tilt 12° | 92.5 | 91.3 | 90.8 | 87.8 |
| Tilt 18° | 42.2 | 41.7 | 32.3 | 48.0 |
| Bumps 8 cm | 34.3 | 40.7 | 38.2 | 44.0 |
| Rough | 57.5 | 53.0 | 53.5 | 57.0 |
| Flat error | 21.0 mm | 20.2 mm | 19.8 mm | 21.0 mm |

Two-seed means against the control:

| Terrain | PREVMORE | STEP17 |
|---|---|---|
| Step 10 cm | 45.1% | **50.7%** |
| Step 11.8 cm | 37.8% | **46.7%** |
| Step 13.5 cm | 29.2% | **36.2%** |
| Step 17 cm | 16.8% | **25.6%** |
| Bumps 8 cm | 36.2% | **42.3%** |
| Tilt 18° | 37.2% | **44.8%** |
| Tilt 12° | 91.7% | 89.6% |
| Rough | 55.5% | 55.0% |

- **Taller steps improved for both seeds**, and nothing else got meaningfully worse.
- **The gap between seeds closed:** s4 on 10 cm steps went from 35.8 to 49.7%, next to s2's 51.7%.

**Problems:**
- **Falls on 17 cm steps are still 15–18%.**
- **A step up ahead is still the main failure:** at 10 cm, s2 falls there 25% of the time and s4 10%, against 1–6% in the other directions. At 17 cm, s2 falls 49% with the step ahead; s4 falls 19% ahead and 30% with the raised side on the right.
- **The punch training shared the GPU with the s4 sweep**, which slowed it to 3.3 s per iteration (normally about 1.7).

**Verdict:** keep the 17 cm step curriculum. Videos are in `logs/MotionTracking/videos_step17_s2_it68500/` (released pose, PREVMORE_s2 and STEP17_s2, including new 13.5 and 17 cm step clips). Step timing for both seeds and the s4 videos are queued after the punch run.

### PUNCH_released: released Horse-stance punch, zero-shot (eval only)

`example/pretrained_horse_stance_punch/model_33000.pt` (v3, ground-relative reference, motion recentred):

| Terrain | Completion / falls |
|---|---|
| Flat | 98.7 / 0.2 |
| Rough | 0.0 / 21.5 |
| Tilt 6° / 12° | 0.0 / 8.5, 0.0 / 2.0 |
| Bumps 4 cm | 0.7 / 21.3 |
| Steps 6.5 / 10 / 13.5 / 17 cm | 77.7 / 2.5, 81.2 / 2.8, 81.8 / 1.0, 82.7 / 1.2 |

- **Flat error:** 17.3 mm.
- **The punch barely moves its feet**, so steps are easy for it even blind.
- **Tilt, bumps and rough fail completely,** mostly by drifting.

### PUNCH_s2: Horse-stance punch with the full recipe (`ft18_P_preview_s2`), running

- **W&B:** `cs47ubpi`.
- **Iterations:** 33,000 → 39,500 (6,500 iterations, twice a pose continuation, because it starts from a flat-ground policy).
- **Warm start:** `logs/warmstart/punch_preview_from_33000.pt` (actor 380 → 749, critic 630 → 1,224). `expand_checkpoint.py` was fixed to accept the released config, which has no `terrain_aware` section.
- **Recipe:** the same as PREVIEW (perception v2, 0.5 s robot-frame preview, fall penalty, 0.7 m cutoff, gentle fine-tuning). The step curriculum tops out at 10 cm, as for PREVIEW.
- **σ:** frozen at the pose run's values, because the released checkpoint doesn't store the adapted σ.
- **Status at 16:30:**
  - Iteration 34,975, about 1.6 s per iteration; it should finish around 18:30.
  - Training levels: flat 3.6, tilt 3.5, bumps 3.2, step 1.8. The low step level is unexpected, since the released punch completes 80% of steps zero-shot; check it once the sweep is in.
- **Queued after training:** the sweep against PUNCH_released, then the step timing for STEP17_s2/s4, then videos (`videos_step17_it68500`, `videos_punch_it39500`).
