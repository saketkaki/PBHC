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
| HARD_s2/4 | `ft19_E_hard_s2/4` | STEP17 + tilt curriculum to 21°, bumps to 8 cm |
| PUNCH17, PUNCH_EARLY, PUNCH_EARLYLR, PUNCH_SWING | `ft20_P_step17_s2`, `ft21_P_early_s2`, `ft22_P_earlylr_s2`, `ft23_P_swing_s2` + `ft24_P_swing_long_s2` / `ft24_P_swing_lr_s2` | Punch: 17 cm step curriculum; + early starts; + higher learning rate; + swing clearance and support-lift gain |
| PUNCH_HONEST, PUNCH_DIVERSE | `ft25_P_honest_s2`, `ft26_P_diverse_s2` | Punch: + only early starts move the level; + 24 terrain columns and placement jitter (continued as `ft27`, `ft29`, `ft30`, `ft32_P_diverse*_s2`; `ft32` is the current best punch) |
| DIVERSE_s2, HONEST_s2 | `ft28_E_diverse_s2`, `ft31_E_honest_s2` + `ft33_E_honest2_s2` | Pose: HARD + 24 terrain columns and placement jitter; + early starts that alone move the level (current best pose, the `terrain_tracking` preset) |
| HOOKS | `ft34_H_warm_s2`, `ft35_H_warm2_s2` | Hooks punch (no released checkpoint), warm-started from the punch terrain policy `ft32` |

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
                          STEP17_s2/s4 → HARD_s2/s4 ──── → 69,250
                          HARD_s2 → DIVERSE_s2 → 70,250 → HONEST_s2 (ft31 → 71,000, ft33 → 72,250, kept)
released punch model_33000 → PUNCH_s2 ──────────────── → 39,500
    └── PUNCH17 → 41,000 → PUNCH_EARLY → 42,000 → PUNCH_SWING
                           PUNCH_EARLY 41,750 → PUNCH_EARLYLR
        PUNCH_SWING → PUNCH_HONEST → 44,750 → PUNCH_DIVERSE → 46,250
            → ft27 → ft29 → ft30 → 50,000 → ft32 → 51,250 (kept)
                └── Hooks punch: ft34 → 52,500 → ft35
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

**Step failure timing** (10 cm steps, 3 × 200 episodes per run):

| Run | Failed | Falls | Fall time (median) | Falls in 1–2.5 s | Falls with the raised side ahead | Failed by raised side: ahead / left / behind / right |
|---|---|---|---|---|---|---|
| PREVMORE_s2 | 265 | 40 | 1.32 s | 95% | 22 of 40 | 83/160, 66/154, 52/156, 64/130 |
| PREVMORE_s4 | 386 | 26 | 1.92 s | 73% | 7 of 26 | 95/160, 138/155, 86/157, 67/128 |
| STEP17_s2 | 288 | 46 | 1.20 s | 85% | 38 of 46 | 97/160, 94/153, 59/156, 38/131 |
| STEP17_s4 | 298 | 26 | 1.24 s | 85% | 15 of 26 | 83/160, 87/153, 65/155, 63/132 |

- **Falls happen in the same horse-stance weight shift as before**, at a median of about 1.2 s.
- **Falls concentrate on steps up ahead even more than before.** For STEP17_s2, 38 of 46 falls had the raised side ahead.
- **Seed 4's gain comes from the left side:** its failures with the raised side on the left dropped from 138 to 87 of about 155. Non-fall failures (drift) are spread over all directions.
- **The share of episodes ending with the feet on a different level from the reference** stays at 31–35%, and shin contact is as common in completed episodes as in failed ones (46–61%), as before.

**Verdict:** keep the 17 cm step curriculum. The next fix should target the step up ahead (a foothold scan, moving reference landing points). Videos:
- `logs/MotionTracking/videos_step17_s2_it68500/`: released pose, PREVMORE_s2 and STEP17_s2, including new 13.5 and 17 cm step clips.
- `logs/MotionTracking/videos_step17_it68500/`: released pose, STEP17_s2 and STEP17_s4. These are attached to both W&B runs.

### PUNCH_released: released Horse-stance punch, zero-shot (eval only)

`example/pretrained_horse_stance_punch/model_33000.pt`, evaluated with protocol v3 and the ground-relative reference.

**First evaluation, invalid on steps.** It was run without `recenter_motion_xy`. The punch motion starts 1.17 m from the origin, at (−0.42, 1.10), so the robot stood away from the step edge: in 410 of 600 episodes per height the edge was more than 20 cm from both feet. The resulting 78–83% step completion measured flat ground. Results are in `metrics/terrain_eval_v3/`.

**Rerun with the motion recentred** (`metrics/terrain_eval_v3_recenter/`, completion / falls):

| Terrain | Completion / falls |
|---|---|
| Flat | 99.2 / 0.0 |
| Rough | 0.0 / 22.3 |
| Tilt 6° / 12° / 18° | 0.0 / 14.0, 0.0 / 1.7, 0.0 / 5.2 |
| Bumps 2 cm / 4 cm | 9.8 / 24.5, 0.0 / 19.7 |
| Steps 6.5 / 10 / 11.8 / 13.5 / 17 cm | 4.3 / 8.8, 18.8 / 14.7, 20.2 / 12.8, 19.5 / 13.3, 20.3 / 11.5 |

- **Flat error:** 17.3 mm.
- **Steps from 10 cm up complete about 20% of the time**, but with large seed spread (± 9–11 points). The completed episodes are mostly the ones that start with both reference feet on the lower side: 56% of those complete, against 7–12% with one or both feet on the raised side. The 6.5 cm step (a ramp after the trimesh conversion) completes only 4%.
- **Tilt, bumps and rough fail completely**, mostly by drifting, as for the released pose policy.
- **Problem:** `scripts/record_pilot_videos.sh` adds `recenter_motion_xy` only for the released pose (A), so the PUNCH_released clips in `videos_punch_it39500` are not recentred either.

### PUNCH_s2: Horse-stance punch with the full recipe (`ft18_P_preview_s2`)

- **W&B:** `cs47ubpi`.
- **Iterations:** 33,000 → 39,500 (6,500 iterations, twice a pose continuation, because it starts from a flat-ground policy).
- **Warm start:** `logs/warmstart/punch_preview_from_33000.pt` (actor 380 → 749, critic 630 → 1,224). `expand_checkpoint.py` was fixed to accept the released config, which has no `terrain_aware` section.
- **Recipe:** the same as PREVIEW (perception v2, 0.5 s robot-frame preview, fall penalty, 0.7 m cutoff, gentle fine-tuning). The step curriculum tops out at 10 cm (rows 4.4–10 cm), as for PREVIEW; it was set up before STEP17's result.
- **σ:** frozen at the pose run's values, because the released checkpoint doesn't store the adapted σ.
- **Training:**
  - Robots started at levels 0–1, since the released checkpoint has no saved levels.
  - Flat, tilt and bumps reached level 3.5–3.6.
  - The step level rose to 1.9 by iteration 35,500 and then stalled at 2.1–2.2 for the last 4,000 iterations. That is roughly the 7.2 cm row, a ramp after the trimesh conversion.

**Results** (v3, 3 seeds, completion / falls; released recentred):

| Terrain | PUNCH_released | PUNCH_s2 |
|---|---|---|
| Flat | 99.2 / 0.0 | 99.3 / 0.0 |
| Tilt 6° | 0.0 / 14.0 | **97.3** / 0.2 |
| Tilt 12° | 0.0 / 1.7 | **77.8** / 1.7 |
| Tilt 15° | 0.0 / 3.0 | **39.7** / 3.2 |
| Tilt 18° | 0.0 / 5.2 | **12.8** / 4.5 |
| Bumps 4 cm | 0.0 / 19.7 | **89.2** / 0.5 |
| Bumps 6 cm | 0.0 / 10.7 | **63.7** / 0.5 |
| Bumps 8 cm | 0.0 / 9.2 | **37.2** / 1.3 |
| Rough | 0.0 / 22.3 | **34.5** / 0.5 |
| Step 6.5 cm | 4.3 / 8.8 | **24.8** / 4.3 |
| Step 10 cm | **18.8** / 14.7 | 9.8 / 6.2 |
| Step 13.5 cm | **19.5** / 13.3 | 7.3 / 12.8 |
| Step 17 cm | **20.3** / 11.5 | 5.8 / 15.3 |
| Flat error | **17.3 mm** | 25.4 mm |

- **The recipe transfers to a second motion on tilt, bumps and rough.** The gains are as large as for the pose: 0 → 78% on 12° tilt, 0 → 89% on 4 cm bumps, and falls drop to at most 1.3% on bumps and rough.
- **Steps from 10 cm up got worse** (about 19–20% → 6–10%), with a high seed spread (± 5–7 points). On steps 70–84% of episodes end by drifting. Even starts with both reference feet on the lower side fail 78% of the time at 10 cm, against 44% for the released punch.
- **Flat error rose** from 17.3 to 25.4 mm. The pose run's flat error ended near the released pose's (about 20 mm).

**Problems:**
- **The step curriculum never got past the ramp rows**, so the run hardly trained on real step edges. The 10 cm ceiling, starting at level 0, and a σ copied from another motion are the likely causes.
- **One seed only.**

**Verdict:** the terrain recipe generalises to the punch on tilt, bumps and rough. Steps need another run: the 17 cm step curriculum, more iterations, and a second seed. Videos are in `logs/MotionTracking/videos_punch_it39500/` (PUNCH_released not recentred there).

## 2026-11-10

From this day on, runs are sized to be just long enough to judge, watched while they train, and stopped early when they are clearly good or clearly bad. Intermediate checkpoints get a quick eval: 10 and 17 cm steps, 18° tilt, 8 cm bumps and flat, 3 seeds × 200 episodes (protocol v3).

### HARD: tilt and bump curriculum ceilings raised (`ft19_E_hard_s2/s4`)

- **W&B:** `p3en7k75` (s2), `vy3xi8ka` (s4).
- **Iterations:** STEP17_s2/s4 from 68,250, planned to 71,500. Stopped at about 69,650; the checkpoint kept is `model_69250.pt` (1,000 iterations).
- **What changed:** `terrain.curriculum_max_difficulty_by_type={step:2.0,tilt:1.75,bumps:2.0}`, so the top rows are 21° tilt and 8 cm bumps (before: 12° and 4 cm). Steps are unchanged (17 cm).
- **Why:** in STEP17 the tilt, bump and flat levels sat at 3.6–3.7 of 4, so those terrains had nothing harder to learn from.
- **Training:** levels dropped in the first 25 iterations (step 3.3 → 2.1–2.5), then recovered and flattened at step 2.8–2.9, tilt 3.5, bumps 3.6 on the harder rows. Falls in training ended at 0.2–0.8%.

**Results** (v3, completion / falls):

| Terrain | STEP17_s2 | HARD_s2 | STEP17_s4 | HARD_s4 | STEP17 mean | HARD mean |
|---|---|---|---|---|---|---|
| Flat | 100.0 / 0.0 | 99.7 / 0.0 | 99.2 / 0.0 | 99.7 / 0.0 | 99.6 / 0.0 | 99.7 / 0.0 |
| Rough | 53.0 / 0.0 | 64.2 / 0.0 | 57.0 / 0.0 | 79.0 / 0.0 | 55.0 / 0.0 | **71.6** / 0.0 |
| Tilt 6° | 99.0 / 0.0 | 99.0 / 0.0 | 97.3 / 0.0 | 99.2 / 0.0 | 98.2 / 0.0 | 99.1 / 0.0 |
| Tilt 12° | 91.3 / 0.0 | 96.3 / 0.0 | 87.8 / 0.0 | 96.2 / 0.0 | 89.6 / 0.0 | **96.2** / 0.0 |
| Tilt 15° | 72.5 / 0.8 | 87.7 / 0.3 | 72.7 / 0.7 | 90.5 / 0.2 | 72.6 / 0.8 | **89.1** / 0.2 |
| Tilt 18° | 41.7 / 6.3 | 71.7 / 1.2 | 48.0 / 4.3 | 75.0 / 1.8 | 44.8 / 5.3 | **73.3** / 1.5 |
| Tilt 24° | 5.5 / 12.8 | 20.5 / 10.2 | 4.5 / 17.7 | 10.3 / 15.2 | 5.0 / 15.2 | 15.4 / 12.7 |
| Bumps 2 cm | 99.5 / 0.0 | 98.3 / 0.0 | 97.3 / 0.0 | 99.2 / 0.0 | 98.4 / 0.0 | 98.8 / 0.0 |
| Bumps 4 cm | 94.5 / 0.0 | 96.0 / 0.0 | 90.8 / 0.0 | 95.8 / 0.2 | 92.7 / 0.0 | **95.9** / 0.1 |
| Bumps 5 cm | 83.5 / 0.0 | 93.2 / 0.0 | 85.0 / 0.2 | 93.3 / 0.0 | 84.2 / 0.1 | **93.2** / 0.0 |
| Bumps 6 cm | 69.2 / 0.0 | 86.2 / 0.0 | 70.7 / 0.3 | 86.3 / 0.2 | 69.9 / 0.2 | **86.2** / 0.1 |
| Bumps 8 cm | 40.7 / 0.8 | 65.3 / 0.0 | 44.0 / 0.8 | 69.8 / 0.3 | 42.3 / 0.8 | **67.6** / 0.2 |
| Step 6.5 cm | 61.3 / 1.8 | 68.0 / 0.5 | 63.2 / 1.3 | 72.5 / 0.8 | 62.3 / 1.6 | **70.2** / 0.7 |
| Step 10 cm | 51.7 / 7.8 | 59.8 / 1.0 | 49.7 / 4.8 | 56.3 / 6.5 | 50.7 / 6.3 | **58.1** / 3.7 |
| Step 11.8 cm | 49.8 / 9.5 | 56.3 / 0.7 | 43.5 / 7.5 | 49.3 / 10.2 | 46.7 / 8.5 | **52.8** / 5.4 |
| Step 13.5 cm | 39.2 / 13.7 | 48.0 / 2.5 | 33.2 / 13.0 | 41.7 / 10.3 | 36.2 / 13.3 | **44.8** / 6.4 |
| Step 17 cm | 28.5 / 17.7 | 35.3 / 3.5 | 22.7 / 14.5 | 30.0 / 12.5 | 25.6 / 16.1 | **32.7** / 8.0 |
| Flat error | 20.2 mm | 22.2 mm | 21.0 mm | 21.8 mm | 20.6 mm | 22.0 mm |

- **Better on every terrain setting for both seeds**, after only 1,000 iterations. The largest gains are where the ceilings were raised (18° tilt +28 points, 8 cm bumps +25) and on held-out rough ground (+17).
- **Steps improved too** (+6 to +9 points at every height), although the step rows did not change.
- **Falls on steps:** seed 2 dropped to 0.5–3.5% at every height; seed 4 stayed at 6.5–12.5% from 10 cm up.

**Problems:**
- **Flat error rose by 1–2 mm** (20.6 → 22.0 mm).
- **Steps still complete only 58% at 10 cm**, and seed 4 still falls on them.
- **Every restart costs step level.** The step level fell by about 0.7–1.0 in the first 25 iterations of each run started today (the actor is frozen for the first 100), while tilt and bump levels did not, and it took hundreds of iterations to recover. The cause is not found.

**Verdict:** current best pose policy. Its settings are the preset `humanoidverse/config/exp/terrain_tracking.yaml` (`+exp=terrain_tracking`), which reproduces this run's saved config exactly except for the checkpoint and the iteration count (1,000).

### PUNCH17: punch with the 17 cm step curriculum (`ft20_P_step17_s2`)

- **W&B:** `4mygljw5`.
- **Iterations:** PUNCH_s2 from 39,250, planned 6,500; stopped at about 41,075 (last checkpoint 41,000).
- **What changed:** `terrain.curriculum_max_difficulty_by_type={step:2.0}`.

Quick eval at 1,000 iterations (`model_40250.pt`, completion / falls):

| Terrain | PUNCH_s2 | PUNCH17 |
|---|---|---|
| Flat | 99.3 / 0.0 | 99.8 / 0.0 |
| Tilt 18° | 12.8 / 4.5 | 27.0 / 2.3 |
| Bumps 8 cm | 37.2 / 1.3 | 52.0 / 1.2 |
| Step 10 cm | 9.8 / 6.2 | 15.2 / 1.8 |
| Step 17 cm | 5.8 / 15.3 | 6.7 / 7.0 |
| Flat error | 25.4 mm | 24.7 mm |

- **Tilt and bumps improved, steps barely.** The step level stalled at 2.2.
- **Where steps fail:** of the failed 10 cm step episodes, 75% end between 0.5 and 1.5 s and nearly all within 2 s, in every step direction. Even starts with both reference feet on the lower side complete only 27%. That is the opening of the motion, where the right foot steps 0.5 m back into the horse stance.
- **Why the step level was inflated:** training starts episodes anywhere in the motion, so only about 15% include the opening step. Episodes that start in the stance complete easily and keep the level up.

**Verdict:** stopped; not a step fix.

### PUNCH_EARLY and PUNCH_EARLYLR: early starts, and a higher learning rate (`ft21_P_early_s2`, `ft22_P_earlylr_s2`)

- **W&B:** `eg1e09bc`, `kbt64ctg`.
- **PUNCH_EARLY:** PUNCH17 from 41,000 with `early_start.prob=0.5`, `max_time=0.8`, `terrain_types=[step,tilt,bumps]`, demotion on. Stopped at about 42,075.
- **PUNCH_EARLYLR:** PUNCH_EARLY from 41,750 with actor and critic learning rates 1e-4 and 3e-4 (5× and 3×). Stopped at about 42,050.
- **Training (PUNCH_EARLY):** the step level fell from 2.2 to 0.9 at once and rose to 1.3 in 1,000 iterations (rows 5.8 and 8.6 cm). Tilt and bump levels rose slightly (3.6).
- **Quick eval of PUNCH_EARLY at 750 iterations:** step 10 cm 14.5 / 3.2, step 17 cm 8.2 / 3.2, tilt 18° 34.5 / 2.3, bumps 8 cm 52.5 / 1.0, flat error 25.8 mm. No gain on steps over PUNCH17.
- **PUNCH_EARLYLR:** lost 0.7 of a step level at its restart (see HARD) and was at 0.64 after 300 iterations; not evaluated.

**Verdict:** early starts make the curriculum honest (the punch cannot do its opening on steps above about 6–9 cm) but do not teach the step. Both stopped.

### Finding: the punch reference cannot be tracked on steps

Forward kinematics of the two reference motions (ankle height above the ground, flat):

| | Pose | Punch |
|---|---|---|
| Opening step of the right foot | 1.4–2.0 s, about 0.45 m | 1.1–1.8 s, about 0.5 m |
| Peak foot lift | 23 cm | 6 cm |
| Starting stance | feet side by side, 0.24 m apart | feet in line, 0.23 m apart front to back |

- **The punch's stepping foot is only 6 cm off the ground.** On any step above 6 cm, the reference foot passes through the riser.
- **The support height jumps mid-step.** The rigid shift picks the supporting foot with softmax((ground − lift) / τ). A foot lifted 6 cm over a 10 cm upper level scores higher than the stance foot on the lower level, so the whole reference rises by the step height while the robot is in single support. The pose's 23 cm lift keeps its swing foot out until it lands.

### PUNCH_SWING: swing clearance and support-lift gain (`ft23_P_swing_s2`, `ft24_P_swing_long_s2`, `ft24_P_swing_lr_s2`)

- **W&B:** `3abbrdtt`, then `6ob6uxjw` (same settings, continued) and `wnkg3i27` (actor and critic learning rates 1e-4 and 3e-4).
- **What changed:** `swing_clearance=True` (raise a lifted reference foot over higher ground near it) and a new option `terrain_aware.support_lift_gain=4` (the lift counts 4× when choosing the supporting foot; default 1 = unchanged behaviour). Early starts kept.
- **Iterations:** PUNCH_EARLY 42,000 → 43,000 (ft23), then → 43,750 (long, stopped) and → 43,525 (higher rate, stopped).

Quick evals (completion / falls):

| Terrain | PUNCH_EARLY (41,750) | ft23 (42,500) | long (43,500) | higher rate (43,500) |
|---|---|---|---|---|
| Flat | 99.8 / 0.0 | 99.2 / 0.0 | 98.5 / 0.0 | 98.7 / 0.0 |
| Tilt 18° | 34.5 / 2.3 | 50.7 / 2.8 | 40.2 / 1.8 | 33.5 / 3.3 |
| Bumps 8 cm | 52.5 / 1.0 | 56.7 / 0.5 | 56.2 / 2.0 | 55.2 / 1.8 |
| Step 10 cm | 14.5 / 3.2 | 16.5 / 3.5 | 19.5 / 3.0 | 11.8 / 4.5 |
| Step 17 cm | 8.2 / 3.2 | 7.7 / 4.5 | 12.0 / 5.8 | 7.5 / 3.0 |

- **Steps improve slowly** (about 3 points per 1,000 iterations at 10 cm).
- **The higher learning rate is worse** on every terrain.
- **What a failing clip shows** (10 cm step, fails at 1.48 s): the robot lifts the stepping leg far above the reference, leans the torso the other way and topples over the stance foot.
- **Layout breakdown** (ft23, 10 cm steps): 21% completed when no reference foot changes level (the pose: 68%), 10% when one does.
- **Problem:** the training step level read 2.3 while the eval completed 20% at 10 cm. Starts after the opening still promoted robots.

**Verdict:** keep both reference options for the punch; not enough alone.

### PUNCH_HONEST: only early starts move the curriculum level (`ft25_P_honest_s2`)

- **W&B:** `0alyeu9c`.
- **What changed:** new option `early_start.levels_only=True` (on the early-start terrain types, only episodes that start in the opening move a robot up or down), and `early_start.prob` 0.5 → 0.7.
- **Iterations:** long 43,750 → 44,750 (stopped).
- **Training:** the step level fell to 0.55 within 250 iterations (the punch could not do its opening on an 8.6 cm step), then rose to 2.5.
- **Quick eval at 750 iterations:** step 10 cm 24.0 / 0.7, step 17 cm 17.5 / 4.7, tilt 18° 50.7 / 1.7, bumps 8 cm 65.5 / 1.2, flat 99.0 / 0.0 (error 24.9 mm).
- **Problem:** training level 2.5 (about half of the from-start episodes complete at roughly 12 cm) against 24% in the eval at 10 cm.

### Finding: a training run has about two step layouts per level

- A step tile has one edge, with one direction and one offset (±0.15 m from the tile centre), and every episode starts at the tile centre.
- With `num_cols: 8` and 30% step tiles, a run has 2–3 step columns: 2–3 layouts per level and 10–15 in total. Tilt has as few directions.
- The eval builds 40 tiles of the evaluated type per seed.
- So the policy learns its own few layouts. This also explains why each pose training seed has its own weak step direction (STEP17, HARD).

### PUNCH_DIVERSE: more columns and placement jitter (`ft26_P_diverse_s2`)

- **W&B:** `0t93nd7b`.
- **What changed:** `terrain.num_cols=24` (about 7 step columns) and a new option `terrain_aware.placement_jitter=0.15` (the motion is shifted by a random xy offset in ±0.15 m per episode). Everything from PUNCH_HONEST kept. Terrain levels could not be restored (different columns), so robots started at levels 0–1; tilt and bump levels were back at 3.65 within 125 iterations.
- **Iterations:** PUNCH_HONEST 44,750 → 46,250 (1,500; 1.2 s per iteration alone on the GPU).

Quick eval at 750 iterations (`model_45500.pt`, completion / falls):

| Terrain | PUNCH_HONEST (44,500) | PUNCH_DIVERSE (45,500) |
|---|---|---|
| Flat | 99.0 / 0.0 | 99.5 / 0.0 |
| Tilt 18° | 50.7 / 1.7 | 57.7 / 2.8 |
| Bumps 8 cm | 65.5 / 1.2 | 65.0 / 0.5 |
| Step 10 cm | 24.0 / 0.7 | **46.5** / 1.7 |
| Step 17 cm | 17.5 / 4.7 | **24.5** / 3.7 |
| Flat error | 24.9 mm | 24.9 mm |

- **10 cm steps nearly doubled in 750 iterations.** The lack of layout variety was the main bottleneck for the punch on steps.
- **The training step level is now 1.4–1.6** (about 9–10 cm), in line with the eval.

**Verdict:** keep.

### PUNCH_DIVERSE continued (`ft27_P_diverse2_s2`, `ft29_P_diverse3_s2`)

- **W&B:** `mkqq4eb0`, `aumd0bza`.
- **ft27:** PUNCH_DIVERSE 46,250 → 47,500 by the saved counter (1,500 iterations), same settings.
- **ft29:** ft27 → 48,750 (1,500 iterations), with the tilt and bump ceilings of HARD added (`{step:2.0,tilt:1.75,bumps:2.0}`).
- **Training (ft29):** step level 2.0 → 2.8 (about 14 cm, from the motion start); tilt dipped to 3.1 on the harder rows and recovered to 3.6; bumps 3.6. Completed episodes 63 → 70%, falls about 1.5%.

Quick evals (completion / falls):

| Terrain | PUNCH_DIVERSE (45,500) | ft27 (47,000) | ft29 (48,250) |
|---|---|---|---|
| Flat | 99.5 / 0.0 | 99.3 / 0.0 | 99.3 / 0.0 |
| Tilt 18° | 57.7 / 2.8 | 60.5 / 1.3 | **77.8** / 0.8 |
| Bumps 8 cm | 65.0 / 0.5 | 73.5 / 0.2 | **84.8** / 0.0 |
| Step 10 cm | 46.5 / 1.7 | 50.2 / 0.5 | **61.5** / 0.7 |
| Step 17 cm | 24.5 / 3.7 | 25.2 / 2.7 | **38.3** / 1.7 |
| Flat error | 24.9 mm | 24.6 mm | 24.8 mm |

- **The punch now matches the pose on steps** (10 cm: 61.5% against 60.3%) and is ahead of it on 8 cm bumps.
- **Raising the tilt and bump ceilings helped the punch as it helped the pose**, and steps improved in the same run.

**Problems:**
- **Flat error is 24.8 mm** against 17.3 mm for the released punch.
- **One training seed.**

**Verdict:** current best punch policy; still improving, continued as `ft30_P_diverse4_s2` (W&B `qf10wga5`).

### DIVERSE: the pose with more columns and placement jitter (`ft28_E_diverse_s2`)

- **W&B:** `lnplx6kg`. Two launches before it failed on shell quoting and left an empty W&B run (`8a8s5tql`).
- **Iterations:** HARD_s2 69,250 → 70,250 (1,000). Levels restarted at 0–1 (different columns) and were back at step 2.9, tilt 3.4, bumps 3.55 by the end.
- **What changed:** `terrain.num_cols=24`, `terrain_aware.placement_jitter=0.15`. No early starts.

**Results** (v3, completion / falls):

| Terrain | HARD_s2 | DIVERSE_s2 |
|---|---|---|
| Flat | 99.7 / 0.0 | 98.8 / 0.0 |
| Rough | 64.2 / 0.0 | **75.3** / 0.0 |
| Tilt 6° | 99.0 / 0.0 | 98.5 / 0.0 |
| Tilt 12° | 96.3 / 0.0 | 94.0 / 0.0 |
| Tilt 15° | 87.7 / 0.3 | 87.5 / 0.0 |
| Tilt 18° | 71.7 / 1.2 | **80.3** / 0.7 |
| Tilt 24° | 20.5 / 10.2 | **30.5** / 4.8 |
| Bumps 2 cm | 98.3 / 0.0 | 98.7 / 0.0 |
| Bumps 4 cm | 96.0 / 0.0 | 95.3 / 0.0 |
| Bumps 5 cm | 93.2 / 0.0 | 91.2 / 0.0 |
| Bumps 6 cm | 86.2 / 0.0 | 85.2 / 0.0 |
| Bumps 8 cm | 65.3 / 0.0 | 67.5 / 0.0 |
| Step 6.5 cm | 68.0 / 0.5 | **74.5** / 0.3 |
| Step 10 cm | 59.8 / 1.0 | 60.3 / 1.5 |
| Step 11.8 cm | 56.3 / 0.7 | 57.8 / 1.2 |
| Step 13.5 cm | 48.0 / 2.5 | **53.5** / 2.3 |
| Step 17 cm | 35.3 / 3.5 | **43.0** / 4.8 |
| Flat error | 22.2 mm | 22.9 mm |

- **Better at the hard end** (rough +11, tilt 18° +9, tilt 24° +10, steps of 13.5 cm and above +5 to +8), unchanged on 10–12 cm steps, and 1–2 points lower on flat, 12° tilt and 4–6 cm bumps.
- **The quick eval at 500 iterations** read 64.0% on 10 cm steps; the final checkpoint reads 60.3%. Differences of about 4 points between checkpoints of one run are noise.

**Verdict:** new best pose checkpoint by a moderate margin (single seed; from here on changes are tested on seed 2 only, with a second seed kept for a final confirmation). Next: the from-start curriculum that helped the punch (`early_start.prob=0.7`, `max_time=1.0`, `levels_only=True`) in `ft31_E_honest_s2` (W&B `qdieofsw`).

### HONEST: the pose with the from-start curriculum (`ft31_E_honest_s2`, `ft33_E_honest2_s2`)

- **W&B:** `qdieofsw`, `mois3a8e`.
- **What changed:** DIVERSE plus `early_start.prob=0.7`, `max_time=1.0`, `terrain_types=[step,tilt,bumps]`, `levels_only=True`. No swing clearance or support-lift gain (the pose lifts its foot 23 cm).
- **Iterations:** ft31: DIVERSE_s2 70,250 → 71,000 by the saved counter (1,000 iterations). ft33: ft31 → 72,250 (1,500 iterations), same settings.
- **Training:** step level 3.0 → 2.65 → 3.2 in ft31 and 3.2 → 2.75 → 2.95 in ft33 (the usual dip after a restart); tilt 3.5, bumps 3.55–3.6. ft33 was flattening by its end: drift terminations 22 → 19–20%, falls about 1%.

**Results of ft31** (v3 full sweep, `model_71000.pt`, completion / falls):

| Terrain | HARD_s2 | DIVERSE_s2 | HONEST_s2 (ft31) |
|---|---|---|---|
| Flat | 99.7 / 0.0 | 98.8 / 0.0 | 99.7 / 0.0 |
| Rough | 64.2 / 0.0 | 75.3 / 0.0 | **80.5** / 0.0 |
| Tilt 6° | 99.0 / 0.0 | 98.5 / 0.0 | 99.0 / 0.0 |
| Tilt 12° | 96.3 / 0.0 | 94.0 / 0.0 | **98.3** / 0.0 |
| Tilt 15° | 87.7 / 0.3 | 87.5 / 0.0 | **94.0** / 0.0 |
| Tilt 18° | 71.7 / 1.2 | 80.3 / 0.7 | **85.8** / 0.0 |
| Tilt 24° | 20.5 / 10.2 | 30.5 / 4.8 | **43.3** / 5.2 |
| Bumps 2 cm | 98.3 / 0.0 | 98.7 / 0.0 | 99.0 / 0.0 |
| Bumps 4 cm | 96.0 / 0.0 | 95.3 / 0.0 | 96.7 / 0.0 |
| Bumps 5 cm | 93.2 / 0.0 | 91.2 / 0.0 | 93.5 / 0.0 |
| Bumps 6 cm | 86.2 / 0.0 | 85.2 / 0.0 | **91.7** / 0.0 |
| Bumps 8 cm | 65.3 / 0.0 | 67.5 / 0.0 | **82.0** / 0.0 |
| Step 6.5 cm | 68.0 / 0.5 | 74.5 / 0.3 | **79.7** / 0.2 |
| Step 10 cm | 59.8 / 1.0 | 60.3 / 1.5 | **64.5** / 0.8 |
| Step 11.8 cm | 56.3 / 0.7 | 57.8 / 1.2 | 60.2 / 1.5 |
| Step 13.5 cm | 48.0 / 2.5 | 53.5 / 2.3 | 56.5 / 1.5 |
| Step 17 cm | 35.3 / 3.5 | 43.0 / 4.8 | **49.8** / 3.5 |
| Flat error | 22.2 mm | 22.9 mm | 23.1 mm |

- **Best or level on all 17 settings.** The largest gains over DIVERSE are bumps 8 cm (+14.5), tilt 24° (+13), tilt 15° (+6.5) and 17 cm steps (+7).
- **Step failures are drift, not falls** (10 cm: 34.7% drift, 0.8% falls). They are worst with the raised side on the robot's left (48% not completed at 10 cm, against 24% on the right) and when one reference foot starts on the upper level (48%, against 17% with none).

**Quick eval of ft33** (`model_72250.pt`, completion / falls; ft31 column from its full sweep):

| Terrain | ft31 (71,000) | ft33 (72,250) |
|---|---|---|
| Flat | 99.7 / 0.0 | 99.7 / 0.0 |
| Tilt 18° | 85.8 / 0.0 | **90.7** / 0.0 |
| Bumps 8 cm | 82.0 / 0.0 | 82.5 / 0.3 |
| Step 10 cm | 64.5 / 0.8 | **69.0** / 0.7 |
| Step 17 cm | 49.8 / 3.5 | **55.2** / 0.8 |
| Flat error | 23.1 mm | **21.5 mm** |

**Problems:**
- **One training seed.**
- **Steps:** a third of 10 cm episodes still lose the motion in the opening step.

**Full sweep of ft33** (`model_72250.pt`, completion / falls): flat 99.7 / 0.0; rough 83.5 / 0.0; tilt 6°, 12°, 15°, 18°, 24°: 99.2 / 0.0, 98.0 / 0.0, 94.7 / 0.0, 90.7 / 0.0, 42.5 / 5.0; bumps 2, 4, 5, 6, 8 cm: 99.8 / 0.0, 98.2 / 0.0, 94.8 / 0.0, 92.3 / 0.0, 82.5 / 0.3; steps 6.5, 10, 11.8, 13.5, 17 cm: 81.7 / 0.0, 69.0 / 0.7, 66.3 / 0.8, 66.0 / 0.7, 55.2 / 0.8; flat error 21.5 mm. Against ft31, steps are 2–9.5 points higher (13.5 cm +9.5), tilt 18° +5, rough +3; tilt 24° is level.

**Verdict:** the recipe is the current best for the pose and is now the preset `humanoidverse/config/exp/terrain_tracking.yaml` (composed and compared with the ft31 config: identical apart from checkpoint and name). ft33 is the best pose checkpoint. Videos: `logs/MotionTracking/videos_honest_it72250/` (released pose against ft33, 10 terrains, on W&B `mois3a8e`).

### PUNCH_DIVERSE continued, rounds 4 and 5 (`ft30_P_diverse4_s2`, `ft32_P_diverse5_s2`)

- **W&B:** `qf10wga5`, `511nkndo`.
- **Settings:** as ft29 (24 columns, placement jitter 0.15, early starts 0.7 / 0.8 s / levels only, swing clearance, support-lift gain 4, ceilings `{step:2.0,tilt:1.75,bumps:2.0}`).
- **Iterations:** ft30: ft29 48,500 → 50,000 by the saved counter (1,500). ft32: ft30 → 51,250 (1,500).
- **Training:** step level 2.8 → 3.0 in ft30 and 3.0 → 2.8 → 3.26 in ft32, still rising at the end; tilt 3.55, bumps 3.65. Drift terminations 31 → 24% in ft32, falls about 1%.

Quick evals (completion / falls):

| Terrain | ft29 (48,250) | ft30 (49,250) | ft32 (51,250) |
|---|---|---|---|
| Flat | 99.3 / 0.0 | 99.7 / 0.0 | 100.0 / 0.0 |
| Tilt 18° | 77.8 / 0.8 | 85.8 / 0.3 | **88.8** / 0.0 |
| Bumps 8 cm | 84.8 / 0.0 | 86.5 / 0.0 | **90.3** / 0.0 |
| Step 10 cm | 61.5 / 0.7 | 63.7 / 0.3 | **69.2** / 0.3 |
| Step 17 cm | 38.3 / 1.7 | 42.3 / 0.7 | **53.8** / 0.8 |
| Flat error | 24.8 mm | 24.0 mm | 25.2 mm |

- **Every terrain improved in both rounds**, and falls are below 1% everywhere tested.
- **Failures at 10 cm** (ft30) are spread evenly over the step layouts, in the opening step (1.0–2.5 s) and the return step (after 4 s); almost all are drift.

**Problems:**
- **Flat error 25 mm** against 17.3 mm for the released punch; it has not improved in any round.
- **One training seed.**

**Full sweeps** (completion / falls):

| Terrain | PUNCH_s2 (ft18) | ft30 (50,000) | ft32 (51,250) |
|---|---|---|---|
| Flat | 99.3 / 0.0 | 99.2 / 0.0 | 100.0 / 0.0 |
| Rough | 34.5 / 0.5 | 88.3 / 0.2 | **92.0** / 0.0 |
| Tilt 6° | 97.3 / 0.2 | 98.8 / 0.0 | 99.7 / 0.0 |
| Tilt 12° | 77.8 / 1.7 | 97.2 / 0.0 | 97.7 / 0.0 |
| Tilt 15° | 39.7 / 3.2 | 94.0 / 0.2 | 96.8 / 0.0 |
| Tilt 18° | 12.8 / 4.5 | 85.3 / 0.3 | **88.8** / 0.0 |
| Tilt 24° | 0.7 / 6.0 | 36.0 / 4.7 | **42.7** / 5.0 |
| Bumps 2 cm | 97.5 / 0.2 | 99.3 / 0.0 | 99.2 / 0.0 |
| Bumps 4 cm | 89.2 / 0.5 | 99.0 / 0.0 | 98.5 / 0.0 |
| Bumps 5 cm | 80.5 / 0.7 | 96.3 / 0.0 | 98.3 / 0.0 |
| Bumps 6 cm | 63.7 / 0.5 | 95.0 / 0.0 | 96.3 / 0.0 |
| Bumps 8 cm | 37.2 / 1.3 | 86.7 / 0.2 | **90.3** / 0.0 |
| Step 6.5 cm | 24.8 / 4.3 | 76.7 / 0.0 | **80.7** / 0.0 |
| Step 10 cm | 9.8 / 6.2 | 66.8 / 0.8 | 69.2 / 0.3 |
| Step 11.8 cm | 9.0 / 6.8 | 62.0 / 0.5 | 63.7 / 0.5 |
| Step 13.5 cm | 7.3 / 12.8 | 55.5 / 0.5 | **61.5** / 0.5 |
| Step 17 cm | 5.8 / 15.3 | 48.5 / 1.0 | **53.8** / 0.8 |
| Flat error | 25.4 mm | 24.4 mm | 25.2 mm |

Videos: `logs/MotionTracking/videos_punchbest_it51250/` (released punch, recentred, against ft32, 10 terrains, on W&B `511nkndo`).

**Verdict:** ft32 is the current best punch checkpoint. The punch and the pose are now level (10 cm steps 69%, 17 cm steps 54–55%, tilt 18° 89–91%, bumps 8 cm 83–90%). Next: the motions without a released checkpoint, Hooks punch first.

### HOOKS: first motion without a released checkpoint (`ft34_H_warm_s2`)

- **W&B:** `88j78nrz`.
- **Motion:** `Hooks_punch.pkl`, 5.8 s. The feet slide apart to about 0.6 m during the first second and back during 5.0–5.8 s, lifting at most 5 cm; the pelvis drops from 0.80 to 0.67 m.
- **Start:** the punch terrain policy `ft32` `model_51250.pt` (same robot and observations, a similar stance), not a from-scratch run.
- **Settings:** the `terrain_tracking` preset with the punch's reference options (`swing_clearance=True`, `support_lift_gain=4`) and `early_start.max_time=1.2`, but with real learning in place of the gentle fine-tune: `schedule=adaptive` from actor 1e-4 / critic 1e-3, a fresh optimizer, `entropy_coef=0.005`, free noise std, adaptive tracking σ restarted from the PBHC defaults, terrain levels restarted at 0–1 (`algo.config.restore_terrain_levels=False`). The 100-iteration actor warm-up is kept.
- **Iterations:** 1,500 (saved counter 51,000 → 52,500).
- **Training:** levels fell to 0.1–0.2 during the warm-up, then climbed fast: step 2.4, tilt 3.2, bumps 3.5 at 250 iterations; step 3.36, tilt 3.64, bumps 3.62 at the end. Drift terminations 82 → 13%, falls 0.2%. Noise std rose from 0.185 to 0.30. The adaptive σ were still tightening (upper body 0.015 → 0.0095, feet 0.010 → 0.0027).

Quick eval (`model_52500.pt`, completion / falls / root-relative body error):

| Terrain | HOOKS (52,500) |
|---|---|
| Flat | 99.2 / 0.0 / 64.4 mm |
| Tilt 18° | 92.8 / 0.0 / 75.3 mm |
| Bumps 8 cm | 88.3 / 0.0 / 71.3 mm |
| Step 10 cm | 72.2 / 0.0 / 75.8 mm |
| Step 17 cm | 61.7 / 0.0 / 83.0 mm |

- **The warm start works:** a new motion completes on every terrain, with no falls, after 1,500 iterations (about 40 minutes).
- **Tracking is loose:** 64 mm on flat ground, against 22–25 mm for the pose and the punch. With the 0.7 m drift cutoff a motion that barely travels can complete while tracked poorly, so completion alone overstates this policy.

**Problems:** no flat-ground baseline exists for this motion (no released checkpoint), so the error can only be compared with the other motions. One seed.

**Verdict:** keep; continued as `ft35_H_warm2_s2` (W&B `bvxw1dsi`): 3,000 more iterations from `model_52500.pt`, optimizer state, learning rates, terrain levels and σ values resumed, no warm-up.

## 2026-11-11

### HOOKS continued (`ft35_H_warm2_s2`)

- **W&B:** `bvxw1dsi`.
- **Iterations:** ft34 52,500 → 55,250 by the saved counter (3,000), same settings, state resumed.
- **Training:** step level 3.3 → 3.56, tilt 3.67, bumps 3.66; drift terminations 17 → 6%, falls 0.3%. Noise std peaked at 0.313 and fell to 0.26. Adaptive σ still tightening at the end (upper body 0.0095 → 0.0042, feet 0.0027 → 0.0019).

Quick evals (completion / falls / root-relative body error):

| Terrain | ft34 (52,500) | ft35 mid (53,750) | ft35 (55,250) |
|---|---|---|---|
| Flat | 99.2 / 0.0 / 64.4 mm | 99.5 / 0.0 / 50.0 mm | 99.8 / 0.0 / **41.1 mm** |
| Tilt 18° | 92.8 / 0.0 / 75.3 mm | 95.5 / 0.0 / 64.5 mm | 96.3 / 0.0 / 57.4 mm |
| Bumps 8 cm | 88.3 / 0.0 / 71.3 mm | 93.3 / 0.0 / 57.0 mm | 92.8 / 0.3 / 48.7 mm |
| Step 10 cm | 72.2 / 0.0 / 75.8 mm | 75.3 / 0.2 / 62.8 mm | 76.3 / 0.7 / 55.3 mm |
| Step 17 cm | 61.7 / 0.0 / 83.0 mm | 62.2 / 0.2 / 72.9 mm | 60.7 / 0.8 / 65.9 mm |

- **Flat error 64 → 41 mm** in 3,000 iterations, about 0.6–1 mm per 100 iterations and not yet flat. Completion is level or up.

**Problems:** still nearly twice the error of the pose and punch (22–25 mm).

**Verdict:** keep; continued as `ft37_H_warm3_s2` (W&B `6343p3vc`, 3,000 more iterations).

### SIDEKICK: first try, as the Hooks punch (`ft36_K_side_s2`)

- **W&B:** `05g1nafr`.
- **Motion:** `Side_kick.pkl`, 6.0 s; the kick is at about 1.5–3.5 s, with the foot up to 1.3 m high and the root moving 0.45 m.
- **Settings:** as HOOKS ft34 (warm start from the punch `ft32`), without swing clearance and support-lift gain, `early_start.max_time=1.5`. 1,500 iterations.
- **Result: failed.** Terrain levels went to 0 and stayed there; 76% of training episodes ended by drift and 4% by falls. Quick eval: 0% completed on every setting, flat included, with episodes ending at 37% of the motion (about 2.2 s, the kick). The upper-body σ tightened to 0.0013 on the standing part before the kick.
- **Cause (likely):** the fixed 0.7 m drift cutoff. A robot that does not yet kick has its foot more than 0.7 m from the reference foot (1.3 m up), so every attempt is cut off at the kick and there is no gradual path to learning it. PBHC's own from-scratch training starts this cutoff at 1.5 m.

**Verdict:** drop these settings for kicks. Retry as `ft38_K_side_far15_s2` (W&B `2xoeimi5`): from `ft36`, 2,000 iterations, drift cutoff fixed at 1.5 m in training (evaluation stays at 0.7 m), σ restarted from the defaults.

### SIDEKICK retry: drift cutoff fixed at 1.5 m (`ft38_K_side_far15_s2`)

- **W&B:** `2xoeimi5`. From `ft36` `model_52500.pt`, settings as `ft36` with `termination_motion_far_threshold=1.5` in training. Stopped at 1,150 of 2,000 iterations.
- **Training:** episodes now get through the kick: step level 0 → 2.57, tilt 3.6; drift terminations 40 → 5%, falls 48 → 16%. The feet σ never tightened (0.0099, the default).
- **Quick eval at 1,000 iterations** (`model_53250.pt`, cutoff 0.7 m): 0% completed on every setting, 0–0.7% falls. All 200 flat episodes of eval seed 0 ended by drift at a median 2.04 s, and in 199 the furthest body was the right ankle, 0.52 m below the reference: the robot stays upright through the kick but does not raise the kicking foot.

**Verdict:** confirms the cause of the `ft36` failure (the cutoff, not balance), but a loose fixed cutoff lets the robot skip the kick. Stopped. Next: `ft39_K_side_farcur_s2` (W&B `ltf7jaxr`), from `model_53250.pt`, 2,500 iterations, with PBHC's own cutoff curriculum from 1.5 m down to 0.7 m (`terminate_when_motion_far_curriculum=True`, `initial_threshold=1.5`, `threshold_min=0.7`; about 1,300 iterations to reach 0.7 m at the default rate), state and σ resumed.

### HOOKS continued again (`ft37_H_warm3_s2`)

- **W&B:** `6343p3vc`. From ft35 `model_55250.pt`, 3,000 iterations (saved counter 55,000 → 58,000), state and σ resumed, with `swing_clearance=True`, `support_lift_gain=4` and `early_start.max_time=1.2` added.
- **Training:** levels unchanged (step 3.56 → 3.57, tilt 3.67, bumps 3.65); drift terminations 3.5–5.8%, falls 0.2%. σ still tightening (upper body 0.0042 → 0.0032, feet 0.0019 → 0.0015). Noise std 0.26 → 0.25.

Quick eval (`model_58000.pt`, completion / falls / root-relative body error):

| Terrain | ft35 (55,250) | ft37 (58,000) |
|---|---|---|
| Flat | 99.8 / 0.0 / 41.1 mm | 100.0 / 0.0 / **37.8 mm** |
| Tilt 18° | 96.3 / 0.0 / 57.4 mm | 98.5 / 0.2 / 51.3 mm |
| Bumps 8 cm | 92.8 / 0.3 / 48.7 mm | 97.0 / 0.0 / 43.0 mm |
| Step 10 cm | 76.3 / 0.7 / 55.3 mm | **80.0** / 0.2 / 49.8 mm |
| Step 17 cm | 60.7 / 0.8 / 65.9 mm | **65.7** / 0.7 / 60.7 mm |

- **Completion up on every terrain;** steps are now above the pose and the punch (69% at 10 cm).
- **Error falls slowly:** 41 → 38 mm on flat ground in 3,000 iterations (64 → 41 in the 3,000 before).

**Problems:** flat error still about 1.5× the pose and punch (22–25 mm), and the gain per iteration is shrinking. The three added settings were not tested separately. One seed.

**Verdict:** keep as the Hooks punch best. Paused here to work on the kicks.

### SIDEKICK third try: cutoff curriculum 1.5 → 0.7 m (`ft39_K_side_farcur_s2`)

- **W&B:** `ltf7jaxr`. From ft38 `model_53250.pt`, 2,500 iterations (53,250 → 55,750), state and σ resumed, `terminate_when_motion_far_curriculum=True`, `initial_threshold=1.5`, `threshold_min=0.7`.
- **Training:** the cutoff reached 0.7 m after about 1,350 iterations and stayed there. Levels dropped while it tightened (step 2.8 → 1.4, tilt 3.5 → 2.4) and recovered (step 2.64, tilt 3.38 at the end). Drift terminations 22%, falls 4.7% at the end. Feet σ never tightened (0.0098); upper body 0.0048.

Quick eval (`model_55750.pt`, cutoff 0.7 m, completion / falls / root-relative body error):

| Terrain | ft38 (53,250) | ft39 (55,750) |
|---|---|---|
| Flat | 0 | 97.2 / 0.0 / 58.6 mm |
| Tilt 18° | 0 | 82.8 / 0.5 / 69.0 mm |
| Bumps 8 cm | 0 | 87.5 / 0.2 / 61.7 mm |
| Step 10 cm | 0 | 53.5 / 1.0 / 65.8 mm |
| Step 17 cm | 0 | 40.5 / 1.2 / 72.0 mm |

- **The cutoff curriculum works:** from 0% to 97% on flat ground at the 0.7 m cutoff.
- **The kick is only partly performed.** Flat ground, eval seed 0, tighter cutoffs: 9.5% complete at 0.5 m and 0% at 0.35 m. At 0.5 m the episodes end at a median 2.0 s (the kick) and the furthest body is the right (kicking) ankle in 104 of 181, the left ankle in 42. So at the kick some body is typically 0.5–0.7 m from the reference: the robot raises the leg just enough to stay inside the cutoff.

**Problems:** completion at 0.7 m overstates this policy; a 1.3 m kick allows a large shortfall inside 0.7 m. One seed.

**Verdict:** keep as a starting point. `ft40_K_side_cont_s2` (W&B `0hib2pas`, same settings with the cutoff fixed at 0.7 m) was started and stopped after 8 minutes, before the tighter-cutoff check, in favour of:

- `ft42_K_side_far05_s2`: from ft39 `model_55750.pt`, 2,500 iterations, cutoff curriculum continued from 0.7 m down to 0.5 m in training (evaluation stays at 0.7 m, plus a flat check at 0.5 m).
- `ft41_R_round_s2` (W&B `fn0ifq01`): Roundhouse kick, warm-started from the side kick ft39 `model_55750.pt`, 3,000 iterations, settings as ft34 plus `early_start.max_time=1.5` and the cutoff curriculum 1.5 → 0.7 m.

### SIDEKICK: training cutoff tightened to 0.5 m (`ft42_K_side_far05_s2`)

- **W&B:** `gbi1b9fs`. From ft39 `model_55750.pt`, 2,500 iterations (saved counter 55,500 → 58,000), cutoff curriculum `initial_threshold=0.7`, `threshold_min=0.5`, state and σ resumed.
- **Training:** the cutoff reached 0.5 m after about 600 iterations. Step level 2.63 → 1.10 → 1.81, tilt 3.40 → 2.66 → 3.33. Drift terminations 25%, falls 4.8% at the end. Feet σ started to tighten (0.0097 → 0.0082).

Quick eval (`model_58000.pt`, cutoff 0.7 m, completion / falls / root-relative body error):

| Terrain | ft39 (55,750) | ft42 (58,000) |
|---|---|---|
| Flat | 97.2 / 0.0 / 58.6 mm | 97.7 / 0.0 / **48.8 mm** |
| Flat, cutoff 0.5 m (seed 0) | 9.5 | **97.0** |
| Tilt 18° | 82.8 / 0.5 / 69.0 mm | 90.3 / 0.7 / 60.4 mm |
| Bumps 8 cm | 87.5 / 0.2 / 61.7 mm | 89.8 / 2.5 / 52.7 mm |
| Step 10 cm | 53.5 / 1.0 / 65.8 mm | 54.8 / **6.3** / 59.1 mm |
| Step 17 cm | 40.5 / 1.2 / 72.0 mm | 39.8 / **7.7** / 66.4 mm |

- **The kick is higher:** 97% of flat episodes now stay within 0.5 m of the reference (9.5% before), and body error fell about 10 mm on every terrain.
- **Falls on steps rose** from about 1% to 6–8%; completion on steps is unchanged.

**Problems:** more falls on steps and bumps. The training step level is lower than before (1.8 against 2.6). One seed.

**Verdict:** keep; tightening the cutoff is what raises the kick. Continued as `ft43_K_side_far035_s2` (W&B `4ckpey5t`): 2,500 iterations, cutoff 0.5 → 0.35 m.

### ROUNDHOUSE first try (`ft41_R_round_s2`)

- **W&B:** `fn0ifq01`. `Roundhouse_kick.pkl` (5.3 s), warm-started from the side kick ft39 `model_55750.pt`, 3,000 iterations (55,500 → 58,500), settings as ft34 plus `early_start.max_time=1.5` and the cutoff curriculum 1.5 → 0.7 m.
- **Training:** cutoff at 0.7 m after about 1,500 iterations. Step level 2.45, tilt 3.38 at the end; drift terminations 23%, falls 0.7%. Feet σ never tightened (0.0099); upper body 0.0107.

Quick eval (`model_58500.pt`, completion / falls / root-relative body error):

| Terrain | ROUNDHOUSE (58,500) |
|---|---|
| Flat | 100.0 / 0.0 / 75.4 mm |
| Flat, cutoff 0.5 m (seed 0) | 0.0 |
| Tilt 18° | 90.3 / 0.3 / 86.3 mm |
| Bumps 8 cm | 95.8 / 0.0 / 80.7 mm |
| Step 10 cm | 66.7 / 0.0 / 83.0 mm |
| Step 17 cm | 47.3 / 0.3 / 86.3 mm |

- **Completes on every terrain with almost no falls** on the first try with the cutoff curriculum.
- **Loosely tracked, as the side kick was at this stage:** 75 mm flat error and 0% at a 0.5 m cutoff.

**Problems:** completion at 0.7 m overstates it. One seed.

**Verdict:** keep; continued as `ft44_R_round_far04_s2` (W&B `b0yd4gb4`): 2,500 iterations, cutoff 0.7 → 0.4 m.

### SIDEKICK: training cutoff tightened to 0.35 m (`ft43_K_side_far035_s2`)

- **W&B:** `4ckpey5t`. From ft42 `model_58000.pt`, 2,500 iterations (saved counter 57,750 → 60,250), cutoff curriculum 0.5 → 0.35 m, state and σ resumed.
- **Training:** cutoff at 0.35 m after about 700 iterations. Step level 1.75 → 0.78 → 1.29, tilt 3.28 → 2.20 → 2.86. Drift terminations 32%, falls 4.4% at the end. σ tightened (feet 0.0081 → 0.0040, upper body 0.0044 → 0.0024).

Quick eval (`model_60250.pt`, cutoff 0.7 m, completion / falls / root-relative body error):

| Terrain | ft39 (cutoff → 0.7) | ft42 (→ 0.5) | ft43 (→ 0.35) |
|---|---|---|---|
| Flat | 97.2 / 0.0 / 58.6 mm | 97.7 / 0.0 / 48.8 mm | **99.2** / 0.2 / **40.0 mm** |
| Flat, cutoff 0.5 m (seed 0) | 9.5 | 97.0 | 99.5 |
| Flat, cutoff 0.35 m (seed 0) | 0.0 | – | **98.5** |
| Tilt 18° | 82.8 / 0.5 / 69.0 mm | 90.3 / 0.7 / 60.4 mm | 84.8 / 4.5 / 51.6 mm |
| Bumps 8 cm | 87.5 / 0.2 / 61.7 mm | 89.8 / 2.5 / 52.7 mm | 85.3 / 5.5 / 44.6 mm |
| Step 10 cm | 53.5 / 1.0 / 65.8 mm | 54.8 / 6.3 / 59.1 mm | 51.3 / **13.2** / 52.6 mm |
| Step 17 cm | 40.5 / 1.2 / 72.0 mm | 39.8 / 7.7 / 66.4 mm | 34.8 / **14.0** / 59.7 mm |

- **Flat ground: the kick is now tracked within 0.35 m in 98.5% of episodes,** and body error fell to 40 mm.
- **Terrain got worse:** falls rose on every terrain (steps 13–14%, tilt and bumps about 5%) and completion fell 3–6 points. The tighter cutoff holds the training terrain levels down (step 1.3 against 2.6 for ft39), so the policy practises less hard terrain.

**Problems:** a trade between kick accuracy and terrain robustness that this schedule does not resolve. One seed.

**Verdict:** two candidates, no single best: `ft42` for terrain (fewer falls), `ft43` for the kick itself. Side kick work stopped here at the user's request (2026-11-11). Untested idea: tighten the cutoff only on flat and easy rows, or let the terrain levels recover with longer training at 0.35 m.

### ROUNDHOUSE: training cutoff tightened to 0.4 m (`ft44_R_round_far04_s2`)

- **W&B:** `b0yd4gb4`. From ft41 `model_58500.pt`, 2,500 iterations (58,250 → 60,750), cutoff curriculum 0.7 → 0.4 m, state and σ resumed.
- **Result: failed.** As the cutoff passed about 0.5 m the terrain levels fell to 0 (step 2.52 → 0, tilt 3.41 → 0) and never recovered; 77% of training episodes ended by drift. Quick eval of `model_60750.pt`: 0% completed on every setting, flat included, with episodes ending at about 35% of the motion (the kick). The σ collapsed onto the part before the kick (feet 0.0098 → 0.0012), as in the first side-kick try (ft36).
- **Cause (likely):** the cutoff tightened faster than the robot learned to raise the leg. The side kick went 0.7 → 0.5 → 0.35 in two runs with 0% → 97% at 0.5 m in between; the roundhouse started from 0% at 0.5 m and was pushed to 0.4 m in about 800 iterations.

**Verdict:** drop this run. The Roundhouse kick best stays `ft41` `model_58500.pt` (completes on every terrain, loosely tracked: 75 mm). Roundhouse work stopped here at the user's request. If resumed: tighten in smaller steps (0.7 → 0.55 first) or slow the curriculum rate.

### New-motion work stopped (2026-11-11)

The user redirected the work to the defense maneuvers on terrain. State of the motions without a released checkpoint:

| Motion | Best checkpoint | Flat | Step 10 cm | Flat error | Note |
|---|---|---|---|---|---|
| Hooks punch | ft37 `model_58000.pt` | 100.0 | 80.0 | 37.8 mm | completes everywhere, falls < 1% |
| Side kick | ft42 `model_58000.pt` (terrain) / ft43 `model_60250.pt` (kick) | 97.7 / 99.2 | 54.8 / 51.3 | 48.8 / 40.0 mm | needs the cutoff curriculum; falls on steps 6% / 13% |
| Roundhouse kick | ft41 `model_58500.pt` | 100.0 | 66.7 | 75.4 mm | kick only loosely tracked |
| Charleston dance, Bruce Lee pose | – | – | – | – | not started |

### Defense on terrain: start (branch `terrain-aware-defense`)

PR #2 (akhilghar's `defense-maneuvers`) is merged into `terrain-aware-tracking`; the terrain work for defense is on the new branch `terrain-aware-defense`. Design: [DEFENSE_MANEUVERS.md](DEFENSE_MANEUVERS.md).

**Dataset.** The 480 attack clips are not committed. `generate_attacks.py --per_attack 60 --seed 0` fails in the `pbhc` env: MuJoCo 3.2.3 reports larger self-penetration than the version the dataset was made with (16.2 mm against 12.2 mm for `body_shot_0000`, limit 15 mm), so every body shot is rejected. Generated in a separate conda env `attackgen` (Python 3.12, MuJoCo 4.0.0, NumPy 1.26): 469 clips equal the committed `index.json`, 11 were resampled (crosses, rear hooks, rear uppercuts). The committed `index.json` is unchanged; the env reads the labels from the clips. `defense_check.py bank / policy / env` pass, and the do-nothing baseline equals the document's (54% hit, 46% blocked).

### DEF-FLAT: first full flat-ground defense run (`def01_flat_s2`)

- **W&B:** `yvdaw6rw` (project `Defense`). The command of DEFENSE_MANEUVERS.md Section 7 with seed 2, 4,096 envs. Stopped at 3,000 of 20,000 iterations (about 65 minutes) because it was already good.
- **Training:** episode length 15 → 937 steps (of 1,000); drift terminations 99% → 21% at 2,000 iterations; falls < 1%; noise std 0.80 → 0.70.

`defense_eval.py`, 256 episodes of 20 s, eval seed 0:

| | Do-nothing guard | 2,000 it | 3,000 it |
|---|---|---|---|
| Hit / blocked / evaded | 54 / 46 / 0 | 0.1 / 85.7 / 14.2 | 0.0 / 88.6 / 11.4 |
| Survived 20 s | – | 99.2 | 100.0 |
| Guard body error | – | 85.8 mm | 56.2 mm |
| Hit, attacker hidden (`--blind freeze`) | – | 15.3 | 32.7 |
| Hit, blocks disabled (`block_radius=0`) | – | 4.0 | – |
| Belief class accuracy during the strike | – | 85% | 86% |

- **Hits go to zero within 2,000 iterations,** with no falls and no attacks interrupted by drift.
- **The policy uses what it sees more as training goes on:** hiding the attacker raises the hit rate to 15% at 2,000 iterations and 33% at 3,000 (calf kick 87% at 2,000).
- **Part of the defense is a posture, not a reaction:** with blocks disabled 96% of attacks still miss at 2,000 iterations, and the guard is held 56–86 mm off the reference.

**Problems:** one training seed, one eval seed. The geometric block rule (7 cm around the forearm axis) is easy to satisfy; no physical contact.

**Verdict:** keep as the flat-ground baseline and warm start. `model_2000.pt` was used for the terrain run because it was the checkpoint available when that run started.

### Terrain code for defense

- `envs/defense/defense.py`: the attack clips stand on the support height of the defender's guard reference (terrain under its feet) when `ground_relative_ref` is on, updated every step; unchanged on flat ground. Terrain curriculum for a held guard: up one row after a survived episode, down after an episode that ended before half time. The attacker keypoint history is refilled on the first step after a reset.
- `config/obs/defense/main_terrain.yaml`: defense observations + height scan (11 × 11), foot scans, privileged scan for the critic. Actor 978, critic 1,470 inputs.
- `config/exp/defense_terrain.yaml`: preset (defense env, `terrain_kungfu`, ground-relative guard, re-centring, jitter 0.15 m, 24 columns, ceilings `{step: 2.0, tilt: 1.75, bumps: 2.0}`, graduate rows 2).
- `scripts/expand_checkpoint.py`: handles the attack-conditioned actor (first layer of its policy MLP; the encoder is unchanged).
- `scripts/defense_eval.py`: `--terrain {flat,step,tilt,bumps} --difficulty D`.
- Check: the expanded flat policy evaluated in the terrain env on flat tiles gives 98.0% survival and 0.1% hits, as in its own env.

### DEF-TERRAIN: first terrain defense run (`def02_terrain_s2`, running)

- **W&B:** `p9ni5abm`. `+exp=defense_terrain`, warm-started from DEF-FLAT `model_2000.pt` expanded to the terrain observations, optimizer state loaded, PPO settings as the flat run, up to 6,000 iterations (saved counter 2,000 → 8,000).
- **Training, first 1,150 iterations:** levels step 0.17 → 2.53, tilt 0.04 → 2.44, bumps 0.33 → 3.36; drift terminations 63 → 45%; falls 1.4%.

`defense_eval.py --terrain`, 256 episodes of 20 s, eval seed 0 (survived / hit rate of resolved attacks / guard body error):

| Terrain | Flat policy (0 it of terrain) | 500 it | 1,000 it | 2,500 it |
|---|---|---|---|---|
| Flat | 98.0 / 0.1 / 88 mm | 96.9 / 0.3 / 120 mm | 98.8 / 0.2 / 94 mm | 99.6 / 0.1 / 70 mm |
| Steps 10 cm | 26.2 / 0.1 / 99 mm | 55.1 / 1.0 / 167 mm | 79.7 / 0.7 / 149 mm | **82.8** / 0.3 / 123 mm |
| Steps 17 cm | 28.9 / 0.4 / 102 mm | 29.7 / 1.6 / 193 mm | 63.7 / 1.8 / 178 mm | **81.2** / 0.8 / 148 mm |
| Tilt 18° | 0.0 / 28.6 / 160 mm | 26.6 / 1.4 / 227 mm | 42.2 / 2.6 / 212 mm | **94.5** / 1.4 / 153 mm |
| Bumps 8 cm | 7.4 / 1.0 / 159 mm | 74.2 / 0.7 / 178 mm | 83.2 / 0.1 / 158 mm | **96.5** / 0.2 / 118 mm |

At 2,500 iterations falls are 5.1% on 10 cm steps, 3.5% on 17 cm steps and 0% elsewhere; training levels are step 3.36, tilt 3.48, bumps 3.60, with drift terminations down to 24%.

Videos of `model_3500.pt` (1,500 it of terrain training), 20 s each: `logs/Defense/videos_def02_it3500/` (flat, steps 10 and 17 cm, tilt 18°, bumps 8 cm; no hits in any clip). `scripts/record_video.py` now keeps a defense run's own episode ends and puts the hit / block / evade counts in the caption.

- **Survival on terrain rises fast** and hits stay below 3% everywhere; the episodes that end, end by drift from the guard (falls 0–4%).
- **The guard is held loosely on terrain** (150–210 mm error).

Final eval, `model_5000.pt` (3,000 it of terrain training): flat 100.0 / 0.0 / 63 mm, steps 10 cm 82.8 / 0.4 / 114 mm, steps 17 cm 82.0 / 1.3 / 140 mm, tilt 18° 98.8 / 0.4 / 126 mm, bumps 8 cm 94.1 / 0.1 / 115 mm; falls 3.1% and 2.7% on steps, 0 elsewhere. Steps did not improve after 2,500 iterations.

**Problems:** steps level off at about 82% survival. One training seed, one eval seed. The policy defends only; it never attacks back (the user's objection, which led to the counterattack work below).

**Verdict:** keep as the terrain defense baseline and the warm start for the counterattacks. Stopped at 3,175 of 6,000 iterations to free the GPU. Videos are in the run on W&B (`videos/…`).

### Counterattacks: design and code

The user's requirement (2026-11-11): the defender must not only dodge; it must counterattack with the kung-fu motions trained on terrain, after each defended attack.

- **Clips** (`scripts/make_defense_counters.py`): Horse-stance punch, Hooks punch, Side kick, Roundhouse kick, each turned and moved to start at the guard's root position and heading, with a 0.6 s blend from the guard and a 0.8 s blend back. 6.7–8.1 s each. Strikes: right fist at 3.50 s (0.97 m high) and 2.87 s (1.16 m), right foot at 2.97 s (1.32 m), left foot at 2.63 s (1.20 m).
- **Env** (`envs/defense/defense.py`, `counter_bank.py`, `env.config.defense.counter`): after a blocked or evaded attack, with probability 0.7, the defender's reference switches from the held guard to a clip and back at its end. The attacker holds its guard and moves (over 0.6 s) to where the clip's strike lands; the target is a point on its pelvis, torso or head vertical at the strike height. The counter lands when the striking hand or foot passes within 0.15 m of the target within ±0.4 s of the strike time. The guard keeps a 0.6 m drift cutoff; the clips use the env's cutoff (0.7 m for the punches).
- **Observation** `counter_command` (6): one-hot of the clip, phase, active flag. **Rewards:** `defense_counter_hit` +5 per landed counter, `defense_counter_proximity` up to +0.1 per step around the strike.
- **Preset** `+exp=defense_terrain_counter`; `expand_checkpoint.py` adds zero-initialised critic heads for new reward terms; `defense_eval.py` and `record_video.py` report counters.
- **Check:** with the reference tracked exactly the striker passes 5–7 cm (mean) from the target for the punches and the side kick on terrain; the roundhouse check gave 18 cm with outliers (not resolved).
- **Limit:** the clips are not turned to face the attacker; the ghost attacker is moved to the strike point.

### DEF-COUNTER stage 1: the two punches (`def03_counter_punch_s2`, running)

- **W&B:** `jogyxnn3`. `+exp=defense_terrain_counter`, clips `[horse_stance_punch, hooks_punch]`, warm-started from DEF-TERRAIN `model_5000.pt` (new inputs and value heads zero), 4,096 envs, about 3.9 s per iteration.
- **Training, first 1,150 iterations:** terrain levels fell when the counters started (step 2.25 → 0.95) and recovered (step 2.80, tilt 3.21, bumps 3.46). Counters that land, of those performed to the end: 0 → 97%. Drift terminations 95 → 35%, falls 2.8%, hit rate 2.3%.

`defense_eval.py --terrain`, 256 episodes of 20 s, eval seed 0. Survived / hit rate / counters landed, of those started:

| Terrain | 500 it | 1,000 it | 2,000 it |
|---|---|---|---|
| Flat | 91.0 / 0.1 / 90.3 | 98.0 / 0.0 / 88.9 | 97.3 / 0.0 / **90.9** |
| Steps 10 cm | 68.0 / 2.7 / 73.2 | 78.9 / 1.8 / 83.1 | 77.3 / 1.0 / **84.3** |
| Steps 17 cm | 60.9 / 3.4 / 65.1 | 68.0 / 3.1 / 75.7 | 74.6 / 2.0 / **87.1** |
| Tilt 18° | 56.6 / 1.6 / 34.9 | 74.2 / 0.6 / 60.4 | 92.2 / 0.1 / **87.3** |
| Bumps 8 cm | 78.5 / 0.4 / 75.4 | 89.1 / 0.3 / 81.2 | 95.3 / 0.1 / **83.3** |

At 2,000 iterations falls are 6.6% and 3.9% on steps and 0 elsewhere; per clip the Hooks punch lands 86–94% and the Horse-stance punch 80–88%. Training levels: step 3.31, tilt 3.49, bumps 3.55; drift terminations 21%.

- **The counters land** on every terrain after 1,000 iterations, 60–89% of those started, and both punches are similar (Hooks a few points higher).
- **Survival is below the terrain-only policy** (steps 79 and 68% against 83 and 82%; tilt 74% against 99%) but still rising.
- About 550 counters and 700 attacks per 256 episodes: a counter takes 7–8 s, so there are fewer attacks per episode than without counters (about 2,700).

**Problems:** steps remain the weakest terrain (75–77% survival, 4–7% falls). The attacker is moved to the strike point, the clip is not aimed. One training seed, one eval seed.

**Verdict:** keep; stopped at 2,075 of 6,000 iterations (`model_7000.pt`) to start the kicks. Videos (side camera): `logs/Defense/videos_def03_it7000/`, logged into the run on W&B.

### DEF-COUNTER stage 2: punches and kicks (`def04_counter_kicks_s2`, running)

- **W&B:** `g4ht9wtr` (two earlier launches failed on a W&B start timeout; `wiemqe8a` is an empty run from one of them). From stage 1 `model_7000.pt`, up to 5,000 iterations.
- **Settings:** clips `[horse_stance_punch, hooks_punch, side_kick ×2, roundhouse_kick ×2]` (kicks sampled twice as often); the kicks' drift cutoff follows PBHC's curriculum from 1.5 m to 0.7 m at a slower rate (`curriculum_degree=1.5e-5`, about 2,100 iterations), the punches keep 0.7 m (`counter.clip_far_threshold`, new) and the guard 0.6 m.
- **Stopped at 350 iterations** (cutoff 1.5 → 1.34 m, kicks 0% landed, punches 97%) when the user reported three problems in the stage 1 videos; the design below replaces the one this run used.

### Rework after the stage 1 videos: attacker on the ground, counters aimed at it

Problems the user saw in the `def03` videos: (1) the red attacker lifts into the air so that the defender's move reaches it; (2) only the reference dots reach the attacker, not the robot itself; (3) the attacker's feet slide.

Causes: the attacker's height followed the ground under the defender, and it was moved to where the counter clip's strike lands (1, 3); the standoff step-in and every new clip's placement translate the whole ghost (3); the counter was scored against a point placed where the reference strikes, within 15 cm (2).

Changes (`envs/defense/defense.py`, `attack_bank.py`, `counter_bank.py`):

- **Attacker feet (`defense.attacker_feet`, on by default).** Each foot stays where it stands while its clip has it in contact, on the terrain under it; the pelvis and upper body are placed as before (clip, standoff, defender position), and the legs shear between them, so the step-in of the standoff becomes a lunge over planted feet. A foot farther than 0.3 m from where the clip has it is re-planted with a lifted step of 0.3 s; a foot the clip lifts (calf kick) follows the clip. Upper-body height: the attacker's own ground, lowered by up to 0.2 m or raised by up to 0.05 m towards the defender's ground so its strikes stay aimed where its legs allow.
- **The attacker is no longer moved for counters.** The counter clip is aimed: turned about its start and shifted along the strike direction so the strike reaches the attacker's trunk, eased in over the clip's blend from the guard and out at its end (limits 100° and 0.4 m). New observation `counter_aim` (yaw, xy shift).
- **Landing is judged on the robot and the attacker's body:** the defender's own striking hand or foot must pass within 0.12 m of the attacker's trunk axis (pelvis to the top of the head) within ±0.4 s of the strike.
- **Clips rebuilt** with 1.5 s blends in and out (8.3–9.7 s), because aiming needs a pivot: the guard is bladed 25° and the motions strike 27–49° to the right of their start heading, so the aim yaw is 40–100° (mean 68°). Counter episodes are 30 s.
- **Checks (128 envs, terrain curriculum tiles):** planted feet move 0.0 mm per step; a planted foot's height above its ground varies by at most 20 mm; with the clip tracked exactly the striker passes 4–7 cm (mean; 90th percentile 8–12 cm) from the trunk axis.

**Limits:** the attacker's legs are sheared, not solved by inverse kinematics (the viewer turns the thigh and shin links to follow). The attacker still cannot be touched. The reference feet of the defender slide during the pivot; the robot has to step.

### DEF-AIM stage 1: aimed counter punches (`def05_aim_punch_s2`, running)

- **W&B:** `ukdg2kn6`. `+exp=defense_terrain_counter` with the rework, punches only, from DEF-COUNTER stage 1 `model_7000.pt` (`counter_aim` inputs zero), up to 5,000 iterations, 30 s episodes.


### DEF-AIM stage 1 stopped, counters moved farther apart (2026-11-11, 10:45 PM)

`def05_aim_punch_s2` (W&B `ukdg2kn6`) stopped at iteration 10,850 (3,850 into the run); last checkpoint `model_10500.pt`.
Survival / attacker hits / counters landed at `model_10000.pt` (30 s episodes): flat 100 / 1.3 / 89.8, step 10 cm
69.9 / 2.8 / 75.0, step 17 cm 69.9 / 3.0 / 76.2, tilt 18° 91.8 / 1.4 / 90.2, bumps 8 cm 98.4 / 1.3 / 86.8.

User feedback on the `model_10000.pt` videos: only the yellow reference dots appear to hit, not the robot's hands;
suggested standing farther away. Measured on flat (`scratchpad/counter_diag.py`): the guard keeps the two pelvises
0.78 m apart, but the aimed clip walked the defender in by 0.33 m, so at the strike the pelvises were 0.45–0.47 m apart
and the fist ended 3–5 cm from the attacker's trunk axis, i.e. inside the ghost. The fist itself did extend (0.41–0.46 m
from the defender's root, reference 0.43–0.49 m), but the two bodies overlapped and hid it.

Changes:
- `counter.aim_depth` 0 → 0.13 m: the strike is aimed at the trunk's surface, pelvises 0.56–0.58 m apart at the strike.
- `counter.hit_radius` 0.12 → 0.17 m (surface of the trunk plus the fist), and new `counter.min_extension: 0.8`: a
  counter lands only if the striker is also at least 80% of the clip's reach away from the defender's own root.
- `defense_counter_proximity` no longer rewards going deeper than `aim_depth`.
- `env.config.draw_reference_markers=False` hides the yellow reference dots (used for the videos).

`def06_apart_punch_s2` (W&B `ofsvbg44`): from `logs/warmstart/def05_10500_apart.pt`, punches only, 2,000 iterations.

### Kung-fu counters paused; counters are now the dataset strikes (2026-11-11, 11:50 PM)

`def06_apart_punch_s2` (W&B `ofsvbg44`) stopped at about iteration 11,450 (950 into the run; last checkpoint
`model_11000.pt`). At `model_11000.pt` (30 s episodes) survival / attacker hits / counters landed: flat 100 / 0.6 / 88.0,
step 10 cm 75.4 / 1.4 / 76.7, step 17 cm 76.6 / 2.0 / 73.5, tilt 18° 86.3 / 1.3 / 85.9, bumps 8 cm 98.8 / 0.4 / 86.5.
With the farther aim the pelvises are 0.55–0.59 m apart at the punch and the fist ends 12–13 cm from the attacker's
centre line. Check videos (no reference dots, white rings on the fist when a counter lands):
`logs/Defense/videos_def06_it11000/`.

User request: use the eight strikes of the attack dataset as the counters for now, instead of the kung-fu motions.

- `scripts/make_defense_counters.py` also builds one counter clip per attack class (jab, cross, lead_hook, rear_hook,
  lead_uppercut, rear_uppercut, body_shot, calf_kick): the class's clip with the median striking distance, 0.3 s blends
  from and to the guard (the dataset clips already start and end in it, feet planted). 2.0–3.3 s per clip.
- Strikers `left_shin` / `right_shin` (75 % from knee to ankle) and a `leg` target for the calf kick: it is aimed at the
  attacker's nearer shin and lands within `counter.leg_hit_radius` (0.12 m) of the attacker's thigh or shin segments.
- `+exp=defense_terrain_strikes`, obs `defense/main_terrain_strikes` (`counter_command` 10 = 8 clips + phase + flag).
- Aim check (`scratchpad/strike_geom.py`, 256 envs): aim yaw −22…25° (the kung-fu clips needed 40–100°), shift towards
  the attacker 0.08–0.26 m (mean per clip), reference fist 11–14 cm from the trunk axis at the strike (calf kick 5.6 cm
  from the leg), reference pelvises 0.58–0.67 m apart.

`def07_strikes_s2` (W&B `6869b32v`): from DEF-TERRAIN `model_5000.pt` expanded to the new inputs
(`logs/warmstart/def02_5000_strikes.pt`), all eight strikes, up to 3,000 iterations (labels 5,000 → 8,000).

### DEF-STRIKES stage 1 (`def07_strikes_s2`, stopped at 1,200 it) and restart (`def08_strikes_ease_s2`) (2026-11-12, 2:00 AM)

`def07_strikes_s2` (W&B `6869b32v`; videos of `model_6000.pt` logged into the run). 30 s episodes, 256 episodes each:

| Terrain | Survived 500 → 1,000 it | Drift at 1,000 it | Attacker hits | Counters landed 500 → 1,000 it |
|---|---|---|---|---|
| Flat | 91.8 → 78.9% | 21.1% | 0.0% | 45.4 → 83.8% |
| Step 10 cm | 53.1 → 40.2% | 57.0% | 0.5% | 43.5 → 81.1% |
| Step 17 cm | 44.1 → 32.4% | 66.8% | 0.9% | 36.3 → 76.0% |
| Tilt 18° | 43.4 → 29.7% | 70.3% | 0.9% | 43.2 → 76.8% |
| Bumps 8 cm | 73.0 → 68.0% | 32.0% | 0.2% | 45.0 → 82.4% |

- The seven hand strikes land 80–97% on every terrain at 1,000 it (the rear-hand ones only after 500 it).
- The calf kick lands 0%: the robot does not throw it; its shin stays 55 cm from the attacker's leg (`scratchpad/calf_diag.py`).
- Survival fell with training: drift terminations. Two causes found. (1) The aim (0.2–0.3 m towards the attacker) was
  eased in over the clip's 0.3 s blend, a move no robot can follow; (2) a robot that does not kick is about 0.5 m from
  the reference leg, close to the 0.7 m cutoff, so calf-kick counters ended episodes.

Changes for `def08_strikes_ease_s2` (W&B `l0ybapr8`, from `def07` `model_6000.pt`, up to 3,000 it):
- The aim of a dataset strike eases in from the clip's start to 0.1 s before the strike and out from 0.3 s after it to
  the clip's end (`counter_meta.aim_span`); the kung-fu clips keep the easing over their blends.
- Drift cutoff: the seven hand strikes keep 0.7 m (`counter.clip_far_threshold`); the calf kick follows the PBHC
  curriculum 1.5 → 0.7 m (degree 1.5e-5).
- `counter.leg_proximity_sigma: 0.3` (0.15 for the trunk): the proximity reward reaches a kick that is not thrown yet.

### `def08_strikes_ease_s2` stopped at 1,175 it; `def09_strikes_kick_s2` (2026-11-12, 4:00 AM)

`def08_strikes_ease_s2` (W&B `l0ybapr8`; videos of `model_7000.pt` logged into the run). 30 s episodes:

| Terrain | Survived 500 → 1,000 it | Drift at 1,000 it | Attacker hits | Counters landed at 1,000 it |
|---|---|---|---|---|
| Flat | 97.7 → 99.6% | 0.4% | 0.0% | 82.8% |
| Step 10 cm | 64.8 → 69.9% | 27.7% | 1.0% | 82.0% |
| Step 17 cm | 53.5 → 64.8% | 33.6% | 0.8% | 77.0% |
| Tilt 18° | 26.6 → 75.4% | 24.6% | 1.0% | 76.1% |
| Bumps 8 cm | 86.7 → 95.3% | 4.7% | 0.0% | 82.7% |

The gentler aim and the calf-kick cutoff curriculum restored survival. The calf kick still lands 0%: the robot keeps
its right foot down (shin 57 cm from the attacker's leg, extension 0.18 m of the clip's 0.33 m).

`def09_strikes_kick_s2` (W&B `142e0ih8`, from `def08` `model_7000.pt`, up to 3,000 it): the calf kick is drawn three
times as often (3 of 10 counters) and `counter.leg_reward_scale: 3.0` triples its hit and proximity rewards.

### 2026-11-12 — DEF-STRIKES: def09 results (calf kick lands), re-anchoring, def10
- `def09_strikes_kick_s2` (W&B `142e0ih8`, from def08 `model_7000`, calf kick sampled 3x, `leg_reward_scale` 3.0), stopped at ~8,350.
  `model_8000`, 256 episodes of 30 s, survived / attacker hits / counters landed (calf kick landed):
  flat 81.6 / 1.6 / 87.0 (92.1), steps 10 cm 42.6 / 2.1 / 84.7 (83.4), steps 17 cm 28.9 / 1.4 / 76.8 (64.5),
  tilt 18 deg 37.9 / 1.0 / 75.9 (78.1), bumps 8 cm 66.4 / 0.7 / 85.8 (92.5). The calf kick now lands (0 % before), but
  survival fell again: drift 18–59 %.
- Diagnosis (scratchpad `drift_diag.py`, `retreat_diag.py`): the robot follows the step-in (root error ≈ 0 at the strike)
  but does not step back: at the end of every clip its root is +0.15…0.22 m towards the attacker (p90 0.3 m) while the
  reference has eased back to the guard's place. Drift ends cluster at the clip's end (0.7 m cutoff) and in the first
  second of the guard after it (0.6 m cutoff); calf kick +0.3 m forward during the whole clip.
- Fix `counter.reanchor` (0.45 m per counter, `reanchor_total` 1.0 m per episode): the aim's shift no longer eases out
  (`CounterBank.hold_shift`), and when the clip ends the guard reference moves to the robot's root
  (`_placement_offset`). The attacker's next clip is placed relative to the robot anyway. Without retraining, def09
  `model_8000` with it: steps 10 cm 42.6 → 60.9 % survival, tilt 18 deg 37.9 → 68.0 %.
- Videos of def09 `model_8000` (recorded with re-anchoring) are in `logs/Defense/videos_def09_it8000/` and in run `142e0ih8`.
- `def10_strikes_reanchor_s2` (W&B `hi21mtti`) continues from def09 `model_8000` with `reanchor: 0.45`.

### 2026-11-12 — DEF-STRIKES: def10 results (re-anchoring), def11
- `def10_strikes_reanchor_s2` (W&B `hi21mtti`, from def09 `model_8000`, `reanchor: 0.45`), stopped at ~10,200.
  `model_10000`, 256 episodes of 30 s, survived / attacker hits / counters landed (calf kick landed):
  flat 99.2 / 0.0 / 92.9 (95.7), steps 10 cm 71.9 / 0.6 / 92.2 (92.8), steps 17 cm 60.5 / 0.6 / 87.4 (84.6),
  tilt 18 deg 85.5 / 1.0 / 83.4 (86.1), bumps 8 cm 98.4 / 0.0 / 88.9 (95.2).
  Earlier checkpoints, survival flat/step10/step17/tilt/bumps: 8500 87.5/60.2/47.3/61.3/88.3; 9000 97.7/71.9/52.3/64.1/93.4;
  9500 91.8/69.5/60.9/55.9/94.5.
- Diagnosis at `model_10000` (scratchpad `drift_diag2.py`): on 10 cm steps the drift ends are falls in progress (head
  farthest, pelvis 9–16 cm low, trunk tipping), median 4.6 s into the episode, half during a counter. On tilt the one
  drift end seen had reached `reanchor_total` 1.0 m (every counter moves robot and attacker ~0.2 m on).
- Videos of `model_10000` (flat, 10 cm steps, 18 deg tilt): `logs/Defense/videos_def10_it10000/`, logged into `hi21mtti`.
- `def11_strikes_steps_s2` (W&B `3hh6zfuw`) continues from def10 `model_10000` with `reanchor_total: 1.5` and
  terrain proportions flat/bumps/tilt/step 0.1/0.2/0.3/0.4 (was 0.1/0.3/0.3/0.3).

### 2026-11-12 — Defense videos were filmed on easy curriculum rows (fixed)
- `scripts/record_video.py` left `terrain.curriculum=True` of the defense exps on; the curriculum builds its own rows and
  ignores `terrain.fixed_difficulty`, and robots start on rows 0–1. So every defense terrain video up to def10's first
  set was on ~6–9 cm steps / 4–8 deg tilt, not the labelled 10 cm / 18 deg (measured slope under the robot 3.7–8.1 deg;
  17.4–17.7 deg with the curriculum off). The scratchpad diagnostics (`drift_diag*.py`, `retreat_diag.py`) had the same
  problem. `scripts/defense_eval.py` sets `terrain.curriculum=False`, so the evaluation numbers are at the stated difficulty.
- Fix: `record_video.py` sets `terrain.curriculum=False` when a fixed difficulty is asked for, and draws the height scan
  (`terrain_aware.debug_draw_height_scan`) by default. The step video is filmed with `--cam-offset -1.0 -2.2 0.7` so the
  step face is towards the camera.
- def10 `model_10000` re-recorded: `videos_def10_it10000/def10_10000_{step,tilt}_fixed.mp4`, logged into `hi21mtti`
  (same keys, newer step). 10 cm steps: one drift end at 2.7 s, 4 of 5 counters land; 18 deg tilt: no episode end, 4 of 6 land.

### 2026-11-12 — DEF-STRIKES: def11 results, def12
- `def11_strikes_steps_s2` (W&B `3hh6zfuw`, from def10 `model_10000`, steps 40 % of tiles, `reanchor_total` 1.5) ran its
  3,000 iterations. Its final weights (~13,000) were saved as `model_12500.pt` (overwriting the 12,500 checkpoint);
  copy: `logs/warmstart/def11_13000_strikes.pt`.
  Final, 256 episodes of 30 s, survived / attacker hits / counters landed: flat 100.0 / 0.1 / 96.6, steps 10 cm
  82.0 / 0.5 / 94.8, steps 17 cm 71.9 / 0.9 / 91.8, tilt 18 deg 88.7 / 1.3 / 88.9, bumps 8 cm 99.2 / 0.6 / 95.2.
  Survival flat/step10/step17/tilt/bumps: 10500 100/73.0/64.8/87.1/99.2; 11000 100/78.5/66.8/78.1/99.6;
  12000 100/80.1/71.5/79.3/99.2.
- Diagnosis on real 17 cm steps (`drift_diag2.py` with `terrain.curriculum=False`): drift ends at 1.3 / 2.5 / 8.3 s
  (quartiles), 123 of 194 in guard, pelvis ~10 cm low, head farthest: the start of the episode on the edge, not the step-in.
- Videos (flat, 10 and 17 cm steps, 18 deg tilt, 8 cm bumps; height scan; real difficulty): `logs/Defense/videos_def11_final/`,
  logged into `3hh6zfuw`. 17 cm steps: one drift end at 2.6 s, then 6 counters land; no episode end on the others.
- `def12_strikes_steps_s2` (W&B `7qff6nph`) continues from the def11 final weights, same settings (labels start at 12500).

### 2026-11-12 — DEF-STRIKES: def12 plateau, guard follows after every attack (def13)
- `def12_strikes_steps_s2` (W&B `7qff6nph`, from def11 final, same settings; labels start at 12500), stopped at ~14,875.
  Survived / attacker hits / counters landed, file `model_13500`: flat 99.6/0.4/96.3, step10 78.9/1.0/95.8,
  step17 66.8/1.3/94.0, tilt 77.0/2.1/92.4, bumps 97.3/0.3/95.6; file `model_14500`: flat 100/0.7/98.0,
  step10 83.6/1.8/95.5, step17 69.1/1.2/93.5, tilt 91.4/1.7/88.9, bumps 99.6/1.0/96.0. Steps plateau (noise ±3–4 points).
  No videos recorded for def12 (no change over def11).
- Checks on `model_14500` (all at the real difficulty):
  - drift cutoff off (`termination.terminate_when_motion_far=False`): step17 78.5 % survive, 21.5 % fall; step10 84.4 / 15.6.
    The drift ends on steps are real falls.
  - `start_diag.py`, step17: 19 % of episode starts end within 4 s; 25 % with a foot on the edge, 11 % feet on different
    levels, 17 % both on one level; 4 % of envs hold a third of the early ends.
  - `counter.prob=0`: step17 survival 32.8 %, drift 66.8 %, falls 0.4 %, guard error 250 mm: without counters nothing
    re-anchors the guard and the robot wanders off it.
- New `counter.reanchor_attacks`: the guard also moves to the robot when an attack ends (`_reanchor`). Without retraining:
  step17 69.1 → 75.4, step10 83.6 → 80.5 (noise), tilt 91.4 → 97.3.
- `def13_strikes_follow_s2` (W&B `b0jpzuex`) continues from def12 `model_14500` (copy `logs/warmstart/def12_14500_strikes.pt`)
  with `reanchor_attacks: True`, 2,000 iterations.

### 2026-11-12 — DEF-STRIKES: def13 (guard follows after every attack) stopped early
- `def13_strikes_follow_s2` (W&B `b0jpzuex`, labels start at 14500) stopped at label ~15,625: training with the guard
  following got worse after the first 500 iterations. Survived / fall / drift, 256 episodes of 30 s:
  - file `model_15000`: flat 99.6/0.4/0.0, step10 84.8/1.6/13.7, step17 78.9/2.7/18.4, tilt 93.8/1.6/4.7, bumps 99.6/0.0/0.4.
  - file `model_15500`: flat 94.5/0.0/5.5, step10 78.1/3.5/18.4, step17 76.6/2.3/21.1, tilt 78.5/3.9/17.6, bumps 98.0/0.0/2.0.
- Check on `model_15500` with `counter.reanchor_total=100` (no cap on the follow): flat 100 % (the 5.5 % drift was the
  1.5 m cap running out: with the guard following, nothing rewards staying in place, so the policy wanders);
  tilt 80.9 % (16.4 % drift remains: more than 0.45 m of slip within one attack).
- Best so far: def13 file `model_15000` (or def12 `model_14500` evaluated with `reanchor_attacks=True`: tilt 97.3).
  Videos of `model_15000`: `logs/Defense/videos_def13_it15000/`.
