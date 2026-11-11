# What We Built, Explained From Scratch

This file explains everything this project added to PBHC, assuming you know nothing about robots, simulation or machine learning. It starts with the basic ideas and builds up. Every number in it comes from our own runs; the full record is in [LOG.md](LOG.md), and the shorter technical summary is in [TERRAIN_AWARE_TRACKING.md](TERRAIN_AWARE_TRACKING.md).

If a word is unfamiliar, check the glossary at the end.

---

## Part 1: The basics

### 1.1 The robot and the motion

The robot is a **Unitree G1**, a humanoid about 1.3 m tall with 23 motors (joints) we control: hips, knees, ankles, waist, shoulders and elbows. The wrists are locked.

The robot's job is to copy a **recorded human motion**. The motion we use most is the **Horse-stance pose**, a kung-fu move about 7 seconds long:

1. The robot starts standing with its feet close together.
2. Around 1–2 seconds in, it shifts its weight onto its left foot.
3. It lifts its right foot (up to 28 cm high), swings it about 50 cm out to the side, and puts it down in a wide stance.
4. It drops its hips into a deep squat (the pelvis goes from 0.80 m down to 0.57 m above the ground) and holds the pose.

The motion was recorded from a person and converted ("retargeted") to the robot's body. The result is a list of poses, 50 per second, saying where every body part should be at each moment. We call this the **reference motion**, or just **the reference**.

A good mental picture: imagine a **ghost robot** doing the move perfectly. The real robot has to keep its own body overlapped with the ghost as closely as it can. In the simulator's viewer, the ghost is actually drawn as small coloured balls at each reference body position.

**One important fact:** the motion was recorded on a **flat floor**. The ghost's feet are always at the height of a flat floor.

### 1.2 The simulator

We never use a real robot. Everything happens in **Isaac Gym**, a physics simulator that runs on the graphics card (GPU). It simulates gravity, contact with the ground, motor torques and so on.

The simulator can run **thousands of robots at once**, each on its own small patch of ground. Our training runs use **1,536 robots in parallel**. They don't see each other; it is just a way to collect experience 1,536 times faster.

The robot's "brain" makes a decision **50 times per second** (every 0.02 s). Each decision is called a **step**. A 7-second motion is therefore 350 steps.

### 1.3 How the robot learns: reinforcement learning

Nobody programs the robot's movements by hand. Instead it **learns by trial and error**, a method called **reinforcement learning (RL)**. Here is the whole idea in plain terms:

- **The policy** is the robot's brain: a neural network (a big adjustable mathematical function). Every step it receives numbers describing the situation and outputs 23 numbers, one target angle per motor.
- **Observations** are the numbers the policy receives: what the robot can "sense". For example its joint angles, how fast it is rotating, which way is down.
- **Actions** are the 23 numbers it outputs.
- **Reward** is a score given after every step: high if the robot did well, low if it did badly. The policy never sees the reward directly; training uses it to adjust the policy.
- **An episode** is one attempt: from the moment the robot is placed at the start of the motion until the motion ends or the attempt is cut off.
- **Termination** is cutting an attempt off early, for example because the robot fell over. The robot then restarts.
- **Training** repeats this millions of times. After each batch of experience (called an **iteration**), the neural network is nudged so that actions that led to high reward become more likely. The algorithm that does the nudging is called **PPO**.

Think of training a dog with treats, except the "dog" makes 50 decisions a second, there are 1,536 of them, and the treat is a number.

### 1.4 The actor and the critic

PPO uses **two** neural networks:

- **The actor** is the policy. It decides the actions. It is what would run on a real robot, so it may only use information a real robot could have.
- **The critic** is a coach used **only during training**. It looks at the situation and predicts how much reward the robot will collect from here on. That prediction helps training tell good decisions from lucky ones.

Because the critic is thrown away after training, it is allowed to see **secret information** that a real robot wouldn't have, such as the exact positions of the ghost's body parts, the exact friction of the floor, or a perfect map of the ground. This is called **privileged information**. Giving the critic privileged information makes training better without making the final robot depend on it.

### 1.5 Motion tracking: how the reward works

The task of following a reference motion is called **motion tracking**. The reward is mostly made of **tracking terms**, each measuring how close one aspect of the robot is to the ghost:

- body **positions** (upper body, lower body, feet, head and hands);
- body **rotations**;
- **joint angles** and joint speeds;
- body **velocities**;
- whether each foot is **touching the ground** when the ghost's foot is.

Each term turns an error into a score with the formula `score = exp(−error / σ)`. Zero error gives a score of 1; large errors give scores close to 0. The number **σ** (sigma) sets how strict the term is: a small σ means even a small error loses most of the score.

There are also **penalties**: for jerky movements, very high motor torques, feet slipping and so on.

**Two ways an episode can end early:**

- **Fall:** the robot's body tilts too far (the simulator checks which way gravity points relative to the pelvis).
- **Drift:** any body part gets more than a set distance from the ghost's matching body part. The robot is "lost". In evaluation we use **0.7 m**.

Each early ending also costs a large one-time penalty (−200 in PBHC).

### 1.6 What PBHC (KungfuBot) is

**PBHC** is the published method we started from (the paper calls it KungfuBot, NeurIPS 2025). It is very good at teaching a humanoid fast, difficult motions like kung-fu and dance **on flat ground**. Its main ideas:

- **Adaptive σ:** it automatically tightens the tracking scores (σ) as the robot gets better.
- **A multi-head critic:** the critic predicts each reward term separately.
- **A shrinking drift limit:** the allowed distance from the ghost starts loose and tightens to 0.3 m during training.
- **One policy per motion,** trained from scratch for about 27 hours.

The authors released trained policies ("checkpoints") for the Horse-stance pose and Horse-stance punch. We start from those.

**What PBHC's actor sees** (all from its own body): its joint angles and joint speeds, how fast its body is rotating, which way gravity points relative to its pelvis, its last actions, the last 4 steps of all of these, and the **motion phase** (a number that says how far through the motion it is: 0 at the start, 1 at the end).

Notice what is missing: **it cannot see the ground at all, and it cannot see the ghost.** It has learned the motion so well on flat ground that it just "knows" what to do at each phase, like a dancer who has memorised a routine on a stage.

The paper itself states its first limitation:

> It lacks environment awareness, such as terrain perception and obstacle avoidance, which restricts deployment in unstructured real-world settings.

That limitation is what this project attacks.

### 1.7 A few more terms we use

- **Checkpoint:** a saved copy of the neural networks at some iteration, so training can continue later or the policy can be tested. The released pose checkpoint is from iteration 50,000.
- **Training from scratch** starts with a random network. **Fine-tuning** starts from an existing trained network and continues training it on a new situation. We fine-tune, because learning the motion itself from scratch takes far longer than we can afford (our 2-hour from-scratch pilots couldn't even do the motion on flat ground).
- **Training seed:** the random number that decides all the random choices in a run. Two runs that differ only in seed can end up noticeably different. That's why the main comparisons up to early November were run with **two or three seeds**. The latest runs (Parts 10 and 11) use one seed, to try more ideas in the same time; the final recipe still needs confirming on a second seed.
- **Curriculum:** starting with easy tasks and making them harder as the robot improves, like school grades.

---

## Part 2: Why uneven ground breaks PBHC

Put the released PBHC policy on uneven ground and it fails almost every time. Starting from the beginning of the motion, out of 600 attempts per terrain, it **completes** the motion:

| Ground | Completed |
|---|---|
| Flat | 98.8% |
| 6° slope | 0% |
| 12° slope | 0% |
| 4 cm bumps | 0.5% |
| Rough ground | 0% |
| 10 cm step | 21% |

There are three separate reasons, and each of our contributions fixes one of them:

1. **The robot can't see the ground** (it has no sensor for it). It moves its feet as if the floor were flat. → *Contribution 3: the height scan.*
2. **The ghost is in the wrong place on uneven ground.** The reward asks for things that are physically impossible. → *Contribution 2: the ground-relative reference.*
3. **The robot can't see where the ghost is.** On flat ground it never needed to, but on uneven ground it gets pushed off the motion and has no way to notice. → *Contribution 4: the reference preview.*

Plus we needed somewhere to train (*Contribution 1: the terrain*), a way to train without wrecking the released policy (*Contribution 5: the training recipe*), and an honest way to measure the result (*Contribution 6: evaluation*).

---

## Part 3: Contribution 1, the terrain playground

### What we built

PBHC only had a flat floor. We built a set of **uneven ground types**, each on a 4 m × 4 m square tile:

| Type | What it looks like | Hardest version we test |
|---|---|---|
| **Tilt** | The whole tile is a flat ramp, sloping through the middle | 24° |
| **Step** | Half the tile is raised, like a single stair; the edge runs near the middle | 17 cm |
| **Bumps** | Smooth random hills and dips, about 30 cm apart | 8 cm high |
| **Rough** | Small random noise in height (from the original code) | Used only for testing |

Each type comes in **difficulty levels**. The difficulty number maps to a size:

| Difficulty | 0.5 | 1.0 | 1.25 | 1.5 | 2.0 |
|---|---|---|---|---|---|
| Step height | 6.5 cm | 10 cm | 11.8 cm | 13.5 cm | 17 cm |
| Tilt | 6° | 12° | 15° | 18° | 24° |
| Bumps | 2 cm | 4 cm | 5 cm | 6 cm | 8 cm |

**Rough ground is never used for training,** only for testing. It checks whether the robot can handle a kind of ground it has never seen.

### Why the feature goes through the middle of the tile

The Horse-stance motion barely travels: the robot moves less than half a metre during the whole thing. So it doesn't walk *across* terrain; it performs *on top of* terrain. We therefore place every motion so it starts at the **centre of its tile**, and design every terrain so the interesting part (the slope, the step edge) **passes through the centre**, right under the robot's feet.

The existing terrain types in robot-locomotion code (pyramids of stairs, for example) have a flat platform in the middle, which would leave our robot standing on flat ground.

### The curriculum

The 1,536 robots in training are spread over a grid of tiles: 5 rows of difficulty, from easy to hard, and 24 columns of terrain types (10% flat, 30% bumps, 30% tilt, 30% step). It was 8 columns at first; "Not enough variety" below explains the change.

- A robot that **completes the motion** moves **up one row** (harder ground).
- A robot whose attempt **ends early** moves **down one row**.
- A robot that completes the hardest row stays in the **two hardest rows**.

So every robot trains near the edge of what it can do. The average row ("terrain level", from 0 to 4) is logged during training and tells us how hard the ground the robots are coping with is.

### Two fixes made along the way

- **Training never reached the test difficulties.** At first the rows only went up to difficulty 0.8, so training never saw the 10 cm step it was tested on. Later we found the same problem one level up: steps topped out at 10 cm in training while tests go to 17 cm. Each terrain type now has its own maximum: steps go up to 17 cm, slopes to 21° and bumps to 8 cm. Raising the step maximum improved tall steps, and raising the slope and bump maximums improved every terrain, steps included (18° slope from 45% to 73% completed, 8 cm bumps from 42% to 68%, two-seed averages).
- **The robots' rows are now saved with the checkpoint.** Before, every time a training run was continued, all robots restarted at the easiest rows and spent much of the run climbing back. A bug also moved every robot down one row at the start of every run. Both are fixed.

### Not enough variety: the robot had memorised about ten steps

This was found late (10 November) and turned out to matter a lot for steps.

- Each step tile has **one edge**, in one random direction, a random distance (up to 15 cm) from the tile centre.
- With 8 columns there were only **2 or 3 step tiles per difficulty row**, so about ten step layouts in the whole of training.
- Every attempt started at the **exact centre** of its tile, so each of those edges was always met at the same place under the robot.

So the robot wasn't learning "steps"; it was learning its own ten steps. That also explains why each training seed had its own weak step direction. Two changes fix it:

- **24 columns instead of 8,** which gives about 7 step tiles per row.
- **Placement jitter:** each attempt shifts the whole motion by a random amount, up to 15 cm in x and y, so the same edge is met at many different places.

For the punch, 10 cm steps went from 24% to 46% completed within 750 iterations of making this change.

### The level only counts attempts that include the hard part

Training starts attempts at random points in the motion, and on uneven ground only the opening (the big step into the stance) is hard. Most attempts skipped it, completed easily and moved their robot up a row, so the terrain level looked much better than the robot really was.

Now 70% of attempts on uneven tiles start within the first second of the motion, and **only those attempts move a robot up or down**. The level then measures the motion from its opening, and the robots spend their time on rows that are hard for the right reason. For the pose this improved or held every one of the 17 test settings (8 cm bumps from 68% to 82%, 24° slope from 31% to 43%).

### A quirk to know about

The simulator turns the ground into a mesh of triangles, and its conversion turns any height change of less than 7.5 cm over 10 cm into a slope. So the **6.5 cm "step" is really a short steep ramp**, not a wall. Steps from about 7.5 cm up are real walls.

---

## Part 4: Contribution 2, the ground-relative reference

This is about **where the ghost is placed**, and it affects almost everything the reward measures.

### What "coordinates" and "frames" mean here

Every position in the simulator is three numbers, **x, y, z**: x and y are where something is on the floor plan, z is how high it is. These numbers are measured from some fixed starting point and directions. That choice is called a **frame**.

- **World frame:** one fixed starting point for the whole simulated world, with z measured straight up from a fixed height. The same everywhere, like latitude, longitude and altitude.
- **Ground-relative:** height measured from **the ground directly underneath**, like "1 m above the floor here", wherever "here" is.

On a flat floor at height 0, the two are the same. On uneven ground they are not.

### The original: the ghost lives in the world frame

The reference motion is a list of poses recorded on a flat floor at height 0. To place it on a robot's tile, PBHC adds a fixed **offset**: the tile's centre for x and y, plus **one fixed height for z**. That height is the **highest point of the ground** in the middle 1 × 1 m of the tile. It is computed once and never changes.

So the ghost is a rigid copy of the flat-floor motion, lifted to one height. It does not know that the ground under it is uneven.

**What that means on a 10 cm step** (side view):

```
 ORIGINAL (world frame)

   ghost feet:   o                 o        <- both at the top-of-step height
                                ___|______  upper level (+10 cm)
   ground:  ___________________|            lower level (0)
                 ^
                 the left ghost foot hangs 10 cm in the air
                 above the lower level
```

**What that means on a 12° slope:**

```
 ORIGINAL (world frame)

   ghost feet:   o ----------------- o      <- one height, about 10 cm above the slope's middle
                                  ___/
   ground:                   ___/
                        ___/                  the slope
                   ___/
   both ghost feet float above the ground, by different amounts
```

We measured this directly. On a 12° slope, the lower ghost foot floats **11 cm** above the ground on average and is more than 9 cm in the air **66%** of the time (on flat ground it is 3–5 cm, which is just the ankle's height above the sole).

### Why that is a real problem: everything compares against the ghost

The ghost's positions are used by:

1. **The position rewards.** They compute "ghost position minus robot position" for every body part, in the world frame. If the ghost's foot is 10 cm in the air, the robot is rewarded for holding its foot 10 cm in the air, which it can't do while standing on it. A robot standing correctly gets a worse score than one doing something odd.
2. **The drift check.** The attempt is cut off when the robot is too far from the ghost. A robot standing correctly on the ground can be cut off for being far from a floating ghost.
3. **The starting position.** Each attempt places the robot where the ghost starts, so it is dropped from the floating height onto the ground.
4. **The critic's view.** The critic sees the robot-minus-ghost differences, so it gets misleading information about how well the robot is doing.

So the original setup **rewards the wrong behaviour on uneven ground**. Even a perfect learner would learn something odd.

### Our fix: move the ghost onto the ground under its feet, every step

**Step 1: forget the fixed height.** The offset keeps the tile centre for x and y, but its z becomes 0.

**Step 2: every step, look at the ground under the ghost's feet.** Each step (50 times a second), we look up the ground height under the ghost's left foot and under its right foot.

**Step 3: compute one "support height".** We combine the two ground heights into one number, the height the ghost should stand at. How we combine them matters:

- An **average** of the two would put the foot on the higher side partly **inside** the step (we measured up to 3.5 cm inside).
- So we use (roughly) the **higher of the two**, so no foot ends up inside the ground.
- But only for feet that are **bearing weight**. A foot the ghost is lifting (swinging through the air) shouldn't pull the whole body up just because it passes over a high spot. So each foot's ground height is lowered by how high the ghost lifts that foot before picking the higher one.
- We use a "soft" version of "the higher of the two", which blends smoothly instead of jumping. In formula form: `support = Σ wᵢ · groundᵢ`, with weights `w = softmax((groundᵢ − gain · liftᵢ) / 2 cm)`. The gain is 1 for the pose and 4 for the punch (see "A motion that barely lifts its foot" below). If you don't know softmax: it gives most of the weight to the largest value, but changes smoothly as the values change.

**Step 4: lift the whole ghost by the support height.** Every body part of the ghost is moved up (or down) by that one number. The ghost's shape doesn't change, only its height.

**Step 5: keep its speeds consistent.** The reward also compares velocities. If the ghost is moved up over time (say, its feet step from the low level to the high level), the ghost is actually moving upward, so its upward speed is adjusted too. Big sudden jumps (a foot crossing a step edge) are capped so they don't produce absurd speeds.

**Step 6: smooth the step edges.** Instead of reading the ground at a single point under each foot, we average it over the foot's length (±4 cm). Otherwise, as a foot crosses a step edge, the whole ghost would jump 10 cm in one step.

**Step 7: measure feet by height above their own ground.** One problem remains. Lifting the whole ghost by one number can't put two feet on two different levels. On a step, if the support height is the upper level, the ghost's lower foot still hangs above the lower level. So the **feet reward** no longer compares world heights. It compares **"how high is the foot above the ground under it"** for the robot and for the ghost. The ghost's lower foot is 3 cm above the lower level (its normal ankle height), and the robot's foot standing on the lower level is also about 3 cm above it, so they match. The robot is rewarded for putting its foot down where it belongs.

```
 OURS (ground-relative)

   ghost feet:   o                 o        <- ghost lifted to the upper level
                                ___|______  upper level (+10 cm)
   ground:  ___________________|            lower level (0)
   robot feet:   x                 x
                 ^
                 the feet reward compares "height above my own ground":
                 ghost foot over the lower level  vs  robot foot on the lower level
```

**On flat ground, all of this does nothing:** every ground height is 0, so every shift is 0. We checked this number by number: on flat ground the new code gives exactly the same ghost, the same rewards and the same observations as the original. So the flat-ground behaviour of PBHC is untouched.

### How much better it is

**The ghost is now somewhere the robot can actually be:**

| Ground | Original: ghost foot height above ground | Ours |
|---|---|---|
| 12° slope | 11.1 cm on average; more than 9 cm in the air 66% of the time | 3.5 cm on average; in the air 0.1%; never inside the ground |
| 10 cm step | 6.3 cm on average | 3.7 cm on average; never inside the ground |

**Training with it, the robot stops falling.** We trained two versions the same way and for the same time, one with the original ghost ("B") and one with ours ("C"):
- Falls dropped from **23–36%** of attempts to **1–7%**.
- Tracking error dropped on every terrain.

**It is also the base for everything after it.** The height scan and the reference preview (Parts 5 and 6) were all trained with the ground-relative ghost. The preview in particular shows the robot the ghost's *shifted* future foot positions, which tell it whether its next foothold is on the upper or lower level.

### What it doesn't do on its own

- **It didn't make the robot complete more motions.** With the ground-relative ghost but no way to see the ground, the robot stopped falling, but its attempts ended by drifting away from the ghost instead. A blind robot still can't place its feet on uneven ground. That needed the height scan.
- **It only fixes heights.** The ghost's legs keep the same joint angles as on flat ground, and its feet land in the same places on the floor plan. For the pose we tried three ways of also bending the ghost's legs to fit the ground (per-foot shifts, inverse kinematics, lifting a swinging foot over a nearby step). None of them made the pose better, so they are off for the pose. The punch does need the third one (next section). Moving *where* the ghost's feet land, away from step edges, hasn't been tried yet.
- **Its effect on its own, alongside the height scan, hasn't been measured.** Every height-scan run used the ground-relative ghost; we never trained one with the original ghost.

### A motion that barely lifts its foot (found with the punch)

The pose lifts its stepping foot 23–28 cm, well above any step we test. The **Horse-stance punch** opens with a 47 cm side-step that lifts the foot only **6 cm**. On a taller step that caused two problems:

- **The ghost's foot went through the step.** A foot lifted 6 cm can't pass over a 10 cm edge, so the ghost's foot passed through the riser, and the robot was being asked to do the same.
- **The swinging foot was taken for a standing foot.** With so little lift, the support-height formula above still gave the swinging foot a lot of weight, so the whole ghost was pulled up as that foot passed over the upper level.

Two options fix this, and the punch uses both: **swing clearance** (a lifted ghost foot is raised over nearby higher ground) and a **support-lift gain of 4** (a foot counts as lifted four times sooner). The pose doesn't need either.

---

## Part 5: Contribution 3, the height scan (letting the robot see the ground)

### What a height scan is

Imagine dropping a grid of strings from the robot's hips down to the ground and measuring how long each string is. Those lengths together form a **height scan**: a small map of the ground around the robot. On flat ground all the strings are the same length; on a step, the strings over the raised half are shorter; on a slope, they get gradually shorter in one direction.

On a real robot this would come from a depth camera or a laser scanner that builds a map of the ground (an **elevation map**). In simulation we read it straight from the terrain.

### Our body scan

- **A grid of 11 × 11 = 121 points,** spaced 10 cm apart, covering 1 m × 1 m centred on the robot's pelvis (from 0.5 m in front to 0.5 m behind, and 0.5 m to each side).
- **The grid turns with the robot.** Its "front" is always the direction the robot faces (we call this the **heading frame**). So "the step is in front of me" always looks the same to the robot, whichever way it is facing in the world.
- **Each number is** `pelvis height − 0.75 m − ground height at that point`, capped at ±1 m. Standing on flat ground at normal height, all 121 numbers are close to the same small value. A raised step under part of the grid makes those numbers smaller. Because each number is measured from the robot's own pelvis, the scan also tells the robot how high it is above the ground.

Our first scan was narrower (only ±0.3 m sideways, 77 points). We measured that in the wide horse stance the feet land 27–41 cm out from the pelvis, beyond that edge, so the robot couldn't see where it was putting its feet. We widened it to ±0.5 m.

### Foot scans: a fine-detail map under each foot

The body scan's 10 cm spacing can't tell exactly where a step edge is relative to a 20 cm-long foot. So each foot also gets its own small, detailed scan:

- **13 × 7 = 91 points per foot,** 2 cm apart, covering the area under the sole: from 8 cm behind the ankle to 16 cm in front of it, and 6 cm to each side.
- **It moves and turns with the foot.**
- **Each number is** `ankle height − 3.5 cm − ground height`, capped at ±30 cm. When the foot is flat on flat ground, all the numbers are close to 0.

With this, the robot can feel, in effect, "my toes are hanging over a step edge" before it puts weight on the foot.

### Exact ground height: a hidden measuring error we fixed

The standard way robot-learning code looks up the ground height takes the **lowest of three neighbouring grid points** (10 cm apart). That is a safe shortcut for walking robots, but it is inaccurate:

- On a 12° slope it reads 1–3 cm too low.
- Near a step edge it reports the lower level for points up to 10 cm onto the upper level, so the step edge appears up to 10 cm away from where it really is.

This lookup was used both for the scans **and** for placing the ghost (Part 4), so the ghost's feet were misplaced near edges too. We replaced it with an **exact** lookup that reads the real surface the simulator uses for collisions. A check against where the robot's weight-bearing feet actually rest: the share of feet more than 2 cm off dropped from 20% to 5% on the 12° slope, and from 16% to 3% on bumps.

### A better map for the coach (critic) only

The critic gets its own scan: **15 × 15 = 225 points** over 1.4 m × 1.4 m, always exact, never noisy. Because the critic is only used during training, it can have a better map than the robot will ever have. A better coach gives better training, and the final robot doesn't depend on it.

### Noise: what if the map isn't perfect?

A real ground map would have errors. We built a **noise model** that can add, separately:
- random errors on each point;
- the whole map being too high or too low;
- the whole map being shifted sideways;
- points that miss an update;
- a delay, so the robot sees the ground as it was a few steps ago.

**Result:** the trained policies shrug off most of these: 5 cm random noise on every point, a 10 cm sideways shift, 60% of points missing, a 100 ms delay. The one error that clearly hurts is the whole map being **5 cm too high or low**, which costs 8–15 points on flat ground, slopes and bumps.

### Proof that the robot really uses what it sees

A fair question: maybe the robot got better just because it trained on uneven ground, and ignores the scan. So we took the same trained policy, with the same weights, and **replaced its scan with "the ground is flat"**:

| Ground | Share of the motion tracked, real scan | Same policy, scan says "flat" |
|---|---|---|
| 6° slope | 94% | 49% |
| 12° slope | 61% | 18% |
| Bumps | 68% | 46% |
| Flat | 99% | 98% |

Without the scan the robot is much worse on uneven ground, and unchanged on flat ground. So it **is** using the scan to read the shape of the ground.

### What the height scan achieved

Adding the scan was what made the robot **complete whole motions on slopes** for the first time. After widening it, adding the foot scans and the exact lookup:
- Completion on the 12° slope went from 64% to 83% (two-seed averages).
- The 18° slope went from 11% to 32%.
- Bumps got better at every height.

A comparison run with the same extra training but the old scan got *worse* on slopes and bumps, so the gain comes from the better scan, not from training longer.

**But steps didn't improve.** Seeing the step wasn't enough. That led to the next contribution.

---

## Part 6: Contribution 4, the reference preview (letting the robot see the ghost)

This was the single biggest improvement.

### The missing piece

Recall that PBHC's actor **can't see the ghost**. It only knows the motion phase ("I am 23% of the way through") and has memorised what to do at each phase. On flat ground that works, because if it does the right thing it stays on the ghost.

On uneven ground, the robot gets pushed around: a foot lands 3 cm higher than expected, the body tips a little, and the robot ends up slightly off the ghost. **It has no way to notice that.** It keeps doing what it memorised, the error grows, and eventually the attempt ends. The critic could see the ghost, but the actor could not.

Picture a dancer following a routine from memory on a moving ship's deck. They can't tell they've slid half a metre to the left, so they don't correct it.

### What we added

A new set of numbers for the actor (and the critic), called the **reference preview**. It tells the robot where the ghost is, and where it is about to be:

- **When:** now, and 0.1, 0.2, 0.3, 0.4 and 0.5 seconds into the future (6 moments).
- **What, at each moment:** where the ghost's pelvis, left foot and right foot are (3 numbers each), and which way the ghost is facing compared with the robot (2 numbers). That makes 11 numbers per moment and 66 in total.
- **Relative to the robot:** all positions are measured from the robot's own pelvis, in the robot's heading frame. So "the ghost's right foot will be 40 cm to my right and 10 cm higher in 0.3 s" is exactly what the robot reads.
- **Ground-relative:** the positions come from the ghost after it has been moved onto the ground (Part 4), so the future foot positions include the height of the ground where they will land.

### Why it works so well

With the preview, the robot can see two things it couldn't before:

1. **Its own error:** "the ghost is 5 cm to my left, so I should move left". It can now correct drift before the attempt is cut off.
2. **What's coming:** "in 0.4 s my right foot needs to be over there, and that spot is on the upper level". Combined with the height scan, it can prepare for a foothold on a different level.

### Results

Compared with the same training continued without the preview (same starting point, same length):

| Ground | Completed without the preview | With the preview |
|---|---|---|
| 10 cm step | 9% | 46% |
| 17 cm step | 2% | 17% |
| Rough (never trained on) | 18% | 52% |
| 8 cm bumps | 18% | 35% |
| Flat | 98% | 99.5% |

It also tracks more precisely: the average error in body position on flat ground dropped from 26 to 22 mm. The gain held up across three training seeds.

### Two things that did *not* work, and what they teach

- **Measuring from the ghost instead of from the robot.** A real robot doesn't know its exact position in the world. So we tried a version that measures the future ghost positions relative to the *current ghost*, which needs no position estimate. **It gave no gain at all.** With only one motion, "where the ghost will be relative to where the ghost is now" is always the same at a given phase, so it tells the robot nothing it didn't already know from the phase. The gain comes entirely from knowing where the ghost is **relative to the robot itself**.
- **Looking further ahead (to 1.0 s).** We expected that seeing the big sideways step coming earlier would help. Instead, steps and slopes got worse in all three runs that used it. Half a second is the better choice.

### The catch for a real robot

The preview needs to know where the robot is relative to where the motion started. In simulation that is exact. A real robot would need to estimate it (with cameras or other sensors), and estimates drift over time. A ground map needs that same estimate, so it isn't a new sensor, but we haven't yet tested how much an imperfect estimate hurts.

---

## Part 7: Contribution 5, the training recipe

Good ideas still fail if training goes wrong. Several of our most useful findings were about **how** to train.

### Problem 1: fine-tuning destroyed the released policy

Our first attempt started from the released policy and trained on uneven ground with normal settings. Within the first 100 updates, **the robot forgot how to do the motion on flat ground**: from 99% completed to 0%. It only partly recovered.

Three things caused this:
- **Steps that were too big.** Each update changed the network too much (a learning rate of 3e-4).
- **A fresh optimizer.** The optimizer is the part of training that decides how big each adjustment is. Started fresh, it makes large adjustments at first.
- **An exploration bonus.** Training normally rewards the robot for trying random variations ("entropy bonus"). That pushed the robot's random jitter from 0.21 up to 0.59 (almost three times as much), and its movements became jerky.

**What we changed ("gentle fine-tuning"):**
- **Coach first:** for the first 100–150 iterations only the critic learns. The critic was trained on flat ground and needs to learn what uneven ground is worth before it starts guiding the actor.
- **Small, fixed learning rate for the actor** (2e-5, about 15 times smaller), brought in gradually over 100 iterations.
- **The released optimizer's state is kept** instead of starting fresh.
- **No exploration bonus,** and the robot's random jitter is frozen at the released policy's level.

This fixed the collapse. Flat-ground precision recovered step by step, to the released policy's level (about 20 mm average body-position error) by early November. The later runs on harder ground are at 22–23 mm.

### Problem 2: new inputs would scramble the network

Adding the height scan and the preview means the network gets new inputs it has never seen. If those inputs were connected with random strengths, the released policy's behaviour would be scrambled at the start.

**What we did:** a tool (`scripts/expand_checkpoint.py`) adds the new inputs with **zero-strength connections**. At the start, the network ignores them completely and behaves exactly like the released policy (we checked: the actions match to within 0.000005). Training then gradually learns to use them. The tool also extends the optimizer's memory for the new connections.

### Problem 3: falling cost the same as drifting

In PBHC, an attempt that ends because the robot **fell** and one that ends because it **drifted** from the ghost both cost the same −200 penalty. So the robot had no reason to give up a bit of tracking in order to stay upright, and as tracking got stricter it fell more.

**What we changed:** drifting now costs a quarter as much as falling. The message is "if you must fail, drift; don't fall".

### Problem 4: the fall penalty never actually happened

Our first run with the new penalty changed nothing. The reason was hidden in PBHC's **shrinking drift limit**: during training, the allowed distance from the ghost shrinks to 0.3 m. At 0.3 m, attempts are cut off for drifting **long before** the robot has a chance to fall. Training saw almost no falls (about 0%), so it never learned what to do in the moments before a fall, and the penalty never applied. Testing, meanwhile, allows 0.7 m, where falls do happen.

**What we changed:** the drift limit stays at **0.7 m** in training, the same as in testing.

**Result:** falls almost disappeared. One run fell at most 7% of the time on every terrain and difficulty we tested, including 17 cm steps and 24° slopes, and falls on the never-seen rough ground dropped from 33% to about 1%.

### Problem 5: adaptive σ gets confused by mixed ground

PBHC tightens σ (the strictness of the tracking scores) based on the average error over all robots. With robots on very different ground, that average doesn't mean much, and the strictness kept rising while robots were struggling. **We freeze σ** at the values the policy had already adapted to.

### Things we tried in training that didn't help

- **Starting more attempts at the hard part of the motion, on its own.** Most failures happen 1–2 seconds in, so we started some training attempts there. That alone didn't improve attempts that start at the beginning of the motion. It did work later, once **only** those attempts were allowed to move robots up and down the curriculum (Part 3).
- **Bigger learning rates.** For the punch we tried rates 3–5 times larger to learn faster. From the same starting point it was worse on every terrain (10 cm steps 12% against 20%).
- **Training longer.** This helped one seed and not the other. The one it didn't help had spent its extra training on easy rows because its curriculum level had been reset, which is what led to saving the levels with the checkpoint.

---

## Part 8: Contribution 6, measuring honestly

How you measure decides what you conclude, and several early conclusions turned out wrong because of how we measured.

### How we test

- **Every test attempt starts at the beginning of the motion**, so it has to get through the hard part. (Training starts attempts at random points in the motion, mostly in the easy parts, which made training numbers look much better than reality.)
- **Each attempt ends in one of three ways:** **completed**, **fall**, or **drift** (more than 0.7 m from the ghost).
- **The test terrains** are flat, rough (never trained on), and steps, slopes and bumps at five difficulties each: 17 settings in total.
- **600 attempts per setting** (3 test seeds × 200). Comparisons up to early November used two or three training seeds; the latest runs use one.

### Fair counting

At first we counted the first 200 attempts to finish. But failures finish *sooner* than successes, so failures were over-counted. Weak policies, which fail early and restart often, looked even worse than they were. With every robot now contributing the same number of attempts, the released policy's score on 10 cm steps went from 13% to 37% of the motion tracked.

### Diagnosing steps

Because steps were the hardest problem, the tests also record for every step attempt:
- which side of the robot the raised part is on at the start (ahead, behind, left or right);
- how close each foot is to the step edge, and whether a foot starts on the raised part;
- when in the motion the attempt failed;
- whether the shin touched the step;
- whether the robot's feet ended on the same level as the ghost's.

These showed, for the runs up to early November:
- **Nearly every step failure happens 1–2.5 seconds in,** during the weight shift and the big sideways step. A robot that gets past that point almost always finishes.
- **The worst case was a step up directly in front of the robot** (10–25% falls at 10 cm). With the varied layouts of Part 3 that is down to 3%, and the weakest direction is now a raised part on the robot's left.
- **Many "drift" endings on steps are really falls in progress:** they happen within half a second, with the pelvis already tipped about 30°, before the fall check (about 50°) triggers.
- **Shin contact is not the cause:** it happens as often in successful attempts as in failed ones.

### Training seeds matter

Two runs with identical settings except the seed differed by about **15 percentage points** on 10 cm steps. So a single run can't show an improvement smaller than that. Several ideas that looked worse on one run were inside that range, which is why we moved to two seeds per idea. Much of that difference came from each run having its own ten step layouts (Part 3).

---

## Part 9: Before and after, side by side

| | Original PBHC | Our version |
|---|---|---|
| **Ground** | Flat only | Slopes, steps, bumps, rough; training curriculum up to 17 cm steps, 21° slopes and 8 cm bumps, with 24 columns of tiles and a random shift of the motion on its tile |
| **Where the ghost is** | One fixed height per robot (world frame) | Moved onto the ground under its feet every step (ground-relative) |
| **Feet reward** | World height | Height above the ground under each foot |
| **What the actor sees** | Its own body + motion phase | + a 121-point ground scan, + a 91-point scan under each foot, + where the ghost is relative to it now and in the next 0.5 s |
| **What the critic sees** | Its body, the ghost, physics settings | + an exact 225-point ground scan, + the ghost preview |
| **Ground height lookup** | Lowest of 3 neighbouring points (1–3 cm low on slopes) | Exact surface |
| **Fall vs drift** | Same penalty | Drift costs a quarter of a fall |
| **Drift limit in training** | Shrinks to 0.3 m | Fixed at 0.7 m (same as testing) |
| **σ (strictness)** | Adapts during training | Frozen |
| **How it's trained** | From scratch, ~27 h | Fine-tuned gently from the released policy, in stages of 0.5–1.7 h |

Actor inputs went from **380 numbers** (original) to **749 numbers**.

---

## Part 10: Results in plain words

Percent of attempts that complete the whole motion, starting from the beginning. The "early November" row is a two-seed average; the "current best" row is one training seed:

| | Flat | Rough (never trained on) | 12° slope | 18° slope | 4 cm bumps | 8 cm bumps | 10 cm step | 17 cm step |
|---|---|---|---|---|---|---|---|---|
| Original PBHC | 98.8 | 0 | 0 | 0 | 0.5 | 0 | 21.0 | 23.2 |
| Ours, early November | 99.6 | 55.0 | 89.6 | 44.8 | 92.7 | 42.3 | 50.7 | 25.6 |
| Ours, current best (10 November) | 99.7 | 80.5 | 98.3 | 85.8 | 96.7 | 82.0 | 64.5 | 49.8 |

What changed between the last two rows: slope and bump maximums raised in the curriculum, 24 columns with placement jitter, and the level counted from early starts only (all in Part 3).

- **Flat ground:** just as good as the original. Precision is slightly lower: about 23 mm average body-position error, against 20 mm for the released policy.
- **Slopes:** from failing every time to completing 98% at 12°, 86% at 18° and 43% at 24°.
- **Bumps:** from failing every time to completing 97% at 4 cm and 82% at 8 cm.
- **Rough ground it has never seen:** from 0% to 80%.
- **Steps:** better, but still the weakest. About two in three attempts complete at 10 cm and half at 17 cm.
- **Falls are now rare:** none on flat, rough, bumps and slopes up to 18°; 1% on 10 cm steps, 3.5% on 17 cm steps, 5% on the 24° slope. When an attempt fails, the robot has usually lost the ghost, not fallen over.

The same recipe trained 1,500 more iterations reads higher again on a shorter test (5 settings): 69% on 10 cm steps, 55% on 17 cm steps, 91% on the 18° slope, and 21.5 mm error on flat ground. Its full test is running.

One oddity: the original PBHC scores higher on 17 cm steps than on 6.5 cm steps, which a blind robot shouldn't do. We haven't found out why, so treat its step numbers with care.

### A second motion: the Horse-stance punch

To show the approach isn't specific to one motion, we applied it to the **Horse-stance punch**, starting from its released policy. Completed / fell, in percent:

| Ground | Released punch | Ours (10 November) |
|---|---|---|
| Flat | 99.2 / 0.2 | 99.2 / 0 |
| Rough (never trained on) | 0 / 22.3 | 88.3 / 0.2 |
| 12° slope | 0 / 1.7 | 97.2 / 0 |
| 18° slope | 0 / 5.2 | 85.3 / 0.3 |
| 8 cm bumps | 0 / 9.2 | 86.7 / 0.2 |
| 10 cm step | 18.8 / 14.7 | 66.8 / 0.8 |
| 17 cm step | 20.3 / 11.5 | 48.5 / 1.0 |

- **The punch is now level with the pose** on every kind of ground, with falls at or below 1%.
- **The pose recipe alone was not enough.** With it, the punch completed only 10% of 10 cm steps. It needed the ghost changes for a low foot lift (Part 4) and the two curriculum changes (Part 3). Those curriculum changes then turned out to help the pose too.
- **It is less precise on flat ground** than the released punch: 24 mm average body-position error against 17 mm. We haven't fixed that.
- **A measuring trap:** the punch's motion file starts 1.17 m away from the origin. Unless the motion is re-centred on its tile, the robot stands away from the step edge and the released punch seems to complete about 80% of steps. An earlier version of this file reported that number; the correct one is about 20%.
- A further 1,500 iterations read 69% on 10 cm steps, 54% on 17 cm steps, 89% on the 18° slope and 90% on 8 cm bumps on the shorter test.

### Motions with no released policy

PBHC ships five more motions without a trained policy: the Hooks punch, side kick, roundhouse kick, Charleston dance and Bruce Lee pose. We have just started on the first (Hooks punch), using the punch's terrain policy as the starting point. There are no results yet.

---

## Part 11: What still doesn't work, and why

- **Steps.** About a third of attempts on 10 cm steps and half on 17 cm steps still fail. The robot rarely falls now; it loses the ghost during the big sideways step 1–2.5 seconds in (and, for the punch, the step back at the end). The foot lands about 50 cm out, right at the edge of the body scan, and the foot scans only cover where the feet are now, not where they are going. The weakest cases are a raised part on the robot's left and starts with one foot already on the upper level.
- **Starts that straddle the edge.** About half the test attempts start with one foot already on the step. Those complete far less often. A real robot would probably not start a kung-fu move standing across a step edge, so these should be reported separately.
- **The small steps are ramps.** Steps below about 7.5 cm are slopes, not walls (Part 3).
- **The punch is less precise on flat ground** than its released policy (24 mm against 17 mm).
- **The newest results are one training seed.**
- **A real robot would need to know where it is.** Both the ground map and the preview depend on that, and we haven't tested how much an imperfect estimate hurts the preview.
- **Simulation only.** Nothing has run on a real robot.

**What we'd try next:**
1. A small ground scan around where each ghost foot **will** land in the next half second.
2. Moving where the ghost's feet land away from step edges.
3. Reporting clean starts and edge-straddling starts separately.
4. Turning the small ramps into real steps.
5. Testing the preview with an imperfect position estimate.
6. More motions, which has started: the Hooks punch first, then the kicks (long time on one leg), the Charleston dance (20 seconds long) and the Bruce Lee pose (very deep crouch), and eventually one policy for several motions.
7. Confirming the final recipe on a second training seed.

---

## Glossary

| Term | Meaning |
|---|---|
| **Actor** | The neural network that decides the robot's actions. It is what would run on a real robot. |
| **Action** | The 23 target joint angles the actor outputs each step. |
| **Adaptive σ** | PBHC's automatic tightening of how strict the tracking scores are. |
| **Checkpoint** | A saved copy of the networks at some point in training. |
| **Completion** | An attempt that reaches the end of the motion without falling or drifting. |
| **Critic** | A second network, used only in training, that predicts future reward. It may see privileged information. |
| **Curriculum** | Training on easy ground first and moving robots to harder ground as they succeed. |
| **Drift** | An attempt cut off because some body part is more than 0.7 m from the ghost. |
| **Elevation map** | A map of ground heights around a robot, built from sensors on a real robot. |
| **Entropy bonus** | A reward for trying random variations, which encourages exploration. |
| **Episode / attempt** | One try at the motion, from the start until it ends or is cut off. |
| **Fall** | An attempt cut off because the robot's body tipped too far. |
| **Fine-tuning** | Continuing to train an already-trained network on a new task. |
| **Frame** | The starting point and directions that positions are measured from. |
| **Ghost** | Our nickname for the reference motion drawn where the robot should be. |
| **Ground-relative** | Heights measured from the ground underneath, not from a fixed world height. |
| **Heading frame** | Directions measured relative to the way the robot faces: "ahead" is always the robot's front. |
| **Height scan** | A grid of ground heights around the robot, given to the policy. |
| **Iteration** | One round of collecting experience and updating the networks. |
| **Learning rate** | How big each update to the network is. |
| **Motion phase** | How far through the motion the robot is, from 0 to 1. |
| **Observation** | The numbers the policy receives each step. |
| **Optimizer** | The part of training that turns the reward signal into network updates. |
| **Placement jitter** | A random shift (up to 15 cm) of the whole motion on its tile each attempt, so a step edge is met at many places. |
| **Policy** | The robot's decision-maker (the actor network). |
| **PPO** | The reinforcement-learning algorithm used to train the networks. |
| **Privileged information** | Information only the critic gets, which a real robot wouldn't have. |
| **Reference motion / reference** | The recorded motion the robot should copy, pose by pose. |
| **Reference preview** | Our new input: where the ghost's pelvis and feet are relative to the robot, now and in the next 0.5 s. |
| **Reinforcement learning (RL)** | Learning by trial and error from a reward signal. |
| **Reward** | The score given each step for how well the robot did. |
| **Seed** | The random number that decides all random choices in a run. |
| **σ (sigma)** | How strict a tracking score is: smaller means stricter. |
| **Step (time)** | One decision of the policy; there are 50 per second. |
| **Support height** | The height we move the ghost to each step: roughly the higher ground under its weight-bearing feet. |
| **Swing clearance** | Raising a lifted ghost foot over nearby higher ground, for motions that lift the foot less than the step height. |
| **Termination** | Cutting an attempt off early (fall or drift). |
| **Tracking** | Making the robot's body follow the reference motion. |
| **World frame** | Positions measured from one fixed point and height for the whole simulated world. |
| **Zero-strength connections** | How we add new inputs so the network initially ignores them and behaves as before. |
