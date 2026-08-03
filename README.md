# vision-pick-place

Vision-guided pick and place with a simulated Franka Panda arm in MuJoCo.

## Overview
- **Days 1–2 Setup** Load the Panda, run the sim loop, hold/reach
  joint targets with PD controller
- **Days 3–4 Reaching a point in space** Damped-least-squares IK (via MuJoCo
  Jacobians): give Cartesian target, solve for joint angles, drive there
  with PD controller.
- **Days 5–6 Perception** Camera in scene, red cube segmented by color,
  back-projected through depth buffer to a world coordinate, reached
  by the arm using vision.
- **Days 7–8 — Full pick and place** State machine chaining approach → descend
  → grasp → lift → transport → release, with minimum-jerk trajectories between
  waypoints. 12/12 randomized trials land the cube in the bin.
- **Days 9–10 — Analysis** Randomized placement, typed
  failure detection with retry-on-recoverable, logged benchmark.
  **98% success (49/50)** over the full workspace. 
- **Gain tuning** Inertia-scaled PD gains replacing the hand-picked ones, found
  by sweeping a 2-parameter (wn, zeta) space. Cycle time **14.7 s → 5.2 s
  (2.8× faster)** with reliability unchanged.

## Things to do in the future:
- **Additional Testing/Analysis** tests with more retries. try to optimize
  success rate vs compute power/time
- **Add more randomness to testing** current tests had all the cubes aligned
  to the axes. add random yaw to see what happens. will have to pull a yaw
  angle from the mask to rotate the gripper.

- **picking out of a pile** a more complex challenge that's pretty common in my
  line of work. if I have a bunch of items in a box, all randomly oriented,
  can I feasibly do this with the existing gripper? usually a different EOAT
  is used for something like this but it would be an interesting challenge. 
- **closed-loop visual system** instead of making an estimate and blindly going for it,
  continuously re-perceive during the descent stage to increase accuracy.


## Run
```bash
python src/sim.py                    # interactive viewer, holds the home pose
python src/sim.py --target 0 -0.5 0 -2.0 0 1.8 0.5   # reach a joint config
python src/sim.py --no-gravity-comp  # watch the arm sag under pure PD

python src/reach_test.py             # headless: prove it converges (PASS/FAIL)
python src/reach_test.py --no-gravity-comp

# Days 3-4 Cartesian IK
python src/reach_point.py --pos 0.5 0.2 0.4               # position-only IK
python src/reach_point.py --pos 0.5 -0.2 0.5 --keep-orient # + hold top-down grasp
python src/reach_point.py --pos 0.5 0.2 0.4 --headless    # verify without a window

# Days 5-6 Perception
python src/perception_test.py        # vision vs ground truth across placements
python src/see_and_reach.py          # SEE the cube, then reach it (viewer)
python src/see_and_reach.py --random --headless   # randomized, scored
python src/debug_render.py           # dump what each camera sees

# Days 7-8 Full pick and place
python src/pick_place.py                        # watch the whole sequence
python src/pick_place.py --headless             # run and score one attempt
python src/pick_place.py --trials 12 --headless # reliability check
```
Reliable cube placements: `x` 0.30–0.80 m, `y` −0.25 to +0.45 m.

```bash
# Days 9-10 Robustness benchmark
python src/benchmark.py --trials 50               # full workspace, with retries
python src/benchmark.py --trials 50 --zone safe   # interior regression check
python src/benchmark.py --trials 50 --retries 0   # what retries are worth
python src/benchmark.py --noise-sweep             # perception-error tolerance
python src/benchmark.py --speed-sweep             # cycle time vs success rate
```
Per-trial CSVs land in `results/`.

```bash
# Gain tuning
python src/analyze_gains.py          # inertia, damping ratio, torque headroom
python src/tune_gains.py             # sweep wn; compares against the old gains
python src/tune_gains.py --zeta-sweep --wn 30
python src/tune_gains.py --speed     # how short a move stays smooth
python src/pick_place.py --speed 1.0 # reproduce the original Day 7-8 timings
```

## Gain tuning
Gains are no longer 14 hand-picked numbers. Each joint is scaled by its own
effective inertia `M_i`, which leaves two physically meaningful knobs:

```
kp_i = wn^2 * M_i            wn   = natural frequency, same for every joint
kd_i = 2 * zeta * wn * M_i   zeta = damping ratio
```

Defaults are `wn = 30 rad/s`, `zeta = 0.8`. Versus the old gains, on the same
pick-and-place path:

| metric | hand-picked | tuned |
|---|---|---|
| trajectory tracking lag | 0.177 rad | **0.071 rad** |
| trajectory overshoot | 2.48% | **1.39%** |
| step-response settling | 1.05 s | **0.32 s** |
| peak torque used | 43% | 43% |

**What actually limits the gains** is not motor torque (peak usage is 43%) and
not the classic explicit-integration limit (kp could go ~100× higher). It is a
discrete-time instability in the *damping* term: because the damper acts on the
previous step's velocity, too large a `kd` makes the correction overshoot and
flip sign every step — a 250 Hz ring, exactly Nyquist for the 2 ms timestep.
The measured boundary is `zeta * wn <~ 27`, and it is sharp: at 27.2 the arm
tracks a smooth trajectory perfectly yet chatters violently the moment anything
disturbs it. The defaults sit at 24, verified chatter-free over 60 random poses.

Note that `zeta` is chosen for margin, not tracking. At equal `zeta*wn` the whole
family — (40, 0.6), (34, 0.7), (30, 0.8), (24, 1.0) — scores the same 96–98% on
the benchmark, so the task cannot pick a winner. What separates them is response
to inputs the trajectory generator never produces: a step command overshoots 31%
at `zeta = 0.6` but only 3.9% at `zeta = 0.8`.

**Speed.** Tighter tracking buys shorter phase durations. `--speed` divides every
duration; 150 trials per level (3 seeds × 50), full workspace:

| speed | 1.0 | 2.0 | 2.5 | **3.0** | 3.5 | 4.0 |
|---|---|---|---|---|---|---|
| cycle time (s) | 14.73 | 7.43 | 6.09 | **5.23** | 4.61 | 4.15 |
| success | 97.3% | 96.7% | 96.0% | **96.7%** | 94.0% | 92.0% |

1.0×–3.0× are statistically indistinguishable, so **3.0 is the default**: a 2.8×
faster cycle for no measurable reliability cost. Past it the extra failures are
mostly `ik_failed`, because each phase seeds IK from the *current* pose and a
hurried arm has not settled when the next phase begins.

## Measured results
50 randomized trials per row, seed 42.

| Configuration | Success | Notes |
|---|---|---|
| full workspace, 2 retries | **49/50 (98%)** | 1 × `ik_failed` at (0.785, +0.372), 0.87 m out, past reach |
| full workspace, 0 retries | 47/50 (94%) | retries recover 3 of the 4 non-reach failures |
| safe interior, 2 retries | 50/50 (100%) | mean placement 2.8 mm from bin centre |

Vision error is 2.3 mm mean / 4.4 mm max; successful placements land 4.6 mm mean
from the bin centre. (Measured at the tuned gains and the default 3.0x speed;
`--speed 1.0` places tighter, ~2 mm, at 2.8x the cycle time.)

**Perception-error tolerance** (12 trials/level, safe zone):

| added noise (mm) | 0 | 5 | 10 | 15 | 20 | 25 | 30 | 40 |
|---|---|---|---|---|---|---|---|---|
| no retries | 100% | 100% | 100% | 100% | 92% | 50% | 25% | 17% |
| 2 retries | 100% | 100% | 100% | 100% | 92% | 92% | 83% | 58% |

The huge jump between 15 and 20 mm is caused by geometry and not tuning.
the gripper opens to 40 mm per side against a 25 mm cube half-width. 
This only leaves 15mm clearance for the finger.
Beyond that a finger strikes the cube instead of passing around it.

Retries roughly double the success rate in the marginal band, because each attempt
re-perceives and is an independent shot. *I want to run more tests on larger
sets of trials with more retries to see how that affects success rate, and if
there is a sweet spot where I'm maximizing success rate and minimizing 
computing time and power.*

## Layout
| File | Purpose |
|------|---------|
| `src/robot.py` | Load Panda; convert arm actuators to direct torque control |
| `src/pd_controller.py` | joint-space PD (+ gravity compensation) |
| `src/sim.py` | Real-time interactive sim loop with the viewer |
| `src/reach_test.py` | Headless convergence check (settling time, overshoot) |
| `src/ik.py` | Damped-least-squares IK solver (MuJoCo Jacobians) |
| `src/reach_point.py` | Cartesian target -> IK -> PD drive, with target marker |
| `src/perception.py` | Color segmentation + depth back-projection -> world xyz |
| `src/perception_test.py` | Scores vision estimates against ground truth |
| `src/see_and_reach.py` | Full loop: see the cube -> IK -> reach it |
| `src/pick_place.py` | Pick-and-place state machine + failure detection/retry |
| `src/benchmark.py` | Randomized trial campaigns, CSV logs, success-rate stats |
| `src/pd_controller.py` | Inertia-scaled PD gains (see Gain tuning) |
| `src/tune_gains.py` | Gain/speed sweeps against smoothness criteria |
| `src/analyze_gains.py` | Inertia, damping ratio, torque headroom per joint |
| `src/debug_render.py` | Save RGB/mask PNGs from each camera |
| `src/inspect_model.py` | Print joints / actuators / dimensions |
| `src/inspect_gripper.py` | Measure fingertip geometry and the true grasp point |


