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

## Things to do in the future:
- **Optimize PD Controller** possibly do this algorithmically. see how fast
  I can get the robot to smoothly run
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
```
Per-trial CSVs land in `results/`.

## Measured results
50 randomized trials per row, seed 42.

| Configuration | Success | Notes |
|---|---|---|
| full workspace, 2 retries | **49/50 (98%)** | 1 × `ik_failed` at (0.785, +0.372), 0.87 m out, past reach |
| full workspace, 0 retries | 48/50 (96%) | retries recovered a `grasp_failed` next to the bin |
| safe interior, 2 retries | 50/50 (100%) | mean placement 0.8 mm from bin centre |

Vision error is 2.3 mm mean / 4.4 mm max; successful placements land 2.0 mm mean
from the bin centre.

**Perception-error tolerance** (12 trials/level, safe zone):

| added noise (mm) | 0 | 5 | 10 | 15 | 20 | 25 | 30 | 40 |
|---|---|---|---|---|---|---|---|---|
| no retries | 100% | 100% | 100% | 100% | 50% | 33% | 25% | 17% |
| 2 retries | 100% | 100% | 100% | 100% | 92% | 75% | 67% | 42% |

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
| `src/debug_render.py` | Save RGB/mask PNGs from each camera |
| `src/inspect_model.py` | Print joints / actuators / dimensions |
| `src/inspect_gripper.py` | Measure fingertip geometry and the true grasp point |


