# vision-pick-place

Vision-guided pick and place with a simulated Franka Panda arm in MuJoCo.

## Week plan
- **Days 1–2 — Get it moving.** Load the Panda, run the sim loop, hold/reach
  joint targets with a hand-written PD controller. *(done)*
- **Days 3–4 — Reach a point in space.** Damped-least-squares IK (via MuJoCo
  Jacobians): give a Cartesian target, solve for joint angles, drive there
  with the PD controller. *(current)*

## Run
```bash
python src/sim.py                    # interactive viewer, holds the home pose
python src/sim.py --target 0 -0.5 0 -2.0 0 1.8 0.5   # reach a joint config
python src/sim.py --no-gravity-comp  # watch the arm sag under pure PD

python src/reach_test.py             # headless: prove it converges (PASS/FAIL)
python src/reach_test.py --no-gravity-comp

# Days 3-4 — Cartesian IK
python src/reach_point.py --pos 0.5 0.2 0.4               # position-only IK
python src/reach_point.py --pos 0.5 -0.2 0.5 --keep-orient # + hold top-down grasp
python src/reach_point.py --pos 0.5 0.2 0.4 --headless    # verify without a window
```

## Layout
| File | Purpose |
|------|---------|
| `src/robot.py` | Load Panda; convert arm actuators to direct **torque** control |
| `src/pd_controller.py` | Hand-written joint-space PD (+ gravity compensation) |
| `src/sim.py` | Real-time interactive sim loop with the viewer |
| `src/reach_test.py` | Headless convergence check (settling time, overshoot) |
| `src/ik.py` | Damped-least-squares IK solver (MuJoCo Jacobians) |
| `src/reach_point.py` | Cartesian target -> IK -> PD drive, with target marker |
| `src/inspect_model.py` | Print joints / actuators / dimensions |


