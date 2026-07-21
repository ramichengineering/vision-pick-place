# vision-pick-place

Vision-guided pick and place with a simulated Franka Panda arm in MuJoCo.

## Week plan
- **Days 1–2 — Get it moving.** Load the Panda, run the sim loop, hold/reach
  joint targets with a hand-written PD controller. *(current)*

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```
The Panda model lives in `models/franka_emika_panda/` (from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)).

## Run
```bash
python src/sim.py                    # interactive viewer, holds the home pose
python src/sim.py --target 0 -0.5 0 -2.0 0 1.8 0.5   # reach a joint config
python src/sim.py --no-gravity-comp  # watch the arm sag under pure PD

python src/reach_test.py             # headless: prove it converges (PASS/FAIL)
python src/reach_test.py --no-gravity-comp
```

## Layout
| File | Purpose |
|------|---------|
| `src/robot.py` | Load Panda; convert arm actuators to direct **torque** control |
| `src/pd_controller.py` | Hand-written joint-space PD (+ gravity compensation) |
| `src/sim.py` | Real-time interactive sim loop with the viewer |
| `src/reach_test.py` | Headless convergence check (settling time, overshoot) |
| `src/inspect_model.py` | Print joints / actuators / dimensions |

## Control notes
The Menagerie Panda ships with position-servo actuators (MuJoCo's built-in PD).
`load_panda()` rewrites the 7 arm actuators into fixed-gain torque motors so the
only feedback loop on the arm is ours: `tau = kp·(q_des − q) − kd·q̇`, plus
optional gravity compensation (`data.qfrc_bias`). Gains live in
`pd_controller.py` — tune them and feel the difference.


