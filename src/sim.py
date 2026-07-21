"""Interactive MuJoCo sim loop: hold/reach a joint target with our own PD.

Run:
    python src/sim.py                 # hold the home pose
    python src/sim.py --target 0 -0.5 0 -2.0 0 1.8 0.5
    python src/sim.py --no-gravity-comp   # feel the arm sag under pure PD

Controls: the viewer is interactive (drag to orbit, scroll to zoom).
Press ESC or close the window to quit.
"""
import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from pd_controller import JointPD
from robot import ARM, ARM_DOF, home_qpos, load_panda, reset_to_home


def parse_args():
    p = argparse.ArgumentParser(description="Panda PD sim loop")
    p.add_argument("--target", type=float, nargs=ARM_DOF, default=None,
                   help="7 target joint angles (rad). Default: home pose.")
    p.add_argument("--no-gravity-comp", action="store_true",
                   help="Disable gravity compensation (pure PD).")
    return p.parse_args()


def main():
    args = parse_args()
    model, data = load_panda()
    reset_to_home(model, data)

    q_home = home_qpos(model)
    q_des = np.array(args.target) if args.target is not None else q_home[:ARM_DOF]

    pd = JointPD(gravity_comp=not args.no_gravity_comp)

    print(f"Gravity compensation: {pd.gravity_comp}")
    print(f"Target (rad): {np.array2string(q_des, precision=3)}")
    print("Launching viewer... (ESC or close window to quit)")

    dt = model.opt.timestep
    with mujoco.viewer.launch_passive(model, data) as viewer:
        last_print = 0.0
        while viewer.is_running():
            step_start = time.perf_counter()

            # --- control at every physics step ---
            data.ctrl[ARM] = pd(model, data, q_des)
            mujoco.mj_step(model, data)

            # periodic tracking report
            if data.time - last_print >= 0.5:
                err = np.linalg.norm(q_des - data.qpos[ARM])
                print(f"t={data.time:6.2f}s   |q_des - q| = {err:.4f} rad")
                last_print = data.time

            viewer.sync()

            # pace the loop to real time
            sleep = dt - (time.perf_counter() - step_start)
            if sleep > 0:
                time.sleep(sleep)


if __name__ == "__main__":
    main()
