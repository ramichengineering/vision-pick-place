"""Headless check that the PD controller reaches a target smoothly.

Runs the physics with no viewer, logs the joint trajectory, and reports:
  - steady-state error   (are we actually there?)
  - overshoot per joint  (did we blow past the target?)
  - settling time        (when did error drop below tolerance for good?)

Usage:
    python src/reach_test.py
    python src/reach_test.py --no-gravity-comp
"""
import argparse

import mujoco
import numpy as np

from pd_controller import JointPD
from robot import ARM, ARM_DOF, home_qpos, load_panda, reset_to_home

SIM_SECONDS = 4.0
SETTLE_TOL = 0.01  # rad, ~0.6 deg


def run(q_des, gravity_comp=True):
    model, data = load_panda()
    reset_to_home(model, data)
    pd = JointPD(gravity_comp=gravity_comp)

    n_steps = int(SIM_SECONDS / model.opt.timestep)
    q_log = np.zeros((n_steps, ARM_DOF))
    t_log = np.zeros(n_steps)

    for k in range(n_steps):
        data.ctrl[ARM] = pd(model, data, q_des)
        mujoco.mj_step(model, data)
        q_log[k] = data.qpos[ARM]
        t_log[k] = data.time
    return t_log, q_log


def report(t_log, q_log, q_start, q_des):
    err_norm = np.linalg.norm(q_log - q_des, axis=1)
    final_err = err_norm[-1]

    # Overshoot: how far past the target each joint went, relative to travel.
    travel = q_des - q_start
    overshoot = np.zeros(ARM_DOF)
    for j in range(ARM_DOF):
        if abs(travel[j]) < 1e-6:
            continue
        # signed progress beyond the target in the direction of travel
        beyond = (q_log[:, j] - q_des[j]) * np.sign(travel[j])
        overshoot[j] = max(0.0, beyond.max()) / abs(travel[j]) * 100.0

    # Settling time: last moment error dips below tol and stays there.
    settled = np.where(err_norm > SETTLE_TOL)[0]
    if len(settled) == 0:
        settle_str = "0.00 s (already within tol)"
    elif settled[-1] + 1 < len(t_log):
        settle_str = f"{t_log[settled[-1] + 1]:.2f} s"
    else:
        settle_str = f"did not settle within {SIM_SECONDS:.0f} s"

    print(f"  steady-state error : {final_err:.5f} rad")
    print(f"  settling time      : {settle_str}  (tol {SETTLE_TOL} rad)")
    print(f"  max overshoot      : {overshoot.max():.1f} %  (per-joint: "
          f"{np.array2string(overshoot, precision=1, suppress_small=True)})")
    ok = final_err < SETTLE_TOL and overshoot.max() < 5.0
    print(f"  RESULT             : {'PASS' if ok else 'NEEDS TUNING'}")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-gravity-comp", action="store_true")
    args = p.parse_args()

    model, _ = load_panda()
    q_start = home_qpos(model)[:ARM_DOF]
    # A clearly different, reachable configuration to reach for.
    q_des = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.8, 0.5])

    print(f"start : {np.array2string(q_start, precision=3)}")
    print(f"target: {np.array2string(q_des, precision=3)}")
    print(f"gravity_comp = {not args.no_gravity_comp}\n")

    t_log, q_log = run(q_des, gravity_comp=not args.no_gravity_comp)
    report(t_log, q_log, q_start, q_des)


if __name__ == "__main__":
    main()
