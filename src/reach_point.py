"""Say where the end-effector should go in Cartesian space; IK + PD take it there.

    python src/reach_point.py --pos 0.5 0.2 0.4              # position-only IK
    python src/reach_point.py --pos 0.5 0.2 0.4 --keep-orient # also hold a top-down grasp
    python src/reach_point.py --pos 0.5 0.2 0.4 --headless    # no window, prints result

Pipeline:  target (x,y,z)  --IK-->  q_des  --PD-->  arm drives there.
A red sphere marks the commanded target in the viewer; the printout reports how
close the end-effector actually got after the controller settles.
"""
import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from ik import IKSolver, tcp_from_data
from pd_controller import JointPD
from robot import ARM, ARM_DOF, home_qpos, load_panda, reset_to_home

SETTLE_SECONDS = 4.0


def parse_args():
    p = argparse.ArgumentParser(description="Cartesian reach via IK + PD")
    p.add_argument("--pos", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"),
                   help="Cartesian target for the end-effector (m, world frame).")
    p.add_argument("--keep-orient", action="store_true",
                   help="Also solve for orientation, holding the home (top-down) grasp.")
    p.add_argument("--headless", action="store_true", help="No viewer; verify and exit.")
    return p.parse_args()


def solve_target(model, data, target_pos, keep_orient):
    reset_to_home(model, data)
    mujoco.mj_forward(model, data)
    _, home_quat = tcp_from_data(model, data)

    solver = IKSolver(model)
    target_quat = home_quat if keep_orient else None
    res = solver.solve(target_pos, target_quat=target_quat, q_init=home_qpos(model))

    print(f"target pos : {np.array2string(np.asarray(target_pos), precision=3)}")
    print(f"IK success : {res.success}  (iters={res.iters}, "
          f"pos_err={res.pos_err*1000:.2f} mm, ori_err={np.degrees(res.ori_err):.2f} deg)")
    print(f"q_des (arm): {np.array2string(res.q[:ARM_DOF], precision=3)}")
    if not res.success:
        print("  WARNING: IK did not converge -- target may be out of reach.")
    return res.q[:ARM_DOF]


def achieved_error(model, data, target_pos):
    mujoco.mj_forward(model, data)
    tcp_pos, _ = tcp_from_data(model, data)
    return np.linalg.norm(tcp_pos - target_pos), tcp_pos


def run_headless(model, data, q_des, target_pos):
    pd = JointPD()
    n = int(SETTLE_SECONDS / model.opt.timestep)
    for _ in range(n):
        data.ctrl[ARM] = pd(model, data, q_des)
        mujoco.mj_step(model, data)
    err, tcp = achieved_error(model, data, target_pos)
    print(f"\nafter settling: TCP={np.array2string(tcp, precision=3)}  "
          f"error={err*1000:.2f} mm  ->  {'PASS' if err < 5e-3 else 'CHECK'}")


def add_target_marker(viewer, pos):
    viewer.user_scn.ngeom = 1
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[0], type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.02, 0, 0], pos=np.asarray(pos, dtype=float),
        mat=np.eye(3).flatten(), rgba=[1, 0, 0, 0.6])


def run_viewer(model, data, q_des, target_pos):
    pd = JointPD()
    dt = model.opt.timestep
    print("Launching viewer... (red sphere = target; ESC to quit)")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        add_target_marker(viewer, target_pos)
        last = 0.0
        while viewer.is_running():
            t0 = time.perf_counter()
            data.ctrl[ARM] = pd(model, data, q_des)
            mujoco.mj_step(model, data)
            if data.time - last >= 0.5:
                err, _ = achieved_error(model, data, target_pos)
                print(f"t={data.time:5.2f}s  EE error = {err*1000:6.2f} mm")
                last = data.time
            viewer.sync()
            sleep = dt - (time.perf_counter() - t0)
            if sleep > 0:
                time.sleep(sleep)


def main():
    args = parse_args()
    model, data = load_panda()
    q_des = solve_target(model, data, args.pos, args.keep_orient)

    reset_to_home(model, data)   # start the drive from home
    if args.headless:
        run_headless(model, data, q_des, args.pos)
    else:
        run_viewer(model, data, q_des, args.pos)


if __name__ == "__main__":
    main()
