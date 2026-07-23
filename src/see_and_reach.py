"""Close the loop: look at the cube, work out where it is, drive the arm there.

The arm is told the cube's location by the CAMERA ONLY -- ground truth is read
afterwards purely to score the result.

    python src/see_and_reach.py                      # viewer
    python src/see_and_reach.py --headless           # verify and exit
    python src/see_and_reach.py --cube 0.55 -0.12 0.025
    python src/see_and_reach.py --random             # random cube placement

Pipeline:  image -> colour segmentation -> depth back-projection -> (x,y,z)
           -> IK -> joint targets -> PD controller -> arm moves.
"""
import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from ik import IKSolver, tcp_from_data
from pd_controller import JointPD
from perception import estimate_cube
from perception_test import set_cube_pose, true_cube_pos
from robot import ARM, ARM_DOF, home_qpos, load_pick_scene, reset_to_home

# Stand off above the cube rather than driving into it (Day 7 does the grasp).
APPROACH_HEIGHT = 0.10
SETTLE_SECONDS = 4.0


def parse_args():
    p = argparse.ArgumentParser(description="Vision-guided Cartesian reach")
    p.add_argument("--cube", type=float, nargs=3, default=[0.5, 0.0, 0.025],
                   metavar=("X", "Y", "Z"), help="Where to place the cube.")
    p.add_argument("--random", action="store_true",
                   help="Place the cube at a random reachable spot.")
    p.add_argument("--camera", default="scene_cam")
    p.add_argument("--headless", action="store_true")
    return p.parse_args()


def perceive(model, data, camera):
    det = estimate_cube(model, data, camera=camera)
    if not det.found:
        raise SystemExit(f"Cube not detected from '{camera}' "
                         f"({det.n_pixels} red px). Is it occluded?")
    print(f"[vision] {det.n_pixels} red px at pixel "
          f"({det.uv[0]:.0f}, {det.uv[1]:.0f})")
    print(f"[vision] estimated cube position: "
          f"{np.array2string(det.position, precision=4)}")
    return det


def plan(model, data, cube_xyz):
    """IK to a point APPROACH_HEIGHT above the perceived cube."""
    goal = np.array(cube_xyz) + np.array([0, 0, APPROACH_HEIGHT])
    reset_to_home(model, data, key="pick_home")
    mujoco.mj_forward(model, data)
    _, down_quat = tcp_from_data(model, data)   # home orientation = gripper down

    solver = IKSolver(model)
    res = solver.solve(goal, target_quat=down_quat,
                       q_init=home_qpos(model, key="pick_home"))
    print(f"[ik]     goal {np.array2string(goal, precision=3)} -> "
          f"success={res.success} iters={res.iters} "
          f"err={res.pos_err*1000:.2f} mm")
    if not res.success:
        print("[ik]     WARNING: did not converge")
    return goal, res.q[:ARM_DOF]


def drive_headless(model, data, q_des):
    pd = JointPD()
    for _ in range(int(SETTLE_SECONDS / model.opt.timestep)):
        data.ctrl[ARM] = pd(model, data, q_des)
        mujoco.mj_step(model, data)


def drive_viewer(model, data, q_des, goal):
    pd = JointPD()
    dt = model.opt.timestep
    print("[sim]    launching viewer (ESC to quit)")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.user_scn.ngeom = 1
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[0], type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.015, 0, 0], pos=goal, mat=np.eye(3).flatten(),
            rgba=[0, 1, 0, 0.5])
        while viewer.is_running():
            t0 = time.perf_counter()
            data.ctrl[ARM] = pd(model, data, q_des)
            mujoco.mj_step(model, data)
            viewer.sync()
            s = dt - (time.perf_counter() - t0)
            if s > 0:
                time.sleep(s)


def main():
    args = parse_args()
    model, data = load_pick_scene()
    reset_to_home(model, data, key="pick_home")

    cube_xyz = args.cube
    if args.random:
        rng = np.random.default_rng()
        cube_xyz = [rng.uniform(0.40, 0.60), rng.uniform(-0.18, 0.18), 0.025]
    set_cube_pose(model, data, cube_xyz)

    # --- 1. SEE (camera only) ---
    det = perceive(model, data, args.camera)

    # --- 2. PLAN (IK from the perceived position) ---
    goal, q_des = plan(model, data, det.position)

    # --- 3. ACT (PD controller drives the arm) ---
    set_cube_pose(model, data, cube_xyz)   # restore cube after IK scratch resets
    if args.headless:
        drive_headless(model, data, q_des)
    else:
        drive_viewer(model, data, q_des, goal)

    # --- score it (ground truth used ONLY here) ---
    mujoco.mj_forward(model, data)
    truth = true_cube_pos(model, data)
    vision_err = np.linalg.norm(det.position - truth) * 1000
    tcp, _ = tcp_from_data(model, data)
    reach_err = np.linalg.norm(tcp - goal) * 1000
    xy_err = np.linalg.norm(tcp[:2] - truth[:2]) * 1000

    print(f"\n[score]  true cube pos      : {np.array2string(truth, precision=4)}")
    print(f"[score]  vision error       : {vision_err:.1f} mm")
    print(f"[score]  TCP vs planned goal: {reach_err:.1f} mm")
    print(f"[score]  TCP xy vs cube xy  : {xy_err:.1f} mm")
    ok = vision_err < 10 and reach_err < 10
    print(f"[score]  RESULT: {'PASS' if ok else 'CHECK'}")


if __name__ == "__main__":
    main()
