"""Verify the vision pipeline against ground truth.

Moves the cube to several known positions, estimates each from the camera image
alone, and reports the error. Ground truth is only used to score the estimate --
never fed into the estimator.

    python src/perception_test.py
    python src/perception_test.py --camera top_cam
    python src/perception_test.py --save   # dump rgb/mask PNGs for inspection
"""
import argparse

import mujoco
import numpy as np

from perception import estimate_cube
from robot import load_pick_scene, reset_to_home

CUBE_QPOS_ADR = None  # resolved at runtime


def set_cube_pose(model, data, xyz):
    """Teleport the cube (free joint) to a world position."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
    adr = model.jnt_qposadr[jid]
    data.qpos[adr:adr + 3] = xyz
    data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)


def true_cube_pos(model, data):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    return data.xpos[bid].copy()


def save_debug(det, tag):
    try:
        from PIL import Image
    except ImportError:
        print("  (install pillow to save debug images)")
        return
    Image.fromarray(det.rgb).save(f"debug_{tag}_rgb.png")
    Image.fromarray((det.mask * 255).astype(np.uint8)).save(f"debug_{tag}_mask.png")
    print(f"  saved debug_{tag}_rgb.png / debug_{tag}_mask.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", default="scene_cam")
    p.add_argument("--save", action="store_true")
    args = p.parse_args()

    model, data = load_pick_scene()
    reset_to_home(model, data, key="pick_home")

    # A spread of reachable cube placements on the floor (z = half-extent).
    targets = [
        (0.50, 0.00, 0.025),
        (0.45, 0.15, 0.025),
        (0.55, -0.15, 0.025),
        (0.60, 0.10, 0.025),
        (0.40, -0.10, 0.025),
    ]

    print(f"camera: {args.camera}\n")
    print(f"{'true (x,y,z)':>24} {'estimated':>24} {'err mm':>8} {'px':>6}")
    print("-" * 66)

    errors = []
    for i, xyz in enumerate(targets):
        set_cube_pose(model, data, xyz)
        truth = true_cube_pos(model, data)
        det = estimate_cube(model, data, camera=args.camera, keep_images=args.save)

        if not det.found:
            print(f"{np.array2string(truth, precision=3):>24}   NOT DETECTED "
                  f"({det.n_pixels} px)")
            continue

        err = np.linalg.norm(det.position - truth) * 1000
        errors.append(err)
        print(f"{np.array2string(truth, precision=3):>24} "
              f"{np.array2string(det.position, precision=3):>24} "
              f"{err:8.1f} {det.n_pixels:6d}")
        if args.save and i == 0:
            save_debug(det, args.camera)

    if errors:
        print("-" * 66)
        print(f"mean error {np.mean(errors):.1f} mm | max {np.max(errors):.1f} mm")
        print("RESULT:", "PASS" if np.max(errors) < 15 else "NEEDS WORK")


if __name__ == "__main__":
    main()
