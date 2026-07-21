"""Load the Panda and convert its arm actuators to direct torque control.

The MuJoCo Menagerie Panda ships with `general` actuators configured as
position servos (an affine bias `[0, -kp, -kv]` = MuJoCo's own built-in PD).
For a hand-written PD controller we want *torque* control instead, so that the
only feedback loop acting on the arm is the one we write.

`load_panda()` rewrites the 7 arm actuators in-memory into fixed-gain motors:

    actuator_force = gainprm[0] * ctrl  ->  ctrl == joint torque (N*m)

The gripper actuator (a tendon position servo) is left untouched so the fingers
simply hold their commanded opening.
"""
from pathlib import Path

import mujoco
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "franka_emika_panda" / "scene.xml"

# The Panda arm is joints 0-6; joints 7-8 are the two gripper fingers.
ARM_DOF = 7
ARM = slice(0, ARM_DOF)


def load_panda(model_path: Path = MODEL_PATH):
    """Return (model, data) with the 7 arm actuators as direct torque motors."""
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # Convert arm actuators from position servos to fixed-gain torque motors.
    for i in range(ARM_DOF):
        model.actuator_gaintype[i] = mujoco.mjtGain.mjGAIN_FIXED
        model.actuator_biastype[i] = mujoco.mjtBias.mjBIAS_NONE
        model.actuator_gainprm[i, :] = 0.0
        model.actuator_gainprm[i, 0] = 1.0   # force = 1.0 * ctrl
        model.actuator_biasprm[i, :] = 0.0
        # The stock ctrlrange is the joint *angle* range; disable it so it does
        # not clamp our torque command. forcerange still limits torque safely.
        model.actuator_ctrllimited[i] = 0

    data = mujoco.MjData(model)
    return model, data


def home_qpos(model) -> np.ndarray:
    """Joint positions from the model's 'home' keyframe (all 9 joints)."""
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    return model.key_qpos[key_id].copy()


def reset_to_home(model, data) -> None:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
