"""A hand-written joint-space PD controller for the Panda arm.

    tau = kp * (q_des - q) - kd * qdot   [ + gravity/Coriolis compensation ]

Per-joint gains are stored as arrays so you can feel how each joint responds.
Gravity compensation (`data.qfrc_bias`, which is gravity + Coriolis +
centrifugal terms) is optional: turn it off to see why a 7-DOF arm under
gravity sags and needs huge kp, and on to see it hold effortlessly.
"""
import numpy as np

from robot import ARM, ARM_DOF


class JointPD:
    def __init__(self, kp=None, kd=None, gravity_comp=True):
        # Defaults: stiffer at the shoulder, softer toward the wrist.
        self.kp = np.array(kp if kp is not None else [600, 600, 600, 600, 250, 150, 50.0])
        self.kd = np.array(kd if kd is not None else [50, 50, 50, 50, 20, 20, 10.0])
        self.gravity_comp = gravity_comp
        assert self.kp.shape == (ARM_DOF,) and self.kd.shape == (ARM_DOF,)

    def __call__(self, model, data, q_des) -> np.ndarray:
        """Compute the torque command for the 7 arm actuators."""
        q = data.qpos[ARM]
        qdot = data.qvel[ARM]
        q_des = np.asarray(q_des)[:ARM_DOF]

        tau = self.kp * (q_des - q) - self.kd * qdot
        if self.gravity_comp:
            # qfrc_bias holds the generalized forces needed to counteract
            # gravity + Coriolis/centrifugal at the current state.
            tau = tau + data.qfrc_bias[ARM]
        return tau
