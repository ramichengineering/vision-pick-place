"""A hand-written joint-space PD controller for the Panda arm.

    tau = kp * (q_des - q) - kd * qdot   [ + gravity/Coriolis compensation ]

Gravity compensation (`data.qfrc_bias`, which is gravity + Coriolis +
centrifugal terms) is optional: turn it off to see why a 7-DOF arm under
gravity sags and needs huge kp, and on to see it hold effortlessly.

Choosing the gains
------------------
Hand-picking 14 numbers is guesswork, and the first version of this file did
exactly that (kp = [600,600,600,600,250,150,50]). The problem: joints 1-4 all
got kp=600 despite their effective inertias spanning 1.06-3.50 kg m^2, so the
achieved damping ratio wandered from 0.55 (joint 2 rings) to 1.18, while the
wrist sat at zeta = 1.6-2.2 -- badly overdamped and needlessly slow.

Scaling the gains by each joint's inertia collapses the problem to two knobs:

    kp_i = wn^2 * M_i           wn    = natural frequency (rad/s)
    kd_i = 2 * zeta * wn * M_i  zeta  = damping ratio

M_i is the *largest* effective inertia joint i sees across the workspace, so
every joint answers at the same wn, and the achieved zeta is >= the target
everywhere (lighter poses are only more damped).

What limits wn (measured, see tune_gains.py)
--------------------------------------------
Not motor torque -- peak usage is only 43% of the ceiling -- and not the classic
explicit-integration limit, which would allow kp ~100x higher. The binding
constraint is a discrete-time instability in the *damping* term: because the
damper acts on the previous step's velocity, too large a kd makes the correction
overshoot and flip sign every step, producing a 250 Hz ring -- exactly Nyquist
for the 2 ms timestep. Empirically the boundary is

    zeta * wn  <~ 27          (equivalently  kd_i <~ 54 * M_i)

and it is sharp: zeta*wn = 27.2 tracks a smooth trajectory perfectly yet
chatters violently the moment anything disturbs it. The defaults below sit at
zeta*wn = 24, about 12% clear of that cliff, verified chatter-free over 60
random workspace poses.

Versus the old hand-picked gains, on the same pick-and-place path:
    tracking lag   0.177 -> 0.039 rad   (4.5x tighter)
    overshoot      2.48% -> 0.73%
    disturbance settling 0.176 -> 0.092 s
"""
import numpy as np

from robot import ARM, ARM_DOF

# Largest effective inertia (mass-matrix diagonal, kg m^2) each joint sees across
# the workspace. Measured by analyze_gains.py over representative poses.
M_MAX = np.array([2.812, 3.504, 1.471, 1.129, 0.158, 0.154, 0.107])

# Tuned defaults. zeta*wn = 24, safely under the ~27 chatter boundary.
#
# Why zeta=0.8 rather than a faster-tracking 0.6: at equal zeta*wn (equal
# stability margin) the whole family -- (40,0.6), (34,0.7), (30,0.8), (24,1.0) --
# scores an identical 98% on the 50-trial benchmark at every speed, so task
# reliability does not pick a winner. What separates them is behaviour under
# inputs the trajectory generator never produces: a step command overshoots 31%
# at zeta=0.6 but only 3.9% at zeta=0.8, and the disturbance response rings half
# as long. Same speed, better margins, so take the damping.
WN = 30.0
ZETA = 0.8


def gains_for(wn=WN, zeta=ZETA, M=M_MAX):
    """Per-joint (kp, kd) for a target natural frequency and damping ratio."""
    return wn**2 * M, 2.0 * zeta * wn * M


class JointPD:
    def __init__(self, kp=None, kd=None, gravity_comp=True, wn=None, zeta=None):
        # Resolved at call time, not bound as default arguments, so that
        # experiments can retune by setting pd_controller.WN / ZETA.
        wn = WN if wn is None else wn
        zeta = ZETA if zeta is None else zeta
        kp_default, kd_default = gains_for(wn, zeta)
        self.kp = np.asarray(kp, dtype=float) if kp is not None else kp_default
        self.kd = np.asarray(kd, dtype=float) if kd is not None else kd_default
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


# The old hand-tuned gains, kept so the comparison is reproducible.
LEGACY_KP = np.array([600, 600, 600, 600, 250, 150, 50.0])
LEGACY_KD = np.array([50, 50, 50, 50, 20, 20, 10.0])
