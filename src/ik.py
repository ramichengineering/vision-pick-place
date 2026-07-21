"""Damped-least-squares inverse kinematics for the Panda, using MuJoCo Jacobians.

Problem: given a Cartesian target for the end-effector, find joint angles q that
put it there. There's no closed-form inverse for a 7-DOF arm, so we solve it
*iteratively* by linearizing around the current guess.

The math
--------
Forward kinematics x = f(q) maps joints -> end-effector pose. Its derivative is
the Jacobian J = df/dq (6 x n): it maps joint velocities to end-effector
velocity,  dx = J dq. We want the joint change dq that produces the pose error
e = x_target - x_current, i.e. solve  J dq = e.

J isn't square/invertible (6 rows, 7 arm cols), and near singularities the naive
pseudo-inverse blows up. Damped least squares (Levenberg-Marquardt) fixes both:

    dq = J^T (J J^T + lambda^2 I)^-1 e

The lambda^2 term trades a little accuracy for stability -- it caps dq near
singularities instead of letting it explode. Iterate q <- q + step*dq until the
error is small, clamping to joint limits each step.
"""
from dataclasses import dataclass

import mujoco
import numpy as np

from robot import ARM, ARM_DOF, load_panda

# TCP: grasp point ~10 cm out from the hand origin along its local +z
# (the gripper's approach axis, pointing toward the fingertips).
HAND_BODY = "hand"
TCP_OFFSET = np.array([0.0, 0.0, 0.10])


def tcp_from_data(model, data):
    """World (position, quaternion) of the TCP given a live MjData (mj_forward done)."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, HAND_BODY)
    R = data.xmat[bid].reshape(3, 3)
    return data.xpos[bid] + R @ TCP_OFFSET, data.xquat[bid].copy()


@dataclass
class IKResult:
    q: np.ndarray          # full qpos solution (all 9 joints)
    success: bool
    iters: int
    pos_err: float         # final position error (m)
    ori_err: float         # final orientation error (rad)


class IKSolver:
    def __init__(self, model, damping=1e-2, step=0.5, pos_tol=1e-3, ori_tol=1e-2,
                 max_iters=200):
        self.model = model
        self.data = mujoco.MjData(model)   # scratch state; never touches the live sim
        self.hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, HAND_BODY)
        self.damping = damping
        self.step = step
        self.pos_tol = pos_tol
        self.ori_tol = ori_tol
        self.max_iters = max_iters

        # Preallocate Jacobian buffers (3 x nv each).
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def tcp_pose(self):
        """Return (world position, world quaternion) of the TCP for current data."""
        d = self.data
        R = d.xmat[self.hand_id].reshape(3, 3)
        pos = d.xpos[self.hand_id] + R @ TCP_OFFSET
        return pos, d.xquat[self.hand_id].copy()

    def solve(self, target_pos, target_quat=None, q_init=None) -> IKResult:
        m, d = self.model, self.data
        target_pos = np.asarray(target_pos, dtype=float)
        use_ori = target_quat is not None

        if q_init is not None:
            d.qpos[:] = q_init
        d.qvel[:] = 0

        err = np.zeros(6)
        neg, qerr = np.zeros(4), np.zeros(4)

        for i in range(self.max_iters):
            mujoco.mj_kinematics(m, d)      # forward kinematics only (fast)
            mujoco.mj_comPos(m, d)          # needed before mj_jac

            tcp_pos, tcp_quat = self.tcp_pose()
            err[:3] = target_pos - tcp_pos

            if use_ori:
                # world-frame orientation error: q_err = target * conj(current)
                mujoco.mju_negQuat(neg, tcp_quat)
                mujoco.mju_mulQuat(qerr, target_quat, neg)
                mujoco.mju_quat2Vel(err[3:], qerr, 1.0)
            else:
                err[3:] = 0.0

            pos_err = np.linalg.norm(err[:3])
            ori_err = np.linalg.norm(err[3:]) if use_ori else 0.0
            if pos_err < self.pos_tol and (not use_ori or ori_err < self.ori_tol):
                return IKResult(d.qpos.copy(), True, i, pos_err, ori_err)

            # Jacobian of the TCP point on the hand body.
            mujoco.mj_jac(m, d, self._jacp, self._jacr, tcp_pos, self.hand_id)
            if use_ori:
                J = np.vstack([self._jacp[:, :ARM_DOF], self._jacr[:, :ARM_DOF]])
                e = err
            else:
                J = self._jacp[:, :ARM_DOF]
                e = err[:3]

            # Damped least squares:  dq = J^T (J J^T + lambda^2 I)^-1 e
            JJt = J @ J.T
            lam2 = self.damping ** 2
            dq = J.T @ np.linalg.solve(JJt + lam2 * np.eye(JJt.shape[0]), e)

            d.qpos[ARM] += self.step * dq
            self._clamp_limits(d)

        return IKResult(d.qpos.copy(), False, self.max_iters, pos_err, ori_err)

    def _clamp_limits(self, d):
        m = self.model
        for j in range(ARM_DOF):
            if m.jnt_limited[j]:
                lo, hi = m.jnt_range[j]
                d.qpos[j] = np.clip(d.qpos[j], lo, hi)


if __name__ == "__main__":
    # Quick self-test: pick a reachable pose, solve, report accuracy.
    model, data = load_panda()
    solver = IKSolver(model)

    # Home-pose TCP as a sanity anchor, then target a point 15 cm forward + down.
    from robot import reset_to_home
    reset_to_home(model, data)
    mujoco.mj_forward(model, data)
    solver.data.qpos[:] = data.qpos
    mujoco.mj_kinematics(model, solver.data)
    home_tcp, home_quat = solver.tcp_pose()
    print(f"home TCP: {np.array2string(home_tcp, precision=3)}")

    target = home_tcp + np.array([0.15, 0.10, -0.20])
    res = solver.solve(target, target_quat=home_quat, q_init=data.qpos)
    print(f"target  : {np.array2string(target, precision=3)}")
    print(f"success={res.success}  iters={res.iters}  "
          f"pos_err={res.pos_err*1000:.2f} mm  ori_err={np.degrees(res.ori_err):.2f} deg")
    print(f"q_arm   : {np.array2string(res.q[:ARM_DOF], precision=3)}")
