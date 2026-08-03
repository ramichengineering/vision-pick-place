"""What the current gains actually are, in physical terms.

A PD gain only means something relative to the inertia it is pushing and the
torque the motor can deliver. This prints, per joint:

  - M_ii, the mass-matrix diagonal (effective inertia), across several poses.
    It varies with configuration -- a stretched-out arm has far more inertia at
    the shoulder than a folded one.
  - the implied damping ratio  zeta = kd / (2*sqrt(kp * M_ii)).
    zeta < 1 rings, zeta = 1 is critically damped, zeta > 1 is sluggish.
  - the undamped natural frequency  wn = sqrt(kp / M_ii), which sets how fast
    the joint can possibly respond.
  - the torque ceiling, which is what ultimately caps the speed.
"""
import mujoco
import numpy as np

from pd_controller import JointPD
from robot import ARM, ARM_DOF, home_qpos, load_pick_scene, reset_to_home

POSES = {
    "home":      [0, 0, 0, -1.571, 0, 1.571, -0.785],
    "extended":  [0, 0.6, 0, -0.6, 0, 1.2, 0],
    "folded":    [0, -0.6, 0, -2.6, 0, 2.2, 0],
    "reach_far": [0, 0.9, 0, -0.3, 0, 1.0, 0],
}


def mass_diag(model, data, q):
    data.qpos[ARM] = q
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, M)
    return np.diag(M)[:ARM_DOF].copy()


def main():
    model, data = load_pick_scene()
    reset_to_home(model, data, key="pick_home")
    pd = JointPD()

    print("effective inertia M_ii (kg m^2), by pose")
    print(f"{'pose':<11}" + "".join(f"{'j'+str(i+1):>9}" for i in range(ARM_DOF)))
    all_M = {}
    for name, q in POSES.items():
        m_ii = mass_diag(model, data, q)
        all_M[name] = m_ii
        print(f"{name:<11}" + "".join(f"{v:9.3f}" for v in m_ii))

    M_lo = np.min(np.array(list(all_M.values())), axis=0)
    M_hi = np.max(np.array(list(all_M.values())), axis=0)
    print(f"{'range x':<11}" + "".join(f"{h/l:8.1f}x" for l, h in zip(M_lo, M_hi)))

    print(f"\ncurrent gains")
    print(f"{'kp':<11}" + "".join(f"{v:9.0f}" for v in pd.kp))
    print(f"{'kd':<11}" + "".join(f"{v:9.0f}" for v in pd.kd))

    print(f"\ndamping ratio zeta = kd / (2*sqrt(kp*M_ii))  [1.0 = critical]")
    for name in POSES:
        z = pd.kd / (2 * np.sqrt(pd.kp * all_M[name]))
        print(f"{name:<11}" + "".join(f"{v:9.2f}" for v in z))

    print(f"\nnatural frequency wn = sqrt(kp/M_ii)  (rad/s), at home")
    wn = np.sqrt(pd.kp / all_M["home"])
    print(f"{'wn':<11}" + "".join(f"{v:9.1f}" for v in wn))
    print(f"{'f (Hz)':<11}" + "".join(f"{v/(2*np.pi):9.1f}" for v in wn))

    print(f"\ntorque ceiling (N m) -- the real speed limit")
    fr = model.actuator_forcerange[:ARM_DOF, 1]
    print(f"{'max tau':<11}" + "".join(f"{v:9.0f}" for v in fr))

    # Gravity load at each pose tells us how much of that ceiling is already spent.
    print(f"\ngravity/bias torque as % of ceiling")
    for name, q in POSES.items():
        data.qpos[ARM] = q
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        frac = 100 * np.abs(data.qfrc_bias[ARM]) / fr
        print(f"{name:<11}" + "".join(f"{v:8.0f}%" for v in frac))

    dt = model.opt.timestep
    print(f"\nstability limit: timestep {dt*1000:.0f} ms -> explicit-integration")
    print(f"rule of thumb wn*dt < ~0.5 gives kp_max ~ M_ii*(0.5/dt)^2")
    print(f"{'kp_max':<11}" + "".join(f"{v:9.0f}" for v in all_M["home"] * (0.5/dt)**2))


if __name__ == "__main__":
    main()
