"""Find the fastest gains that still move smoothly.

Parameterization
----------------
Hand-picking 14 numbers is guesswork. A second-order system has only two
meaningful knobs, so scale the gains by each joint's inertia instead:

    kp_i = wn^2 * M_i          natural frequency wn (rad/s), same for every joint
    kd_i = 2 * zeta * wn * M_i  damping ratio zeta (1.0 = critically damped)

M_i is the *largest* effective inertia the joint sees across the workspace, so
the achieved zeta is >= the target everywhere (lighter poses are only more
damped). That guarantees no ringing anywhere, which is what "smooth" means here.

wn then becomes a single speed dial, and the search is one-dimensional.

What limits wn
--------------
Not numerical stability (kp could go ~100x higher before the 2 ms timestep
complains) but *torque*. Joints 5-7 have only a 12 N m ceiling, and gravity
already spends a fifth of it. Once the controller commands more torque than the
motor can deliver, the command is clipped, the arm stops tracking the plan, and
the motion is no longer smooth or predictable. So the search raises wn until
torque saturates or overshoot appears.

    python src/tune_gains.py              # sweep wn, report the frontier
    python src/tune_gains.py --zeta-sweep # check the damping choice
    python src/tune_gains.py --speed      # with the best gains, how fast?
"""
import argparse

import mujoco
import numpy as np

from ik import IKSolver, tcp_from_data
from pd_controller import LEGACY_KD, LEGACY_KP, M_MAX, JointPD, gains_for
from robot import ARM, ARM_DOF, home_qpos, load_pick_scene, reset_to_home

SETTLE_TOL = 0.01     # rad, the same tolerance used since day 1
OVERSHOOT_LIMIT = 2.0  # percent of travel
SAT_LIMIT = 0.5        # percent of steps allowed to hit the torque ceiling


def gains_from(wn, zeta, M=M_MAX):
    """kp, kd for a target natural frequency and damping ratio."""
    return gains_for(wn, zeta, M)


def minimum_jerk(s):
    s = np.clip(s, 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def waypoints(model):
    """Joint configurations along a realistic pick-and-place path."""
    solver = IKSolver(model)
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = home_qpos(model, key="pick_home")
    mujoco.mj_forward(model, scratch)
    quat = tcp_from_data(model, scratch)[1]

    q_home = home_qpos(model, key="pick_home")[:ARM_DOF]
    pts = [("pre-grasp", (0.50, 0.0, 0.125)),
           ("grasp",     (0.50, 0.0, 0.025)),
           ("lift",      (0.50, 0.0, 0.205)),
           ("over-bin",  (0.40, -0.35, 0.22)),
           ("release",   (0.40, -0.35, 0.10))]
    qs = [("home", q_home)]
    q_prev = home_qpos(model, key="pick_home")
    for name, p in pts:
        r = solver.solve(np.array(p), target_quat=quat, q_init=q_prev)
        qs.append((name, r.q[:ARM_DOF].copy()))
        q_prev = r.q.copy()
    return qs


def simulate_move(model, data, kp, kd, q_start, q_goal, move_time,
                  settle_time=1.0, payload=False):
    """Track a minimum-jerk move and return performance metrics."""
    reset_to_home(model, data, key="pick_home")
    data.qpos[ARM] = q_start
    data.qvel[:] = 0
    if payload:
        # Park the cube in the gripper so the wrist carries a real load.
        adr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        adr = model.jnt_qposadr[adr]
        mujoco.mj_forward(model, data)
        tcp, _ = tcp_from_data(model, data)
        data.qpos[adr:adr + 3] = tcp
        data.ctrl[7] = 0.0
    mujoco.mj_forward(model, data)

    pd = JointPD(kp=kp, kd=kd)
    limits = model.actuator_forcerange[:ARM_DOF, 1]
    dt = model.opt.timestep
    n = int((move_time + settle_time) / dt)

    q_log = np.zeros((n, ARM_DOF))
    tau_log = np.zeros((n, ARM_DOF))
    travel = q_goal - q_start

    for k in range(n):
        s = minimum_jerk((k * dt) / move_time) if move_time > 0 else 1.0
        q_des = q_start + s * travel
        tau = pd(model, data, q_des)
        tau_log[k] = tau
        data.ctrl[ARM] = tau
        mujoco.mj_step(model, data)
        q_log[k] = data.qpos[ARM]

    # --- metrics ---
    err = np.linalg.norm(q_log - q_goal, axis=1)
    final_err = float(err[-1])

    overshoot = np.zeros(ARM_DOF)
    for j in range(ARM_DOF):
        if abs(travel[j]) < 1e-6:
            continue
        beyond = (q_log[:, j] - q_goal[j]) * np.sign(travel[j])
        overshoot[j] = max(0.0, beyond.max()) / abs(travel[j]) * 100

    over_tol = np.where(err > SETTLE_TOL)[0]
    settle = (over_tol[-1] + 1) * dt if len(over_tol) and over_tol[-1] + 1 < n else (
        0.0 if len(over_tol) == 0 else np.inf)

    # Torque headroom: how close the command came to the motor ceiling.
    usage = np.abs(tau_log) / limits
    sat_pct = 100.0 * np.mean(np.any(usage >= 1.0, axis=1))
    peak_usage = float(usage.max())

    # Tracking lag during the move itself (not just the endpoint).
    move_steps = max(1, int(move_time / dt))
    s_arr = minimum_jerk(np.arange(move_steps) * dt / move_time)
    q_ref = q_start + s_arr[:, None] * travel
    lag = float(np.abs(q_log[:move_steps] - q_ref).max())

    return dict(final_err=final_err, settle=float(settle),
                overshoot=float(overshoot.max()), sat_pct=float(sat_pct),
                peak_usage=peak_usage, lag=lag)


def evaluate(model, data, kp, kd, move_time, payload=False):
    """Worst-case metrics across every leg of the pick-and-place path."""
    qs = waypoints(model)
    worst = dict(final_err=0, settle=0, overshoot=0, sat_pct=0, peak_usage=0, lag=0)
    for (_, qa), (_, qb) in zip(qs[:-1], qs[1:]):
        m = simulate_move(model, data, kp, kd, qa, qb, move_time, payload=payload)
        for k in worst:
            worst[k] = max(worst[k], m[k])
    return worst


def smooth(m):
    """Does this configuration count as smooth?"""
    return (m["overshoot"] <= OVERSHOOT_LIMIT and m["sat_pct"] <= SAT_LIMIT
            and m["final_err"] < SETTLE_TOL and np.isfinite(m["settle"]))


def sweep_wn(model, data, zeta, move_time, payload):
    print(f"wn sweep   zeta={zeta}  move_time={move_time}s  payload={payload}")
    print(f"{'wn':>5} {'kp(j2)':>8} {'kd(j2)':>7} {'over%':>7} {'settle':>7} "
          f"{'lag':>7} {'peakT':>7} {'sat%':>6}  verdict")
    best = None
    for wn in [10, 15, 20, 25, 30, 35, 40, 50, 60, 70]:
        kp, kd = gains_from(wn, zeta)
        m = evaluate(model, data, kp, kd, move_time, payload)
        ok = smooth(m)
        st = "inf" if not np.isfinite(m["settle"]) else f"{m['settle']:.2f}"
        print(f"{wn:>5} {kp[1]:8.0f} {kd[1]:7.0f} {m['overshoot']:7.2f} {st:>7} "
              f"{m['lag']:7.4f} {m['peak_usage']:7.2f} {m['sat_pct']:6.1f}  "
              f"{'smooth' if ok else 'REJECT'}")
        if ok:
            best = (wn, kp, kd, m)
    return best


def sweep_zeta(model, data, wn, move_time, payload):
    print(f"\nzeta sweep   wn={wn}  move_time={move_time}s")
    print(f"{'zeta':>6} {'over%':>7} {'settle':>7} {'lag':>7} {'peakT':>7}  verdict")
    for z in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4]:
        kp, kd = gains_from(wn, z)
        m = evaluate(model, data, kp, kd, move_time, payload)
        st = "inf" if not np.isfinite(m["settle"]) else f"{m['settle']:.2f}"
        print(f"{z:>6.1f} {m['overshoot']:7.2f} {st:>7} {m['lag']:7.4f} "
              f"{m['peak_usage']:7.2f}  {'smooth' if smooth(m) else 'REJECT'}")


def sweep_speed(model, data, wn, zeta, payload):
    """With gains fixed, how short can the move get before it stops being smooth?"""
    kp, kd = gains_from(wn, zeta)
    print(f"\nspeed sweep   wn={wn} zeta={zeta} payload={payload}")
    print(f"{'move_t':>7} {'over%':>7} {'lag':>7} {'peakT':>7} {'sat%':>6}  verdict")
    fastest = None
    for t in [2.5, 2.0, 1.6, 1.2, 1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
        m = evaluate(model, data, kp, kd, t, payload)
        ok = smooth(m)
        print(f"{t:>7.2f} {m['overshoot']:7.2f} {m['lag']:7.4f} "
              f"{m['peak_usage']:7.2f} {m['sat_pct']:6.1f}  "
              f"{'smooth' if ok else 'REJECT'}")
        if ok:
            fastest = t
    return fastest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zeta", type=float, default=1.0)
    p.add_argument("--wn", type=float, default=30.0)
    p.add_argument("--move-time", type=float, default=1.6)
    p.add_argument("--no-payload", dest="payload", action="store_false",
                   default=True, help="Run without the cube in the gripper.")
    p.add_argument("--zeta-sweep", action="store_true")
    p.add_argument("--speed", action="store_true")
    args = p.parse_args()

    model, data = load_pick_scene()

    if args.zeta_sweep:
        sweep_zeta(model, data, args.wn, args.move_time, args.payload)
    elif args.speed:
        sweep_speed(model, data, args.wn, args.zeta, args.payload)
    else:
        # Baseline: what the original hand-picked gains achieve on the same test.
        # Reference LEGACY_* explicitly -- JointPD() now defaults to the tuned
        # gains, so it would silently compare the new gains against themselves.
        m = evaluate(model, data, LEGACY_KP, LEGACY_KD, args.move_time, args.payload)
        print(f"baseline (hand-picked) : over={m['overshoot']:.2f}% "
              f"lag={m['lag']:.4f} peakT={m['peak_usage']:.2f} "
              f"sat={m['sat_pct']:.1f}%  {'smooth' if smooth(m) else 'REJECT'}\n")
        sweep_wn(model, data, args.zeta, args.move_time, args.payload)


if __name__ == "__main__":
    main()
