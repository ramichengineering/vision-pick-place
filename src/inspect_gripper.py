"""Measure the gripper: fingertip geometry, tendon, and the true grasp point.

The TCP used for reaching (10 cm out from the hand) was never checked against
where the fingers actually close. For grasping that matters: the cube has to end
up BETWEEN the pads, not past them.
"""
import mujoco
import numpy as np

from robot import load_pick_scene, reset_to_home

m, d = load_pick_scene()
reset_to_home(m, d, key="pick_home")
mujoco.mj_forward(m, d)

hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
R = d.xmat[hand].reshape(3, 3)
hand_pos = d.xpos[hand]


def in_hand_frame(world_pt):
    return R.T @ (world_pt - hand_pos)


print(f"hand world pos: {np.array2string(hand_pos, precision=4)}")
print("\n-- finger bodies, in HAND frame --")
for name in ("left_finger", "right_finger"):
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    print(f"  {name:14s} {np.array2string(in_hand_frame(d.xpos[bid]), precision=4)}")

print("\n-- geoms on the fingers, in HAND frame (z = along approach axis) --")
for g in range(m.ngeom):
    bid = m.geom_bodyid[g]
    bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid)
    if bname and "finger" in bname:
        gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"
        local = in_hand_frame(d.geom_xpos[g])
        print(f"  {gname:34s} {np.array2string(local, precision=4)} "
              f"size={np.array2string(m.geom_size[g], precision=4)}")

print("\n-- finger joints --")
for jname in ("finger_joint1", "finger_joint2"):
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
    adr = m.jnt_qposadr[jid]
    print(f"  {jname}: qpos={d.qpos[adr]:.4f} range={m.jnt_range[jid]} "
          f"axis={m.jnt_axis[jid]}")

print("\n-- gripper actuator (index 7) --")
print(f"  gaintype={m.actuator_gaintype[7]} biastype={m.actuator_biastype[7]}")
print(f"  gainprm[:3]={m.actuator_gainprm[7,:3]}  biasprm[:3]={m.actuator_biasprm[7,:3]}")
print(f"  ctrlrange={m.actuator_ctrlrange[7]}  forcerange={m.actuator_forcerange[7]}")
print(f"  current ctrl={d.ctrl[7]}")

print("\n-- cube --")
cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
print(f"  half-extents={np.array2string(m.geom_size[cid], precision=4)} "
      f"(full width {2*m.geom_size[cid][0]:.3f} m)")
print(f"  mass={m.body_mass[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,'cube')]:.4f} kg")
