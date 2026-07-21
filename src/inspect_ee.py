"""Find a good end-effector reference (sites/bodies) and print joint limits."""
from robot import ARM_DOF, load_panda, reset_to_home
import mujoco
import numpy as np

model, data = load_panda()
reset_to_home(model, data)
mujoco.mj_forward(model, data)

print("Sites:")
for i in range(model.nsite):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
    print(f"  {i}: {name:20s} pos(world)={np.array2string(data.site_xpos[i], precision=3)}")

print("\nBodies (last few):")
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    if name and ("hand" in name or "link7" in name or "finger" in name):
        print(f"  {i}: {name:20s} pos(world)={np.array2string(data.xpos[i], precision=3)}")

print("\nArm joint limits (rad):")
for j in range(ARM_DOF):
    lo, hi = model.jnt_range[j]
    print(f"  joint{j+1}: [{lo:+.3f}, {hi:+.3f}]  limited={bool(model.jnt_limited[j])}")
