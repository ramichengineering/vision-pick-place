"""Sanity-check the pick scene: keyframe padding, cube pose, camera params."""
from pathlib import Path

import mujoco
import numpy as np

PICK_SCENE = Path(__file__).resolve().parent.parent / "models" / "franka_emika_panda" / "pick_scene.xml"

m = mujoco.MjModel.from_xml_path(str(PICK_SCENE))
d = mujoco.MjData(m)

kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
print("key_qpos (padded):", np.array2string(m.key_qpos[kid], precision=3))

mujoco.mj_resetDataKeyframe(m, d, kid)
mujoco.mj_forward(m, d)
cube_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")
print("cube xpos after reset:", np.array2string(d.xpos[cube_bid], precision=4))
print("cube quat after reset:", np.array2string(d.xquat[cube_bid], precision=4))

print("\nCameras:")
for i in range(m.ncam):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i)
    print(f"  {i}: {name:10s} fovy={m.cam_fovy[i]:.1f} pos={np.array2string(d.cam_xpos[i], precision=3)}")
    print(f"       xmat=\n{np.array2string(d.cam_xmat[i].reshape(3,3), precision=3)}")
