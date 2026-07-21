"""Quick introspection of the Panda model: joints, actuators, dimensions."""
from pathlib import Path

import mujoco

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "franka_emika_panda" / "scene.xml"


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    print(f"nq (positions): {model.nq} | nv (velocities): {model.nv} | nu (actuators): {model.nu}")
    print(f"timestep: {model.opt.timestep}")

    print("\nJoints:")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        print(f"  {i}: {name}")

    print("\nActuators:")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"  {i}: {name}")


if __name__ == "__main__":
    main()
