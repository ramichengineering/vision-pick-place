"""Dump RGB + mask images from each camera so you can see what the robot sees."""
import numpy as np
from PIL import Image

from perception import render_rgb_depth, segment_red
from perception_test import set_cube_pose
from robot import load_pick_scene, reset_to_home

model, data = load_pick_scene()
reset_to_home(model, data, key="pick_home")
set_cube_pose(model, data, (0.5, 0.0, 0.025))

for cam in ("scene_cam", "top_cam"):
    rgb, depth = render_rgb_depth(model, data, camera=cam, width=640, height=480)
    mask = segment_red(rgb)
    Image.fromarray(rgb).save(f"debug_{cam}_rgb.png")
    Image.fromarray((mask * 255).astype(np.uint8)).save(f"debug_{cam}_mask.png")
    print(f"{cam}: {int(mask.sum())} red px | depth range "
          f"{np.nanmin(depth):.2f}-{np.nanmax(depth):.2f} m")
