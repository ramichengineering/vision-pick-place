"""Pixels -> world coordinates: find the red cube using RGB + depth.

The pipeline, and the geometry behind it
----------------------------------------
1. RENDER. MuJoCo gives us an RGB image and a depth image from a named camera.
   Depth is the distance from the camera along its optical axis (metres).

2. SEGMENT. The cube is the only strongly red thing in the scene, so a simple
   per-pixel test (red clearly dominates green and blue) isolates it. No
   training, no detector -- just a colour threshold.

3. BACK-PROJECT. A pixel plus its depth is enough to recover a 3D point, given
   the camera intrinsics. MuJoCo cameras are ideal pinholes, so the focal length
   in pixels comes straight from the vertical field of view:

       f = (H/2) / tan(fovy/2)

   For pixel (u, v) with depth d, in the CAMERA frame:

       x_cam =  (u - cx) * d / f
       y_cam = -(v - cy) * d / f     (image rows go down, camera +y goes up)
       z_cam = -d                    (MuJoCo cameras look down their local -z)

4. TRANSFORM. MuJoCo gives the camera's world pose each step. The columns of
   cam_xmat are the camera's axes expressed in world coordinates, so:

       p_world = cam_pos + R_cam @ p_cam

Back-projecting every masked pixel yields a point cloud of the cube's *visible
surface*. Note that its centroid is not the cube's centre -- it is biased toward
the camera, because we only ever see the faces pointing at us. `estimate_cube`
corrects for that (see `_refine_center`).
"""
from dataclasses import dataclass

import mujoco
import numpy as np

CUBE_HALF = 0.025  # half-extent of the cube (m), used to de-bias the surface centroid


@dataclass
class Detection:
    found: bool
    position: np.ndarray      # estimated cube CENTRE in world coords (3,)
    surface_centroid: np.ndarray
    n_pixels: int
    uv: tuple                 # (col, row) pixel centroid of the mask
    rgb: np.ndarray = None
    mask: np.ndarray = None


def render_rgb_depth(model, data, camera="scene_cam", width=640, height=480):
    """Return (rgb uint8 HxWx3, depth float HxW in metres) from a named camera."""
    with mujoco.Renderer(model, height=height, width=width) as r:
        r.update_scene(data, camera=camera)
        rgb = r.render().copy()
        r.enable_depth_rendering()
        r.update_scene(data, camera=camera)
        depth = r.render().copy()
    return rgb, depth


def segment_red(rgb, min_red=90, dominance=1.6):
    """Boolean mask of strongly-red pixels."""
    img = rgb.astype(np.int16)
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    return (R > min_red) & (R > dominance * G) & (R > dominance * B)


def camera_intrinsics(model, cam_id, width, height):
    """Pinhole focal length (px) and principal point for a MuJoCo camera."""
    fovy = np.deg2rad(model.cam_fovy[cam_id])
    f = (height / 2.0) / np.tan(fovy / 2.0)
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    return f, cx, cy


def backproject(model, data, cam_id, mask, depth):
    """Masked pixels + depth -> Nx3 array of world-frame points."""
    H, W = depth.shape
    f, cx, cy = camera_intrinsics(model, cam_id, W, H)

    v, u = np.nonzero(mask)              # rows, cols
    d = depth[v, u]
    good = np.isfinite(d) & (d > 0)      # drop sky / invalid depth
    u, v, d = u[good], v[good], d[good]
    if u.size == 0:
        return np.empty((0, 3))

    # camera frame
    pts_cam = np.stack([(u - cx) * d / f,
                        -(v - cy) * d / f,
                        -d], axis=1)

    # camera -> world
    R = data.cam_xmat[cam_id].reshape(3, 3)
    return data.cam_xpos[cam_id] + pts_cam @ R.T


def _refine_center(pts_world, cam_pos, top_tol=0.006):
    """Recover the cube's CENTRE from its visible surface.

    The naive fix -- push the surface centroid back along the view ray by half a
    cube -- leaves a systematic bias, because the visible surface is a mix of the
    top face and one or two side faces, so the true offset is not a clean half
    extent.

    Both cameras look down on the cube, so the *top face* is fully visible. That
    gives a far better estimator, assuming the cube rests flat (top face level):
        - its x,y centroid IS the cube's x,y centre (whole face is seen)
        - its height is the cube's top, so centre_z = top_z - CUBE_HALF

    Falls back to the view-ray correction if too few top-face points survive.
    """
    z_top = pts_world[:, 2].max()
    top = pts_world[pts_world[:, 2] > z_top - top_tol]

    if top.shape[0] >= 10:
        cx, cy = top[:, 0].mean(), top[:, 1].mean()
        cz = top[:, 2].mean() - CUBE_HALF
        return np.array([cx, cy, cz])

    centroid = pts_world.mean(axis=0)
    ray = centroid - cam_pos
    ray /= np.linalg.norm(ray)
    return centroid + ray * CUBE_HALF


def estimate_cube(model, data, camera="scene_cam", width=640, height=480,
                  keep_images=False, min_pixels=150) -> Detection:
    """Full pipeline: render -> segment -> back-project -> world position.

    `min_pixels` guards against partially-occluded views: a sliver of the cube
    still back-projects to *something*, but the estimate is badly biased (a 20px
    detection was seen to be off by 140mm). Better to report nothing than a lie.
    """
    mujoco.mj_forward(model, data)   # ensure camera/body poses are current
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    rgb, depth = render_rgb_depth(model, data, camera, width, height)
    mask = segment_red(rgb)

    n = int(mask.sum())
    if n < min_pixels:   # too few pixels to trust
        return Detection(False, np.zeros(3), np.zeros(3), n, (np.nan, np.nan),
                         rgb if keep_images else None, mask if keep_images else None)

    pts = backproject(model, data, cam_id, mask, depth)
    if pts.shape[0] == 0:
        return Detection(False, np.zeros(3), np.zeros(3), n, (np.nan, np.nan),
                         rgb if keep_images else None, mask if keep_images else None)

    surface = pts.mean(axis=0)
    center = _refine_center(pts, data.cam_xpos[cam_id])
    v, u = np.nonzero(mask)
    return Detection(True, center, surface, n, (u.mean(), v.mean()),
                     rgb if keep_images else None, mask if keep_images else None)
