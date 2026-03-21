import asyncio
import os
import numpy as np
from pxr import Usd, UsdGeom, Vt

from omni.ext.mobility_gen.reader import Reader


def _quat_to_rotmat(q):
    # q = [x,y,z,w]
    x, y, z, w = q
    # normalize
    n = x * x + y * y + z * z + w * w
    if n == 0:
        return np.eye(3)
    s = 2.0 / n
    xx = x * x * s
    yy = y * y * s
    zz = z * z * s
    xy = x * y * s
    xz = x * z * s
    yz = y * z * s
    wx = w * x * s
    wy = w * y * s
    wz = w * z * s
    R = np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ], dtype=np.float32)
    return R


def transform_points(points: np.ndarray, position, orientation):
    if points is None:
        return None
    pts = np.asarray(points, dtype=np.float32)
    if pts.size == 0:
        return pts
    if orientation is None or position is None:
        return pts
    try:
        q = orientation
        t = np.asarray(position, dtype=np.float32)
        R = _quat_to_rotmat(q)
        pts_w = (R @ pts[:, :3].T).T + t.reshape(1, 3)
        # if there are extra columns (intensity/color), preserve them
        if pts.shape[1] > 3:
            rest = pts[:, 3:]
            pts_w = np.hstack([pts_w, rest])
        return pts_w
    except Exception:
        return pts


class PointCloudPlayer:
    """Player that can replay multiple pointcloud sensors, mapping intensity
    and RGB to displayColor when available. Creates one Points prim per
    sensor under /World/PointCloudReplay.
    """

    def __init__(self, recording_path: str, prim_root: str = "/World/PointCloudReplay"):
        self.recording_path = recording_path
        self.reader = Reader(recording_path)

        self.stage = Usd.Stage.GetCurrent()
        if self.stage is None:
            raise RuntimeError("No current USD stage available")

        self.prim_root = prim_root
        if not self.prim_root:
            self.prim_root = "/World/PointCloudReplay"

        # ensure parent
        try:
            self.parent = UsdGeom.Xform.Define(self.stage, self.prim_root)
        except Exception:
            self.parent = None

        # create per-sensor prims
        self.sensors = []
        self.points_prims = {}
        for name in self.reader.pointcloud_names:
            xform_path = os.path.join(self.prim_root, name)
            pts_path = os.path.join(xform_path, "points")
            UsdGeom.Xform.Define(self.stage, xform_path)
            pts = UsdGeom.Points.Define(self.stage, pts_path)
            self.sensors.append(name)
            self.points_prims[name] = pts

        if len(self.sensors) == 0:
            raise RuntimeError("No pointcloud sensors found in recording")

        self.playing = False
        self.index = 0
        self.fps = 10.0

        # sensor visibility models for UI
        self.sensor_models = {name: True for name in self.sensors}

    def _set_points(self, name: str, pts_world: np.ndarray):
        pts_prim = self.points_prims.get(name)
        if pts_prim is None or pts_world is None:
            return
        # set points positions
        arr = Vt.Vec3fArray()
        for p in pts_world[:, :3].astype(np.float32):
            arr.append(tuple(p.tolist()))
        pts_prim.GetPointsAttr().Set(arr)

        # try to set colors if available (r,g,b) or use intensity
        # determine if pts_world has rgb columns (r,g,b) after xyz
        cols = pts_world.shape[1]
        colors = None
        if cols >= 6:
            # assume r,g,b are columns 3,4,5
            rgb = pts_world[:, 3:6]
            # normalize if in 0-255
            if rgb.max() > 1.0:
                rgb = rgb / 255.0
            colors = Vt.Vec3fArray()
            for c in rgb.astype(np.float32):
                colors.append(tuple(c.tolist()))
        elif cols == 4:
            # intensity -> grayscale
            intensity = pts_world[:, 3]
            # normalize intensity
            if intensity.max() > 1.0:
                intensity = intensity / (intensity.max() + 1e-6)
            colors = Vt.Vec3fArray()
            for v in intensity.astype(np.float32):
                colors.append((v, v, v))

        if colors is not None:
            pts_prim.GetDisplayColorAttr().Set(colors)

    async def _play_loop(self):
        while self.playing:
            self.step()
            await asyncio.sleep(1.0 / max(0.001, self.fps))

    def play(self, fps: float = 10.0):
        if self.playing:
            return
        self.fps = float(fps)
        self.playing = True
        asyncio.ensure_future(self._play_loop())

    def pause(self):
        self.playing = False

    def step(self, forward: bool = True):
        if forward:
            self.index = (self.index + 1) % len(self.reader)
        else:
            self.index = (self.index - 1) % len(self.reader)

        # load points and metadata for all sensors
        pts_dict = self.reader.read_state_dict_pointcloud(self.index)
        for sensor_name, pts in pts_dict.items():
            if sensor_name not in self.sensors:
                continue
            meta = self.reader.read_pointcloud_metadata(sensor_name, self.index)
            if meta is not None and 'position' in meta and 'orientation' in meta:
                pos = meta.get('position', None)
                ori = meta.get('orientation', None)
                pts_w = transform_points(pts, pos, ori)
            else:
                pts_w = pts

            # only update if sensor visibility enabled
            if self.sensor_models.get(sensor_name, True):
                self._set_points(sensor_name, pts_w)

    def goto(self, index: int):
        self.index = max(0, min(index, len(self.reader) - 1))
        # show this frame
        # set index then call step with forward True but avoid changing index
        cur = self.index
        self.step(forward=True)
        self.index = cur

    def set_visibility(self, sensor_name: str, visible: bool):
        self.sensor_models[sensor_name] = bool(visible)

    def set_fps(self, fps: float):
        self.fps = float(fps)

    def get_length(self):
        return len(self.reader)

