"""
Small demo to build the ForkliftRobot, attach sensors and exercise manual/auto modes.

Run this inside Isaac Sim Kit Python (Kit must be running). Example:

    # inside Kit python (or run via an extension entrypoint)
    python examples/test_forklift_demo.py

This script:
  - creates a new stage and world
  - builds the `ForkliftRobot` from the registry at `/World/robot`
  - enables sensors and switches to `auto` mode with a short path
  - steps the simulation for a few frames and prints basic sensor buffer info

Notes:
  - This is a smoke test and requires Isaac/Omniverse APIs (pxr, omni, isaacsim).
  - The mobility_gen writer/reader will pick up the sensor Buffers created here
    because cameras expose `rgb` buffers and the Lidar exposes `pointcloud`.
"""

import asyncio
import os
import sys
import numpy as np

# Friendly import guard: this demo must run inside Isaac Sim / Omniverse Kit Python
try:
    from omni.ext.mobility_gen.utils.global_utils import new_stage, new_world, get_world
    from omni.ext.mobility_gen.robots import ROBOTS
except Exception as e:  # ModuleNotFoundError / ImportError when running outside Kit
    print("ERROR: This demo must be run using Isaac Sim (Omniverse Kit) Python.\n")
    print("Common ways to run:\n"
          "  1) Open Isaac Sim, open the Script Editor (Window -> Script Editor) and run this file's contents.\n"
          "  2) From a terminal inside the Kit environment run: /path/to/kit/python3 examples/test_forklift_demo.py\n"
          "     (find the Kit python with `which python3` inside the Kit terminal or check sys.executable inside Kit).\n\n")
    print("Debug info:")
    print("  sys.executable:", sys.executable)
    print("  caught:", repr(e))
    # Exit cleanly so the user sees the message instead of a long traceback
    sys.exit(1)


async def main():
    # create clean stage and world
    new_stage()
    world = new_world(physics_dt=0.01)
    await world.initialize_simulation_context_async()

    # build forklift
    ForkliftClass = ROBOTS.get("ForkliftRobot")
    robot = ForkliftClass.build("/World/robot")

    # switch to auto and give a short square path
    robot.set_control_mode("auto")
    path = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
    robot.set_auto_path(path)

    # step simulation a few frames and show sensor buffer states
    n_steps = 120
    for i in range(n_steps):
        # robot decides/apply commands based on control_mode
        robot.write_action(robot.physics_dt)

        # step physics (no render to keep it fast here)
        world.step(render=False)

        # update module buffers from sim
        robot.update_state()

        if (i % 20) == 0:
            pose = robot.get_pose_2d()
            pc_info = None
            try:
                if robot.lidar is not None:
                    pts = robot.lidar.pointcloud.get_value()
                    pc_info = None if pts is None else f"{pts.shape[0]} points"
            except Exception:
                pc_info = "(error reading pointcloud)"

            rgb_info = None
            try:
                if robot.front_stereo is not None and hasattr(robot.front_stereo, "left"):
                    img = robot.front_stereo.left.rgb_image.get_value()
                    rgb_info = None if img is None else f"rgb {img.shape}"
            except Exception:
                rgb_info = "(error reading rgb)"

            print(f"step={i} pose=({pose.x:.2f},{pose.y:.2f},{pose.theta:.2f}) pc={pc_info} rgb={rgb_info}")

    print("Demo finished. If the script ran inside Kit you should see a robot built at /World/robot and sensors attached.")


if __name__ == "__main__":
    try:
        asyncio.get_event_loop().run_until_complete(main())
    except Exception as e:
        print("This demo must be run in Isaac Sim Kit (Kit Python). Error:", e)
