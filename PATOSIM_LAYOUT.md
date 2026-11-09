# PatoSim organization inside MOD_patosim

## Main entrypoints

- `exts/omni.ext.patosim`: the main Isaac Sim extension -- robot, sensors,
  scenarios (navigation + autonomous), recording. Opens with **all sensors
  off by default** so navigation stays lightweight/fluid; tick the sensor
  checkboxes before a recording session.
- `exts/omni.ext.patosim_worldbuilder`: a second, standalone, lightweight
  extension for preparing/validating a world **before** a recording session
  -- scene loading with **no robot spawned**, dataset-object import, and
  check/apply tools for sonar semantic reflectance and physics collisions.
  Reuses `omni.ext.patosim.build`'s pure scene-building helpers instead of
  duplicating them; does not pull in the main extension's robot/sensor/
  recording UI. See `exts/omni.ext.patosim_worldbuilder/docs/README.md`.
- `scripts/launch_patosim.sh`: launches Isaac Sim with only `omni.ext.patosim` enabled.
- `scripts/launch_world_builder.sh`: launches Isaac Sim with only
  `omni.ext.patosim_worldbuilder` enabled (the lightweight world-prep tool).
- `scripts/launch_sim.sh`: compatibility wrapper that now forwards to `launch_patosim.sh`.
- `scripts/register_oceansim_assets.sh`: registers the OceanSim asset directory into the integrated extension.
- `link_app.sh`: links the local `app` folder to `/mnt/external/isaac/isaac-sim-5.1` by default.

## Extension structure

- `exts/omni.ext.patosim/assets`: icon and preview shown in Isaac Sim.
- `exts/omni.ext.patosim/config`: extension manifest and asset registration helper.
- `exts/omni.ext.patosim/docs`: integrated extension docs.
- `exts/omni.ext.patosim/demo`: sample RGB, depth and waypoint files used by OceanSim tooling.
- `exts/omni.ext.patosim/media`: figures and GIFs used by docs.
- `exts/omni.ext.patosim/omni/ext/patosim`: main extension package, derived from the MobilityGen structure.
- `exts/omni.ext.patosim/omni/ext/patosim/oceansim`: OceanSim modules, sensors and utilities integrated into the main package.
- `exts/omni.ext.patosim/isaacsim/oceansim`: compatibility symlink pointing to `omni/ext/patosim/oceansim`.

## Legacy material

- `legacy_exts/omni.ext.mobility_gen`: previous standalone MobilityGen extension, moved out of `exts` so Isaac no longer loads it directly.

## Notes

- The runtime extension is now only `omni.ext.patosim`.
- The visual identity shown in Isaac is now `PatoSim`, not `MobilityGen`.
- OceanSim code now lives under the same extension package while keeping a compatibility path for older imports.
