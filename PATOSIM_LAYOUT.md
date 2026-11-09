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
- `link_app.sh --path <isaac-sim-5.1>`: links the local `app` folder to the Isaac Sim installation
  (without `--path`: `$ISAAC_SIM_PATH`, else `/home/bevlog/isaac-sim`).
- `scripts/assets/downloadAll.py`: downloads the public Hugging Face assets (`PDI-DL/PatoSimAssets`) into
  `assets/models`; the code also finds the same folders directly under `assets/`. `scripts/assets/assets.sh`
  works on a git clone of the private `PDI-DL/PDI_3DUW` dataset.
- `tests/unit` (pytest, no Isaac) and `tests/integration` (`./app/python.sh <script>`, Isaac headless).

## Extension structure

- `exts/omni.ext.patosim/assets`: icon and preview shown in Isaac Sim.
- `exts/omni.ext.patosim/config`: extension manifest and asset registration helper.
- `exts/omni.ext.patosim/docs`: integrated extension docs.
- `exts/omni.ext.patosim/demo`: sample RGB, depth and waypoint files used by OceanSim tooling.
- `exts/omni.ext.patosim/media`: figures and GIFs used by docs.
- `exts/omni.ext.patosim/omni/ext/patosim`: main extension package, derived from the MobilityGen structure.
- `exts/omni.ext.patosim/omni/ext/patosim/oceansim`: OceanSim modules, sensors and utilities integrated into the
  main package, imported as `omni.ext.patosim.oceansim`.
- `isaacsim/oceansim` and `exts/omni.ext.patosim/isaacsim/oceansim`: relative links to that package, kept for old
  `isaacsim.oceansim` imports (the code no longer uses them).

## Legacy material

- `legacy_exts/omni.ext.mobility_gen`: previous standalone MobilityGen extension, moved out of `exts` so Isaac no longer loads it directly.

## Notes

- The runtime extension is now only `omni.ext.patosim`.
- The visual identity shown in Isaac is now `PatoSim`, not `MobilityGen`.
- OceanSim code now lives under the same extension package and is imported through the extension's own Python namespace.
