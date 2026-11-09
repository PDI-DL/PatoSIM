# Changelog

# PatoSim (main)

## 2026-11-09

- OceanSim code imported as `omni.ext.patosim.oceansim` everywhere (branch `update_imports`): the sonar,
  camera, DVL and barometer no longer depend on an OceanSim/PatoSIM copy in `app/extsUser`
- Assets found in `assets/models/` (`scripts/assets/assets.sh`) or directly in `assets/`; the example world
  ("Use MHL Scene") is resolved from the repo or `$PATOSIM_DEV_SCENE` instead of a fixed path
- `isaacsim/oceansim` compatibility link is now relative (it pointed to one machine's home folder)
- `path_planner/` sources stay versioned; only its build outputs are ignored
- Integration checks run on any machine (paths relative to the repo, optional legacy recording) and confirm
  the sonar code comes from the repo
- README rewritten for the underwater workflow: Isaac Sim 5.1, assets, ROV scenarios and keys, World Builder,
  tests; `docs/PatoSim_novidades.pdf` summarizes the new features

## 2026-10-24

- Lightweight navigation: every ROV sensor starts off (Config and window); turn on only what a recording needs
- New `omni.ext.patosim_worldbuilder` extension (`scripts/launch_world_builder.sh`): load a world without a
  robot, insert dataset objects, check/apply semantic classes, sonar reflectivity and collisions
- `tests/integration/`: Isaac Sim headless checks of the full ROV scenario, live recording and World Builder

## 2026-10-05

- Sonar `raw` normalization (keeps the range falloff) as the default; raw sonar saved during live recording
- Underwater lidar profiles (Voyis Insight ranges) with Beer-Lambert return probability
- Gamepad 6-DOF teleoperation scenario; sensor extrinsics (`sensors_extrinsics.json`); 3D occupancy map stack
- Asset unit-scale compensation; pure unit tests (`tests/unit`, 32 tests)

# MobilityGen (upstream)

# main

- Added support for surface normals image in replay rendering
- Added support for instance ID segmentation rendering
- Added camera world pose to state

# ref-4.5

Release targeting Isaac Sim 4.5

- Updated to use isaacsim.* rather than omni.* import calls
- Added minor fixes to API discrepancies
- Added required extension loading for replay rendering
- Updated README to include setup instructions for Isaac Sim 4.5

# ref-4.2

Release targeting Isaac Sim 4.2

- Added modular state capture
- Added state writer class
- Added state reader class
- Added support for H1
- Added support for Jetbot
- Added support for Spot 
- Added support for Carter
- Added keyboard teleop scenario
- Added gamepad teleop scenario
- Added path following scenario
- Added random acceleration scenario
- Added initial OV extension for launching scenarios and recording data
- Added replay rendering scripts
- Added data loading examples
- Added workflow documentation
