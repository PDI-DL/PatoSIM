# PatoSim

PatoSim is the single integrated Isaac Sim extension in this repository. It keeps the project structure and main workflow derived from MobilityGen, while incorporating the OceanSim modules, sensors and underwater rendering utilities into the same extension package.

## What is inside

- Scenario building, robot control, recording and replay from the former MobilityGen workflow.
- OceanSim sensor modules such as underwater camera, imaging sonar, DVL and barometer.
- Underwater image rendering utilities and color-picker tooling kept inside the same extension tree.
- Documentation, demo assets and media required by the OceanSim-compatible tooling.

## Package layout

- `omni.ext.patosim`: main Isaac Sim extension package.
- `omni.ext.patosim.oceansim`: integrated OceanSim code now living inside the main extension.
- `isaacsim.oceansim`: compatibility path kept as a symlink to the integrated OceanSim package.

## Main commands

```bash
./link_app.sh
./scripts/register_oceansim_assets.sh /path/to/OceanSim_assets
./scripts/launch_patosim.sh
```

## OceanSim docs kept in the repo

- [Installation](subsections/installation.md)
- [Running OceanSim](subsections/running_example.md)
- [Building Your Own Digital Twins with OceanSim](subsections/building_own_digital_twin.md)
