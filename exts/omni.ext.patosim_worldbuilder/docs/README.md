# PatoSim World Builder

A lightweight, standalone Isaac Sim extension for preparing and validating
underwater worlds **before** running a data-recording scenario. It spawns
**no robot** and loads none of the main PatoSim extension's sensor/recording
UI -- it only reuses the pure scene-building helpers already in
`omni.ext.patosim.build` (see `../../omni.ext.patosim/omni/ext/patosim/build.py`).

## Launch

```bash
./scripts/launch_world_builder.sh
```

## What it does

1. **Load World** -- references a scene USD under `/World/scene`, no robot.
2. **Import Dataset Object** -- lists the same asset folders the main
   extension's dataset-object picker uses
   (`assets/models/{plataforms,statues_temples,scenario}`), lets you set
   position/rotation/scale/reflectivity and insert one into the loaded world.
3. **Check / Apply Semantic Classes** -- reports how many mesh prims are
   missing the `class`/`reflectivity` semantic labels the sonar model's
   `compute_intensity()` looks up by semantic id, and lets you apply a
   default reflectivity to every mesh that doesn't have one yet.
4. **Check / Apply Physics (Collisions)** -- reports how many geometry prims
   are missing a collision API (needed for sonar/lidar raycasts and ROV
   collision checks to register hits), and lets you apply collision to the
   whole scene.

Run the checks, fix what's missing, and only then switch to the main
`omni.ext.patosim` extension to build an actual robot/recording scenario
against the same scene.

See `docs/plano_upgrade_simulacao_subaquatica.md` and
`docs/relatorio_implementacao_upgrade_underwater_v2.md` in the repo root
`docs/` folder for the full rationale.
