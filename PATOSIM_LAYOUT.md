# PatoSIM organization inside MOD_patosim

## Main entrypoints

- `exts/omni.ext.patosim`: Isaac Sim extension in the same style as the MobilityGen extension layout.
- `scripts/launch_patosim.sh`: launches Isaac Sim with the PatoSIM extension enabled.
- `scripts/register_oceansim_assets.sh`: registers the OceanSim asset directory into `asset_path.json`.
- `link_app.sh`: links the local `app` folder to the Isaac Sim 5.1 install at `/mnt/external/isaac/isaac-sim-5.1` by default.

## Extension structure

- `exts/omni.ext.patosim/assets`: extension icon and preview image.
- `exts/omni.ext.patosim/config`: extension manifest and asset registration helper.
- `exts/omni.ext.patosim/docs`: imported OceanSim documentation.
- `exts/omni.ext.patosim/demo`: sample RGB, depth and waypoint files used by the UI.
- `exts/omni.ext.patosim/media`: figures and GIFs used by the documentation.
- `exts/omni.ext.patosim/isaacsim/oceansim/modules`: UI-facing OceanSim modules.
- `exts/omni.ext.patosim/isaacsim/oceansim/sensors`: underwater sensor implementations.
- `exts/omni.ext.patosim/isaacsim/oceansim/utils`: rendering, math and asset-path helpers.

## Notes

- The original OceanSim python package path `isaacsim.oceansim` was preserved to minimize code churn.
- The extension assets were normalized to `assets/` to match the MobilityGen-style extension layout.

## A fazer. 
* Oraganizar e revisar organizacao dos arquivso
* Ajustar link assets que nao estao funcionando
* Revisar escritps

