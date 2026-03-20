import json
import os


OCEANSIM_ASSET_PATH = None


def _get_json_path() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "asset_path.json"))


def get_oceansim_assets_path() -> str:
    global OCEANSIM_ASSET_PATH

    if OCEANSIM_ASSET_PATH is not None:
        return OCEANSIM_ASSET_PATH

    env_path = os.environ.get("OCEANSIM_ASSET_PATH")
    if env_path:
        env_path = os.path.abspath(env_path)
        if not os.path.isdir(env_path):
            raise FileNotFoundError(
                f"OCEANSIM_ASSET_PATH points to a missing directory: {env_path}"
            )
        OCEANSIM_ASSET_PATH = env_path
        return env_path

    json_path = _get_json_path()

    if not os.path.isfile(json_path):
        raise FileNotFoundError(
            f"'asset_path.json' not found at {json_path}. "
            "Run 'scripts/register_oceansim_assets.sh <path_to_assets>' or set OCEANSIM_ASSET_PATH."
        )

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"'asset_path.json' is not valid JSON: {e}") from e

    if "asset_path" not in json_data:
        raise KeyError(
            f"'asset_path.json' at {json_path} does not contain the required 'asset_path' key."
        )

    asset_path = os.path.abspath(json_data["asset_path"])

    if not os.path.isdir(asset_path):
        raise FileNotFoundError(
            f"The provided asset path does not exist: {asset_path}. "
            "Run 'scripts/register_oceansim_assets.sh <path_to_assets>' or update asset_path.json."
        )

    OCEANSIM_ASSET_PATH = asset_path
    return asset_path


if __name__ == "__main__":
    print("OceanSim assets are configured at", get_oceansim_assets_path())
