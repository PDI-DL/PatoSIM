from pathlib import Path
import os

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import LocalTokenNotFoundError, RepositoryNotFoundError

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

HF_REPO_ID = "PDI-DL/PatoSimAssets"

LOCAL_DOWNLOAD_PATH = Path(__file__).resolve().parents[2] / "assets" / "models"


def authenticate(api: HfApi) -> str | None:
    """Attempts to authenticate with Hugging Face using the local token.

    Download does not require authentication for public repositories, so
    failures are logged as warnings and execution continues regardless.

    Returns the username string on success, or None if unauthenticated.
    """
    try:
        name = api.whoami()["name"]
        print(f"# Logged in as: {name}")
        return name
    except LocalTokenNotFoundError:
        print("# No Hugging Face login detected. Continuing with public access...")
    except Exception as exc:
        print(f"# Could not validate Hugging Face login. Continuing anyway... ({exc})")
    return None


def resolve_repo_type(api: HfApi, repo_id: str) -> str | None:
    """Detects the type of the remote Hugging Face repository.

    Tries 'model' first, then 'dataset', returning the first match.
    Returns None if the repository is not found under either type.
    """
    for repo_type in ("model", "dataset"):
        try:
            api.repo_info(repo_id=repo_id, repo_type=repo_type)
            print(f"# Repository '{repo_id}' found as {repo_type} repo")
            return repo_type
        except RepositoryNotFoundError:
            continue
    print(f"### Error: repository '{repo_id}' not found as model or dataset")
    return None


def validate_local_path(path: Path) -> bool:
    """Ensures the destination directory exists, creating it if necessary.

    Returns True if the path is ready for use, or False if creation failed.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as exc:
        print(f"### Error: could not create destination folder ###\n{exc}")
        return False


def download(repo_id: str, repo_type: str, local_path: Path) -> bool:
    """Downloads the full repository snapshot to the given local directory.

    Uses snapshot_download to sync all remote files locally. Existing files
    are not removed before download to avoid data loss on network failures.

    Returns True on success, or False if the download raised an exception.
    """
    print(f"# Starting download from: {repo_id}")
    print(f"Destination: {local_path}")
    try:
        result = snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=str(local_path),
        )
        print(f"\n\n### Download completed successfully! Files saved to: {result} ###")
        return True
    except Exception as exc:
        print(f"\n### Error: download failed ###\n{exc}")
        return False


def main() -> int:
    """Entry point for the download script.

    Authenticates if a local token is available, resolves the repository
    type, ensures the destination folder exists, and runs the download.

    Returns 0 on success, or 1 if any step fails.
    """
    api = HfApi()

    authenticate(api)

    repo_type = resolve_repo_type(api, HF_REPO_ID)
    if not repo_type:
        return 1

    if not validate_local_path(LOCAL_DOWNLOAD_PATH):
        return 1

    return 0 if download(HF_REPO_ID, repo_type, LOCAL_DOWNLOAD_PATH) else 1


if __name__ == "__main__":
    raise SystemExit(main())