from pathlib import Path
import os

from huggingface_hub import HfApi
from huggingface_hub.utils import LocalTokenNotFoundError, RepositoryNotFoundError

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

HF_REPO_ID = "PDI-DL/PatoSimAssets"
HF_REPO_TYPE = "dataset"

LOCAL_FOLDER_PATH = Path(__file__).resolve().parents[2] / "assets" / "models"


def authenticate(api: HfApi) -> str | None:
    """Attempts to authenticate with Hugging Face using the local token.

    Upload requires a valid authenticated session, so failures are treated
    as hard errors and None is returned to abort the script.

    Returns the username string on success, or None if authentication fails.
    """
    try:
        name = api.whoami()["name"]
        print(f"# Logged in as: {name}")
        return name
    except LocalTokenNotFoundError as exc:
        print(f"### Error: token not found ###\n{exc}")
    except Exception as exc:
        print(f"### Error: authentication failed ###\n{exc}")
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


def validate_local_folder(path: Path) -> bool:
    """Checks that the source folder exists before attempting an upload.

    Returns True if the path exists, or False if it does not.
    """
    if not path.exists():
        print(f"### Error: local folder does not exist ###\n{path}")
        return False
    return True


def upload(api: HfApi, local_path: Path, repo_id: str, repo_type: str) -> bool:
    """Uploads the local folder to the remote Hugging Face repository.

    Uses upload_large_folder to handle large datasets reliably, supporting
    resumable uploads and automatic chunking for folders of any size.

    Returns True on success, or False if the upload raised an exception.
    """
    print(f"# Starting upload to: {repo_id}")
    print(f"Source: {local_path}")
    try:
        api.upload_large_folder(
            folder_path=str(local_path),
            repo_id=repo_id,
            repo_type=repo_type,
        )
        print("\n### Upload completed successfully ###")
        return True
    except Exception as exc:
        print(f"\n### Error: upload failed ###\n{exc}")
        return False


def main() -> int:
    """Entry point for the upload script.

    Authenticates the user, resolves the repository type, validates the
    local source folder, and runs the upload.

    Returns 0 on success, or 1 if any step fails.
    """
    api = HfApi()

    if not authenticate(api):
        return 1

    repo_type = resolve_repo_type(api, HF_REPO_ID)
    if not repo_type:
        return 1

    if not validate_local_folder(LOCAL_FOLDER_PATH):
        return 1

    return 0 if upload(api, LOCAL_FOLDER_PATH, HF_REPO_ID, repo_type) else 1


if __name__ == "__main__":
    raise SystemExit(main())