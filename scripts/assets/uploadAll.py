from pathlib import Path
import os

from huggingface_hub import HfApi
from huggingface_hub.utils import LocalTokenNotFoundError, RepositoryNotFoundError


os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


# O repositório PDI_3DUW está publicado como "model repo" no Hugging Face.
HF_REPO_ID = "PDI-DL/PDI_3DUW"

# Resolve /.../MOD_patosim/assets/models a partir de scripts/assets/uploadAll.py
LOCAL_FOLDER_PATH = Path(__file__).resolve().parents[2] / "assets" / "models"


def resolve_repo_type(api: HfApi, repo_id: str) -> str:
    """Detecta o tipo do repositório remoto antes do upload.

    Hoje o repositório existe como model repo, mas o fallback para dataset
    mantém o script reutilizável caso a organização do hub mude depois.
    """

    for repo_type in ("model", "dataset"):
        try:
            api.repo_info(repo_id=repo_id, repo_type=repo_type)
            return repo_type
        except RepositoryNotFoundError:
            continue

    raise RepositoryNotFoundError(
        f"Could not find Hugging Face repository '{repo_id}' as model or dataset."
    )


def main() -> int:
    api = HfApi()

    # Upload exige autenticação válida; aqui o login é obrigatório.
    try:
        user = api.whoami()["name"]
        print(f"### Logged as {user}")
    except LocalTokenNotFoundError as exc:
        print(f"##### Error: user is not logged in or authentication failed ###\n{exc}")
        return 1
    except Exception as exc:
        print(f"##### Error: not logged in or token is invalid ###\n{exc}")
        return 1

    try:
        repo_type = resolve_repo_type(api, HF_REPO_ID)
        print(f"### {HF_REPO_ID} is available as a Hugging Face {repo_type} repo")
    except RepositoryNotFoundError as exc:
        print(f"##### Error: repository not found ###\n{exc}")
        return 1
    except Exception as exc:
        print(f"##### Error while validating repository ###\n{exc}")
        return 1

    if not LOCAL_FOLDER_PATH.exists():
        print(f"##### Error: local folder does not exist ###\n{LOCAL_FOLDER_PATH}")
        return 1

    print(f"### Starting upload for {HF_REPO_ID}...")
    print(f"### Source: {LOCAL_FOLDER_PATH}")

    try:
        # upload_large_folder envia o conteúdo inteiro da pasta preservando a
        # estrutura de diretórios. Não usamos flags destrutivas aqui.
        api.upload_large_folder(
            folder_path=str(LOCAL_FOLDER_PATH),
            repo_id=HF_REPO_ID,
            repo_type=repo_type,
        )
    except Exception as exc:
        print(f"##### Error: upload failed ###\n{exc}")
        return 1

    print(f"##### Upload complete! Files sent to: {HF_REPO_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
