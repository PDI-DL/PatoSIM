from pathlib import Path
import os

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import LocalTokenNotFoundError, RepositoryNotFoundError


os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


# Este repositório do Hugging Face hospeda modelos 3D, mas está publicado
# como "model repo" e não como "dataset repo".
HF_REPO_ID = "PDI-DL/PDI_3DUW"

# Resolve /.../MOD_patosim/assets/models a partir de scripts/assets/downloadAll.py
LOCAL_DOWNLOAD_PATH = Path(__file__).resolve().parents[2] / "assets" / "models"


def resolve_repo_type(api: HfApi, repo_id: str) -> str:
    """Detecta o tipo real do repositório no Hugging Face.

    O script antigo fixava repo_type="dataset", o que gera 404 para este repo.
    Aqui tentamos primeiro como model, que é o tipo usado hoje, e mantemos um
    fallback para dataset caso o repositório seja reorganizado no futuro.
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

    # Repositórios públicos podem ser baixados sem login. O whoami() é usado
    # apenas para informar quando existe uma sessão autenticada disponível.
    try:
        user = api.whoami()["name"]
        print(f"### Logged as {user}")
    except LocalTokenNotFoundError:
        print("### No Hugging Face login detected. Continuing with public access...")
    except Exception as exc:
        print(f"### Could not validate Hugging Face login. Continuing anyway... ({exc})")

    try:
        repo_type = resolve_repo_type(api, HF_REPO_ID)
        print(f"### {HF_REPO_ID} is available as a Hugging Face {repo_type} repo")
    except RepositoryNotFoundError as exc:
        print(f"##### Error: repository not found ###\n{exc}")
        return 1
    except Exception as exc:
        print(f"##### Error while validating repository ###\n{exc}")
        return 1

    LOCAL_DOWNLOAD_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"### Starting download for {HF_REPO_ID}...")
    print(f"### Destination: {LOCAL_DOWNLOAD_PATH}")

    try:
        # snapshot_download sincroniza o conteúdo remoto dentro do diretório
        # local informado. Não removemos a pasta antes do download para evitar
        # perda local desnecessária em caso de falha na rede ou no servidor.
        download_path = snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type=repo_type,
            local_dir=str(LOCAL_DOWNLOAD_PATH),
        )
    except Exception as exc:
        print(f"##### Error: download failed ###\n{exc}")
        return 1

    print(f"##### Download complete! Files saved to: {download_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
