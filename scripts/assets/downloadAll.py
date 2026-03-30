import os
import shutil
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import LocalTokenNotFoundError, RepositoryNotFoundError

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

hf_repo_id = "PDI-DL/PDI_3DUW"

current = __file__

api = HfApi()

try:
    user = api.whoami()["name"]
    print(f"### Logged as {user}")
    try:
        api.repo_info(repo_id=hf_repo_id, repo_type="dataset")
        print(f"### {hf_repo_id} is a valid Repo ID")
        try:
            for i in range(3):
                # print(current)
                local_download_path = os.path.dirname(current)
                current = local_download_path
            else:
                local_download_path += "/assets/models/"


            if os.path.exists(local_download_path):
                print(f"##### Directory already exists at: {local_download_path}. Removing it...")
                shutil.rmtree(local_download_path)
                print("##### Directory removed.")

            print(f"### Starting download for {hf_repo_id}...")


            download_path = snapshot_download(
                repo_id=hf_repo_id,
                repo_type="dataset",
                local_dir=local_download_path,
            )

            print(f"##### Download complete! Files saved to: {download_path}")
        except Exception as e:
            print(f"##### Error: download failed ###\n {e}")
    except RepositoryNotFoundError:
        print(f"##### Error: invalid Repo ID ###")
    except Exception as e:
        print(f"##### Error: Repo ID ### \n{e}")
except LocalTokenNotFoundError as e:
    print(f"##### Error: user is not logged in or authentication failed ### \n{e}")
except Exception as e:
    print(f"##### Error: not logged in or token is invalid: ### \n{e}")
