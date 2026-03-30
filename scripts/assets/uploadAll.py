import os
import shutil
from huggingface_hub import HfApi
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
                    local_folder_path = os.path.dirname(current)
                    current = local_folder_path
                else:
                    local_folder_path += "/assets/models/"

                if not os.path.exists(local_folder_path):
                    print(f"Directory does not exist at: {local_folder_path}... Stopping upload")
                    raise Exception(f"No directory at {local_folder_path}")

                print(f"### Starting upload for {hf_repo_id}...")

                api.upload_large_folder(
                    folder_path=local_folder_path,
                    repo_id=hf_repo_id,
                    repo_type="dataset",  
                )

                print(f"##### Upload complete! Files saved to: {hf_repo_id}")
        except Exception as e:
            print(f"##### Error: upload failed ###\n {e}")
    except RepositoryNotFoundError:
        print(f"##### Error: invalid Repo ID ###")
    except Exception as e:
        print(f"##### Error: Repo ID ### \n{e}")
except LocalTokenNotFoundError as e:
    print(f"##### Error: user is not logged in or authentication failed ### \n{e}")
except Exception as e:
    print(f"##### Error: not logged in or token is invalid: ### \n{e}")

