# import huggingface_hub
# print(huggingface_hub.__version__)

import os
from huggingface_hub import HfApi

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
api = HfApi()

current = __file__

for i in range(3):
    # print(current)
    local_folder_path = os.path.dirname(current)
    current = local_folder_path
else:
    local_folder_path += "/assets/models/"

hf_repo_id = "PDI-DL/PDI_3DUW"

print(f"### Starting upload for {hf_repo_id}...")

api.upload_large_folder(
    folder_path=local_folder_path,
    repo_id=hf_repo_id,
    repo_type="dataset",  
)

print("### Upload complete!")