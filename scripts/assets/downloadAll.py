import os
from huggingface_hub import snapshot_download

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

hf_repo_id = "PDI-DL/PDI_3DUW"

current = __file__

for i in range(3):
    # print(current)
    local_download_path = os.path.dirname(current)
    current = local_download_path
else:
    local_download_path += "/assets/models/"

print(f"### Starting download for {hf_repo_id}...")


download_path = snapshot_download(
    repo_id=hf_repo_id,
    repo_type="dataset",
    local_dir=local_download_path,
)

print(f"### Download complete! Files saved to: {download_path}")