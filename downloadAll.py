import os
from huggingface_hub import snapshot_download

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

hf_repo_id = "PDI-DL/PDI_3DUW"
local_download_path = f"{os.getcwd()}/modelos_os" 

print(f"### Starting download for {hf_repo_id}...")

# snapshot_download downloads all files in the repository
download_path = snapshot_download(
    repo_id=hf_repo_id,
    repo_type="dataset",
    local_dir=local_download_path,
)

print(f"### Download complete! Files saved to: {download_path}")