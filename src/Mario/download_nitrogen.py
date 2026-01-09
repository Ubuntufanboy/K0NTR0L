from huggingface_hub import hf_hub_download
import os

model_id = "nvidia/NitroGen"
filename = "ng.pt"
local_dir = "/workspace/K0NTR0L/src/Mario/Nitrogen"

print(f"Downloading {filename} from {model_id} to {local_dir}...")
hf_hub_download(repo_id=model_id, filename=filename, local_dir=local_dir)
print("Download complete.")
