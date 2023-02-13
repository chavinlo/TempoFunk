from huggingface_hub import HfApi
import os
api = HfApi()
path = "/workspace/TempoFunk/models"
directory = sorted(os.listdir(path))
for folder in directory:
    if folder in ["MaSDV", "masdv-latest", "tmp"]:
        continue
    new_path = os.path.join(path, folder)
    versions = sorted(os.listdir(new_path))
    for version in versions:
        final_path = os.path.join(new_path, version, "unet.pt")
        print("Uploading:", final_path)
        api.upload_file(
            path_or_fileobj=final_path,
            path_in_repo=f"{folder}/{version}.pt",
            repo_id="chavinlo/TempoFunk",
            repo_type="model",
            revision="starry_pop"
        ),