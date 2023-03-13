from huggingface_hub import HfApi
import os
api = HfApi()
path = "/workspace/disk/models"
directory = sorted(os.listdir(path))
for folder in directory:
    if folder in ["latest", "make-a-stable-diffusion-video-timelapse", "tmp", "v8-19_lr3.3333333333333335e-05"]:
        continue
    new_path = os.path.join(path, folder)
    files = sorted(os.listdir(new_path))
    for file in files:
        final_path = os.path.join(new_path, file)
        print("Uploading:", final_path)
        api.upload_file(
            path_or_fileobj=final_path,
            path_in_repo=f"{folder}/{file}",
            repo_id="chavinlo/TempoFunk",
            repo_type="model",
            revision="starry_pop"
        )