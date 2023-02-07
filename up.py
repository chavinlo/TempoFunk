from huggingface_hub import login
login()
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="/workspace/diffused-video-trainer/models/500/unet.pt",
    path_in_repo="unet.pt",
    repo_id="chavinlo/test",
    repo_type="model",
)