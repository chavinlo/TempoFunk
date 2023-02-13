from huggingface_hub import login
login()
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    repo_id="chavinlo/DataFunk",
    folder_path="/workspace/TempoFunk/data/raw",
    path_in_repo="v1/",
    repo_type="dataset"
)