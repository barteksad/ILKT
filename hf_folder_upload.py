from huggingface_hub import HfApi
api = HfApi()

# to upload manually
api.upload_folder(
    folder_path="/home2/faculty/bsadlej/ILKT/outputs/2024-06-13/13-51-37/ILKTModel",
    repo_id="ILKT/2024-06-13_13-48-40",
)