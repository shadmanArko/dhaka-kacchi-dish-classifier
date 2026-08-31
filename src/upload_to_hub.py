from huggingface_hub import HfApi, create_repo

REPO_ID = "shadmanArko/dhaka-kacchi-dish-classifier"
LOCAL_MODEL_DIR = "model_export"

api = HfApi()

# Creates the repo if it doesn't exist yet; harmless if it already does
create_repo(REPO_ID, exist_ok=True, repo_type="model")

api.upload_folder(
    folder_path=LOCAL_MODEL_DIR,
    repo_id=REPO_ID,
    repo_type="model",
)

print(f"\nUploaded. View it at: https://huggingface.co/{REPO_ID}")