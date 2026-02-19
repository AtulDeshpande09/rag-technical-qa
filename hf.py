from huggingface_hub import HfApi

api = HfApi()

# Upload LoRA
api.upload_folder(
    folder_path="models/mistral_lora",
    repo_id="AtulDeshpande/mistral-interview-assistant",
    path_in_repo="lora"
)

# Upload merged model
api.upload_folder(
    folder_path="models/mistral_merged",
    repo_id="AtulDeshpande/mistral-interview-assistant",
    path_in_repo="merged"
)
