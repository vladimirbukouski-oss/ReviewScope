#!/usr/bin/env python3
"""
Download models from Hugging Face or cloud storage if not present locally.
"""
import os
from pathlib import Path


def check_models_exist():
    """Check if models are already downloaded."""
    sentiment_model = Path("./models/sentiment/final/model.safetensors")
    rating_model = Path("./models/rating/final/model.safetensors")

    return sentiment_model.exists() and rating_model.exists()


def main():
    """Download models if needed."""
    if check_models_exist():
        print("[✓] Models already exist, skipping download")
        return

    print("[!] Models not found locally")
    print("[!] You need to either:")
    print("    1. Include models in Docker image (2.2GB)")
    print("    2. Upload models to cloud storage (S3/R2) and download on startup")
    print("    3. Use Hugging Face Hub to host models")
    print()
    print("[!] For now, proceeding without models (will fail on first request)")
    print("[!] Set HF_MODEL_REPO env var to auto-download from Hugging Face")

    # Example: Download from Hugging Face (if you upload your models there)
    hf_repo = os.getenv("HF_MODEL_REPO")
    if hf_repo:
        print(f"[→] Downloading from Hugging Face: {hf_repo}")
        from huggingface_hub import snapshot_download

        try:
            snapshot_download(
                repo_id=hf_repo,
                local_dir="./models",
                local_dir_use_symlinks=False
            )
            print("[✓] Models downloaded successfully")
        except Exception as e:
            print(f"[✗] Failed to download models: {e}")


if __name__ == "__main__":
    main()
