"""
upload_to_hf.py
===============
Uploads the bot to Hugging Face Spaces (mastertheo7277/usbot) via the official
Hugging Face API, automatically handling binary files (fonts, images) without git-lfs errors.

Usage:
  python upload_to_hf.py
  python upload_to_hf.py hf_yourTokenHere
"""

import os
import sys
from huggingface_hub import HfApi

REPO_ID = "mastertheo7277/usbot"

token = os.getenv("HF_TOKEN")
if not token and len(sys.argv) > 1:
    token = sys.argv[1].strip()

if not token:
    print("👉 Get a Write token from: https://huggingface.co/settings/tokens")
    token = input("Enter your Hugging Face Access Token (write): ").strip()

if not token:
    print("❌ No token provided. Exiting.")
    sys.exit(1)

api = HfApi(token=token)

print(f"🚀 Uploading project to Hugging Face Space: https://huggingface.co/spaces/{REPO_ID} ...")

try:
    api.upload_folder(
        repo_id=REPO_ID,
        folder_path=".",
        repo_type="space",
        commit_message="Deploy Textbox Bot to Hugging Face Space",
        ignore_patterns=[
            ".git*",
            ".venv*",
            "venv*",
            "*.pyc",
            "__pycache__*",
            "scratch*",
            "9to5_reel_bot*",
            ".env*",
            "*.mp4",
            "*.mov",
            "*.pid",
            "node_modules*",
            "user_support_*.png"
        ]
    )
    print(f"\n✅ Upload complete! Check your Space build at:\n   https://huggingface.co/spaces/{REPO_ID}")
except Exception as e:
    print(f"\n❌ Upload failed: {e}")
