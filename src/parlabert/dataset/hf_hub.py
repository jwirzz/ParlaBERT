import os

from huggingface_hub import HfApi

DATASET_REPO = os.environ.get("HF_DATASET_REPO")

TOKEN_HELP = "HF_TOKEN is not set - create a token with write access"


def api() -> HfApi:
    token = os.environ.get("HF_TOKEN")

    if not token:
        raise SystemExit(TOKEN_HELP)

    return HfApi(token=token)
