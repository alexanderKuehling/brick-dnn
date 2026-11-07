#!/usr/bin/env bash
# Builds the online demo and uploads it to a static Hugging Face Space (free, no server).
# Creates the Space if it does not exist yet.
#
#   deploy/huggingface/deploy.sh [space]        default space: alexKlg/brick-dnn
#
# Needs Node.js (see frontend/.nvmrc) and the hf CLI with a login (`hf auth login`).
set -euo pipefail

SPACE="${1:-alexKlg/brick-dnn}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

cd "$ROOT/frontend"
npm run build:demo
cp "$ROOT/deploy/huggingface/README.md" dist-demo/README.md

# `hf upload` would first try to create the repository as a Gradio Space, which needs a paid
# plan, even if the Space exists. So use the Python library the hf CLI runs on instead.
HF_PYTHON="$(sed -n '1s/^#!//p' "$(command -v hf)")"
$HF_PYTHON - "$SPACE" "$ROOT/frontend/dist-demo" <<'PY'
import sys

from huggingface_hub import HfApi

space, folder = sys.argv[1], sys.argv[2]
api = HfApi()
api.create_repo(space, repo_type="space", space_sdk="static", exist_ok=True)
api.upload_folder(
    repo_id=space,
    repo_type="space",
    folder_path=folder,
    # Build files have hashed names: remove the ones this upload replaces
    # (and the stylesheet of Hugging Face's Space template).
    delete_patterns=["assets/*", "style.css"],
    commit_message="Deploy online demo",
)
PY
echo "Done: https://huggingface.co/spaces/$SPACE"
