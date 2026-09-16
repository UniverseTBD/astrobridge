from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from huggingface_hub import HfApi, hf_hub_download

from .config import DEFAULT_HF_DATA_REPO


def get_hf_token() -> Optional[str]:
    """Retrieve the Hugging Face token from environment variables or HF cache."""
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )
    if token:
        return token.strip()
    try:
        from huggingface_hub import get_token

        cached = get_token()
        if cached:
            return cached.strip()
    except Exception:
        pass
    return None


def upload_caption_file(
    local_path: Union[str, Path],
    hf_repo: Optional[str] = None,
    hf_path: Optional[str] = None,
    commit_message: Optional[str] = None,
    token: Optional[str] = None,
) -> str:
    """Upload a generated captions file to the specified Hugging Face dataset repository."""
    local_file = Path(local_path)
    if not local_file.is_file():
        raise FileNotFoundError(f"Local caption file does not exist: {local_file}")

    repo_id = hf_repo or DEFAULT_HF_DATA_REPO
    path_in_repo = hf_path or f"evals/captions/{local_file.name}"

    auth_token = token or get_hf_token()
    if not auth_token:
        raise ValueError(
            "Hugging Face write token is required to upload files. "
            "Please set HF_TOKEN in your environment, in a .env file, or run 'huggingface-cli login'."
        )

    msg = commit_message or f"Upload captions: {local_file.name}"
    api = HfApi(token=auth_token)

    print(f"Uploading {local_file} -> {repo_id}:{path_in_repo} ...")
    result = api.upload_file(
        path_or_fileobj=str(local_file),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=msg,
    )
    print(f"Successfully uploaded to {repo_id}:{path_in_repo}")
    return str(result)


def check_and_download_existing(
    hf_repo: str,
    hf_path: str,
    local_path: Union[str, Path],
    token: Optional[str] = None,
) -> bool:
    """Check if the caption file exists on Hugging Face Hub, and if so, download it locally.

    Returns True if downloaded from HF, False otherwise.
    """
    dest = Path(local_path)
    if dest.is_file() and dest.stat().st_size > 0:
        return False

    auth_token = token or get_hf_token()
    api = HfApi(token=auth_token)

    try:
        exists = api.file_exists(repo_id=hf_repo, filename=hf_path, repo_type="dataset")
    except Exception as exc:
        print(f"Note: Could not check Hugging Face for remote cached captions: {exc}")
        return False

    if not exists:
        return False

    print(f"Found remote cached captions on Hugging Face: {hf_repo}:{hf_path}")
    print(f"Downloading to {dest} ...")
    try:
        downloaded = hf_hub_download(
            repo_id=hf_repo,
            filename=hf_path,
            repo_type="dataset",
            token=auth_token,
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(downloaded, dest)
        print(f"Successfully downloaded cached captions to {dest}")
        return True
    except Exception as exc:
        print(f"Warning: Failed to download existing captions from HF ({hf_repo}:{hf_path}): {exc}")
        return False


def get_git_commit_sha() -> Optional[str]:
    """Retrieve current Git commit SHA if inside a git repository."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        )
        return out.strip()
    except Exception:
        return None


def write_run_metadata(
    output_dir: Union[str, Path],
    suite_config: Dict[str, Any],
    models_run: List[str],
) -> Path:
    """Write run_metadata.json into output_dir recording runtime details."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = out_dir / "run_metadata.json"

    data = {
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit_sha(),
        "python_version": sys.version,
        "suite_name": suite_config.get("suite_name", "unknown"),
        "benchmark": suite_config.get("benchmark", "all"),
        "hf_repo": suite_config.get("hf_repo", DEFAULT_HF_DATA_REPO),
        "hf_subdir": suite_config.get("hf_subdir", "evals/captions"),
        "models_run": models_run,
    }

    with open(metadata_path, "w") as f:
        json.dump(data, f, indent=2)

    return metadata_path


__all__ = [
    "get_hf_token",
    "upload_caption_file",
    "check_and_download_existing",
    "get_git_commit_sha",
    "write_run_metadata",
]

