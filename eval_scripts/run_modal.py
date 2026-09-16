from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
import modal

APP_NAME = "astrobridge-evals"
REPO_ROOT = Path(__file__).resolve().parent.parent

app = modal.App(APP_NAME)


def download_qwen() -> None:
    import os
    from huggingface_hub import snapshot_download

    model_id = "meta-models/Muse-Glimmer-30B"
    print(f"Pre-downloading base model weights: {model_id}")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    snapshot_download(repo_id=model_id, token=token)


ENV_FILE = REPO_ROOT / ".env"
eval_secrets: list[modal.Secret] = []
if ENV_FILE.is_file():
    eval_secrets.append(modal.Secret.from_dotenv(path=ENV_FILE))
else:
    hf_tokens = {k: os.environ[k] for k in ("HF_TOKEN", "HUGGINGFACE_TOKEN") if k in os.environ}
    if hf_tokens:
        eval_secrets.append(modal.Secret.from_dict(hf_tokens))

eval_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install_from_pyproject(str(REPO_ROOT / "pyproject.toml"))
    # .run_function(download_qwen, secrets=eval_secrets)
    .add_local_dir(str(REPO_ROOT / "src"), remote_path="/root/astrobridge-eval/src")
    .add_local_dir(str(REPO_ROOT / "configs"), remote_path="/root/astrobridge-eval/configs")
    .add_local_dir(str(REPO_ROOT / "eval_configs"), remote_path="/root/astrobridge-eval/eval_configs")
    .add_local_dir(str(REPO_ROOT / "eval_scripts"), remote_path="/root/astrobridge-eval/eval_scripts")
    .add_local_dir(str(REPO_ROOT / "scripts"), remote_path="/root/astrobridge-eval/scripts")
    .add_local_file(str(REPO_ROOT / "pyproject.toml"), remote_path="/root/astrobridge-eval/pyproject.toml")
)


MODAL_GPU = os.environ.get("MODAL_GPU", "A100-80GB")


@app.function(
    image=eval_image,
    gpu=MODAL_GPU,
    timeout=86400,
    secrets=eval_secrets,
)
def run_command_remote(
    script: str, args: str, output_path: str | None = None
) -> bytes | None:
    workdir = "/root/astrobridge-eval"
    python_bin = shutil.which("python") or sys.executable
    cmd = [python_bin, script] + shlex.split(args)
    print(f"Running on Modal remote: {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{workdir}/src:{workdir}"
    subprocess.run(cmd, cwd=workdir, env=env, check=True)

    if output_path:
        target = Path(output_path)
        if not target.is_absolute():
            target = Path(workdir) / target

        if target.is_file():
            print(f"Reading output file {target} ({target.stat().st_size} bytes) to download locally...")
            return target.read_bytes()
        elif target.is_dir():
            print(f"Archiving output directory {target} to download locally...")
            import io
            import tarfile

            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tar:
                tar.add(str(target), arcname=target.name)
            return buf.getvalue()
        else:
            raise FileNotFoundError(f"Expected remote output not found at: {target}")
    return None


@app.local_entrypoint()
def main(
    script: str = "eval_scripts/generate_captions.py",
    args: str = "--responder eval_configs/responders/astrobridge.yaml --output eval_results/cached_captions/astrobridge_captions.jsonl --benchmark all",
    output: str | None = None,
    remote_output: str | None = None,
) -> None:
    if remote_output:
        target_remote = remote_output
    else:
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--output", type=str, default=None)
        parser.add_argument("--output-dir", type=str, default=None)
        parsed, _ = parser.parse_known_args(shlex.split(args))
        target_remote = parsed.output or parsed.output_dir

    if not target_remote:
        raise ValueError("Cannot determine remote output path from arguments.")

    target_local = output or target_remote

    print(f"Dispatching script '{script}' to Modal with args: {args}")
    print(f"Remote output: '{target_remote}' -> Local destination: '{target_local}'")
    result_bytes = run_command_remote.remote(
        script=script, args=args, output_path=target_remote
    )

    if not result_bytes:
        raise RuntimeError(
            f"Modal execution finished, but returned no data for remote output '{target_remote}'"
        )

    dest = Path(target_local)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if result_bytes[:2] == b"\x1f\x8b":  # gzip compressed archive
        import io
        import tarfile

        with tarfile.open(fileobj=io.BytesIO(result_bytes), mode="r:gz") as tar:
            tar.extractall(dest.parent)
        print(f"Successfully downloaded and extracted output directory to {dest}")
        return

    dest.write_bytes(result_bytes)
    print(f"Successfully downloaded output file to {dest} ({len(result_bytes)} bytes)")
