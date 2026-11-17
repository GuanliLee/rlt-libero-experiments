#!/usr/bin/env python3
"""Download the paired released RLT_a checkpoints at fixed revisions."""
import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

from huggingface_hub import HfApi

MODELS = [
    ("AlphaBrainGroup/qwenoft-5traj-libero-goal",
     "91ea0817f556bf50d64008187519de19458f8188", "qwenoft_5traj"),
    ("AlphaBrainGroup/alphabrain-rlt-5traj-alltasks-libero-goal",
     "b5c0080e46f4b900c049e76bd9f3df68befac12a", "rlt_a"),
]


def download(model):
    repo, revision, folder = model
    destination = ROOT / "checkpoints" / folder
    print(f"Downloading {repo}@{revision} -> {destination}", flush=True)
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    # The configured proxy is slow for large files. Use the public HF mirror
    # directly, but validate against metadata from the original Hugging Face API.
    lines = []
    for entry in info.siblings:
        path = destination / entry.rfilename
        if (path.is_file() and path.stat().st_size == entry.size
                and not Path(str(path) + ".aria2").exists()):
            continue  # All existing files are hashed below, including small files.
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://hf-mirror.com/{repo}/resolve/{revision}/{quote(entry.rfilename)}"
        lines.extend([url, f"  dir={path.parent}", f"  out={path.name}"])
        if entry.lfs:
            lines.append(f"  checksum=sha-256={entry.lfs.sha256}")
    if lines:
        aria_input = ROOT / "manifests" / f"{folder}.aria2.txt"
        aria_input.write_text("\n".join(lines) + "\n")
        download_env = {key: value for key, value in os.environ.items()
                        if key.lower() not in {"http_proxy", "https_proxy", "all_proxy"}}
        subprocess.run([
            "aria2c", "--input-file=" + str(aria_input), "--continue=true",
            "--max-connection-per-server=16", "--split=16", "--min-split-size=8M",
            "--max-concurrent-downloads=2", "--file-allocation=none",
            "--auto-file-renaming=false", "--allow-overwrite=true",
            "--max-tries=8", "--retry-wait=3", "--connect-timeout=15", "--timeout=60",
            "--summary-interval=30", "--console-log-level=warn",
            "--show-console-readout=false",
            "--download-result=hide",
        ], env=download_env, check=True)
    files = []
    for entry in info.siblings:
        path = destination / entry.rfilename
        size = path.stat().st_size
        if entry.size is not None and size != entry.size:
            raise RuntimeError(f"Size mismatch: {path}: {size} != {entry.size}")
        sha = hashlib.sha256()
        git_hash = hashlib.sha1(f"blob {size}\0".encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                sha.update(block)
                git_hash.update(block)
        checksum = sha.hexdigest()
        expected = getattr(entry.lfs, "sha256", None) if entry.lfs else None
        if expected and checksum != expected:
            raise RuntimeError(f"SHA256 mismatch: {path}")
        if not expected and entry.blob_id and git_hash.hexdigest() != entry.blob_id:
            raise RuntimeError(f"Git blob hash mismatch: {path}")
        files.append({"path": str(path.relative_to(ROOT)), "bytes": size,
                      "sha256": checksum, "hub_lfs_sha256": expected})
    record = {"repo_id": repo, "revision": revision,
              "download_endpoint": "https://hf-mirror.com",
              "verification_metadata_endpoint": "https://huggingface.co",
              "verified_at": datetime.now(timezone.utc).isoformat(), "files": files}
    output = ROOT / "manifests" / f"{folder}.json"
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Verified {repo}: {sum(f['bytes'] for f in files):,} bytes", flush=True)
    return record


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(download, MODELS))
    print("Both checkpoints downloaded and verified.", flush=True)
