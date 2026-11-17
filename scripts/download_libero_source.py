#!/usr/bin/env python3
"""Fetch missing official files at one pinned commit and verify every Git blob."""
import concurrent.futures
import hashlib
import json
import time
import urllib.request
from pathlib import Path

root = Path(__file__).resolve().parents[1]
commit = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
tree = json.loads((root / "manifests/libero-git-tree.json").read_text())
files = [item for item in tree["tree"] if item["type"] == "blob"]

def matches(data, item):
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == item["sha"]

def fetch(item):
    path = root / "src/LIBERO" / item["path"]
    if path.is_file() and matches(path.read_bytes(), item):
        return "cached", len(path.read_bytes())
    url = f"https://raw.githubusercontent.com/Lifelong-Robot-Learning/LIBERO/{commit}/{item['path']}"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=45) as response:
                data = response.read()
            if not matches(data, item):
                raise ValueError(f"Git blob mismatch: {item['path']}")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".download")
            temporary.write_bytes(data)
            temporary.replace(path)
            if item.get("mode") == "100755":
                path.chmod(0o755)
            return "downloaded", len(data)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)

done = 0
downloaded = 0
started = time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
    for status, size in pool.map(fetch, files):
        done += 1
        downloaded += size if status == "downloaded" else 0
        if done % 50 == 0 or done == len(files):
            print(f"Verified {done}/{len(files)} files, downloaded {downloaded / 1048576:.1f} MiB, elapsed {time.monotonic() - started:.1f}s", flush=True)
