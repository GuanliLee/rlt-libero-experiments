#!/usr/bin/env python3
"""Verify the official source archive against the pinned GitHub Git tree."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
commit = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
tree = json.loads((root / "manifests/libero-git-tree.json").read_text())
assert not tree.get("truncated"), "GitHub tree was truncated"
files = [item for item in tree["tree"] if item["type"] == "blob"]
failures = []
for item in files:
    path = root / "src/LIBERO" / item["path"]
    if not path.is_file():
        failures.append(f"Missing {item['path']}")
        continue
    data = path.read_bytes()
    digest = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    if digest != item["sha"]:
        failures.append(f"SHA mismatch {item['path']}")
if failures:
    raise RuntimeError("\n".join(failures))

manifest = {
    "repository": "https://github.com/Lifelong-Robot-Learning/LIBERO",
    "commit": commit,
    "source_url": f"https://raw.githubusercontent.com/Lifelong-Robot-Learning/LIBERO/{commit}/",
    "retrieval": "official source files, with every Git blob verified against the pinned GitHub recursive Git tree; initial files recovered from an interrupted official codeload archive",
    "verified_source_files": len(files),
    "source_file_bytes": sum(item.get("size", 0) for item in files),
    "training_demonstrations_downloaded": False,
}
(root / "manifests/libero-source.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
