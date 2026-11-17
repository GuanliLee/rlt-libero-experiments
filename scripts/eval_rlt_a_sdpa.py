#!/usr/bin/env python3
"""Run the pinned AlphaBrain RLT_a evaluator with an in-memory SDPA override.

The downloaded checkpoint files and the upstream checkout remain unchanged.
All command-line arguments are forwarded to the upstream eval_libero.py.
"""
from pathlib import Path
import json
import os
import struct
import sys


ROOT = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get("RLT_SOURCE_ROOT", ROOT / "src" / "AlphaBrain")).resolve()
if not (REPO / "AlphaBrain").is_dir():
    raise FileNotFoundError(f"Invalid AlphaBrain source directory: {REPO}")
sys.path.insert(0, str(REPO))

from AlphaBrain.model.framework import base_framework


_read_mode_config = base_framework.read_mode_config
_from_pretrained = base_framework.BaseFramework.from_pretrained


def _read_mode_config_sdpa(checkpoint):
    checkpoint = Path(checkpoint)
    weights = checkpoint / "model.safetensors"
    if not weights.is_file() or weights.with_suffix(weights.suffix + ".aria2").exists():
        raise FileNotFoundError("Finish downloading model.safetensors before evaluation.")
    if not any(
        (checkpoint / name).is_dir() and any((checkpoint / name).iterdir())
        for name in ("vlm_pretrained", "qwen_pretrained")
    ):
        raise FileNotFoundError(
            "Expected a complete self-contained QwenOFT checkpoint with "
            "qwen_pretrained/ or vlm_pretrained/. Finish its download first."
        )
    config, norm_stats = _read_mode_config(checkpoint)
    config["framework"]["qwenvl"]["attn_implementation"] = "sdpa"
    return config, norm_stats


base_framework.read_mode_config = _read_mode_config_sdpa


def _from_pretrained_checked(cls, checkpoint, **kwargs):
    model = _from_pretrained(checkpoint, **kwargs)
    weights = Path(checkpoint) / "model.safetensors"
    with weights.open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        if not 0 < header_size < 16 * 1024 * 1024:
            raise ValueError("Invalid safetensors header size")
        header = json.loads(stream.read(header_size))
    supplied = {
        "qwen_vl_interface." + key[4:] if key.startswith("vlm.") else key
        for key in header if key != "__metadata__"
    }
    expected = set(model.state_dict())
    audit = {
        "source_directory": str(REPO),
        "checkpoint": str(Path(checkpoint).resolve()),
        "supplied_keys": len(supplied),
        "expected_keys": len(expected),
        "missing_keys": sorted(expected - supplied),
        "unexpected_keys": sorted(supplied - expected),
    }
    vlm_config = model.qwen_vl_interface.model.config
    audit["attention_implementation"] = {
        "vlm": getattr(vlm_config, "_attn_implementation", None),
        "text": getattr(getattr(vlm_config, "text_config", None), "_attn_implementation", None),
        "vision": getattr(getattr(vlm_config, "vision_config", None), "_attn_implementation", None),
    }
    (ROOT / "logs" / "alphabrain_weight_key_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    if audit["missing_keys"] or audit["unexpected_keys"]:
        raise RuntimeError(f"Checkpoint key mismatch; refusing partial loading: {audit}")
    print(f"Checkpoint key audit passed: {len(expected)} matched tensors.", flush=True)
    print(f"Attention implementations: {audit['attention_implementation']}", flush=True)
    return model


base_framework.BaseFramework.from_pretrained = classmethod(_from_pretrained_checked)


if __name__ == "__main__":
    import torch
    from AlphaBrain.training.reinforcement_learning.eval.eval_libero import main

    main()
    if torch.cuda.is_initialized():
        gib = 1024 ** 3
        print(
            "PyTorch peak GPU memory: "
            f"allocated={torch.cuda.max_memory_allocated() / gib:.2f} GiB, "
            f"reserved={torch.cuda.max_memory_reserved() / gib:.2f} GiB "
            "(excludes separate LIBERO processes).",
            flush=True,
        )
