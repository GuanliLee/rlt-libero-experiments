#!/usr/bin/env python3
"""Initialize LIBERO-Goal and verify AlphaBrain's real subprocess protocol."""
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("LIBERO_HOME", str(ROOT / "src/LIBERO"))
os.environ.setdefault("LIBERO_PYTHON", str(ROOT / "envs/libero/bin/python"))
os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / "config/libero"))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / "cache/numba"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "cache/xdg"))

proxy_file = ROOT / "src/AlphaBrain/AlphaBrain/training/reinforcement_learning/envs/libero_env.py"
spec = importlib.util.spec_from_file_location("rlt_libero_proxy", proxy_file)
proxy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proxy)
output = ROOT / "outputs/libero_smoke"
output.mkdir(parents=True, exist_ok=True)

suite_info = proxy.get_suite_info("libero_goal")
env = proxy.LiberoEnv()
try:
    obs = env.reset("libero_goal", task_id=0, initial_state_idx=0, seed=42)
    obs["primary_image"].save(output / "primary_initial.png")
    obs["wrist_image"].save(output / "wrist_initial.png")
    initial_state = obs["state"].tolist()
    assert obs["state"].shape == (8,)
    assert obs["primary_image"].size == (256, 256)
    assert obs["wrist_image"].size == (256, 256)
    for _ in range(10):
        obs, reward, done = env.step(np.zeros(7, dtype=np.float32))
    obs["primary_image"].save(output / "primary_after_10_steps.png")
    result = {
        "status": "passed",
        "scope": "simulation and subprocess protocol; no model policy evaluated",
        "task_suite": "libero_goal",
        "task_id": 0,
        "initial_state_idx": 0,
        "seed": 42,
        "task_description": env.task_description,
        "suite_info": suite_info,
        "state_shape": list(obs["state"].shape),
        "initial_state": initial_state,
        "primary_image_size": list(obs["primary_image"].size),
        "wrist_image_size": list(obs["wrist_image"].size),
        "simulation_steps": 10,
        "reward_after_steps": reward,
        "done_after_steps": done,
        "render_backend": os.environ["MUJOCO_GL"],
        "libero_python": os.environ["LIBERO_PYTHON"],
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
finally:
    env.close()
