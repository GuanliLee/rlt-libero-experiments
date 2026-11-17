"""Generate LIBERO paths for the current checkout."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
libero = root / "src/LIBERO/libero/libero"
if not libero.is_dir():
    raise SystemExit("Clone LIBERO into src/LIBERO first")
dataset = root / "datasets/libero"
dataset.mkdir(parents=True, exist_ok=True)
lines = {
    "benchmark_root": libero,
    "bddl_files": libero / "bddl_files",
    "init_states": libero / "init_files",
    "assets": libero / "assets",
    "datasets": dataset,
}
target = root / "config/libero/config.yaml"
target.write_text("".join(f"{key}: {value}\n" for key, value in lines.items()))
print(target)
