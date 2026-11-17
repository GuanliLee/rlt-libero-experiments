# Experiment artifacts

The [v1.0 release](https://github.com/GuanliLee/rlt-libero-experiments/releases/tag/v1.0) contains both archives:

| Archive | Contents |
| --- | --- |
| `rlt-trained-checkpoints.tar.gz` | All 17 local `.pt` files from `results/` |
| `rlt-replay-datasets.tar.gz` | All 35 local `.npz` files from `results/` |

Download both files to the repository root, verify them with `sha256sum -c artifacts/SHA256SUMS`, then run `tar -xzf` on each. The paths inside the archives restore their original locations under `results/`. Per-file hashes are in the two adjacent `.sha256` manifests.

The separate upstream `checkpoints/pi05_libero` model is available from its original publisher and is not part of these local experiment artifacts.
