# AlphaBrain 单卡推理环境

准备日期：2026-09-08。目标为已发布的 QwenOFT + RLT_a 权重在 LIBERO 中推理；未安装完整训练依赖，也未训练模型。

## 路径与版本

- 源码：`../src/AlphaBrain`，固定 commit `604924beb77b04b0da49326dfae6ea423a27d28a`；未发现 `AGENTS.md`，源码工作区无修改。
- Python：`../envs/python/cpython-3.11.15-linux-x86_64-gnu`。
- 独立环境：`../envs/alphabrain`，指向上述根目录内的 Python 运行时。
- PyTorch `2.7.1+cu128`、torchvision `0.22.1+cu128`、Transformers `4.57.0`、Accelerate `1.5.2`、NumPy `1.26.4`。
- 最小依赖入口：`alphabrain-inference-requirements.in`；实际安装的 56 个包：`alphabrain-env-freeze.txt`。
- 安装脚本：`../scripts/install_alphabrain_env.sh`。脚本采用官方 PyTorch wheel 与清华 PyPI 镜像；关闭代理只作用于其安装子命令。

## 来源及本地复用

Torch CPython 3.11 CUDA 12.8 wheel 已独立下载至 `../cache/wheels`，对照官方响应提供的 SHA256 验证。具体路径、大小、来源、校验值见 `alphabrain-torch-wheel.json`。

14 个 NVIDIA 运行库从现有 `graspnet` 环境复制，版本逐项匹配该 Torch wheel 的 `Requires-Dist`；均为不依赖 Python 小版本的 `py3-none` wheel。Triton 3.3.1 从现有 Python 3.11 的 `openpi` 环境复制，wheel ABI 为 `cp311`。没有复制 Python 3.10 的 Torch 或 Triton，也没有修改任何来源环境。

复制后的 15 个包、710 条文件记录先按原 wheel `RECORD` 校验通过。两个可选 Triton CLI 的 shebang 随后指向本独立环境，并更新其本地 `RECORD`。全部来源、匹配版本和入口调整见 `alphabrain-local-cuda-reuse.json`。

## 运行适配

`../scripts/eval_rlt_a_sdpa.py` 保留上游评估参数，导入固定版本的 `eval_libero.py`，在内存中覆盖 `framework.qwenvl.attn_implementation=sdpa`。没有改动下载的 checkpoint 或上游源码，也没有安装 Flash Attention。

该入口要求完整的本地 self-contained checkpoint；若基础权重仍有 `.aria2` 下载标记便停止，避免读取未完成文件。模型加载后，它会将 safetensors header 中的全部权重名称与模型的 `state_dict` 比对；任何 missing/unexpected key 都会报错，拒绝上游 `strict=False` 回退产生的部分加载。审计结果写入 `../logs/alphabrain_weight_key_audit.json`，同时记录实际 attention implementation。

正常评估结束会打印 PyTorch 的显存 allocated/reserved 峰值；该数值不包含单独的 LIBERO 进程。

## 已完成检查

- `uv pip check`：56 个包，依赖兼容检查通过。
- `eval_rlt_a_sdpa.py --help`：退出码 0；见 `../logs/alphabrain_eval_help.log`。
- `QwenOFT` 模块单独导入：通过；见 `../logs/alphabrain_qwenoft_import.log`。
- RTX 5090（compute capability 12.0）上的 BF16 矩阵乘、BF16 SDPA：通过；见 `../logs/alphabrain_cuda_smoke.json`。

上游框架扫描会输出缺少 `snntorch` 的可选模块提示；该包属于 NeuroVLA，QwenOFT/RLT_a 的实际导入检查通过。完整权重加载与仿真结果由主流程的 `../scripts/run_preview.sh` 和对应结果目录记录。

检查入口：

```bash
/opt/shared_assets/8t/lgl/RLT/envs/alphabrain/bin/python \
  /opt/shared_assets/8t/lgl/RLT/scripts/eval_rlt_a_sdpa.py --help
```
