# 可迁移的 FlowHMR GPU 后端

本包提供固定 NVIDIA PyTorch 25.01 基础环境和从已验证环境导出的 Python 依赖快照；Motius 源码、Blender 外部 worker 和启动适配器一起打包。推理仍由统一 FlowHMR pipeline 执行，普通版和低显存版均可使用同一个镜像。镜像不包含模型权重、视频、用户凭据或 SMPL-H 文件。

## 支持范围

目标是 Linux x86_64、Python 3.12、NVIDIA GPU 和 NVIDIA Container Toolkit，适合在支持 NVIDIA 容器的本地工作站、Koala、Taiji 或其他 Linux 平台部署。需要与 CUDA 12.8 兼容的 NVIDIA 驱动。Blender 可以和计算运行时安装在不同的环境；此适配器使用本机 Docker daemon，因此 Blender 与 Docker daemon 必须访问同一套本地文件路径。不同主机的远程调用需要额外服务层，不能把本适配器当作远程协议。

Windows / macOS、非 NVIDIA GPU 和消费级 GPU 尚未实机验证；镜像格式不保证跨架构运行。图形界面仍在宿主机运行，显存和磁盘预检在 GPU 容器内执行。

## 构建与安装

1. 在目标 Linux 主机安装 Docker 和 NVIDIA Container Toolkit，检查 `docker run --rm --gpus all nvcr.io/nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi`。
2. 解压 `motius-flowhmr-0.3.16-runtime-context.tar.gz`，在解压目录运行：

   ```bash
   docker build -t motius-flowhmr:0.3.16 .
   ```

   构建会拉取 `nvcr.io/nvidia/pytorch:25.01-py3`。依赖快照随构建上下文提供，不需要在容器构建时运行 pip。构建时检查 Python、PyTorch、TorchVision、NumPy、SciPy 版本和 TorchVision 编译算子；不匹配时明确失败。若环境要求不可变基础镜像，使用该版本已验证的 digest 传入 `--build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch@sha256:...`，并记录实际构建后的 `docker image inspect` ID。
3. Blender 安装标准版或低显存版插件。在 **Options → Backend setup → External Python** 指向解压目录中的 `deploy/flowhmr/container_python.py`；保持脚本可执行，并确保宿主机有 Python 3.10+。**Motius checkout** 指向宿主机当前项目目录，用于解析已有文件。容器实际执行的代码来自镜像快照。
4. 指定已授权 neutral SMPL-H、模型检查点以及可写的 Job output 和 Weight cache。路径自动以相同绝对路径挂载：输入和权重只读、当前任务目录及权重缓存可写。默认 Hugging Face 模型需要在线下载；关闭 Blender 联网权限时容器使用 `--network none`，只读取已缓存的资源。
5. 点击 **Check Environment** 后按正常队列流程使用。队列服务是宿主机的轻量 Python 进程，只有每个推理 worker 在容器中执行。每次尝试使用独立容器名；取消只删除该尝试的容器。

同一镜像可用 `docker save motius-flowhmr:0.3.16 -o motius-flowhmr-image.tar` 导出，在另一个满足条件的 Linux NVIDIA 主机用 `docker load -i motius-flowhmr-image.tar` 导入。也可以把构建上下文复制到目标平台重新构建；不要把模型权重或私有凭据放入构建上下文。

如果不使用 Blender，可在平台自己的容器任务中直接运行镜像内的 `/opt/motius/blender_extension/motius_flowhmr/worker.py --request <request.json>`；请求的 `project_root` 设置为 `/opt/motius`，视频、检查点、body model 和 output 路径需在该容器中存在。请求/结果协议与 Blender 队列相同。

## 运行验证与限制

`runtime-overlay.tar.gz.json` 包含冻结的包版本、目标 Python/架构、基础镜像和依赖快照 SHA-256；上下文旁的 JSON 包含源码与整个压缩包的 SHA-256。`verify_runtime.py` 可用于检查依赖和 ABI。不同 GPU、驱动及模型输入仍需要真实运行验收；环境快照不保证逐位相同或任何显卡均有足够显存。

构建上下文和依赖快照已做完整性、源码身份、启动适配器与宿主隔离验证。实际 Docker 镜像构建及另一个 NVIDIA 主机上的完整视频结果是否通过，以随包验证报告为准；不要把“构建上下文已打包”表述为“所有平台已实测”。

## 与 ONNX / TensorRT 的关系

此交付物封装当前统一 PyTorch 推理实现与依赖，并不包含 ONNX 图或 TensorRT 引擎，也不宣称它们带来的加速。迁移包的目标是复用同一套执行代码、精度策略和任务协议；它仍受操作系统、CPU 架构、NVIDIA 驱动及显存约束。后续若针对特定网络子模块导出 ONNX 或构建 TensorRT 引擎，需要分别验证每种形状、精度、输出与运行平台，并保留 PyTorch 后备路径。[TensorRT 引擎兼容性文档](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html)说明了平台、版本与 GPU 硬件兼容条件，不能假设单个引擎自动适配所有机器。
