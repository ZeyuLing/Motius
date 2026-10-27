# Blender 原生运行时安装验证

2026-10-27 在 Blender 4.5.3 LTS、Linux x86_64、NVIDIA H200 上，从空运行时目录使用标准插件包启动 **Install / update runtime**。安装器通过固定 SHA-256 校验 uv 0.9.11，安装独立 CPython 3.12.12、PyTorch 2.6.0 / CUDA 12.4、TorchVision 0.21.0 和固定版本的 FlowHMR 依赖；后端源码来自插件 ZIP。Blender 进程未导入 Torch。安装器在独立目录下载固定 Hugging Face revision `e77d64daaf23a030d6908c843b2f4e36888097b3` 的 18 个模型和许可文件，权重总计约 9.66 GB（十进制）。插件观察到实际字节进度；Python 和依赖显示阶段与正在下载的包，不虚构总体百分比。

测试选择已有授权 neutral SMPL-H。安装完成后实际运行 **Check environment**，核对 CUDA、TorchVision 算子、模型文件、FFmpeg/FFprobe 和 SMPL-H，再在 Blender 任务队列中对用户完整 750 帧视频执行标准模式 FP16 推理，并手动导入 52 关节骨架。重复两次从独立空目录启动安装并下载真实权重；第二轮完整安装、检查、推理与导入耗时约 189 秒，仅反映该机器和当时网络，不代表一般安装时间。

另用最终标准插件包从第三个空目录启动安装，在真实权重下载开始后退出 Blender 进程，再启动另一个 Blender 进程。插件按 PID、进程启动时间及命令行重新连接原安装任务，持续到安装和环境检查完成；两次 Blender 进程均正常退出，重连测试成功。

第二轮推理内部计时 123.22 秒，PyTorch 峰值 allocated 3.90 GiB、reserved 4.34 GiB；这些数字不是驱动级进程峰值。与此前同视频严格 FP32 的世界关节平均 / 最大差为 1.317 / 8.016 mm，手部旋转平均 / 最大差 0.057° / 1.282°。这只是固定视频下的数值比较，不代表真实动作准确度或其他硬件的表现。

另外在干净 PATH 的独立虚拟环境实际下载并找到静态 FFmpeg 和 FFprobe，验证 SHA-256；该测试模拟 CUDA 可用，端到端 Blender 运行使用系统 `/usr/bin` 的 FFmpeg/FFprobe 和真实 GPU。主测试日志不包含用户凭据；`outputs/validation/flowhmr/product-release-20261027/shared-standard-20261027/native-installer-validation/` 保存结果摘要和 profile。

自动安装目前只开放 Linux x86_64 / NVIDIA CUDA；Windows、macOS、AMD GPU、消费级 NVIDIA GPU及无网首次安装均未验证。插件不携带或下载 SMPL-H，用户需要自行提供已授权的 neutral 模型。安装中断后权重缓存可复用；已完成的旧环境不会被新安装覆盖或删除。若更改默认运行时目录，应在 Blender 中保存偏好设置，以便下次启动后找到该目录。
