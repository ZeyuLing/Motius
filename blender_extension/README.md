# FlowHMR Blender extension

[安装与使用说明](motius_flowhmr/README.md)。支持标准和加速两种模式；两种安装包
使用同一份实现，只有新场景的默认推理模式不同。

## Build

```bash
python tools/package_flowhmr_blender.py
```

输出 `dist/flowhmr_blender/` 下的 standard / accelerated ZIP 和 `checksums.json`。
安装包根目录包含 `blender_manifest.toml` 和 `__init__.py`，可用 Blender 4.2+
的 Install from Disk 安装。ZIP 不包含 Motius、Torch、模型权重或 body model。
后端必须使用 Motius 0.3.10 或更新的 checkout 和外部环境；使用本轮 SAM 常量/无用输出优化需更新到 0.3.11。

## Validation

```bash
pytest -q tests/test_flowhmr_blender.py
blender --command extension validate dist/flowhmr_blender/motius-flowhmr-0.2.7-standard.zip
blender --command extension validate dist/flowhmr_blender/motius-flowhmr-0.2.7-accelerated.zip
blender -b --factory-startup --python-exit-code 1 \
  --python tests/blender_flowhmr_smoke.py -- \
  --motion /path/to/motion.blender.npz --output-dir /tmp/flowhmr-blender-test \
  --external-python /path/to/motius/venv/bin/python
```

Blender smoke 检查真实 30 帧 FlowHMR 输出的 52 关节位置和姿态、不同 rest roll
与比例的合成 Mixamo 骨架、骨长保留、水平原地模式、原 Action 保留、FBX 导出/
再导入、异步外部环境检查、取消以及卸载清理。它不等价于任意生产角色的视觉验收。
已有 GPU 推理模式对照见
[FlowHMR 轻量验证记录](../docs/architecture/validation/flowhmr-lightweight.json)。
0.2.3 的完整 Fast SAM 后端与真实视频对照见
[Fast SAM 验证记录](../docs/architecture/validation/flowhmr-fast-sam-20261026.md)。
每个推理任务会在输出目录保存 `inference_profile.json`，含模型加载、各阶段耗时和
PyTorch 峰值分配/保留显存；记录会区分是否使用预处理缓存，不包含 Blender 内的
网格导入、重定向和接地耗时。
安装包校验、两种模式的实际安装/导入、数值误差及源文件校验和见
[Blender 验证记录](../docs/architecture/validation/flowhmr-blender.json)。

## Layout

- `ui.py`：面板、设置、操作和 Blender 主线程定时回调。
- `jobs.py`：独立进程、取消、环境隔离和日志。
- `worker.py`：外部 Python 中调用 Motius，保存原 capture 与交换数据。
- `motion_data.py`：52 关节协议、坐标变换、Mixamo 名称映射；不加载 pickle。
- `animation.py`：原生动画、实际预测网格、基于目标 rest basis 的重定向与 Action。
- `grounding_math.py` / `grounding.py`：接触检测、足锁、双腿 IK 与实际蒙皮脚底复核。
- `preprocessing_cache.py`：同一视频与前端权重的 SHA-256 缓存校验。
- `defaults.py`：构建时写入 standard / accelerated 默认模式。

0.2.4 的全帧低显存后端实测见
[750 帧显存验证](../docs/architecture/validation/flowhmr-memory-20261026.md)。

0.2.5 接通逐阶段进度、速率和 ETA，并默认合批 8 帧提取身体/双手特征。
[完整视频速度与输出对照](../docs/architecture/validation/flowhmr-batches-20261026.md)。

0.2.6 的加速模式自动测量并选择相机注意力内核，保留所有帧与全局键。
[VGGT 全片速度和输出对照](../docs/architecture/validation/flowhmr-vggt-20261026.md)。

0.2.7 让独立角色共享场景逐帧更新，各自求解和完整复核接地；有跨对象依赖时保留串行处理。
配合 Motius 0.3.11，750 帧后端实测 104.30 秒；仍未达到 60 秒，未降低帧数、手部特征或精度设置。
[纯计算等待、动作差异和真实 Blender 对照](../docs/architecture/validation/flowhmr-extreme-20261026.md)。
