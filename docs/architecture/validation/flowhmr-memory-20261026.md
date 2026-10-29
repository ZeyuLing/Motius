# FlowHMR 全帧低显存验证 · 2026-10-26

Motius 0.3.8 / Blender extension 0.2.4。用户提供的 25 秒视频，1280×720、30 fps、750 帧。全程提取身体与双手特征，VGGT interval=1，没有独立时间窗、减少帧数、模型量化或新增精度降级。

## 实测结果

| 完整后端流程 | 原始仓库 | 上一版 Fast SAM | Fast SAM + streamed VGGT |
| --- | ---: | ---: | ---: |
| PyTorch 峰值分配 | 39.98 GiB | 27.80 GiB | **5.39 GiB** |
| PyTorch 峰值保留 | 59.49 GiB | 47.29 GiB | **5.46 GiB** |
| 包含模型加载的后端耗时 | 413.60 s | 253.94 s | **257.46 s** |

新流程在 H200 上设置 **6 GiB 的 PyTorch 分配器限额**，完整跑通并解码全部 750 帧。分配峰值比上一版下降 **80.62%**。单次耗时增加 1.39%；这不是多次测量的统计结论。新进程包含 Python 导入的总时间为 260.73 秒。

驱动层 `nvidia-smi` 每秒采样，本次进程峰值 **6284 MiB（6.14 GiB）**，包含框架外 CUDA 占用；采样可能漏掉短暂峰值。CPU 峰值 RSS **10.58 GiB**。没有将容器原有的 1546 MiB 服务进程算作本次推理显存；该服务始终保留，未停止。

**这证明了 H200 上的显存预算可行，不是消费级显卡的实卡验证。** 没有测量 RTX/T4 的耗时、内核兼容性或 Blender 同时运行的总显存。完整流程仍约 4 分 17 秒，速度问题仍在。750 帧的结果不构成长视频固定显存承诺：全局键值和残差随帧数增长。

完整后端分段：加载 2.64 s，转码 2.15 s，检测跟踪 9.60 s，相机 71.24 s，身体/双手特征（包含加载）168.43 s，采样解码 1.40 s。时间不包含 Blender 导入、重定向、接地或播放准备。

## 实现

- Pipeline 控制 CPU/GPU 模块生命周期；网络 bundle 仅包含网络计算。VGGT 每次只将当前模块放在 GPU，计算完成后移回 CPU。
- DINO 编码器和帧内块按 8 帧计算，保留所有帧及原顺序。
- 跨帧块先构造完整、不可变的 K/V，再分批计算查询；每个查询的 softmax 始终覆盖全片原有键。残差在无梯度评估模式下原地更新，FFN 按 token 分批。
- K/V 按原有 AMP 注意力实际使用的 dtype 保存，避免保留 FP32 副本后再创建 AMP 副本。训练与标准推理路径保留。
- 只保留 CameraHead 使用的最终 camera/register 前缀；相机头使用 256 个查询一批的全局注意力。

独立相机阶段测试同样完整处理 750 帧，耗时 73.68 s，峰值分配 5.39 GiB、CPU RSS 7.50 GiB。第一轮限额测试在注意力转换临时缓存上 OOM，修正 K/V 存储后才通过；未通过降低帧数绕过限制。

## 原实现输出对照

沿用上次报告的独立原始仓库输出、相同视频、模型权重与 20 步采样、seed=0、TF32=False、无动作后处理。跟踪框逐位相同。参考代码与权重身份见 [Fast SAM 报告](flowhmr-fast-sam-20261026.md) 及本次 JSON 的 provenance。

| 指标 | 相对原始仓库 | 相对上一版 Fast SAM |
| --- | ---: | ---: |
| 所有 52 关节局部旋转平均差 | 0.03283° | 0.03330° |
| 所有 52 关节局部旋转最大差 | 1.24672° | 1.14610° |
| 根平移最大差 | 3.94664 mm | 3.51493 mm |
| 所有 52 关节世界位置平均差 | 2.19038 mm | 1.81253 mm |
| 所有 52 关节世界位置最大差 | 5.37048 mm | 4.67318 mm |

相机旋转最大差 0.02719°，内参最大绝对差 1.7146 像素。VGGT 相机平移没有米制标定，其差值保留在 JSON 的原生单位中，不能解释成毫米。身体、左手、右手 token 最低余弦相似度分别为 0.99999979、0.99642057、0.99618542。

**输出并非完全一致，也不应将上述差异归结为已验证的原实现重跑噪声。** 改变分批计算会改变浮点舍入，手框条件链也会放大中间差异；本测试没有真人动捕真值，不能证明精度无损或提高。要求严格保留原计算调度时可选 `vggt_memory_mode='reference'`；完整原始 SAM 路径另设 `sam_backend='official'`。

## 使用与复测

`--lightweight` 默认启用 Fast SAM 和 streamed VGGT，Blender 的加速模式同样生效。可显式指定：

```bash
python tools/infer_flowhmr.py /path/to/checkpoint /path/to/video.mp4 \
  --lightweight --sam-backend fast --vggt-memory-mode streamed \
  --vggt-frame-interval 1 --profile
```

相机单项限额测试：

```bash
PYTHONPATH=. python tools/validate_flowhmr_memory.py \
  --assets /path/to/asset_paths.json --video /path/to/input_30fps.mp4 \
  --output outputs/validation/flowhmr/memory-camera --frames 750 --budget-gib 6
```

完整对照 worker 使用 `tools/validate_flowhmr_fast_sam.py --variant fast --frames 750 --vggt-memory-mode streamed --memory-budget-gib 6`，其余视频、检查点、body model、regressor、assets 参数与原始对照一致。没有使用预计算前端缓存。

77 项回归测试通过、1 项跳过；两种插件 ZIP 均通过 Blender 4.5.3 extension validate。测试覆盖完整键注意力、非零残差、批次/帧顺序、register 块、微批大小约束、评估限制和异常清理。

- [完整数据与执行身份](flowhmr-memory-20261026.json)
- 本地原始数组、报告与逐帧 CSV：`outputs/validation/flowhmr/memory-20261026/`
- `fast/result.npz` 包含可供 Blender 使用的完整 750 帧和 52 关节旋转；`results.tar.gz` 保存原始特征、相机和动作数组。
