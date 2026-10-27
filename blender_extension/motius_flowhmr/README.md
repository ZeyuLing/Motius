# Motius FlowHMR for Blender

在视频中选择一个人物，将身体与双手动作导入 Blender，并重定向到多个 Mixamo 角色。
适用于 Blender 4.2+；已在 Blender 4.5.3 LTS / Linux 实测。

[产品视频与图文快速开始](https://github.com/ZeyuLing/Motius/tree/main/docs/flowhmr-blender)。桌面使用不需要 Docker：插件可一键安装隔离的外部 Python、依赖和模型权重。

## 选人、动作与角色

- **Detect people**：运行真实 YOLOX / ByteTrack，显示处理帧数。完成后点 **Choose person**。
- 在人物框内部点击选择。框重叠时按 **Tab** 或再次点击切换候选，绿色框表示当前选择；
  箭头或滚轮查看后续帧，按 Enter 确认。背景空白点击不会猜测最近的人。
- 源视频预览保留实际检测框。轨迹与视频 SHA 绑定；目标离开画面、端点缺失或长时间丢失会停止推理，
  应裁剪到连续可见片段后重新检测。短缺口只在同一轨迹内插值。ByteTrack 不能保证任意遮挡、交叉中的身份正确，须检查轨迹。
- **Add capture to queue**：保存当前视频、选人轨迹和参数，加入持久化任务队列；显示真实阶段、计数和阶段 ETA。
  推理期间可以继续准备其他视频、导入角色和编辑场景。完成后由用户在队列点击 **Import result**，不自动修改场景。
- 对一个角色点击 **Apply motion to character → Plant feet on the floor**；对多个角色点击
  **Animate and ground all characters**，分别重定向并根据各自实际脚底求解接地，再并排排列。
- **Prepare & play at original speed**：先缓存角色视口及源视频，再按场景原始帧率播放。
  本示例 750 帧 / 30 fps = 25 秒；不会用慢放或跳帧补偿显示性能。播放期间显示 Cached viewport，
  结束后恢复可编辑的三维场景。Esc/Space 停止；缓存准备可按 Esc 取消。
  修改姿态、材质、视角或范围后缓存失效；重复播放可复用有效缓存。若显示端无法维持原速，
  明确停止并提示，不自动改变播放速度。缓存仅用于显示，不修改动作、接地结果或导出数据。
  预览缓存按三维视口最高 960 像素宽、源图像最高 640 像素宽保存，限制估算显存不超过 3 GiB。
- 验证过的前处理缓存是可选项。缓存保持原始 bbox / 特征完全配对，人物框须在每帧有至少 0.85 IoU，
  且不存在与缓存 ROI 重叠超过 0.25 IoU 的其他检测轨迹；否则拒绝复用。新动作仍实际执行模型采样。

## 推理模式

只需安装 `motius-flowhmr-0.2.12.zip`。在面板中选择 **Standard** 或 **Lightweight**：Standard 使用官方 SAM 裁剪和单手解码；Lightweight 使用 Fast SAM GPU 预处理与双手合批。两种模式均包含在同一安装包，可在提交任务前切换。

普通模式保留官方 SAM 的 CPU 图像裁剪和左右手分开解码，使用逐帧身体与双手 token；两种模式共用帧合批、流式 VGGT、注意力查询分块、模块分阶段驻留和推理无用输出的消除。低显存模式使用 Fast SAM 图像预处理与双手合批：
每帧上传一次 RGB 图像，身体与手部裁剪、翻转、像素缩放和归一化在 GPU 执行；
左右手按 `[left, right]` 合成一次网络调用，保留每帧真实的身体与双手 token。
VGGT 默认仍处理所有帧，原始 SAM 网络、权重和 decoder 中间预测保持不变。
少量仿射元数据保留原来的 CPU 计算，以匹配 OpenCV 的参考裁剪几何。

完整视频对照见仓库的 `docs/architecture/validation/flowhmr-fast-sam-20261026.md`。
0.3.6 的约 366 秒结果属于之前的部分优化，不能代表本版 Fast SAM。
每次实际运行的记录见输出目录的 `inference_profile.json`。全视频逐帧 VGGT
仍可能需要较高显存；H200 数据不代表消费级显卡的速度或兼容性。

单个安装包含两种模式，**无需安装两个插件**。
面板可以随时切换两种模式；已有 `.blend` 保存的模式会继续保留。
两种模式默认使用相同权重、种子和采样配置。上游配置在该模型中使用 BF16 SAM 主干和 BF16 VGGT 聚合；动作采样使用 FP32。不同分批和后端可能产生浮点舍入差异。FP32 选项只改变前端计算，SAM 权重仍源自 BF16 存储。加速版不使用平均手部特征，
不默认稀疏相机采样，也不新增量化、改变原始网络精度设置或默认启用 `torch.compile`。
短视频不保证更快，加速版输出有浮点差异，不能宣称逐位一致。

## 安装与环境

1. 在 Blender **Edit → Preferences → Get Extensions → Install from Disk…** 安装并启用插件 ZIP；在 3D Viewport 按 **N → FlowHMR**。
2. 选择已获授权的 **Neutral SMPL-H (.npz)**：52 个关节、16 个 shape 系数。此文件不包含在插件或下载权重里，男女模型不能替代 neutral 模型。
3. 确认 Blender 在线访问已开启，点击 **Install / update runtime**。插件在后台下载隔离的 Python 3.12、CUDA PyTorch、固定依赖与 FlowHMR 模型（模型文件约 9.7 GB），显示步骤、当前下载项及权重字节进度；Blender 保持可操作。下载需要额外磁盘空间。已有权重可在更新或中断后复用。
4. 完成后点击 **Check environment**。安装器检查 CUDA、TorchVision 算子、模型文件和所选 SMPL-H；若系统没有 FFmpeg/FFprobe，会下载并校验视频工具。用户自行提供 NVIDIA 驱动。
5. 在 **Options → Backend setup** 可更改运行时目录，或为开发用途手动配置 **External Python**、**Motius checkout** 和权重缓存。默认目录为 `~/.cache/motius/flowhmr-runtime`；自动安装和推理不会往 Blender 内置 Python 安装 Torch。

安装期间可以继续使用或关闭 Blender；重新打开后插件会接续显示仍在运行的安装任务。若更改运行时目录，请保存 Blender 偏好设置，使下次启动仍能找到该目录。

自动安装适配 **Linux / Windows x86_64 与 NVIDIA GPU**；完整流程在 Blender 4.5.3 / Linux / H200 上验证。Windows 和消费级 NVIDIA GPU 尚未实机验证，macOS 和 AMD GPU 不支持 CUDA 推理。默认遵循上游 BF16 精度；Options 中可显式选择 FP16 混合精度或 FP32 前端计算。安装包不含权重或 SMPL-H；首次安装需联网，下载耗时不计入推理时间。Blender 关闭联网时不会启动自动安装，推理只读取已有本地权重。

## 视频 → 动画

1. 在 **Your performance** 选择视频，点击 **Preview**，再点击 **Detect people**。
2. 在队列的检测结果点击 **Choose person**，在实际检测框内部选人，检查后续帧并按 Enter 确认。
3. 在 **Capture the motion** 选择 Standard 或 Lightweight，然后点击 **Add capture to queue**。
   保持 **Camera interval = 1**、**Frame limit = 0**，处理完整视频。
4. 完成后在队列点击 **Import result**，再开启 **Show raw reconstruction**，检查真实的 FlowHMR 输出与原视频。
5. 用 **Add Mixamo character** 添加 FBX / GLB / glTF 角色。单角色使用
   **Apply motion to character → Plant feet on the floor**；多角色使用
   **Animate and ground all characters**。原始 Action 保留，使用 **Original / Grounded** 比较。
6. 点击 **Prepare & play at original speed**。显示缓存准备另计；播放结果时收起侧栏，
   让全部角色进入视口。显示缓存不改变骨骼动作或导出结果。
7. 选择目标角色及 Action，点击 **Export animated FBX**。

推理在独立队列服务中运行，同一队列串行执行，避免任务互抢显存。队列数据库默认位于 `~/.cache/motius/flowhmr/queue.sqlite3`，可在启动 Blender 前用 `MOTIUS_FLOWHMR_QUEUE` 指定；多个 Blender 实例使用同一数据库时共享队列。不同数据库是不同调度域，不会协调同卡任务，建议使用默认共享数据库。

**Pause after current** 暂停后续任务，当前任务继续；**Resume queue** 恢复或重连；等待中的任务可用 ↑/↓ 调整顺序，活动任务可取消，失败/取消/中断的任务可重试。History 和分页保留全部任务。每个 case 独立保存选择和参数，任务完成后选择 **Import result** 导入当前场景。

关闭 Blender、切换 `.blend` 或禁用插件不会停止后台任务。机器重启或运行器中断后，已完成结果会恢复，未完成任务标记 Interrupted，等待用户 Retry；运行器发现上一工作进程仍存活时先等待或按请求取消，不启动重复推理。进程身份使用创建时间校验，避免误杀 PID 被复用后的其他进程。

执行时重新检查空闲显存。两种模式都会按当前资源降低 SAM/VGGT 帧批量和查询/FFN 工作块；不会抽帧、减少双手特征、更改权重、dtype、种子或采样步数。CUDA OOM 最多自动重试4次，每次新开进程并缩小工作块。标准模式也会自动降低工作批量重试；始终保留官方 SAM，不自动切换为 Fast SAM。仍失败时保留错误。

```text
<Job output>/<timestamp>-<id>/
  submission.json             # 提交参数快照
  selected-subjects.json       # 选中轨迹快照（视频身份继续校验）
  attempt-001/
    request.json              # 该次尝试的工作批量和完整参数
    preflight.json            # GPU/可用显存/视频规模/调度决策
    status.json               # 当前进度、结果路径或结构化错误
    job.log                   # 完整日志
    result.npz                # 原始 capture
    motion.blender.npz        # 52关节交换格式
    surface.blender.npz       # 可选预测网格
  attempt-002/                # OOM或手动重试；保留上次输出
```

导入、骨架创建和 Action 烘焙仍由 Blender 主线程执行；这是用户点击 Import result 后的明确操作。推理计时不包含该操作。

## 接地与足锁

**Plant feet on the floor** 以目标蒙皮网格的实际脚底高度判断支撑，锁定支撑脚的
水平位置，使用双腿两段 IK 修正骨盆与腿部旋转，并处理摆动脚穿地。
它保留目标骨长、手部姿态和根部水平轨迹；腾空段保留跳跃高度，避免把每帧都压到地面。
**Floor height** 使用 Blender 世界坐标米制 Z 值。结果烘焙为独立 Action，
原始重定向 Action 保留，可随时切换 Raw / Grounded。

接地需要具有蒙皮网格的可编辑标准 Mixamo 角色。接触检测是运动启发式，
自遮挡、特殊脚型和持续滑行动作应检查支撑判定；显示的误差针对检测到的脚底点，
不能替代任意角色或动作的视觉验收。导出时使用当前选择的 Action。

## 复用经过校验的预处理

开启 **Reuse video preprocessing** 并选择含 `frontend-cache.json` 的目录，
会逐一校验原视频、跟踪/相机/全手部特征文件及前端权重的 SHA-256、模式和相机间隔。
校验成功后仍执行新的 FlowHMR 动作采样与网格重建，不读取旧的 `result.npz` 充当新结果。
缓存仅适用于同一视频和相同前端配置；默认关闭。完整首次运行仍执行所有前端。

## 已有推理结果

在 Options 选择已有 FlowHMR `result.npz`，点击
**Prepare Existing Capture**。这一步使用外部 Python 和相同 neutral 模型，
重建全部 52 关节的 FK 与旋转，再导入 Blender。纯转换可把 Device 改为 `cpu`。
**Import Motion** 直接读取的是 `motion.blender.npz`，不是原始 `result.npz`。

## 坐标、重定向与参数

- FlowHMR 的米制 Y-up 坐标转换为 Blender Z-up、朝向 -Y，时间轴使用 30 fps。
  原生导入保留身体、手指旋转与 FK 关节位置，不从 22 个关节点猜测手指或头部朝向。
- Mixamo 使用目标骨架的 rest basis 转换姿态，保留角色原骨长。根位移默认按腿长
  比例缩放并从角色 rest 髋部位置开始；可关闭缩放。关闭 Root motion 可原地播放，
  仍保留上下移动。缺少手指/脚趾/锁骨时保留可映射部分。
- 标准 Mixamo FBX 导入后的 90° 对象旋转与 0.01 单位缩放会正确保留；
  世界坐标的姿态与位移会先转换到 armature 空间，不必手动应用统一缩放与旋转。
- 重定向要求标准 T-pose 的可编辑 Mixamo 骨架。Rigify、IK 控制骨架、镜像/非均匀
  缩放和仍激活的约束/NLA 不在直接支持范围；请先处理。不同身材可能需要后续足部
  IK 或接触修正。插件支持接地和足锁，不生成 Mixamo 角色网格或蒙皮。
- 导入会创建新 Action，原 Action 保留在数据块中；人工导入支持 Blender Undo。
  可在 Action Editor 中切换；任务完成时不自动导入。
- **Camera interval 默认 1**。大于 1 可降低 VGGT 显存，但会改变相机预测，面板会提示。
  加速注意力无法消除长视频 VGGT 的显存增长。
- **Frame limit** 可先测试短片段；**Seed / Motion post-processing** 在比较两种模式时
  应保持一致。后处理可能放大微小数值差异。

此前 Motius 的 H200 单次测量：30 帧完整视频的峰值分配显存从 9.70 降到
7.23 GiB，最大旋转差 0.0265°、根位移差 0.122 mm；3,789 帧**采样阶段**
从 11.68 降到 2.81 GiB、8.73 降到 4.76 秒。长采样重复使用 30 帧的已测特征，
不包含实时 SAM/VGGT，不能视为完整长视频显存保证或所有硬件的性能承诺。

## 来源与许可

参考 [Motion2MixamoRig](https://github.com/MaxLiu22/Motion2MixamoRig) 的外部进程与
角色导入工作流，以及 [FlowHMR issue #4](https://github.com/flowhmr/flowhmr/issues/4)
的工程优化讨论；实现复用 Motius 的原生 FlowHMR 推理管线，不打包参考仓库源码。
插件代码 GPL-3.0-or-later；模型、前端、SMPL-H、角色各自遵循原始许可。
FlowHMR 的教育、研究、非营利使用限制依然适用，见 `NOTICE.md`。

### Live inference progress (0.2.6)

Tracking, camera encoding/layers, body/hand feature extraction and motion sampling
report their own completed counts and totals. The panel refreshes about twice a
second; measured stage speed and approximate remaining time appear after enough
samples are available. Loading phases show elapsed time without a fake percentage.
Accelerated mode batches eight independent frames by default; Options → Frames
per feature batch can select 1 to restore the previous frame-by-frame schedule.
All camera frames and both hands are retained. Batch sizes can change floating
point outputs; this is not a bitwise-equivalence claim. Requires Motius 0.3.10.

### Camera execution (0.2.6)

Accelerated mode measures the installed default/cuDNN attention implementations
on a real long-sequence input and selects cuDNN only when materially faster.
Unsupported kernels fall back before updating the residual. All camera frames
and global keys remain visible. Results can differ by floating-point rounding;
selected backend and fallback details are recorded in the result metadata.
