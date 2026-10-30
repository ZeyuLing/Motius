# 把视频里的表演带进 Blender

**FlowHMR，集成于 Motius。** 选择视频中的人物，检查身体与双手重建，将同一段表演应用到多个 Mixamo 角色，处理脚底接触，再继续编辑或导出动画。

主片约 **2 分 14 秒**：先看结果，再看实际操作；画面只添加操作步骤说明。

当前插件 0.2.14 的按钮名称和队列交互以本文为准；上方影片保留此前录制的操作界面。

支持 Blender 4.2+。本次发布流程在 Blender 4.5.3 LTS / Linux 中录制。支持具有可编辑标准 Mixamo 骨架和蒙皮网格的 FBX、GLB、glTF 角色；其他控制骨架需要先转换。

[观看产品发布片](../../assets/model_zoo/flowhmr/blender_demo/motius-flowhmr-product-release-1080p.mp4) · [安装与后端配置](../../blender_extension/motius_flowhmr/README.md#安装与环境) · [性能与数值对照](../architecture/validation/flowhmr-extreme-20261026.md)

![同一段表演，三个 Blender 角色](../../assets/model_zoo/flowhmr/blender_demo/product-release/cast-ready.png)

## 第一次使用前

安装统一插件包后，在 Blender 的 3D Viewport 按 **N → FlowHMR** 打开面板；Standard 和 Lightweight 可在推理前选择。

在面板选择已授权的 **Neutral SMPL-H (.npz)**，开启 Blender 在线访问并点击 **Install / update runtime**。插件会在外部进程安装 Python、CUDA 依赖和模型权重，并显示安装与下载进度。自动安装适配 Linux / Windows x86_64 / NVIDIA GPU；Windows 和消费级显卡尚待实机验证，首次下载的模型约 9.7 GB；不需要 Docker，也不修改 Blender 自带 Python。

如需自备环境，可在 **Options → Backend setup** 指定 **External Python**、**Motius checkout** 和权重缓存。选择本机可写的 **Job output**；使用相对目录前先保存 `.blend` 文件。

点击 **Check Environment**，确认成功后开始。保持默认 **Camera interval = 1**、**Frame limit = 0**，完整处理视频。默认关闭 **Reuse video preprocessing**。

## 1. 选择视频和表演者

在 **Your performance** 选择视频，点击 **Preview** 查看，再点击 **Detect people**。

在 **Task queue** 的检测任务中点击 **Choose person**，直接点击人物所在框的内部。绿色框表示已选中；框重叠时按 **Tab** 或再次点击切换候选。用方向键或滚轮检查其他帧，确认目标身份一致后按 **Enter**。

![点击人物，确认实际选中框](../../assets/model_zoo/flowhmr/blender_demo/product-release/person-selected.png)

背景空白点击不会自动选择最近人物。不要只看第一帧：交叉、遮挡或离开画面可能造成轨迹中断或身份错误。短缺口只在同一轨迹内插值；长缺口会被拒绝。必要时裁剪到目标连续可见的片段后重新检测。

## 2. 生成动作，并看清处理进度

在 **Capture the motion** 选择模式，点击 **Add capture to queue**。每次提交保存当前视频、人物轨迹和参数快照；可以继续选择下一个视频并提交检测或推理。

面板显示真实处理阶段、已完成计数、经过时间，以及有足够样本后的阶段速率和预计剩余时间。不同阶段的计数单位不同，不能把一个阶段的百分比理解为整个任务的百分比。可以取消任务；Blender 不需要关闭。

加速模式保留每帧的身体和两只手特征，以及默认逐帧的 VGGT 相机计算。它通过原生 GPU 预处理、合批、工作空间限制、模型分阶段驻留和无用计算消除节省资源；网络输出仍有浮点差异，并非逐位无损。

任务完成后不会自动导入。在 **Task queue → Completed and failed tasks** 找到该任务，点击 **Import result**，再检查重建。

队列支持 **Pause after current、Resume queue、↑ / ↓、Cancel、Retry**。暂停不终止正在运行的任务；关闭 Blender、加载其他 `.blend` 或禁用插件不会取消后台任务。只有明确点击取消才终止对应工作进程。重启 Blender 后自动重新连接队列；机器重启或运行器异常中断的任务会标为 Interrupted，使用 Retry 新开一次尝试。结果由用户选择导入到当前场景，不依赖提交时的场景。

执行前按当时空闲显存检查并选择 SAM/VGGT 工作批量；CUDA OOM 会释放失败进程，再用更小批量重跑。不会减少帧数、改变采样步数、权重或网络 dtype；标准模式不自动切换低显存模式。每次尝试的检查、日志和输出单独保存。

## 3. 先检查 FlowHMR 原始重建

完成后，默认显示实际预测的 SMPL-H 网格。保持 **Show raw reconstruction** 开启，先与源视频对照，再进入角色重定向。

![原视频与实际 FlowHMR 输出对照](../../assets/model_zoo/flowhmr/blender_demo/product-release/raw-reconstruction.png)

点击 **Prepare & play at original speed**，插件先准备源视频与视口的显示缓存，再按原帧率播放。缓存准备是独立的计算步骤；重复播放可复用有效缓存。缓存不会修改动作或导出数据。

显示性能不足时，插件会停止并提示，不会自动慢放或跳帧来冒充原速。修改视角、角色、姿态或范围后需要重新准备显示缓存。

## 4. 将表演应用到角色，并接地

点击 **Add Mixamo character**，导入 FBX、GLB 或 glTF。角色应具有可编辑的标准 Mixamo 骨架和蒙皮网格。

单个角色：选中目标，依次点击 **Apply motion to character → Plant feet on the floor**。

多个角色：添加角色后点击 **Animate and ground all characters**。每个角色按照自己的骨架和真实脚底求解，完成后并排排列。独立角色会共享场景更新；有不适合共享执行的依赖时保留串行路径。

完成后播放检查支撑、抬脚和腾空。**Original / Grounded** 切换原始重定向与接地 Action；原 Action 保留，接地结果可以继续编辑。接地是启发式接触求解，特殊脚型、滑步、遮挡引发的错误仍需要人工检查。

## 5. 播放、编辑和导出

播放结果时收起插件侧栏，让所有角色完整进入视口。需要原视频对照时保留源视频区域。

动画仍是 Blender 中可编辑的骨架与 Action。选择目标角色和需要导出的 Action，点击 **Export animated FBX**，保存到目标位置。显示缓存不会代替或烘进导出的骨骼动作。

## 速度与显存，按实际条件看

推理速度以处理完整视频的平均 **毫秒/帧**和**帧/秒**报告；数字包含模型加载，排除单独的人物检测、Blender 导入、重定向和操作时间。显存使用推理进程的驱动采样峰值。具体设备、视频、精度与两种模式的同条件结果见[性能对照](../flowhmr-blender/README.md#measured-performance)。

帧均值分摊了模型加载等固定成本；相机全局注意力的开销也随视频长度变化，因此不应直接外推为任意视频的固定每帧耗时。显存受帧数、批量、显卡和其他程序影响；当前没有消费级 GPU 实测。

## 遇到问题时

**没有选到想要的人物**：检查高亮框和多个时间位置；重叠时切换候选。轨迹身份不确定时重新检测或缩短片段。

**推理报显存不足**：关闭其他 GPU 应用，减小 **Frames per feature batch** 或先用短片段确认环境。增大 Camera interval 会改变预测，不能作为保持相同效果的默认解决办法。原视频、视口显示缓存和推理网络的内存也需要分别考虑。

**脚仍然滑动或穿地**：确认角色绑定、目标骨架和 Floor height 正确，比较 Original / Grounded，再检查接触判定。接地不是任意运动的物理解算器。

**完成后没有自动导入**：结果保存在任务目录。切换到 Object Mode，纠正目标后导入 `motion.blender.npz`；不需要丢掉已经生成的 capture。

**需要反馈错误**：保留任务目录中的 `request.json`、`status.json`、`job.log`、`inference_profile.json`；不要在公开问题中贴凭据或私有绝对路径。`result.npz` 是原始 capture，`motion.blender.npz` 是 Blender 交换格式。

[下载中文字幕](../../assets/model_zoo/flowhmr/blender_demo/motius-flowhmr-product-release.zh.srt)。主片中的开场与原始重建片段均通过了最终编码视频的双计数器校验；三角色的完整运动范围也已逐帧检查。

## 集成来源

- 原始方法与模型：[FlowHMR](https://github.com/flowhmr/flowhmr)。Motius 将其集成到统一网络与推理架构中。
- Blender 工作流参考：[Motion2MixamoRig](https://github.com/MaxLiu22/Motion2MixamoRig)。
- 工程优化讨论：[FlowHMR issue #4](https://github.com/flowhmr/flowhmr/issues/4)。

插件、模型、前端、SMPL-H 和角色分别遵循各自许可。使用前阅读[插件 NOTICE](../../blender_extension/motius_flowhmr/NOTICE.md)；集成不改变上游的使用范围，也不代表上游作者为本插件背书。
