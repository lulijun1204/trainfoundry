# UCF101：视频文件与官方划分校验

这个示例读取 `model_data/multimodal/video/ucf101` 中的 RAR 视频包和 ZIP split 包，
不修改原始数据。

```bash
# 默认：ffprobe 每条视频，并用 ffmpeg 完整解码每一帧
uv run python -m pipeline.examples.data_processing.ucf101.step_01_validate_file_format

# 快速预检：只探测容器和流元数据
uv run python -m pipeline.examples.data_processing.ucf101.step_01_validate_file_format --probe-only

uv run python -m pipeline.examples.data_processing.ucf101.step_02_validate_data
```

## Step 01：文件级校验

| 层次 | 常规校验 |
| --- | --- |
| RAR/ZIP | magic bytes、成员路径安全、重复成员、ZIP CRC、完整解压 |
| 文件审计 | SHA-256、归档数量、视频数量（官方应为 13,320） |
| 视频探测 | ffprobe 可识别 AVI、存在视频流、codec、宽高、时长、帧率为正 |
| 完整解码 | ffmpeg 将视频流逐帧解码到 null sink，发现中后段损坏 |

`ffprobe` 很快，但主要读取容器和流信息；它成功不代表所有帧都可解码。因此默认执行
完整解码，`--probe-only` 仅适合作为快速预检。

## Step 02：数据级校验

| 对象 | 常规校验 |
| --- | --- |
| 类别 | `classInd.txt` 为 1..101、ID/名称唯一 |
| 三组 split | 训练行的类别 ID 正确、引用的视频存在、每折覆盖完整数据集 |
| 精确泄漏 | 同一视频不能同时进入一折的 train/test |
| group 泄漏 | `v_Class_gXX_cXX.avi` 的同一 `gXX` 不能跨 train/test |
| 内容泄漏 | 默认计算视频 SHA-256，检测不同文件名但字节相同的跨集合副本 |

类别以目录名和 `classInd.txt` 为准。官方数据存在 `HandstandPushups` 与文件名前缀
`HandStandPushups` 这类大小写差异，所以文件名只用于解析 `gXX`，不承担类别真值。

UCF101 官方说明同一 group 的视频具有相似背景和视角，因此按单条视频随机切分会造成
数据泄漏；应使用官方三组 train/test split。

## 有效性与质量不要混在一起

本例不以分辨率、时长、运动量、镜头切换、审美分或内容安全分拒绝视频。这些属于下游
任务相关的质量筛选。视频可完整解码是硬有效性；“时长是否适合训练”是策略配置。
