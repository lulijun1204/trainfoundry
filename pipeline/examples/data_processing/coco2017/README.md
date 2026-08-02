# COCO 2017：图片文件与标注校验

这个示例读取 `model_data/multimodal/image/coco2017` 中的原始 ZIP，不修改数据，
也不生成新的 DatasetVersion。

```bash
uv run python -m pipeline.examples.data_processing.coco2017.step_01_validate_file_format
uv run python -m pipeline.examples.data_processing.coco2017.step_02_validate_data
```

## Step 01：文件级校验

| 层次 | 常规校验 |
| --- | --- |
| ZIP 容器 | magic bytes、路径穿越、符号链接、加密成员、重复成员、异常压缩比、CRC |
| 审计信息 | 文件大小、SHA-256、成员数量 |
| 图片编码 | 后缀与实际 JPEG 格式一致、Pillow `verify()` |
| 完整解码 | 重新打开后 `load()` 每张图片，捕获截断和像素流损坏 |
| 图片属性 | 宽高为正、色彩模式、EXIF Orientation 合法 |
| JSON 文件 | UTF-8 和 JSON 语法可解析 |

`verify()` 只检查文件结构，因此随后必须重新打开并 `load()`；只读图片头部不能证明
所有像素都可解码。

## Step 02：数据级校验

| 对象 | 常规校验 |
| --- | --- |
| image | `id`/文件名唯一、文件存在、标注宽高与实际图片一致、无孤儿文件 |
| category | `id` 唯一、名称存在、annotation 引用有效 |
| bbox/area | 有限数值、面积非负、框尺寸为正且不越界 |
| segmentation | polygon 坐标成对且至少三点；RLE size 与图片一致 |
| keypoints | `(x, y, v)` 三元组、可见性、边界、`num_keypoints` 一致 |
| caption | 非空字符串、无异常控制字符 |
| 多任务标注 | instances、captions、person_keypoints 的 image 元数据一致 |

## 有效性与质量不要混在一起

本例只回答“文件能否可靠读取、样本和标注能否正确关联”。分辨率阈值、长宽比偏好、
模糊度、美学分、NSFW、文本图片相似度、近重复等依赖训练目标，应作为后续可配置的
质量筛选。有效样本不等于高质量样本。
