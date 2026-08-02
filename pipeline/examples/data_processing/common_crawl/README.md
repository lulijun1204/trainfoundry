# Common Crawl：WET 文件与正文数据校验
这个目录展示 `model_data/text/common_crawl/*.warc.wet.gz` 实际如何被读取和
校验。程序直接流式读取 gzip，不需要提前解压，也不会修改原文件。

当前只实现两个步骤：

```bash
uv run python -m pipeline.examples.data_processing.common_crawl.step_01_validate_file_format
uv run python -m pipeline.examples.data_processing.common_crawl.step_02_validate_data
```

## Step 01：文件格式校验

| 校验 | 目的 |
| --- | --- |
| 文件后缀 `.warc.wet.gz` | 确认输入符合 CC WET 文件约定 |
| gzip magic bytes `1f 8b` | 防止只改后缀的伪 gzip 文件 |
| 完整读取 gzip | 触发 CRC 和截断校验，不需要落地解压文件 |
| WARC/1.0 或 WARC/1.1 | 确认容器版本 |
| Header 与空行边界 | 确认每条 WARC record 可以解析 |
| `Content-Length` 与实际 body | 防止记录错位或正文截断 |
| `WARC-Type` | 区分 `warcinfo`、`conversion` 等记录 |
| SHA-256、压缩大小、记录数 | 提供文件级审计结果 |

## Step 02：数据校验

`warcinfo` 是文件元信息，会统计但跳过；只有 `WARC-Type: conversion` 是候选
网页正文。

| 校验 | 错误码 |
| --- | --- |
| `WARC-Target-URI` 存在且为绝对 HTTP(S) URL | `MISSING_TARGET_URI` / `INVALID_TARGET_URI` |
| `WARC-Record-ID` 存在 | `MISSING_RECORD_ID` |
| `WARC-Date` 是带时区的 ISO-8601 时间 | `MISSING_WARC_DATE` / `INVALID_WARC_DATE` |
| `Content-Type` 为 `text/plain` | `INVALID_CONTENT_TYPE` |
| body 是严格 UTF-8 | `INVALID_UTF8` |
| 正文非空 | `EMPTY_TEXT` |
| 不包含换行、回车、制表符以外的控制字符 | `CONTROL_CHARACTER` |

输出中的关系为：

```text
total_records = valid_records + rejected_records + skipped_records
```

本阶段不执行语言识别、广告/成人内容识别、长度阈值、去重、清洗或 Lance
写入。这些属于后续质量过滤和标准化，不属于当前的“数据是否能够被可靠读取”
校验。
