# WikiText：从原始 JSONL 到 Lance

这个目录只展示数据实际发生了什么，不使用 Pipeline、Operator 或元数据
抽象。输入是已经下载的 `model_data/text/wikitext_2_raw`，原始文件不会被
修改；输出写到 Git 忽略的 `model_data/learning/`。

按顺序运行：

```bash
uv run python -m pipeline.examples.data_processing.wikitext.step_01_inspect_raw
uv run python -m pipeline.examples.data_processing.wikitext.step_02_validate
uv run python -m pipeline.examples.data_processing.wikitext.step_03_clean_and_standardize
uv run python -m pipeline.examples.data_processing.wikitext.step_04_write_lance
uv run python -m pipeline.examples.data_processing.wikitext.step_05_read_lance
```

## 每一步观察什么

1. `step_01` 同时展示磁盘上的 JSONL 字节和 `json.loads` 后的 Python 值。
2. `step_02` 扫描三个 split 的每一行。语法错误、错误类型、缺少 `text`、
   空文本和非法控制字符都会给出明确原因；本示例把 WikiText 的空分隔行
   视为非训练样本并过滤。
3. `step_03` 执行确定性的 Unicode NFC、首尾空白和 WikiText `@-@`、
   `@,@`、`@.@` 标记处理，然后显示固定的 Arrow Schema 和首批数据。
4. `step_04` 每 1,024 行生成一个 `RecordBatch`，流式写入 Lance，不把全部
   数据加载到内存。目标已存在时默认重建；显式传 `--no-overwrite`
   才会保留并回读已有输出。
5. `step_05` 从 Lance 回读 Schema、总行数和样本，确认写入结果可用。

最终 Schema：

| 字段 | Arrow 类型 | 含义 |
| --- | --- | --- |
| `text` | string | 清洗后的训练文本 |
| `source_split` | string | 原始 train / validation / test |
| `source_line` | int64 | 原始 JSONL 行号，可回溯问题 |
| `is_heading` | bool | 是否为 WikiText 标题行 |
| `character_count` | int32 | 清洗后字符数 |

这是学习示例，不代表所有文本数据都应该采用同一规则。例如空文本在
WikiText 中可表示段落边界；这里过滤它，是为了清楚展示“业务规则决定
保留还是拒绝”，而不是声称空行永远无效。
