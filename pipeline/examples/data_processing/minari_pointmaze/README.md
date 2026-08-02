# Minari PointMaze：Robot 轨迹数据校验

这个学习示例直接读取 `model_data/robot/D4RL/pointmaze/umaze-v2`，执行质量校验并输出
统计结果；它不修改原始 HDF5，也不产生新的 DatasetVersion。

```bash
uv run python -m pipeline.examples.data_processing.minari_pointmaze.step_01_validate_file_format
uv run python -m pipeline.examples.data_processing.minari_pointmaze.step_02_validate_data
```

## 数据组成

| 对象 | 作用 | 核心内容 |
| --- | --- | --- |
| `metadata.json` | 数据集级描述 | dataset id、Minari 版本、episode/step 总数、Gymnasium observation/action space |
| `main_data.hdf5` 根属性 | 可执行契约 | 总量、space schema、生成器与版本 |
| `episode_N` | 一条完整轨迹 | observations、actions、rewards、terminations、truncations、infos |
| transition `t` | 一次环境转移 | `observation[t] + action[t] -> reward[t] + observation[t+1]` |

因此，长度关系不是所有数组都相等：action/reward/终止标记是 `T`，observation 和
info 是 `T + 1`。最后多出的状态是执行最后一个 action 后得到的 next observation。

## Step 01：文件级校验

| 层次 | 校验内容 | 目的 |
| --- | --- | --- |
| 文件身份 | 路径、普通文件、大小、SHA-256 | 固定本次扫描的输入，支持复现与传输后比对 |
| 实际格式 | HDF5 magic bytes；metadata UTF-8/JSON | 不相信文件后缀，尽早发现错文件 |
| HDF5 完整性 | 遍历并读取每个 dataset、禁止 soft/external link | 发现截断、损坏或隐式外部依赖 |
| 物理画像 | dataset 数、shape、dtype、chunk、compression、Fletcher32 | 了解实际布局；未启用 checksum 只记录，不直接判错 |
| 元数据一致性 | JSON 与 HDF5 的总量、space、版本和 dataset id 对账 | 防止“文件可读，但描述的是另一份数据” |

`SHA-256` 应作为 quality result 保存下来。它是整个原始文件的内容身份，不是样本质量分；
以后重新下载、搬运或入湖时再次计算，哈希相同才说明字节级内容未变化。

## Step 02：数据级校验

### Minari 通用硬校验

| 对象 | 校验内容 |
| --- | --- |
| episode | 命名/id 唯一，必需字段齐全，非空，root/episode 总量一致 |
| 时间轴 | action/reward/flags 为 `T`；observation/info 为 `T+1` |
| space | Dict key、Box shape/dtype/bounds 与声明的 Gymnasium space 一致 |
| 数值 | observation、action、info、reward 无 NaN/Inf |
| 边界 | 非末步不出现 boundary；末步 terminated/truncated 恰好一个为真 |
| 汇总属性 | total_steps 必须一致；可选 reward 缓存统计存在时必须可重算一致 |

### PointMaze 语义硬校验

| 关系 | 约束 |
| --- | --- |
| 位置 | `achieved_goal == observation[..., :2]` |
| 目标 | `desired_goal == infos/goal` |
| 稀疏奖励 | reward 只能是 0/1 |
| 成功对齐 | `reward[t] == infos/success[t + 1]`，不是 `success[t]` |

### 画像与 Warning

episode 长度、return、成功率、action 范围属于数据画像；完全重复轨迹记为 Warning。
reward min/max/mean/std/sum 全部缺失时只记录覆盖率，不拒绝有效轨迹；只缺一部分时记 Warning。
这些信息帮助发现分布异常，但不应使用任意阈值把本来有效的训练样本直接判坏。

## 为什么分两步

文件级回答“包是否完整、可读、与描述一致”；数据级回答“轨迹能否按训练语义正确解释”。
生产系统可把两步放在同一个 executor 中，但结果应分别保留，便于定位下载损坏与数据语义错误。
