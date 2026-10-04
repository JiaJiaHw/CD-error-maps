# 输出 schema（v1）

## 新运行目录

每次 generate 写入一个不存在的新目录。省略 `--out` 时在 `output_root` 下生成独立运行目录；不提供静默覆盖或续写已有运行。相对路径按项目根解析。输出不能与任何输入发生路径冲突。

```text
<run>/
  error_rgb/<dataset>/<model>/<output_id>
  error_class/<dataset>/<model>/<output_id>
  pairs.csv
  pixel_counts.csv
  summary.csv
  palette.json
  legend.png
  effective_config.json
  validation_report.json
  run_manifest.json
  COMPLETE.json
```

源 sample ID 以 `.png` 结尾时保留完整名称及子目录；否则在**完整 ID 后**附加 `.png`，例如 `A.mask` 输出为 `A.mask.png`。不删除原扩展名，不使用 basename。输出路径碰撞（包括大小写）会报错。

## 图片、索引和颜色

`error_class` 是单通道 `L` PNG；TN=0、TP=1、FP=2、FN=3、IGNORE=255。该索引协议固定，不随色板变化。`error_rgb` 是对应 RGB PNG，逐像素使用 `palette.json` 色板，尺寸与源 GT/Prediction 完全一致。PNG 无损，图内不加图例、文字或边框。

默认颜色 TN 黑、TP 白、FP 红 `[255,0,0]`、FN 绿 `[0,255,0]`、IGNORE 灰 `[128,128,128]`。配置可覆盖颜色，但每类 RGB 必须合法且互不相同。`legend.png` 单独展示各类与颜色。

## 配对、统计和报告

| 文件 | 用途 |
|---|---|
| `pairs.csv` | `(dataset,model,sample_id)`、输入/输出对应路径、文件 SHA-256 及可用来源信息 |
| `pixel_counts.csv` | 每图 TN/TP/FP/FN、ignored、valid、width、height |
| `summary.csv` | 按 dataset/model 累计像素与 micro 指标，并记录处理数量 |
| `palette.json` | 类别、固定索引与 RGB 颜色记录 |
| `effective_config.json` | 解析后实际配置；本地绝对路径仅留在本地运行结果 |
| `validation_report.json` | 本轮所选完整组的输入预检查结果；预览也先全组检查 |
| `run_manifest.json` | 范围、preview/全组状态、配对与统计记录、环境/版本、输入与产物哈希和运行状态 |
| `COMPLETE.json` | 完成封印；只在生成及回读验收通过后最后写入 |

`run_manifest.json` 保存全部运行产物（除自身和 COMPLETE）的哈希；`COMPLETE.json` 进一步保存原始 `run_manifest.json` 字节哈希，形成完成封印。`verify` 只读，验收 JSON 输出到终端，不额外写入验证报告，不改变这些封印。

CSV 的固定字段如下；`provenance` 列是 JSON 对象字符串，源路径以实际本地绝对路径记录，输出图片路径相对于运行目录：

```text
pairs.csv:
dataset,model,sample_id,rank,gt_path,prediction_path,gt_sha256,prediction_sha256,
gt_encoding,prediction_encoding,error_rgb,error_class,error_rgb_sha256,error_class_sha256,provenance

pixel_counts.csv:
dataset,model,sample_id,TN,TP,FP,FN,ignored,valid,width,height

summary.csv:
dataset,model,images,TN,TP,FP,FN,ignored,valid,precision,recall,f1,iou,accuracy,undefined_metrics
```

`summary.csv` 的 `undefined_metrics` 列列出零分母指标名。`run_manifest.json` 中的主要字段为 `scope`、`preview`、`limit_per_combination`、`validated_pair_count`、`generated_pair_count`、`pairs`、`summary`、`artifact_hashes`、`versions` 和 `status`；`versions` 记录工具/Git/Python/NumPy/Pillow/平台及实际命令。逐图只输出像素计数，首版不提供逐图指标 CSV 或 macro 平均。

每个有效像素恰好属于四类之一：`TP+TN+FP+FN=valid`。`valid+ignored=width×height`。指标从计数计算，分组指标使用组内累计计数：

```text
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2TP / (2TP + FP + FN)
IoU       = TP / (TP + FP + FN)
accuracy  = (TP + TN) / valid
```

分母为零的指标以 JSON `null` 和 CSV 空值记录。忽略像素不进入分母。分组汇总是 micro 统计，不是逐图 macro 平均。固定子集汇总只说明该集合，不代表完整 test 指标。

`--limit N` 按每组清单順序输出前 N 个样本，报告标记 preview，保存预检查完整范围和实际输出范围；不得把预览产物当作完整组验收结果。

## 回读验收与失败

`python run.py verify --run <run>` 检查完成封印及报告结构，核对各产物/源输入的 SHA-256，回读 PNG，对照索引重算逐图计数、RGB 与色板、分组汇总及配对记录，并根据源 GT/Prediction 重新分类验证。

输入需要保持在报告记录的位置。文件发生变化、丢失、图像损坏、索引非法、RGB/统计/配对/哈希不一致或完成封印无效都会验收失败。哈希提供可复查的完整性记录；它不是数字签名。

运行开始后会记录进行中的状态。异常或中断时保留已生成的可检查文件，尽量记录失败原因；硬终止可能留下进行中状态。任何未写入且通过验证的 `COMPLETE.json` 的运行都不能视为完成。修复输入后使用新的运行目录，不覆盖旧结果。
