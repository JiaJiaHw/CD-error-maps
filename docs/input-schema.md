# 输入 schema（v1）

## 根路径和配置

配置是 UTF-8 JSON，`schema_version` 为 `1`。配置文件路径、配置内相对路径、CLI 的相对输出路径均相对于项目根解析，绝对根路径可用于覆盖。清单内的图片路径必须是可移植的 POSIX 相对路径，使用 `/`；禁止绝对路径、反斜杠、`.`/`..` 段、路径逃逸和大小写造成的碰撞。不会移动、修改或重采样输入。

| 字段 | 含义 |
|---|---|
| `samples_root` | GT 图片根；固定导出也包含 T1/T2 |
| `selection_file` | 固定导出的原 `selection.json`；通用 CSV 模式设为 `null` |
| `samples_manifest` | 通用 CSV 路径；固定 selection 模式下不额外选择通用 CSV |
| `prediction_root` | Prediction 图片根 |
| `prediction_manifest` | 显式 Prediction 映射 CSV，或 `null` |
| `prediction_path_template` | 同名路径模板；和显式映射二选一 |
| `output_root` | 省略 `--out` 时新运行目录的父目录 |
| `datasets`、`models` | 默认所选范围，二者取笛卡尔积 |
| `model_aliases` | 可选模型别名说明元数据，不自动改写 CLI 名称或文件路径 |
| `combinations` | 可选明确范围，形如 `[{"dataset":"WHU","model":"CDMamba"}]` |
| `gt_encoding`、`prediction_encoding` | 分别是 `binary_01` 或 `binary_0255` |
| `encodings` | 可选按 dataset/model 覆盖编码，见下例 |
| `gt_value_maps` | 可选按 dataset 显式声明 GT 背景/变化原像素值，仅作用于 GT |
| `ignore` | `null`，或 `{"gt_value":128}` 等显式第三值 |
| `palette` | TN/TP/FP/FN/IGNORE 的不同 RGB 三元组，每通道 0～255 |
| `expected_selection_sha256` | 可选原 selection 字节 SHA-256 |
| `expected_manifest_sha256` | 可选通用样本 CSV 原始字节 SHA-256 |
| `expected_prediction_manifest_sha256` | 可选 Prediction 映射 CSV 原始字节 SHA-256 |
| `expected_counts` | 可选每数据集样本数约束，如 `{"MY-DATASET":2}` |
| `expected_size` | 可选共同尺寸约束 `[width,height]` |

本次实验已完成随机冻结包导入和真实全量误差图验收，数量、尺寸、冻结哈希和已确认的 GT 值映射保存在被忽略的 `configs/local.fixed100.json`。公开配置示例不固定这些个人实验约束；自己的数据应填写自己的验收值，不沿用其他集合的哈希。

```json
{
  "encodings": {
    "MY-DATASET": {
      "MY-MODEL": {
        "gt_encoding": "binary_0255",
        "prediction_encoding": "binary_01"
      }
    }
  }
}
```

## 固定 selection 适配

读取原 `selection.json` 的 `datasets[dataset].selection`，保留 rank、sample ID 和文件元数据。GT 路径使用 `files.GT.target`，相对于 `samples_root`；`files.GT.source` 仅作为来源记录。原 JSON 和各 `samples.csv` 不重写。

固定导出必须带 `selection.sha256`、`EXPORT_COMPLETE.json`、各数据集 `samples.csv` 和 T1/T2/GT。导入验收检查原始字节哈希、CSV 与 selection 一致性、文件集合、完整解码、尺寸以及对应哈希。`selection.json` 中旧阶段文字不改变冻结集合；不能据此重新选择样本。

可选的 `datasets[dataset].test_txt_sha256` 是严格的 64 位十六进制 SHA-256 字符串。字段存在时，包内必须有 `<dataset>/test.txt`；UTF-8 文件每行一个完整原始 sample ID，ID 与顺序须精确等于该数据集的 `selection`。工具首先校验文件原字节哈希，再核对 ID/顺序，不去扩展名、改大小写、去前导零、排序或跳过空行。缺失文件、非法摘要、哈希不符或重新封印后仍错序都报错。

通过检查的 `test.txt` 加入 `manifest_hashes`，因此 `validate`、`generate` 和 `verify` 会检查列表在读取或生成期间是否变化。旧 selection 没有这个可选字段仍兼容。该列表用于上游模型测试列表适配；随机策略与种子仅由上游第一次选样决定，本工具只消费冻结清单。

`validate --samples-only` 检查样本清单（包含显式封印的 `test.txt`）与 GT，可在 Prediction 尚未准备时执行；它不重复导入阶段对 T1/T2 的完整验收，也不声称验收 Prediction。

## 通用样本 CSV

必须有 `dataset,sample_id,gt_relative_path`。可选 `rank,gt_sha256,width,height`。UTF-8（包括 BOM）输入可读；普通样本 CSV 的其他字段不参与首版配对或验收，来源信息可放在显式 Prediction 映射中。

```csv
dataset,sample_id,gt_relative_path,rank,gt_sha256,width,height
MY-DATASET,0001.png,MY-DATASET/GT/0001.png,1,,,
MY-DATASET,Subset/Case_02.png,MY-DATASET/GT/Subset/Case_02.png,2,,,
```

每个 `(dataset,sample_id)` 唯一。ID 作为完整字符串，不按数字转换、不去扩展名、不扁平化。处理顺序沿用清单；rank 作为记录，不用于猜测配对。可选哈希一旦提供便必须一致，宽高一旦提供便必须与解码结果一致。

## Prediction 路径模板

默认 `{dataset}/{model}/{sample_id}`，相对于 `prediction_root`。只允许已明确声明的模板变量；不会猜配 stem、编号或排序。如果清单 ID 是 `Subset/Case_02.png`，预测目录必须保留相同相对子目录和文件名。

同名模式按清单找全量文件，并核对所选模型组目录中的额外图片。无关 metadata 可保留；额外预测图片必须查明来源，不能默默纳入或忽略。

## 显式 Prediction CSV

名称不同或者布局不同，设置 `prediction_manifest`，同时将 `prediction_path_template` 设为 `null`。

```csv
dataset,model,sample_id,prediction_relative_path,prediction_sha256
MY-DATASET,MY-MODEL,0001.png,MY-DATASET/MY-MODEL/0001_pred.png,
MY-DATASET,MY-MODEL,Subset/Case_02.png,MY-DATASET/MY-MODEL/Subset/Case_02_pred.png,
```

必填四列为 `dataset,model,sample_id,prediction_relative_path`；可选 `prediction_sha256` 和来源字段，如 `acquisition_method`、`checkpoint_sha256`、`config_sha256`、`provenance_reference`。

配对键始终为 `(dataset,model,sample_id)`。所选组合必须精确覆盖对应样本 ID 集合：缺失、重复、未知 ID、一对多、多对一、输入输出路径冲突都会报错。未选择的组合不被声称已验收。

## 图片与 ignore

图片必须是可完整解码的 PNG，模式为 `L`（uint8 单通道）。`binary_01` 只允许 0/1；`binary_0255` 只允许 0/255。单值背景或单值前景合法。GT 和 Prediction 尺寸相同；如有清单/配置尺寸声明，也要满足声明。

同源 GT 若已证实用多个原像素值表达同一类，可在个人配置显式声明：

```json
{
  "gt_value_maps": {
    "LEVIR-CD": {"background": [0], "change": [254, 255]}
  }
}
```

该规则只覆盖所列数据集的 GT 类值解释；没有规则的数据集继续严格使用原 `gt_encoding`。每个规则只能有 `background`、`change` 两个非空列表，元素须为不重复的 0～255 整数（不接受布尔值），两类不可重叠。未声明的像素值仍拒绝；这不是阈值转换，也不会改写 GT 文件、selection、CSV 或任一 Prediction。映射本身不能声明 ignore；独立 `ignore` 保持既有规则，并且不得与映射背景/变化值冲突。Prediction 始终按独立二值编码检查，不接受 GT 映射，即使像素位于 GT ignore 区域。

非空的 `gt_value_maps` 原声明进入验证报告，映射涉及的样本/配对 JSON 记录 `gt_value_map`。生成运行还将规则保存于 `effective_config.json`；配置和验证报告纳入 artifact 哈希及完成封印。`verify` 重读规则、重新解码输入并核对原输入快照，映射篡改或生成期间改变规则均导致失败。没有映射字段的既有二值配置和运行保持兼容。公开模板不默认启用映射，个人实验必须记录它的来源证据。

默认没有 ignore。仅 GT 可声明不等于自身背景/前景值的第三值，范围 0～255。该值的像素在输出为 IGNORE=255，不参与统计指标。Prediction 即使位于 GT ignore 区域也必须是合法二值，不接受概率、脏值或另一个 ignore 值。

P/RGB/RGBA、JPEG、网格图和概率图均须在上游按已知语义规范导出；本工具不自动转换。路径、编码、尺寸和哈希错误的退出码为非零；使用错误不能留下完成标记。
