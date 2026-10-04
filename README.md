# CD-error-maps

用于遥感变化检测对比实验的清单驱动误差图工具。读取 GT、Prediction 和样本清单，生成 TP/TN/FP/FN 彩色 PNG、稳定类别索引 PNG、逐图统计及可回读验收的运行记录。

目前已完成 LEVIR-CD、SYSU-CD、WHU 各 100 个冻结 test 样本、每个数据集 10 个模型的本地真实实验：30 组、3000 对输入，生成并验收 3000 张 RGB 误差图和 3000 张类别索引图。真实输入、结果与来源证据保存在本地，不包含在本仓库中。固定子集统计用于诊断，不替代完整 test 的正式指标。

工具也支持简单 CSV 清单。上游仅在首次选样时随机，后续操作固定完整 sample ID；本工具消费已经冻结、验收的输入，不筛选样本、不运行模型、不调整 Prediction，也不制作最终论文排版图。合成示例仅用于独立测试。

## 快速开始

运行依赖为 Python、NumPy、Pillow 和标准库，无需 PyTorch、CUDA 或 GPU。项目支持 Python 3.9 及以上版本，提供专用 Conda 环境定义 [environment.yml](environment.yml)，与模型环境隔离。

首次使用时，在项目根执行：

```powershell
conda env create --prefix ./.conda --file environment.yml
conda activate ./.conda
python -m pip install --no-deps --no-build-isolation -e .
python run.py --help
```

已经存在本项目 `.conda` 环境时，直接激活，无需重新创建：

```powershell
conda activate ./.conda
python run.py --help
```

下文的 `python` 表示本项目环境的解释器。源码入口 `python run.py ...` 只需要运行依赖；editable 安装后，也可使用 `cd-error-maps ...` 或 `python -m cd_error_maps ...`。离线机器使用方法见 [离线说明](docs/offline-use.md)。

Windows 的 Python 3.9.23 / NumPy 1.26.4 / Pillow 10.4.0 已完成 120 项测试：118 项通过，2 项真实符号链接测试因系统权限跳过。Linux Python 3.9、3.12 的验证按对应提交的 [GitHub Actions](https://github.com/JiaJiaHw/CD-error-maps/actions) 实际结果记录；历史验证与本地真实验收范围见 [开发进度](DEVELOPMENT_PROGRESS.md)。macOS 尚未验证。

配置中的相对路径、配置文件的相对路径和命令行相对输出路径均按**项目根**解析，与终端当前工作目录无关。可从任意目录使用 `python <项目根>/run.py ...`。

## 目录与数据边界

```text
CD-error-maps/
  run.py
  pyproject.toml
  environment.yml                  # 可提交的专用环境定义
  .conda/                          # 本地环境，不提交
  src/cd_error_maps/                # 通用工具代码
  configs/
    fixed_samples.example.json     # 公开配置示例
    local.fixed100.json            # 本地实验配置，不提交
  data/
    fixed_samples/
      selection.json               # 冻结清单，不提交
      selection.sha256
      EXPORT_COMPLETE.json
      LEVIR-CD/{samples.csv,test.txt,T1/,T2/,GT/}
      SYSU-CD/{samples.csv,test.txt,T1/,T2/,GT/}
      WHU/{samples.csv,test.txt,T1/,T2/,GT/}
    predictions/<dataset>/<model>/ # 真实 Prediction，不提交
  outputs/                         # 独立运行目录，不提交
  examples/                        # 公开模板与独立合成生成器
  docs/
  tests/
```

Git 在 `data/`、`outputs/` 内只跟踪 `.gitkeep` 占位文件。真实图片、原始清单、来源元数据、Prediction、结果、个人配置、环境、缓存和临时文件均不提交。没有全局忽略 PNG；明确独立的合成示例可以提交。`examples/synthetic_workspace/` 是被忽略的本地演示工作区。

## 固定 100 样本使用流程

### 1. 准备配置和固定样本

本次数据根使用 `data/fixed_samples/`。`selection.json` 决定冻结 ID、顺序、对应路径和已记录的哈希；各数据集的 `samples.csv` 用于交叉核对；`test.txt` 每行一个完整原始 ID，供上游模型测试列表适配。保留原始文件字节、名称和完整相对路径，不编辑或重新排序这些清单。

首次配置可复制 [固定样本示例](configs/fixed_samples.example.json) 到被忽略的 `configs/local.fixed100.json`，再根据实际输入设置根路径、编码、组合范围，以及已验收的数量、尺寸和 selection 哈希。已有本地实验配置时直接复用。代码和公开配置不硬编码个人冻结 SHA、100 张或 256×256 约束。

本地 GT 按 `samples_root + selection 中 files.GT.target` 读取；清单中的服务器 `source` 绝对路径仅作为来源记录。T1/T2 用于对应关系检查与后续展示，误差分类直接读取 GT 和 Prediction。

同源 GT 若已经确认多个原像素值表示变化，可在个人配置中显式声明。例如本次 LEVIR-CD 使用：

```json
{
  "gt_value_maps": {
    "LEVIR-CD": {"background": [0], "change": [254, 255]}
  }
}
```

这个例子仅解释该数据集 GT 的已确认原值，保持原图片字节与冻结哈希；未知值仍报错。Prediction 始终使用自己的严格二值编码。不要仅为绕过非法编码错误启用映射，映射须有来源证据。约束和报告记录见 [输入 schema](docs/input-schema.md)。

selection 的 `datasets[dataset].test_txt_sha256` 存在时，工具检查 `<dataset>/test.txt` 的原字节哈希、ID 和顺序，并纳入生成及回读的输入快照。旧格式没有此字段仍可读取。上游选样只随机一次，后续导出、预测和误差图均消费同一冻结集合，不能混用其他样本版本。

先做不依赖 Prediction 的检查：

```powershell
python run.py validate --config configs/local.fixed100.json --samples-only
```

该命令验证样本清单、显式封印的 `test.txt` 和 GT，不读取 T1/T2，也不声称验证了 Prediction。导入时的全量验收另外检查 T1/T2/GT 共 900 张原图。

在另一份检出中导入固定包，可使用 `python run.py import-fixed --help` 查看参数。导入先校验压缩包、检查路径及文件类型，在独立暂存区验收后合并相同文件；只去掉唯一最外层目录，拒绝覆盖不一致的已有文件。保留源压缩包和校验文件。

### 2. 放入对应的 Prediction

以 `WHU × CDMamba` 为例，将冻结清单内的全部预测放入：

```text
data/predictions/WHU/CDMamba/<原始sample_id>
```

默认模板 `{dataset}/{model}/{sample_id}` 相对于 `prediction_root`。ID 为 `02833.png` 时预测也应为 `02833.png`；ID 为 `Subset/Case_02.png` 时保留子目录、大小写和扩展名。**不按排序、编号或 basename 猜配。**

目录和 CLI 使用稳定模型名：`BIT_CD`、`CDMamba`、`ChangeFine`、`ChangeFormer`、`ChangeViT`、`FC-Siam-diff`、`RSMamba`、`SNUnet-CD`、`BAN`、`ChangerEx-S50`。`BIT`/`BIT-CD` 对应 `BIT_CD`，`RSM-CD` 对应 `RSMamba`，`Siam-NestedUNet`/`SNUNet-CD`/`SNUnet` 对应 `SNUnet-CD`；别名仅是说明元数据，CLI 不自动转换。

名称不同、扩展名变化或布局不同时，用 [显式 Prediction CSV](examples/predictions.template.csv) 记录完整 `(dataset, model, sample_id)` 与 `prediction_relative_path`，配置 `prediction_manifest`，并将 `prediction_path_template` 设为 `null`。

若上游包多一层 `Prediction/`，可将 mask 放入默认模型目录，或明确将模板改为 `{dataset}/{model}/Prediction/{sample_id}`。保留上游来源元数据，目录层级必须在模板或映射中明确声明。

输入须为独立、无损、单通道 PNG mask；JPEG、拼图、batch 网格、带字可视化和概率图需在上游按已知语义规范导出。上游负责来源与空间对应验收，本工具继续检查完整配对、编码、尺寸和哈希。

### 3. 验证、预览、生成和回读

新输入或新配置使用以下完整流程；已经验收的运行可以直接保留和复查，无需为了重新执行示例覆盖结果：

```powershell
python run.py validate --config configs/local.fixed100.json --dataset WHU --model CDMamba
python run.py generate --config configs/local.fixed100.json --dataset WHU --model CDMamba --limit 3 --out outputs/WHU-CDMamba-preview-001
python run.py verify --run outputs/WHU-CDMamba-preview-001
python run.py generate --config configs/local.fixed100.json --dataset WHU --model CDMamba --out outputs/WHU-CDMamba-full-001
python run.py verify --run outputs/WHU-CDMamba-full-001
```

预览先检查所选组合的完整清单，再按清单顺序每组输出前 `--limit` 个样本，并在报告中标记为 preview。人工核对 FP/FN 方向与空间对应后，再生成全组。每次 `--out` 必须为新的目录；省略时在 `output_root` 下自动建立独立运行目录。

处理几个明确组合时，重复使用 `--pair`：

```powershell
python run.py validate --config configs/local.fixed100.json --pair WHU:CDMamba --pair SYSU-CD:BIT_CD
python run.py generate --config configs/local.fixed100.json --pair WHU:CDMamba --pair SYSU-CD:BIT_CD --out outputs/selected-pairs-001
python run.py verify --run outputs/selected-pairs-001
```

重复 `--dataset` 和 `--model` 表示两个列表的笛卡尔积；不能与 `--pair` 混用。未指定筛选时使用配置声明的范围。配对键为 `(dataset, model, sample_id)`，所选组合须精确覆盖完整清单，缺失、重复、额外图片或路径冲突都报错，不跳过缺失输入。

`validate` 默认只读并向终端输出 JSON。需要完整报告时，可追加 `--report outputs/validation-WHU-CDMamba-001.json`；目标必须是新文件，且不能位于输入数据根内。成功退出码为 0，验证或用法错误为 2，中断为 130。

## 编码、类别和色板

GT 与 Prediction 分别声明编码，不猜阈值、不自动转换：

| 编码 | 未变化 | 变化 |
|---|---:|---:|
| `binary_01` | 0 | 1 |
| `binary_0255` | 0 | 255 |

接受 PNG 单通道 `L` 模式，尺寸必须完全一致。全背景、全前景均合法。GT 显式值映射是上文所述的可选规则，Prediction 不使用该映射。

默认 `ignore: null`，**输入 `binary_0255` 的 255 表示变化，不能默认当 ignore**。可显式配置 GT 的第三值，例如 `"ignore": {"gt_value": 128}`；不能与背景、变化值或 GT 映射值冲突。Prediction 在全部像素上均须满足自己的二值编码。

| 类别 | 类别索引 PNG 数值 | 含义 | 默认 RGB |
|---|---:|---|---|
| TN | 0 | GT 未变化，Prediction 未变化 | `[0, 0, 0]` 黑 |
| TP | 1 | GT 变化，Prediction 变化 | `[255, 255, 255]` 白 |
| FP | 2 | GT 未变化，Prediction 变化，误报 | `[255, 0, 0]` 红 |
| FN | 3 | GT 变化，Prediction 未变化，漏报 | `[0, 255, 0]` 绿 |
| IGNORE | 255 | 已显式声明的 GT 忽略区域 | `[128, 128, 128]` 灰 |

类别索引固定，颜色可配置。类别索引 PNG 的 255 表示 IGNORE，与输入二值 PNG 的 255 语义不同。图例单独保存；误差图无文字、边框、缩放或插值。

## 输出与复查

每个独立运行目录包含：

```text
<run>/
  error_rgb/<dataset>/<model>/<output_id>
  error_class/<dataset>/<model>/<output_id>
  legend.png
  palette.json
  pairs.csv
  pixel_counts.csv
  summary.csv
  effective_config.json
  validation_report.json
  run_manifest.json
  COMPLETE.json
```

RGB 与类别索引均为无损 PNG。以 `.png` 结尾的原始 ID 保留完整名称与子目录；其他 ID 在完整 ID 后附加 `.png`，不丢失原扩展名。`pairs.csv` 保存输入输出映射与哈希，`pixel_counts.csv` 保存逐图像素计数，`summary.csv` 按数据集/模型汇总 micro precision、recall、F1、IoU 和 accuracy。零分母以 JSON `null` / CSV 空值表示。

所有图像满足 `TP + TN + FP + FN = valid`、`valid + ignored = width × height`。详细字段、指标和封印关系见 [输出 schema](docs/output-schema.md)。

`verify` 核对完成封印、产物与输入哈希，回读索引/RGB、重算逐图与汇总统计，并根据源 GT/Prediction 重新分类。验收 JSON 输出到终端，不修改原运行目录。只有生成及回读全部成功，才最后写入 `COMPLETE.json`；失败或中断不能视为完成。

**保留报告记录位置上的源输入及完整运行目录。** 删除或迁移输入会影响旧运行回读；不要修改封印报告来改变路径。后续变更使用新配置和新运行，维护与清理原则见 [维护说明](docs/maintenance.md)。

## 独立合成演示与测试

演示与真实数据完全隔离。生成器使用手写 2×2 数组，建立两个合成数据集、两个合成模型、8 对输入；`expected_counts.json` 是手算答案，覆盖四类方向、全背景/全前景和两种编码。

```powershell
python examples/make_synthetic_example.py
python run.py validate --config examples/synthetic_workspace/config.json
python run.py validate --config examples/synthetic_workspace/config.explicit.json
python run.py generate --config examples/synthetic_workspace/config.json --out examples/synthetic_workspace/outputs/demo-001
python run.py verify --run examples/synthetic_workspace/outputs/demo-001
python -m unittest discover -s tests -v
```

生成器拒绝覆盖已有 `examples/synthetic_workspace/`；已有示例可直接复用，生成新运行时使用未占用目录。测试另覆盖 ignore、非法编码与模式、尺寸、缺失/重复 ID、路径冲突、哈希、冻结测试列表、GT 映射、输出回读、统计和失败状态。CI 只使用合成数据，不下载真实样本、模型或权重。

## 使用自己的 GT 和 Prediction

1. 将 GT 放到自己的 `samples_root`，建立 [样本 CSV](examples/samples.template.csv)，必填 `dataset,sample_id,gt_relative_path`。
2. 将 Prediction 放到 `prediction_root`；同名使用明确路径模板，不同名使用 [显式映射 CSV](examples/predictions.template.csv)。
3. 复制 [通用配置](examples/config.generic.json) 到被忽略的 `configs/local.custom.json`，修改根路径、数据集、模型和编码。
4. 用自己的完整清单替换模板示例行，然后执行：

```powershell
python run.py validate --config configs/local.custom.json
python run.py generate --config configs/local.custom.json --out outputs/custom-001
python run.py verify --run outputs/custom-001
```

CSV ID 按字符串读取，保留大小写、前导零、扩展名和子目录。清单中的相对路径使用 `/`，禁止绝对路径、`..`、非可移植路径和越出数据根的链接。配置与字段细节见 [输入 schema](docs/input-schema.md)。

## 常见问题

| 错误或现象 | 处理 |
|---|---|
| `Prediction not ready`、缺失预测 | 准备所选组的完整预测，核对模型名与 ID；也可先用 `--samples-only` 检查 GT |
| 重复/未知 ID、额外图片 | 查明上游导出或显式映射，确保精确覆盖冻结清单，不删改冻结清单 |
| 非法编码、P/RGB/RGBA 模式 | 按上游已确认的语义导出单通道 PNG；GT 多值规则须有来源证据，不用阈值或自动灰度转换绕过错误 |
| GT/Prediction 尺寸不一致 | 返回上游检查保存与对应关系，不在本工具自动 resize |
| 哈希不符 | 核对来源与传输，不修改期望哈希绕过错误 |
| 路径碰撞或越界 | 使用唯一完整 ID、明确相对路径和独立输出目录 |
| 输出目录已存在 | 使用新的运行目录，保留旧报告供复查 |
| `verify` 失败、缺少 COMPLETE | 查看报告、核查完整产物与源输入；失败或中断的目录不能视为完成 |

版本功能记录见 [CHANGELOG](CHANGELOG.md)。日后修改代码或扩展输入时，保留已验收数据、配置与完整运行证据，依据改动范围执行必要测试，并使用新的运行目录。
