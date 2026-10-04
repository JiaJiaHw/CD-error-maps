# CD-error-maps

用于遥感变化检测对比实验的清单驱动误差图工具。读取 GT、Prediction 和样本清单，生成 TP/TN/FP/FN 彩色 PNG、稳定类别索引 PNG、逐图统计及可回读验收的运行记录。

当前优先支持 LEVIR-CD、SYSU-CD、WHU 已冻结的各 100 个 test 样本，也支持简单 CSV 清单。本工具不筛选样本、不运行模型、不调整 Prediction，不制作最终论文排版图。

本工具读取从真实 test 随机选择一次并冻结的集合。旧按图片文件大小排序的集合已退役；后续生成始终复用验收后的完整 ID，不重新抽样。实际实验状态、输入和结果保存在本地验收记录中。

**工具开发与原图导入可以在没有真实 Prediction 时完成。真实 Prediction 尚未准备时，普通配对验证会明确报错；合成示例仅用于独立测试，不代表真实模型结果。**

## 环境与运行

运行依赖只有 Python、NumPy、Pillow 和标准库，无需 PyTorch、CUDA 或 GPU。项目提供专用 Conda 环境定义 `environment.yml`，不复用或修改模型环境。首版已在 Windows 的 Python 3.9.23 / NumPy 1.26.4 / Pillow 10.4.0 中实测；首版 [Linux CI](https://github.com/JiaJiaHw/CD-error-maps/actions/runs/36952784905) 的 Python 3.9、3.12 两组均已通过全部 91 项测试。新增冻结 `test.txt` 适配及其当前验证范围见 [开发进度](DEVELOPMENT_PROGRESS.md)，macOS 尚未验证。

安装 Conda 后，在项目根打开终端，直接创建并激活项目专用环境：

```powershell
conda env create --prefix ./.conda --file environment.yml
conda activate ./.conda
python -m pip install --no-deps --no-build-isolation -e .
python run.py --help
```

下文的 `python` 表示已激活的项目 Conda 解释器。首版以仓库源码和 editable 安装为运行方式；上述安装后也可用 `cd-error-maps ...` 或 `python -m cd_error_maps ...`。`python run.py ...` 本身只需要环境中的运行依赖。离线环境见 [离线使用](docs/offline-use.md)。

配置中的相对路径和命令行相对输出路径都按**项目根**解析，与终端当前工作目录无关。可从任意目录使用 `python <项目根>/run.py ...`；配置文件本身的相对路径也按项目根解析。

## 目录与数据边界

```text
CD-error-maps/
  run.py
  pyproject.toml
  environment.yml                   # 专用 Conda 环境定义，可提交
  .conda/                           # 项目本地环境，不提交
  src/cd_error_maps/
  configs/
    fixed_samples.example.json     # 可提交的通用固定样本配置示例
    local.fixed100.json            # 本次实验配置，仅本地保存
  data/
    fixed_samples/
      selection.json               # 原始冻结清单，仅本地保存
      selection.sha256
      EXPORT_COMPLETE.json
      LEVIR-CD/{samples.csv,test.txt,T1/,T2/,GT/}
      SYSU-CD/{samples.csv,test.txt,T1/,T2/,GT/}
      WHU/{samples.csv,test.txt,T1/,T2/,GT/}
    predictions/<dataset>/<model>/ # 真实单通道 Prediction，仅本地保存
  outputs/                         # 每次生成的新运行目录，仅本地保存
  examples/                        # 公开模板和独立合成生成器
  docs/
  tests/
```

Git 在 `data/`、`outputs/` 内只跟踪 `.gitkeep` 占位文件。真实图片、原始清单、来源元数据、预测、结果、个人配置、环境、缓存和临时文件不提交。没有全局忽略 PNG：公开的合成示例可在明确独立的位置提交。`examples/synthetic_workspace/` 是被忽略的本地演示工作区。

## 本次固定 100 样本怎么用

### 1. 检查固定样本

本次本地数据根是项目内的 `data/fixed_samples/`。`selection.json` 决定冻结 ID、顺序、对应路径和已记录的哈希；每个数据集的 `samples.csv` 用于交叉核对。不要编辑这些原始文件。

如果同源 GT 已确认多个原像素值表示变化类，可在个人配置显式增加 `"gt_value_maps": {"LEVIR-CD": {"background": [0], "change": [254, 255]}}`。工具只在读取该数据集 GT 时解释这些值，保持原图字节和冻结 SHA；未声明值继续报错，Prediction 仍严格使用自己的二值编码。规则进入输入报告、运行配置和 `verify` 封印检查。映射的来源及 ignore 冲突约束见 [输入 schema](docs/input-schema.md)。

新随机导出还为每个数据集提供 `test.txt`：每行一个原始 sample ID，顺序与 selection 完全一致，供服务器模型测试列表适配复用。selection 的 `datasets[dataset].test_txt_sha256` 声明其原字节哈希时，本工具检查文件存在、哈希及 ID/顺序，并将它纳入生成与回读的输入快照。上游仅在首次选样时随机；推理、提取预测、误差图生成均读取同一冻结清单，不能再次随机。旧格式没有这个字段仍可读取，但本次新实验须使用新随机导出的清单和对应 Prediction，不能混用退役集合。

本地 GT 按 `samples_root + selection 中 files.GT.target` 读取。清单中的服务器 `source` 绝对路径只保留为来源记录。T1/T2 供人工核对及后续展示；误差图分类直接使用 GT 和 Prediction。

在已导入数据并具备本地配置时，先做不依赖 Prediction 的检查：

```powershell
python run.py validate --config configs/local.fixed100.json --samples-only
```

该命令只验证样本清单（包括显式封印的 `test.txt`）和 GT，不读取 T1/T2，也不声称验证了任何 Prediction。导入时的全量验收另行检查 T1/T2/GT 共 900 张原图。固定约束（本次数量、尺寸、新冻结 SHA-256）在重新导入后写入本地配置；不要沿用退役集合的哈希。通用代码不硬编码这些约束。

如需在另一份本地检出中导入固定包，`import-fixed` 会先校验压缩包、检查路径和文件类型，在独立暂存区验收后合并相同文件；不会覆盖不一致的既有文件。用 `python run.py import-fixed --help` 查看参数。源压缩包和校验文件保留，导入只去掉唯一最外层目录。

### 2. 放入已经验收的真实 Prediction

以一个 `WHU × CDMamba` 组合为例，把固定清单内的所有 Prediction 放入：

```text
data/predictions/WHU/CDMamba/<原始sample_id>
```

默认模板是 `{dataset}/{model}/{sample_id}`，相对于 `prediction_root`。例如 ID 是 `02833.png`，预测也应是 `02833.png`；ID 是 `Subset/Case_02.png`，保留子目录、大小写和扩展名。**不能按排序、编号或 basename 猜配。**

模型目录使用项目配置中的稳定名称；模型别名仅是说明元数据，CLI 不会自动转换别名，目录和命令都按规范名称输入。文件名不同、扩展名变化或另有保存布局时，建立显式 CSV 映射，填写完整 `(dataset, model, sample_id)` 和 `prediction_relative_path`，设置 `prediction_manifest` 并将 `prediction_path_template` 设为 `null`。模板见 [Prediction 映射](examples/predictions.template.csv)。

当前候选目录名为 `BIT_CD`、`CDMamba`、`ChangeFine`、`ChangeFormer`、`ChangeViT`、`FC-Siam-diff`、`RSMamba`、`SNUnet-CD`、`BAN`、`ChangerEx-S50`。其中 `BIT`/`BIT-CD` 对应 `BIT_CD`，`RSM-CD` 对应 `RSMamba`，`Siam-NestedUNet`/`SNUNet-CD`/`SNUnet` 对应 `SNUnet-CD`。候选目录骨架不表示这些模型已经取得真实预测。

若上游包中还保留一层 `Prediction/`，可将其中的 mask 放到上述默认模型目录，或把模板明确改为 `{dataset}/{model}/Prediction/{sample_id}`。同时保留上游来源元数据；额外目录层必须在模板或映射中明确声明。

上游下载的是独立、无损、单通道 PNG mask，不是 JPEG、拼图、batch 网格、带字可视化或概率图。数量、来源和空间对应应由上游导出验收确认；本工具进一步检查清单、编码、尺寸和哈希。

### 3. validate → 预览 → 全组 generate → verify

```powershell
python run.py validate --config configs/local.fixed100.json --dataset WHU --model CDMamba
python run.py generate --config configs/local.fixed100.json --dataset WHU --model CDMamba --limit 3 --out outputs/WHU-CDMamba-preview-001
python run.py verify --run outputs/WHU-CDMamba-preview-001
python run.py generate --config configs/local.fixed100.json --dataset WHU --model CDMamba --out outputs/WHU-CDMamba-full-001
python run.py verify --run outputs/WHU-CDMamba-full-001
```

预览仍先检查所选组合的完整清单，再按清单顺序每组生成前 `--limit` 个样本。它不允许缺失配对，不改变冻结集合，运行报告标记为 preview。人工核对预览的 FP/FN 方向和空间对应后，再生成全组。每次 `--out` 必须是不存在的新目录；省略时由工具在 `output_root` 下建立独立运行目录。

一次处理几个明确组合，用重复的 `--pair`：

```powershell
python run.py validate --config configs/local.fixed100.json --pair WHU:CDMamba --pair SYSU-CD:BIT_CD
python run.py generate --config configs/local.fixed100.json --pair WHU:CDMamba --pair SYSU-CD:BIT_CD --out outputs/selected-pairs-001
python run.py verify --run outputs/selected-pairs-001
```

重复 `--dataset` 与 `--model` 表示两组列表的笛卡尔积。`--pair` 与这两个筛选参数不能混用。未指定筛选时使用配置声明的范围；只选择已经准备好的组合。当前空 Prediction 目录会报告 `Prediction not ready`，这表示输入尚未准备，不表示算法或工具开发失败。

`validate` 默认只读并把摘要 JSON 输出到终端；需要保留完整验证报告时可追加 `--report outputs/validation-WHU-CDMamba-001.json`，报告路径必须是新的文件且不能位于输入数据根内。验证/用法错误退出码为 2，中断退出码为 130，成功为 0。

## 编码、分类和色板

GT 与 Prediction 分别声明编码，不猜阈值、不自动转换：

| 编码 | 未变化 | 变化 |
|---|---:|---:|
| `binary_01` | 0 | 1 |
| `binary_0255` | 0 | 255 |

首版只接受 PNG 的单通道 `L` 模式，尺寸须完全一致。全背景和全前景都是合法输入。默认 `ignore: null`，**二值输入的 255 是变化类，不能默认当 ignore**。可显式配置 GT 的第三值，如 `"ignore": {"gt_value": 128}`；它不能等于该编码的背景或前景值。Prediction 在所有像素处都必须满足自身二值编码。

| 类别 | 类别索引 PNG 数值 | 含义 | 默认 RGB |
|---|---:|---|---|
| TN | 0 | GT 未变化，Prediction 未变化 | `[0, 0, 0]` |
| TP | 1 | GT 变化，Prediction 变化 | `[255, 255, 255]` |
| FP | 2 | GT 未变化，Prediction 变化，误报 | `[255, 0, 0]` |
| FN | 3 | GT 变化，Prediction 未变化，漏报 | `[0, 255, 0]` |
| IGNORE | 255 | 已显式声明的 GT 忽略区域 | `[128, 128, 128]` |

类别索引固定，颜色可在配置中修改。索引 PNG 的 255 表示 IGNORE，和输入 `binary_0255` 的 255 语义不同。图例单独保存；误差图无文字、边框、缩放或插值。

逐图和汇总均满足 `TP + TN + FP + FN = valid`，`valid + ignored = width × height`。汇总指标从组内累计像素计算 micro precision、recall、F1、IoU 和 accuracy；零分母用 JSON `null` / CSV 空值表示。这些固定子集统计用于诊断，不替代完整 test 的正式指标。

## 输出与复查

每个运行目录保存彩色图、索引图、独立图例、色板、配对清单、逐图/分组统计、有效配置、输入验证报告、运行报告、输入输出 SHA-256 和完成标记。详见 [输出 schema](docs/output-schema.md)。

`verify` 回读输出，核对文件哈希、类别/RGB 对应、逐图与汇总统计、配对信息，并重新检查源输入。验收结果以 JSON 输出到终端，不修改原运行目录。迁移或删除输入后，应先恢复报告中记录的数据路径再验收。只有全部生成及回读验收成功才写 `COMPLETE.json`；失败或中断的运行不能视为完成。

## 独立合成演示与测试

以下示例完全独立于真实固定样本和真实 Prediction 目录。生成器用手写的 2×2 数组建立两个合成数据集、两个合成模型、8 对输入，覆盖四类方向、全背景/全前景及两种编码；`expected_counts.json` 是手算答案。

```powershell
python examples/make_synthetic_example.py
python run.py validate --config examples/synthetic_workspace/config.json
python run.py validate --config examples/synthetic_workspace/config.explicit.json
python run.py generate --config examples/synthetic_workspace/config.json --out examples/synthetic_workspace/outputs/demo-001
python run.py verify --run examples/synthetic_workspace/outputs/demo-001
python -m unittest discover -s tests -v
```

生成器拒绝覆盖已有 `examples/synthetic_workspace/`，已有示例可直接复用。测试还检查 ignore、非法编码/模式、尺寸不一致、缺失/重复 ID、路径冲突、哈希不符、输入输出回读、统计一致性和失败状态。CI 只使用合成数据，不下载真实样本、模型或权重。

## 使用自己的 GT 与 Prediction

1. 将 GT 放到自己的 `samples_root`，建立 [样本 CSV](examples/samples.template.csv)：必填 `dataset,sample_id,gt_relative_path`。
2. 将 Prediction 放到 `prediction_root`。同名时使用明确模板；名称不同则使用 [显式映射 CSV](examples/predictions.template.csv)。
3. 复制 [通用配置](examples/config.generic.json) 到被忽略的 `configs/local.custom.json`，修改根路径、数据集、模型和编码。
4. 执行以下命令。模板中的示例行只是格式说明，应替换为自己的完整清单。

```powershell
python run.py validate --config configs/local.custom.json
python run.py generate --config configs/local.custom.json --out outputs/custom-001
python run.py verify --run outputs/custom-001
```

CSV ID 按字符串读取，保留大小写、前导零、扩展名和子目录。全部清单相对路径使用 `/`；禁止绝对路径、`..`、非可移植路径和越出数据根的链接。更多字段、按组合覆盖编码、显式 ignore 和配置含义见 [输入 schema](docs/input-schema.md)。

## 常见问题

| 错误或现象 | 处理 |
|---|---|
| `Prediction not ready`、缺失预测 | 下载所选组的完整预测集合，核对模型目录名和 ID；可先运行 `--samples-only` 检查 GT |
| 重复/未知 ID、额外图片 | 修正上游导出或显式映射，确保所选组精确覆盖清单，不删改冻结样本清单 |
| 非法编码、P/RGB/RGBA 模式 | 根据上游真实语义导出明确编码的单通道 PNG；不要依赖自动转灰度或阈值 |
| GT/Prediction 尺寸不一致 | 返回上游查明保存与对应关系，不在本工具自动 resize |
| 哈希不符 | 核对文件来源与传输；不要修改期望哈希来绕过错误 |
| 路径碰撞或越界 | 使用唯一 ID、显式相对路径和独立输出目录；检查大小写和子目录 |
| 输出目录已存在 | 使用新的运行目录；保留旧报告供检查 |
| `verify` 失败、缺少 COMPLETE | 查看运行报告，核查文件和来源；完整成功后再把运行视为完成 |

真实实验完成状态以实际配对验证、生成与 verify 记录为准；合成测试只验证工具行为。
