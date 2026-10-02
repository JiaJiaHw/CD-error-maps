# 离线与源码使用

工具运行不访问网络，不依赖模型框架、GPU、训练环境或在线服务。项目使用由 `environment.yml` 创建的专用 Conda 环境；真实数据和 Prediction 由用户独立准备。

## 创建项目专用 Conda 环境

在与运行平台一致的在线环境，直接为本项目创建 `.conda`，无需盘点或复用原模型环境。环境定义采用 Python 3.9、NumPy 1.26.4、Pillow 10.4.0，并包含 editable 安装所需的 pip/setuptools。

```powershell
conda env create --prefix ./.conda --file environment.yml
conda activate ./.conda
python -m pip install --no-deps --no-build-isolation -e .
python run.py --help
```

这些命令在项目根执行，Windows/Linux/macOS 使用各自平台的 Conda 安装。激活后以下 `python` 均属于本项目环境；源码入口不要求安装项目包。`environment.yml` 可以提交，`.conda/` 和环境内的实际文件不提交。CI 的版本矩阵是声明的自动验证任务，是否通过应查看实际运行记录。

## 为离线机器准备依赖

离线服务器不能直接联网执行 `conda env create`。应在匹配目标操作系统和 CPU 架构的在线环境创建本项目专用 Conda 环境，再按已验证的离线环境迁移流程转移、恢复和自检；Windows 环境不能直接复制到 Linux。

若目标已有本项目专用的 Python 3.9 Conda 环境、仅需补充运行依赖，也可在匹配目标 Python、操作系统和 CPU 架构的在线环境下载 wheel：

```text
python -m pip download --only-binary=:all: --dest wheelhouse numpy==1.26.4 Pillow==10.4.0
```

该例使用项目固定依赖版本。Windows wheel 不能直接用于 Linux；应在匹配平台下载，不修改模型训练环境。

将代码与对应 `wheelhouse/` 传到离线机器后，激活本项目专用环境再安装：

```text
python -m pip install --no-index --find-links wheelhouse numpy==1.26.4 Pillow==10.4.0
python run.py --help
python -m unittest discover -s tests -v
```

从源码运行无需构建 wheel、联网安装项目或下载真实数据。`wheelhouse/`、`.conda/` 和本地配置被忽略，不提交。

## 路径和本地输入

配置根路径可覆盖，默认是项目内 `data/fixed_samples/`、`data/predictions/`、`outputs/`。相对路径按项目根定位，不依赖终端目录。清单内图片路径保持 POSIX 相对形式，不引用原清单里的服务器 `source` 去读取本地图片。

迁移真实固定导出时保留原 selection、samples.csv、文件内容与名称，并验证传输哈希。迁移 Prediction 时保留完整 ID 和对应映射/来源记录。生成结果中的绝对源路径若随迁移变化，应在新配置及新运行中重新验证输入，旧运行仍按其记录的位置验收。

## 不依赖真实 Prediction 的自检

```text
python examples/make_synthetic_example.py
python run.py validate --config examples/synthetic_workspace/config.json
python run.py generate --config examples/synthetic_workspace/config.json --out examples/synthetic_workspace/outputs/offline-demo-001
python run.py verify --run examples/synthetic_workspace/outputs/offline-demo-001
```

示例目录与真实实验数据隔离，全部图片来自手写合成数组。运行成功只能证明该环境下的工具流程可用，不能证明真实 Prediction 已完成来源与配对验收。
