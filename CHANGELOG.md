# 版本记录

## 0.1.0（2026-10-04 整理）

- 清单驱动的 `validate`、`generate`、`verify`；支持固定 selection 和简单通用 CSV。
- 明确路径模板或显式 Prediction 映射，保留完整 ID，严格检查缺失、重复、尺寸、编码、哈希及路径冲突。
- 安全固定包导入；支持封印的 `test.txt` 原字节哈希与 ID/顺序核对。
- GT/Prediction 独立二值编码，可选 GT 原值映射与显式 GT ignore。
- 稳定类别索引与可配置色板；默认 TN 黑、TP 白、FP 红、FN 绿、IGNORE 灰。
- 无损 RGB/索引 PNG、独立图例、逐图像素计数、分组 micro 指标、配对/哈希记录和完成封印。
- 独立运行目录，拒绝静默覆盖；回读产物与源输入，失败或中断不标为完成。
- 中文文档、通用模板、专用 Conda 环境定义、独立手算合成示例及 Python 3.9/3.12 CI。

验证与真实实验完成范围见 [开发进度](DEVELOPMENT_PROGRESS.md)。真实数据及结果仅在本地保存；本记录不表示创建了 Git tag、GitHub Release 或软件包发布。
