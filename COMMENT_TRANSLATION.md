# Chinese-English comment translation / 中英文注释转换说明

This document explains how to translate comments and docstrings in SKD-EGP while preserving the code's experimental meaning. It is a translation guide, not a claim that all comments have been converted. 本文说明 SKD-EGP 代码注释和文档字符串的中英互译原则；目前源码注释仍为中英文混用。

## Scope / 转换范围

- Translate human-facing `#` comments and triple-quoted docstrings when an English or Chinese edition is needed.
- Keep executable code, variable names, environment-variable keys, file names, JSON keys, entity labels, shell commands, and numeric values unchanged.
- Do not translate string literals that appear in logs, errors, metadata, or serialized outputs as part of a comment-only change; those are runtime behavior or data contracts.
- Preserve the exact meaning of experimental constraints, especially whether a setting is locked by `experiment_profile.py` or only defaults in `config.py`.

只转换供人阅读的 `#` 注释和文档字符串。变量名、配置键、数据集标签、文件名、路径、数值和可执行代码保持不变。日志、异常信息及输出数据中的字符串不属于“仅翻译注释”的范围。

## Terminology / 术语对照

| 中文 | English | Meaning in this repository |
| --- | --- | --- |
| 命名实体识别 | named entity recognition (NER) | Span detection and entity typing |
| 结构化知识蒸馏 | structured knowledge distillation (KD) | Teacher-to-student span, boundary, and nested-structure losses |
| 教师模型 / 学生模型 | teacher model / student model | 12-layer teacher / 8-layer full-experiment student |
| 高效 GlobalPointer | Efficient GlobalPointer | The efficient task head selected by `GP_EFFICIENT=1` |
| 标准 GlobalPointer | Standard GlobalPointer | The baseline task head selected by `GP_EFFICIENT=0` |
| 在线数据增强 | online data augmentation | Applied during training, rather than pre-generated offline data |
| 序列拼接 | sequence concatenation | One online augmentation operation |
| 位置偏移 | position shift | One online augmentation operation; retain the code's precise conditions |
| Token 掩码 | token masking | Masking of eligible non-special tokens |
| 滑动窗口 | sliding window | Windowed processing of long text |
| 半开区间 | half-open interval | `[start_idx, end_idx)`; end index is excluded |
| 闭区间 | inclusive interval | `[start, end]`; end index is included |
| 最佳检查点 | best checkpoint | Dev-selected `best_model.pth` |
| 原始训练集 | raw training split | Original human-annotated Train, not generated data |

Use `Dev` for the development/validation split and `Test` for the test split. Preserve dataset spellings (`CMeEE-V2`, `CLUENER`, `ChiNesE`) and identifiers such as `RUN_NAME`, `TRAIN_ROLE`, `GP_EFFICIENT`, `USE_EIGHT_LAYER_ENCODER`, and `DISTILLATION_TEACHER_CHECKPOINT` exactly.

## Examples / 示例

Chinese to English:

```python
# 训练长文本也使用滑窗监督；窗口 offset_mapping 保持原始字符坐标。
# Use sliding-window supervision for long training texts; offset_mapping keeps the original character coordinates.
```

English to Chinese:

```python
# Validate the teacher checkpoint metadata before loading its weights.
# 加载教师模型权重前，先校验检查点元数据。
```

These paired lines illustrate wording only. When actually converting a source file, replace the old comment with the chosen language (or keep both only where bilingual maintenance is intentional); avoid accumulating duplicate comments throughout the code.

## Review / 校核

1. Compare each translated sentence with the adjacent code and `experiment_profile.py`; do not infer behavior from an outdated comment alone.
2. Check boundary terminology carefully: input data uses `[start_idx, end_idx)`, while some internal GlobalPointer operations use an inclusive end index.
3. Review the diff to ensure only comments or docstrings changed. For a comment-only translation, a Python syntax check such as `python -m compileall <experiment-directory>` can catch accidental syntax damage, but it cannot verify the semantic accuracy of a translation.

逐条对照实际代码和实验 profile；特别检查教师模型协议、数据边界及术语。注释翻译后的语法检查只能发现代码损坏，仍需人工核对译文是否准确。
