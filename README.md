# SKD-EGP

SKD-EGP is a Chinese named entity recognition (NER) research codebase covering CMeEE-V2, CLUENER, and ChiNesE. Each dataset has a 12-layer Standard GlobalPointer baseline and an 8-layer Efficient GlobalPointer full experiment with online augmentation and structured knowledge distillation (KD). The six experiment directories are independent; their settings are fixed by each directory's `experiment_profile.py`.

SKD-EGP 是面向 CMeEE-V2、CLUENER 和 ChiNesE 的中文命名实体识别研究代码。每个数据集分别提供 12 层 Standard GlobalPointer 基线和 8 层 Efficient GlobalPointer + 在线增强 + 结构化知识蒸馏的 full 实验。六个实验目录相互独立，具体设置由各目录的 `experiment_profile.py` 固定。

Repository: <https://github.com/kaggleoldeight/SKD-EGP>

## Repository layout / 仓库结构

| Dataset | Baseline | Full experiment |
| --- | --- | --- |
| CMeEE-V2 | `CMeEE-V2-baseline/` | `CMeEE-V2-full/` |
| CLUENER | `CLUENER-baseline/` | `CLUENER-full/` |
| ChiNesE | `ChiNesE-baseline/` | `ChiNesE-full/` |

Each directory contains `config.py`, `experiment_profile.py`, `train_CME.py`, `predict_CME.py`, model/data modules, and analysis scripts. `train_CME.py` and `predict_CME.py` have no command-line arguments; paths and training settings come from `config.py` and the profile. Some paths, including `TRAIN_FILE`, `DEV_FILE`, `TEST_FILE`, and `RUN_NAME`, can be supplied through environment variables, but profile-locked settings take precedence.

每个目录都有独立的配置、训练、预测和分析脚本。训练和预测入口不接收命令行参数；请先核对相应目录的 `config.py` 与 `experiment_profile.py`。部分路径可通过环境变量设置，但 profile 锁定的实验开关不能靠外部环境变量覆盖。

## Installation / 安装

```bash
git clone https://github.com/kaggleoldeight/SKD-EGP.git
cd SKD-EGP
python -m pip install -r requirements.txt
```

`requirements.txt` pins the main versions visible in the supplied environment: PyTorch 2.8.0, Transformers 4.57.1, NumPy 2.3.2, tqdm 4.66.2, Matplotlib 3.10.5, and jsonlines 4.0.0. Packages absent from that environment listing are left unpinned. The supplied environment listed a `2.8.0+cu128` PyTorch build; for GPU use, install the build matching your CUDA driver from the [official PyTorch instructions](https://pytorch.org/get-started/locally/). A CUDA-capable PyTorch build does not by itself guarantee an available GPU.

依赖文件只固定已核实的主要版本。GPU 环境应先按官方说明安装匹配驱动的 PyTorch 版本；是否能实际使用 GPU，以 `torch.cuda.is_available()` 为准。

## Required assets / 运行前准备

The repository does **not** contain the pretrained model in any of the six directories. Place a local 12-layer `chinese-roberta-wwm-ext` model (configuration, tokenizer, and weights) at:

```text
<experiment-directory>/pretrained_models/chinese-roberta-wwm-ext/
```

The code uses local `AutoConfig`, `AutoTokenizer` (fast tokenizer), and `AutoModel` loading. The expected encoder depth is checked against 12 layers. Each experiment also reads its own `datasets/` directory:

| Dataset | Train | Dev / Eval | Test |
| --- | --- | --- | --- |
| CMeEE-V2 | `CMeEE-V2_train.json` | `CMeEE-V2_dev.json` | `CMeEE-V2_test.json` |
| CLUENER | `CLUENER_train.json` | `CLUENER_dev.json` | `CLUENER_test.json` |
| ChiNesE | `ChiNesE_train.json` | `ChiNesE_eval.json` | `ChiNesE_test.json` |

At the checked public revision (`721281d`), CMeEE-V2 and CLUENER baseline directories include converted data, while the three full directories have no `datasets/` directory. The ChiNesE baseline directory does not include the required `ChiNesE_train.json`, `ChiNesE_eval.json`, and `ChiNesE_test.json`. Supply the correct dataset for **each** experiment directory; do not use similarly named files from a different dataset.

公开仓库的三个 full 目录缺少数据目录；ChiNesE-baseline 也缺少所需的 ChiNesE 训练、验证和测试文件。各实验目录都缺少预训练模型，因此下载代码后还需要自行准备模型与相应数据。各目录不能混用其他数据集的文件。

For CLUENER, `prepare_cluener_dataset.py` converts the original `train.json`, `dev.json`, and `test.json` JSONL files into the expected JSON files. For ChiNesE, `prepare_chinese_dataset.py` expects `new_train.json`, `new_eval.json`, and `new_test.json`. Run `python <experiment-directory>/prepare_cluener_dataset.py --help` or `python <experiment-directory>/prepare_chinese_dataset.py --help` to set `--raw-dir` and `--output-dir` as needed. CMeEE-V2 uses the JSON files named in the table.

Converted records are JSON arrays with `text` and `entities`; CLUENER and ChiNesE records also require `dataset_name`. Entity spans use half-open character offsets `[start_idx, end_idx)`, meaning `text[start_idx:end_idx]` is the entity text. The conversion scripts handle their source datasets' inclusive end offsets.

转换后的数据为 JSON 数组，实体跨度使用字符半开区间 `[start_idx, end_idx)`。CLUENER 和 ChiNesE 的原始 JSONL 需要先经对应转换脚本处理，不能直接作为训练输入。

### Full-experiment teacher / full 实验教师模型

All three `*-full` profiles enable KD and point to `<experiment-directory>/teacher_checkpoints/Teacher_12L_woKD_PS/best_model.pth` by default. This checkpoint is **not** in the public repository. Training validates the teacher checkpoint together with `best_model.pth.meta.json` and its `train_params.json`; it checks the dataset, 12-layer architecture, Efficient GlobalPointer head, training protocol, and file hashes. The published `*-baseline` configurations use a Standard GlobalPointer head and are **not** interchangeable with this KD teacher. The public repository therefore does not, by itself, provide everything needed to reproduce the full training runs.

三个 full profile 默认启用 KD，但公开仓库未提供教师权重及配套元数据。公开的 baseline 配置是 Standard GlobalPointer，不能直接当作 full 实验所需的 Efficient GlobalPointer 教师模型。

## Training and prediction / 训练与预测

After placing the required assets in the corresponding experiment directory, run one experiment at a time from the repository root. For example:

```bash
python CMeEE-V2-baseline/train_CME.py
python CMeEE-V2-baseline/predict_CME.py
```

Replace `CMeEE-V2-baseline` with another experiment directory only after preparing its own data, model, and (for full experiments) validated teacher checkpoint. Training selects the best checkpoint on Dev and writes `outputs/<RUN_NAME>/best_model.pth`, its `.meta.json`, training logs, and other artifacts under that experiment directory. Prediction loads that best checkpoint and writes a dataset-specific `*_predict_result.json` under the same directory's `datasets/`. CLUENER and ChiNesE prediction additionally write `*_test_predictions.jsonl`. ChiNesE also has `evaluate_test.py` for a labeled Test split after Dev-based model selection.

训练以 Dev 选出的最佳模型为预测入口，输出位于对应实验目录下的 `outputs/<RUN_NAME>/`。同一 `RUN_NAME` 下已有训练产物时，代码默认拒绝再次训练；重跑应使用新的 `RUN_NAME`。训练结束还会按配置启动效率评测，可能增加运行时间。没有真实数据、预训练模型或教师权重时，以上命令无法完成完整实验。

## Comment language / 注释语言

The code contains both Chinese and English comments. For a terminology guide and rules for translating comments without changing experimental behavior, see [COMMENT_TRANSLATION.md](COMMENT_TRANSLATION.md). This guide documents the translation process; it does not claim that every source comment has already been translated.

代码注释中英文混用。术语对照和注释翻译原则见 [COMMENT_TRANSLATION.md](COMMENT_TRANSLATION.md)；该说明不代表源码中的全部注释已经完成翻译。
