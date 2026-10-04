# -*- coding: utf-8 -*-
"""
@Time: 2021/8/27 16:12
@Auth: Xhw0
@File: entity_extract.py
@Description: 实体抽取.
"""
import os
import sys

# PYTHONHASHSEED只有在解释器启动前生效。直接运行本文件时，先读取不依赖
# PyTorch的实验profile并按锁定seed重启一次，避免“日志写了seed但哈希随机性
# 未固定”。子进程用内部标记防止循环重启。
if __name__ == "__main__":
    from experiment_profile import LOCKED_ENV as _EARLY_LOCKED_ENV
    _expected_hash_seed = str(_EARLY_LOCKED_ENV.get("SEED", "42"))
    if (
        os.environ.get("PYTHONHASHSEED") != _expected_hash_seed
        and os.environ.get("HAEGP_HASHSEED_REEXEC") != "1"
    ):
        _child_env = os.environ.copy()
        _child_env["PYTHONHASHSEED"] = _expected_hash_seed
        _child_env["HAEGP_HASHSEED_REEXEC"] = "1"
        os.execve(sys.executable, [sys.executable, os.path.abspath(__file__)], _child_env)

import random
import numpy as np
import csv
import gc
import hashlib
import json
import platform
import subprocess
from data_loader import EntDataset, load_data, ent2id, id2ent, validate_cmeee_records, load_dataset_records
from data_augmenter import SequenceAugmenter
from transformers import AutoConfig, AutoTokenizer, AutoModel
import transformers
from torch.utils.data import DataLoader, get_worker_info
import torch
import torch.nn.functional as F
from GlobalPointer import GlobalPointer
from tqdm import tqdm
from config import (
    DATASET_NAME,
    PRETRAINED_MODEL_DIR,
    EXPECTED_PRETRAINED_ENCODER_LAYERS,
    DATA_DIR,
    TRAIN_FILE,
    DEV_FILE,
    TEST_FILE,
    DEVICE,
    BATCH_SIZE,
    ENT_CLS_NUM,
    GP_HEAD_SIZE,
    GP_EFFICIENT,
    LEARNING_RATE,
    NUM_WORKERS,
    EPOCHS,
    WEIGHT_DECAY,
    WARMUP_RATIO,
    RNN_HIDDEN_SIZE,
    RNN_LAYERS,
    USE_BIGRU,
    MAX_LEN,
    SEED,
    RUN_NAME,
    TRAIN_ROLE,
    SELECTION_SPLIT_ROLE,
    USE_SEQUENCE_CONCAT,
    USE_POSITION_SHIFT,
    USE_TOKEN_MASK,
    TOKEN_MASK_SCOPE,
    LEGACY_CONCAT_ROUTING,
    V12_STUDENT_CONCAT_ROUTING,
    USE_EXPLICIT_DATALOADER_GENERATOR,
    CONCAT_PROB,
    POSITION_SHIFT_PROB,
    POSITION_SHIFT_PREFIXES,
    TOKEN_MASK_PROB,
    SEQUENCE_CONCAT_SEPARATOR,
    EXPERIMENT_PROFILE,
    DECODE_THRESHOLD,
    TRAIN_STRIDE,
    PREDICT_STRIDE,
    USE_SLIDING_WINDOW_DEV_EVAL,
    REMOVE_HOLDOUT_TEXT_OVERLAP,
    SAVE_TOPK_MODELS,
    TOPK_MODELS,
    ALLOW_RUN_DIR_REUSE,
    USE_EIGHT_LAYER_ENCODER,
    USE_NESTED_STRUCTURE_DISTILLATION,
    STUDENT_NUM_HIDDEN_LAYERS,
    DISTILLATION_TEACHER_CHECKPOINT,
    DISTILLATION_TEACHER_TRAIN_ROLE,
    DISTILLATION_TEACHER_TRAIN_FILE,
    DISTILLATION_TEACHER_GP_EFFICIENT,
    DISTILLATION_TEACHER_USE_BIGRU,
    DISTILLATION_TEACHER_USE_SEQUENCE_CONCAT,
    DISTILLATION_TEACHER_USE_POSITION_SHIFT,
    DISTILLATION_TEACHER_USE_TOKEN_MASK,
    DISTILLATION_TEACHER_TOKEN_MASK_SCOPE,
    DISTILLATION_TEACHER_LEGACY_CONCAT_ROUTING,
    DISTILLATION_TEACHER_SEED,
    DISTILLATION_TEACHER_DEVICE,
    DISTILLATION_TEMPERATURE,
    DISTILLATION_SPAN_WEIGHT,
    DISTILLATION_BOUNDARY_WEIGHT,
    DISTILLATION_NESTED_WEIGHT,
    DISTILLATION_NESTED_TOP_K,
    AUTO_RUN_EFFICIENCY_BENCHMARK,
    AUTO_EFFICIENCY_DEVICES,
    EFFICIENCY_BENCHMARK_TIMEOUT_SECONDS,
    EFFICIENCY_DIR,
)
from config import TRAIN_LOG_CSV, TRAIN_STEP_LOG_CSV, TRAIN_CLASS_LOG_CSV, TRAIN_STRUCTURE_LOG_CSV, TRAIN_PARAMS_JSON, ONLINE_AUGMENTATION_LOG_CSV, BEST_MODEL_PATH, BEST_MODELS_DIR, BEST_MODELS_META
from train_logger import append_epoch_log, append_step_log, append_class_epoch_log, append_structure_epoch_log, save_params, save_best_model, save_topk_model, verify_topk_models
from transformers import get_linear_schedule_with_warmup
from predict_CME import load_model as load_prediction_model, predict_batch
from adversarial import AdversarialTrainer
from config import (
    ADV_TRAIN_METHOD,
    ADV_EMB_NAME,
    FGM_EPSILON,
    PGD_EPSILON,
    PGD_ALPHA,
    PGD_STEPS,
    PGD_START_EPOCH,
)
from dataset_policy import (
    DATASET_NAME as POLICY_DATASET_NAME,
    FOCUS_CLASS_PRIORITY,
)

if DATASET_NAME != POLICY_DATASET_NAME:
    raise RuntimeError(
        f"数据集策略与config不一致: policy={POLICY_DATASET_NAME}, config={DATASET_NAME}"
    )

train_cme_path = (
    os.path.abspath(TRAIN_FILE)
    if os.path.isabs(TRAIN_FILE)
    else os.path.join(DATA_DIR, TRAIN_FILE)
)
eval_cme_path = (
    os.path.abspath(DEV_FILE)
    if os.path.isabs(DEV_FILE)
    else os.path.join(DATA_DIR, DEV_FILE)
)
device = DEVICE


def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _update_train_params(values):
    if not os.path.isfile(TRAIN_PARAMS_JSON):
        raise FileNotFoundError(
            f"效率评测前缺少训练参数文件: {TRAIN_PARAMS_JSON}"
        )
    with open(TRAIN_PARAMS_JSON, "r", encoding="utf-8") as f:
        params = json.load(f)
    params.update(values)
    save_params(params, TRAIN_PARAMS_JSON)


def run_automatic_efficiency_benchmark():
    """训练进程释放模型后，在独立进程评测并严格核验效率输出。"""
    if not AUTO_RUN_EFFICIENCY_BENCHMARK:
        _update_train_params({
            "automatic_efficiency_benchmark_enabled": False,
            "efficiency_benchmark_completed": False,
            "efficiency_benchmark_skip_reason": "disabled_by_configuration",
        })
        return
    if not os.path.isfile(BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"效率评测需要训练后的best_model.pth: {BEST_MODEL_PATH}"
        )

    os.makedirs(EFFICIENCY_DIR, exist_ok=True)
    status_path = os.path.join(EFFICIENCY_DIR, "efficiency_status.json")
    head_name = "efficient_gp" if GP_EFFICIENT else "standard_gp"
    output_prefix = os.path.join(EFFICIENCY_DIR, f"efficiency_{head_name}")
    result_json = output_prefix + ".json"
    result_csv = output_prefix + ".csv"
    benchmark_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "benchmark_efficiency.py"
    )
    if not os.path.isfile(benchmark_script):
        raise FileNotFoundError(f"缺少效率评测脚本: {benchmark_script}")
    command = [
        sys.executable,
        benchmark_script,
        "--checkpoint",
        BEST_MODEL_PATH,
        "--efficient",
        "true" if GP_EFFICIENT else "false",
        "--use-bigru",
        "true" if USE_BIGRU else "false",
        "--devices",
        AUTO_EFFICIENCY_DEVICES,
        "--output-prefix",
        output_prefix,
    ]
    save_params({
        "status": "running",
        "checkpoint": os.path.abspath(BEST_MODEL_PATH),
        "checkpoint_sha256": file_sha256(BEST_MODEL_PATH),
        "devices": AUTO_EFFICIENCY_DEVICES,
        "output_prefix": os.path.abspath(output_prefix),
    }, status_path)
    _update_train_params({
        "automatic_efficiency_benchmark_enabled": True,
        "efficiency_benchmark_completed": False,
        "efficiency_benchmark_devices": AUTO_EFFICIENCY_DEVICES,
        "efficiency_benchmark_timeout_seconds": EFFICIENCY_BENCHMARK_TIMEOUT_SECONDS,
        "efficiency_benchmark_status_file": os.path.abspath(status_path),
    })

    try:
        completed = subprocess.run(
            command,
            check=False,
            timeout=EFFICIENCY_BENCHMARK_TIMEOUT_SECONDS,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"benchmark_efficiency.py退出码={completed.returncode}"
            )
        if not os.path.isfile(result_json) or not os.path.isfile(result_csv):
            raise FileNotFoundError(
                "效率评测未同时生成JSON和CSV: "
                f"json={result_json}, csv={result_csv}"
            )
        if os.path.getsize(result_csv) <= 0:
            raise RuntimeError(f"效率CSV为空: {result_csv}")
        with open(result_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        measurements = payload.get("measurements")
        if not isinstance(measurements, list) or not measurements:
            raise RuntimeError("效率JSON没有任何有效measurement")
        actual_checkpoint_hash = file_sha256(BEST_MODEL_PATH)
        if payload.get("checkpoint_sha256") != actual_checkpoint_hash:
            raise RuntimeError(
                "效率结果引用的权重哈希与best_model不一致: "
                f"result={payload.get('checkpoint_sha256')}, "
                f"actual={actual_checkpoint_hash}"
            )
        expected_architecture = head_name
        if payload.get("architecture") != expected_architecture:
            raise RuntimeError(
                "效率结果架构与当前实验不一致: "
                f"result={payload.get('architecture')}, "
                f"expected={expected_architecture}"
            )
        result_info = {
            "automatic_efficiency_benchmark_enabled": True,
            "efficiency_benchmark_completed": True,
            "efficiency_benchmark_devices": AUTO_EFFICIENCY_DEVICES,
            "efficiency_benchmark_timeout_seconds": (
                EFFICIENCY_BENCHMARK_TIMEOUT_SECONDS
            ),
            "efficiency_benchmark_measurement_count": len(measurements),
            "efficiency_benchmark_json": os.path.abspath(result_json),
            "efficiency_benchmark_json_sha256": file_sha256(result_json),
            "efficiency_benchmark_csv": os.path.abspath(result_csv),
            "efficiency_benchmark_csv_sha256": file_sha256(result_csv),
            "efficiency_benchmark_status_file": os.path.abspath(status_path),
        }
        save_params({"status": "completed", **result_info}, status_path)
        _update_train_params(result_info)
        print(
            "Efficiency benchmark verified: "
            f"measurements={len(measurements)}, directory={os.path.abspath(EFFICIENCY_DIR)}"
        )
    except Exception as exc:
        failure = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "checkpoint": os.path.abspath(BEST_MODEL_PATH),
            "devices": AUTO_EFFICIENCY_DEVICES,
            "timeout_seconds": EFFICIENCY_BENCHMARK_TIMEOUT_SECONDS,
        }
        save_params(failure, status_path)
        _update_train_params({
            "automatic_efficiency_benchmark_enabled": True,
            "efficiency_benchmark_completed": False,
            "efficiency_benchmark_error_type": type(exc).__name__,
            "efficiency_benchmark_error": str(exc),
            "efficiency_benchmark_status_file": os.path.abspath(status_path),
        })
        raise RuntimeError(
            f"训练已完成，但自动效率评测失败；详情见 {status_path}: {exc}"
        ) from exc


PRETRAINED_IDENTITY_FILENAMES = (
    "config.json",
    "vocab.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


def pretrained_artifact_fingerprint(model_dir):
    """用小型配置/词表文件标识预训练底座，避免用绝对路径判断同一模型。"""
    fingerprints = {}
    for filename in PRETRAINED_IDENTITY_FILENAMES:
        path = os.path.join(model_dir, filename)
        if os.path.isfile(path):
            fingerprints[filename] = file_sha256(path)
    if "config.json" not in fingerprints:
        raise FileNotFoundError(
            f"预训练模型目录缺少 config.json，无法建立可迁移身份指纹: {model_dir}"
        )
    return fingerprints


def resolve_data_path(path_or_name):
    """数据文件名相对 DATA_DIR 解析；绝对路径保持不变。"""
    if not path_or_name:
        raise ValueError("数据文件名不能为空")
    return (
        os.path.abspath(path_or_name)
        if os.path.isabs(path_or_name)
        else os.path.abspath(os.path.join(DATA_DIR, path_or_name))
    )


def audit_raw_training_source(records, train_path):
    """记录并核验本发布版只使用原始人工标注Train。"""
    if not isinstance(records, list):
        raise ValueError("训练JSON顶层必须是list")
    expected_path = resolve_data_path(DISTILLATION_TEACHER_TRAIN_FILE if TRAIN_ROLE == DISTILLATION_TEACHER_TRAIN_ROLE else TRAIN_FILE)
    if os.path.abspath(train_path) != os.path.abspath(expected_path):
        raise ValueError(
            "训练文件与当前实验profile锁定的原始Train不一致："
            f"actual={train_path}, expected={expected_path}"
        )
    return {
        "training_source_contract_checked": True,
        "training_data_kind": "raw_train",
        "training_rows": len(records),
        "training_file_sha256": file_sha256(train_path),
    }


def append_online_augmentation_epoch_log(path, epoch, stats):
    """逐轮追加在线增强真实生效率；概率仅是配置，不能代替此日志。"""
    stats = dict(stats)
    ordered = {
        "epoch": int(epoch),
        **{key: int(value) for key, value in sorted(stats.items())},
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    write_header = not os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(ordered))
        if write_header:
            writer.writeheader()
        writer.writerow(ordered)


def summarize_online_augmentation(stats):
    summary = {key: int(value) for key, value in sorted(stats.items())}
    rate_specs = {
        "concat_application_rate": ("concat_applied", "concat_attempted"),
        "position_shift_application_rate": (
            "position_shift_applied", "position_shift_attempted"
        ),
        "token_mask_application_rate": ("token_masked", "token_mask_eligible"),
        "entity_token_mask_application_rate": (
            "entity_tokens_masked", "entity_token_mask_eligible"
        ),
    }
    for rate_name, (numerator, denominator) in rate_specs.items():
        den = int(stats.get(denominator, 0))
        summary[rate_name] = (
            int(stats.get(numerator, 0)) / den if den else None
        )
    return summary


def set_seed(seed=42):
    """
    设置随机种子，确保实验可复现
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 设置PyTorch的确定性模式（可能影响性能，但确保可复现）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # 设置环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed} for reproducibility")

def worker_init_fn(worker_id):
    """
    为每个 DataLoader worker 设置不同随机种子，避免多进程同轨迹。
    """
    # torch.initial_seed() 已包含worker_id；不要再次相加。
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    worker_info = get_worker_info()
    if worker_info is not None:
        worker_augmenter = getattr(worker_info.dataset, "augmenter", None)
        if worker_augmenter is not None:
            worker_augmenter.seed_worker(worker_seed)


def audit_training_window_gold_coverage(data, tokenizer, max_len, stride):
    """统计重叠训练窗口仍无法完整容纳的 Gold（例如实体本身超长）。"""
    total_entities = 0
    uncovered_entities = 0
    affected_samples = 0
    total_windows = 0
    safe_stride = min(max(0, int(stride)), max(0, max_len - 3))
    for item in data:
        encoded = tokenizer(
            item[0],
            max_length=max_len,
            truncation=True,
            return_offsets_mapping=True,
            return_overflowing_tokens=safe_stride > 0,
            stride=safe_stride if safe_stride > 0 else 0,
        )
        offsets = encoded["offset_mapping"]
        if not offsets or not isinstance(offsets[0], (list, tuple)) or (
            offsets[0] and isinstance(offsets[0][0], int)
        ):
            windows = [offsets]
        else:
            windows = offsets
        total_windows += len(windows)
        sample_uncovered = 0
        for start, end, _label in item[1:]:
            total_entities += 1
            covered_in_one_window = any(
                any(ts <= start < te for ts, te in window)
                and any(ts <= end < te for ts, te in window)
                for window in windows
            )
            if not covered_in_one_window:
                uncovered_entities += 1
                sample_uncovered += 1
        if sample_uncovered:
            affected_samples += 1
    return {
        "train_gold_entities_total": total_entities,
        "train_windows_per_epoch": total_windows,
        "train_gold_entities_uncovered_by_sliding_windows": uncovered_entities,
        "train_samples_with_uncovered_gold": affected_samples,
    }


def remove_holdout_text_overlap(train_data, holdout_paths):
    """仅用 holdout 正文做精确去重，不读取其标签参与训练或选模。"""
    holdout_texts = set()
    holdout_hashes = {}
    for path in holdout_paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"开启了训练集去污染，但找不到 holdout 文件: {path}")
        holdout_hashes[os.path.abspath(path)] = file_sha256(path)
        holdout_texts.update(
                row.get("text", "")
                for row in load_dataset_records(path)
                if row.get("text")
            )
    removed = [item for item in train_data if item[0] in holdout_texts]
    cleaned = [item for item in train_data if item[0] not in holdout_texts]
    return cleaned, {
        "holdout_text_decontamination_enabled": True,
        "holdout_files_sha256": holdout_hashes,
        "train_rows_before_holdout_decontamination": len(train_data),
        "train_rows_removed_by_exact_holdout_text": len(removed),
        "train_rows_after_holdout_decontamination": len(cleaned),
        "removed_train_text_sha256": [
            hashlib.sha256(item[0].encode("utf-8")).hexdigest() for item in removed
        ],
    }


def multilabel_categorical_crossentropy(y_pred, y_true, reduction="mean"):
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = y_pred - (1 - y_true) * 1e12 # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    loss = neg_loss + pos_loss
    if reduction == "none":
        return loss
    if reduction != "mean":
        raise ValueError(f"不支持的 reduction={reduction!r}")
    return loss.mean()

def loss_fun(logits, labels, sample_weights=None):
    """
    logits/labels: (batch_size, ent_type_size, seq_len, seq_len)
    """
    batch_size, ent_type_size = logits.shape[:2]
    flat_logits = logits.reshape(batch_size * ent_type_size, -1)
    flat_labels = labels.float().reshape(batch_size * ent_type_size, -1)
    per_head_loss = multilabel_categorical_crossentropy(
        flat_logits,
        flat_labels,
        reduction="none",
    ).reshape(batch_size, ent_type_size)
    per_sample_loss = per_head_loss.mean(dim=1)
    if sample_weights is None:
        return per_sample_loss.mean()
    sample_weights = sample_weights.to(
        device=per_sample_loss.device,
        dtype=per_sample_loss.dtype,
    ).reshape(-1)
    if sample_weights.numel() != batch_size:
        raise ValueError(
            "sample_weights 与 batch_size 不一致："
            f"weights={sample_weights.numel()}, batch={batch_size}"
        )
    denominator = sample_weights.sum()
    if denominator <= 0:
        raise ValueError("监督样本权重之和必须大于0")
    return (per_sample_loss * sample_weights).sum() / denominator


def _transformer_parts(encoder):
    """返回 BERT/RoBERTa 风格编码器的 embeddings 与 Transformer 层列表。"""
    embeddings = getattr(encoder, "embeddings", None)
    encoder_stack = getattr(encoder, "encoder", None)
    layers = getattr(encoder_stack, "layer", None) if encoder_stack is not None else None
    if embeddings is None or layers is None:
        raise TypeError(
            "当前层抽取实现只接受具有 encoder.embeddings 和 "
            "encoder.encoder.layer 的 BERT/RoBERTa 风格模型"
        )
    return embeddings, layers


def select_teacher_layer_indices(teacher_layers, student_layers):
    """均匀选取 Teacher 层；首层和末层都保留，且不使用实验阈值。"""
    teacher_layers = int(teacher_layers)
    student_layers = int(student_layers)
    if not (0 < student_layers < teacher_layers):
        raise ValueError(
            "学生层数必须严格小于 Teacher 层数："
            f"student={student_layers}, teacher={teacher_layers}"
        )
    indices = [
        int(round(x))
        for x in np.linspace(0, teacher_layers - 1, num=student_layers)
    ]
    if len(indices) != student_layers or len(set(indices)) != student_layers:
        raise RuntimeError(f"层映射生成失败: {indices}")
    return indices


def build_encoder_from_pretrained(pretrained_model_dir, target_layers):
    """
    从完整预训练编码器构建目标深度。

    轻量学生始终使用与 KD 初始化相同的均匀层映射；这样 w/o KD 与 Full
    的结构和预训练层选择一致，唯一差异才是 Teacher/KD 损失。
    """
    encoder = AutoModel.from_pretrained(pretrained_model_dir)
    _embeddings, full_layers = _transformer_parts(encoder)
    full_depth = len(full_layers)
    target_layers = int(target_layers)
    if target_layers == full_depth:
        return encoder, list(range(full_depth))
    layer_indices = select_teacher_layer_indices(full_depth, target_layers)
    encoder.encoder.layer = torch.nn.ModuleList(
        [full_layers[index] for index in layer_indices]
    )
    encoder.config.num_hidden_layers = target_layers
    return encoder, layer_indices


def initialize_student_from_teacher(
    student_model,
    teacher_model,
    *,
    copy_task_head=True,
):
    """用Teacher初始化编码器；仅在GP头同构时复制任务头。"""
    student_embeddings, student_layers = _transformer_parts(student_model.encoder)
    teacher_embeddings, teacher_layers = _transformer_parts(teacher_model.encoder)
    layer_indices = select_teacher_layer_indices(
        len(teacher_layers),
        len(student_layers),
    )
    student_embeddings.load_state_dict(teacher_embeddings.state_dict(), strict=True)
    for student_index, teacher_index in enumerate(layer_indices):
        student_layers[student_index].load_state_dict(
            teacher_layers[teacher_index].state_dict(),
            strict=True,
        )

    if not copy_task_head:
        return layer_indices

    # 同构时复用已训练任务头；Standard/Efficient异构蒸馏只复制编码器，
    # 两者输出均为 [B,C,L,L]，仍可执行后续logit KD。
    student_state = student_model.state_dict()
    teacher_state = teacher_model.state_dict()
    for name, teacher_tensor in teacher_state.items():
        if name.startswith("encoder."):
            continue
        if name not in student_state or student_state[name].shape != teacher_tensor.shape:
            raise ValueError(f"压缩 Teacher 与学生任务头不同构: {name}")
        student_state[name] = teacher_tensor.detach()
    student_model.load_state_dict(student_state, strict=True)
    return layer_indices


def validate_distillation_teacher_metadata(checkpoint):
    """Fail closed：只接受同数据集、原始Train、完整10轮的12层无KD Teacher。"""
    if not checkpoint:
        raise ValueError("开启嵌套结构感知蒸馏时必须设置DISTILLATION_TEACHER_CHECKPOINT")
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"未找到压缩Teacher权重: {checkpoint}")
    if os.path.abspath(checkpoint) == os.path.abspath(BEST_MODEL_PATH):
        raise ValueError("压缩Teacher权重不能与当前Run的学生输出路径相同")
    meta_path = checkpoint + ".meta.json"
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"压缩Teacher缺少架构元数据: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    teacher_role = meta.get("train_role")
    if teacher_role != DISTILLATION_TEACHER_TRAIN_ROLE:
        raise ValueError(
            "压缩Teacher必须由05号包以distillation_teacher角色训练；"
            f"当前role={teacher_role!r}"
        )
    expected = {
        "dataset_name": DATASET_NAME,
        "gp_efficient": DISTILLATION_TEACHER_GP_EFFICIENT,
        "use_bigru": DISTILLATION_TEACHER_USE_BIGRU,
        "ent_cls_num": ENT_CLS_NUM,
        "gp_head_size": GP_HEAD_SIZE,
        "rnn_hidden_size": RNN_HIDDEN_SIZE,
        "rnn_layers": RNN_LAYERS,
        "use_eight_layer_encoder": False,
        "use_nested_structure_distillation": False,
        "use_sequence_concat": DISTILLATION_TEACHER_USE_SEQUENCE_CONCAT,
        "use_position_shift": DISTILLATION_TEACHER_USE_POSITION_SHIFT,
        "use_token_mask": DISTILLATION_TEACHER_USE_TOKEN_MASK,
        "token_mask_scope": DISTILLATION_TEACHER_TOKEN_MASK_SCOPE,
        "legacy_concat_routing": (
            DISTILLATION_TEACHER_LEGACY_CONCAT_ROUTING
        ),
        "encoder_num_hidden_layers": EXPECTED_PRETRAINED_ENCODER_LAYERS,
        "pretrained_encoder_num_hidden_layers": EXPECTED_PRETRAINED_ENCODER_LAYERS,
        "training_source_contract_checked": True,
        "training_data_kind": "raw_train",
    }
    mismatches = {
        key: (meta.get(key), value)
        for key, value in expected.items()
        if meta.get(key) != value
    }
    if mismatches:
        raise ValueError(f"压缩Teacher架构或数据元数据不符: {mismatches}")
    student_pretrained_fingerprint = pretrained_artifact_fingerprint(PRETRAINED_MODEL_DIR)
    teacher_pretrained_fingerprint = meta.get("pretrained_artifact_sha256")
    if teacher_pretrained_fingerprint != student_pretrained_fingerprint:
        raise ValueError("压缩Teacher与学生的预训练配置/词表指纹不一致")
    actual_hash = file_sha256(checkpoint)
    if meta.get("checkpoint_sha256") != actual_hash:
        raise ValueError("压缩Teacher权重与元数据哈希不一致")
    checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint))
    run_dir = os.path.dirname(checkpoint_dir) if os.path.basename(checkpoint_dir) == "best_models" else checkpoint_dir
    teacher_train_params_path = os.path.join(run_dir, "train_params.json")
    if not os.path.isfile(teacher_train_params_path):
        raise FileNotFoundError(f"压缩Teacher缺少train_params.json: {teacher_train_params_path}")
    with open(teacher_train_params_path, "r", encoding="utf-8") as f:
        teacher_train_params = json.load(f)
    expected_teacher_train_path = resolve_data_path(DISTILLATION_TEACHER_TRAIN_FILE)
    if not os.path.isfile(expected_teacher_train_path):
        raise FileNotFoundError(f"找不到声明的Teacher原始训练文件: {expected_teacher_train_path}")
    expected_teacher_train_hash = file_sha256(expected_teacher_train_path)
    completion_expected = {
        **expected,
        "epochs": EPOCHS,
        "actual_completed_epochs": EPOCHS,
        "completed_full_configured_epochs": True,
        "train_role": DISTILLATION_TEACHER_TRAIN_ROLE,
        "seed": DISTILLATION_TEACHER_SEED,
        "concat_prob": (
            CONCAT_PROB if DISTILLATION_TEACHER_USE_SEQUENCE_CONCAT else 0.0
        ),
        "position_shift_prob": (
            POSITION_SHIFT_PROB if DISTILLATION_TEACHER_USE_POSITION_SHIFT else 0.0
        ),
        "token_mask_prob": (
            TOKEN_MASK_PROB if DISTILLATION_TEACHER_USE_TOKEN_MASK else 0.0
        ),
        "train_file_sha256": expected_teacher_train_hash,
        "dev_file_sha256": file_sha256(eval_cme_path),
    }
    completion_mismatches = {
        key: (teacher_train_params.get(key), value)
        for key, value in completion_expected.items()
        if teacher_train_params.get(key) != value
    }
    if completion_mismatches:
        raise ValueError(
            "压缩Teacher未通过原始Train/完整10轮/同Dev证据核验："
            f"{completion_mismatches}"
        )
    meta = dict(meta)
    meta["validated_teacher_role"] = teacher_role
    meta["validated_teacher_protocol"] = (
        "12l_wokd_egp_raw_concat_legacytokenmask_no_er_no_position_shift"
    )
    return (
        meta,
        actual_hash,
        file_sha256(meta_path),
        teacher_train_params_path,
        file_sha256(teacher_train_params_path),
    )


def _valid_span_mask(student_logits, teacher_logits, attention_mask):
    """只蒸馏模型实际允许的非 padding、非特殊符号、上三角 span。"""
    seq_len = student_logits.size(-1)
    valid_tokens = attention_mask.bool()
    mask = valid_tokens.unsqueeze(1).unsqueeze(3) & valid_tokens.unsqueeze(1).unsqueeze(2)
    upper = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=student_logits.device)
    )
    mask = mask & upper.unsqueeze(0).unsqueeze(0)
    # GlobalPointer 使用 -1e12 屏蔽 CLS/SEP/padding/下三角；同时检查两模型，
    # 防止无效位置进入 softmax 后污染分布。
    mask = mask & (student_logits > -5e11) & (teacher_logits > -5e11)
    return mask


def _masked_distribution_kl(student_scores, teacher_scores, mask):
    """最后一维上的 masked KL；无合法候选的行被显式排除。"""
    valid_rows = mask.any(dim=-1)
    if not valid_rows.any():
        return student_scores.sum() * 0.0
    student_rows = student_scores[valid_rows]
    teacher_rows = teacher_scores[valid_rows]
    mask_rows = mask[valid_rows]
    minus_inf = torch.finfo(student_rows.dtype).min
    student_log_prob = F.log_softmax(
        student_rows.masked_fill(~mask_rows, minus_inf),
        dim=-1,
    )
    teacher_prob = F.softmax(
        teacher_rows.masked_fill(~mask_rows, minus_inf),
        dim=-1,
    )
    return F.kl_div(
        student_log_prob,
        teacher_prob,
        reduction="none",
    ).sum(dim=-1).mean()


def span_distribution_distillation_loss(
    student_logits,
    teacher_logits,
    valid_span_mask,
    temperature,
):
    """按实体类型对全部合法 span 的相对分布做温度 KL 蒸馏。"""
    batch_size, ent_types = student_logits.shape[:2]
    student_flat = (student_logits / temperature).reshape(batch_size * ent_types, -1)
    teacher_flat = (teacher_logits / temperature).reshape(batch_size * ent_types, -1)
    mask_flat = valid_span_mask.reshape(batch_size * ent_types, -1)
    return (
        _masked_distribution_kl(student_flat, teacher_flat, mask_flat)
        * (temperature ** 2)
    )


def boundary_distribution_distillation_loss(
    student_logits,
    teacher_logits,
    valid_span_mask,
    temperature,
):
    """对 span 分布沿 end/start 聚合，分别蒸馏起点与终点分布。"""
    minus_inf = torch.finfo(student_logits.dtype).min
    student_scaled = (student_logits / temperature).masked_fill(
        ~valid_span_mask, minus_inf
    )
    teacher_scaled = (teacher_logits / temperature).masked_fill(
        ~valid_span_mask, minus_inf
    )
    student_start = torch.logsumexp(student_scaled, dim=-1)
    teacher_start = torch.logsumexp(teacher_scaled, dim=-1)
    start_mask = valid_span_mask.any(dim=-1)
    student_end = torch.logsumexp(student_scaled, dim=-2)
    teacher_end = torch.logsumexp(teacher_scaled, dim=-2)
    end_mask = valid_span_mask.any(dim=-2)
    start_loss = _masked_distribution_kl(
        student_start,
        teacher_start,
        start_mask,
    )
    end_loss = _masked_distribution_kl(
        student_end,
        teacher_end,
        end_mask,
    )
    return 0.5 * (start_loss + end_loss) * (temperature ** 2)


def nested_joint_confidence_distillation_loss(
    student_logits,
    teacher_logits,
    labels,
    valid_span_mask,
    temperature,
    top_k,
):
    """蒸馏严格包含 span 对的联合置信度；候选为 Gold 与 Teacher Top-K 并集。"""
    if top_k <= 0:
        raise ValueError("DISTILLATION_NESTED_TOP_K 必须为正整数")
    batch_size, _ent_types, seq_len, _ = student_logits.shape
    student_prob = torch.sigmoid(student_logits / temperature)
    teacher_prob = torch.sigmoid(teacher_logits / temperature)
    squared_error_sum = student_logits.sum() * 0.0
    nested_pair_count = 0

    for batch_index in range(batch_size):
        gold_candidates = torch.nonzero(
            labels[batch_index] > 0,
            as_tuple=False,
        )
        if gold_candidates.numel() > 0:
            gold_is_valid = valid_span_mask[
                batch_index,
                gold_candidates[:, 0],
                gold_candidates[:, 1],
                gold_candidates[:, 2],
            ]
            gold_candidates = gold_candidates[gold_is_valid]

        flat_mask = valid_span_mask[batch_index].reshape(-1)
        valid_count = int(flat_mask.sum().item())
        k = min(int(top_k), valid_count)
        top_candidates = torch.empty(
            (0, 3),
            dtype=torch.long,
            device=student_logits.device,
        )
        if k > 0:
            flat_scores = teacher_logits[batch_index].detach().reshape(-1)
            masked_scores = flat_scores.masked_fill(~flat_mask, -torch.inf)
            top_indices = torch.topk(masked_scores, k=k, sorted=False).indices
            spans_per_type = seq_len * seq_len
            ent_type = torch.div(
                top_indices,
                spans_per_type,
                rounding_mode="floor",
            )
            remainder = top_indices.remainder(spans_per_type)
            start = torch.div(remainder, seq_len, rounding_mode="floor")
            end = remainder.remainder(seq_len)
            top_candidates = torch.stack((ent_type, start, end), dim=-1)

        candidates = torch.cat((gold_candidates, top_candidates), dim=0)
        if candidates.numel() == 0:
            continue
        candidates = torch.unique(candidates, dim=0)
        if candidates.size(0) < 2:
            continue

        entity_types = candidates[:, 0]
        starts = candidates[:, 1]
        ends = candidates[:, 2]
        # 行为 outer，列为 inner；同边界不同类型不算严格包含。
        strictly_contains = (
            (starts.unsqueeze(1) <= starts.unsqueeze(0))
            & (ends.unsqueeze(0) <= ends.unsqueeze(1))
            & (
                (starts.unsqueeze(1) != starts.unsqueeze(0))
                | (ends.unsqueeze(1) != ends.unsqueeze(0))
            )
        )
        if not strictly_contains.any():
            continue

        student_candidates = student_prob[
            batch_index,
            entity_types,
            starts,
            ends,
        ]
        teacher_candidates = teacher_prob[
            batch_index,
            entity_types,
            starts,
            ends,
        ]
        student_joint = (
            student_candidates.unsqueeze(1)
            * student_candidates.unsqueeze(0)
        )[strictly_contains]
        teacher_joint = (
            teacher_candidates.unsqueeze(1)
            * teacher_candidates.unsqueeze(0)
        )[strictly_contains]
        squared_error_sum = squared_error_sum + F.mse_loss(
            student_joint,
            teacher_joint,
            reduction="sum",
        )
        nested_pair_count += int(strictly_contains.sum().item())

    if nested_pair_count == 0:
        return student_logits.sum() * 0.0
    return (
        squared_error_sum / nested_pair_count
    ) * (temperature ** 2)

def calc_per_class_counts(y_pred, y_true, threshold=0.0):
    """
    返回每个类别的 TP/Pred/Gold 计数，用于计算类别级 P/R/F1。
    y_pred/y_true shape: (batch, ent_type_size, seq_len, seq_len)
    """
    y_pred_np = y_pred.data.cpu().numpy()
    y_true_np = y_true.data.cpu().numpy()
    ent_type_size = y_pred_np.shape[1]
    tp = np.zeros(ent_type_size, dtype=np.int64)
    pred = np.zeros(ent_type_size, dtype=np.int64)
    gold = np.zeros(ent_type_size, dtype=np.int64)

    for cls_id in range(ent_type_size):
        pred_set = set(
            (b, s, e) for b, s, e in zip(*np.where(y_pred_np[:, cls_id, :, :] > threshold))
        )
        gold_set = set(
            (b, s, e) for b, s, e in zip(*np.where(y_true_np[:, cls_id, :, :] > 0))
        )
        tp[cls_id] += len(pred_set & gold_set)
        pred[cls_id] += len(pred_set)
        gold[cls_id] += len(gold_set)

    return tp, pred, gold


def _entity_signature(entity):
    return (
        entity["type"],
        int(entity["start_idx"]),
        int(entity["end_idx"]),
    )


def _entity_buckets(entities):
    """按当前实体集合计算嵌套角色与字符长度分桶。"""
    signatures = [_entity_signature(e) for e in entities]
    buckets = {
        "nested": set(),
        "non_nested": set(),
        "inner": set(),
        "outer": set(),
        "crossing_overlap": set(),
        "length_1_2": set(),
        "length_3_5": set(),
        "length_6_10": set(),
        "length_11_plus": set(),
        "nesting_depth_1": set(),
        "nesting_depth_2": set(),
        "nesting_depth_3_plus": set(),
    }
    for i, sig in enumerate(signatures):
        _, s, e = sig
        is_inner = False
        is_outer = False
        is_crossing = False
        for j, other in enumerate(signatures):
            if i == j:
                continue
            _, os_, oe_ = other
            if os_ <= s and e <= oe_ and (os_, oe_) != (s, e):
                is_inner = True
            if s <= os_ and oe_ <= e and (os_, oe_) != (s, e):
                is_outer = True
            if s < oe_ and os_ < e:
                contains_either = (
                    (os_ <= s and e <= oe_)
                    or (s <= os_ and oe_ <= e)
                )
                if not contains_either:
                    is_crossing = True
        if is_inner or is_outer:
            buckets["nested"].add(sig)
        else:
            buckets["non_nested"].add(sig)
        if is_inner:
            buckets["inner"].add(sig)
        if is_outer:
            buckets["outer"].add(sig)
        if is_crossing:
            buckets["crossing_overlap"].add(sig)

        depth = 1 + sum(
            1
            for j, other in enumerate(signatures)
            if i != j
            and other[1] <= s
            and e <= other[2]
            and (other[1], other[2]) != (s, e)
        )
        if depth == 1:
            buckets["nesting_depth_1"].add(sig)
        elif depth == 2:
            buckets["nesting_depth_2"].add(sig)
        else:
            buckets["nesting_depth_3_plus"].add(sig)

        length = e - s
        if length <= 2:
            buckets["length_1_2"].add(sig)
        elif length <= 5:
            buckets["length_3_5"].add(sig)
        elif length <= 10:
            buckets["length_6_10"].add(sig)
        else:
            buckets["length_11_plus"].add(sig)
    return buckets


def _accumulate_structure_counts(structure, pred_entities, gold_entities):
    """正确匹配按 Gold 属性归桶，未匹配预测才按预测侧属性计 FP。"""
    pred_set = {_entity_signature(e) for e in pred_entities}
    gold_set = {_entity_signature(e) for e in gold_entities}
    matched = pred_set & gold_set
    pred_buckets = _entity_buckets(pred_entities)
    gold_buckets = _entity_buckets(gold_entities)
    for name in structure:
        gold_bucket = gold_buckets[name]
        matched_in_bucket = matched & gold_bucket
        false_positives_in_pred_bucket = pred_buckets[name] - matched
        structure[name]["pred"] += len(false_positives_in_pred_bucket | matched_in_bucket)
        structure[name]["gold"] += len(gold_bucket)
        structure[name]["tp"] += len(matched_in_bucket)


def evaluate_dev_with_prediction_pipeline(model, tokenizer, eval_records):
    """用与最终测试相同的字符级滑窗预测进行Dev选模与分桶分析。"""
    texts = [row["text"] for row in eval_records]
    predictions = predict_batch(
        texts,
        tokenizer,
        model,
        max_len=MAX_LEN,
        batch_size=BATCH_SIZE,
        threshold=DECODE_THRESHOLD,
        stride=PREDICT_STRIDE if USE_SLIDING_WINDOW_DEV_EVAL else 0,
    )

    total_tp = total_pred = total_gold = 0
    cls_tp = np.zeros(ENT_CLS_NUM, dtype=np.int64)
    cls_pred = np.zeros(ENT_CLS_NUM, dtype=np.int64)
    cls_gold = np.zeros(ENT_CLS_NUM, dtype=np.int64)
    name_to_id = {name: idx for idx, name in id2ent.items()}
    structure = {
        name: {"tp": 0, "pred": 0, "gold": 0}
        for name in (
            "nested", "non_nested", "inner", "outer", "crossing_overlap",
            "length_1_2", "length_3_5", "length_6_10", "length_11_plus",
            "nesting_depth_1", "nesting_depth_2", "nesting_depth_3_plus",
        )
    }

    for pred_entities, row in zip(predictions, eval_records):
        gold_entities = row.get("entities", [])
        pred_set = {_entity_signature(e) for e in pred_entities}
        gold_set = {_entity_signature(e) for e in gold_entities}
        matched = pred_set & gold_set
        total_tp += len(matched)
        total_pred += len(pred_set)
        total_gold += len(gold_set)

        for type_name, _s, _e in pred_set:
            if type_name in name_to_id:
                cls_pred[name_to_id[type_name]] += 1
        for type_name, _s, _e in gold_set:
            if type_name in name_to_id:
                cls_gold[name_to_id[type_name]] += 1
        for type_name, _s, _e in matched:
            if type_name in name_to_id:
                cls_tp[name_to_id[type_name]] += 1

        # 否则“内层实体预测正确但外层漏掉”会因预测侧暂时
        # 呈 non-nested 而在相关桶中丢失 TP。
        _accumulate_structure_counts(structure, pred_entities, gold_entities)

    precision = total_tp / total_pred if total_pred else 0.0
    recall = total_tp / total_gold if total_gold else 0.0
    f1 = 2 * total_tp / (total_pred + total_gold) if total_pred + total_gold else 0.0
    return f1, precision, recall, cls_tp, cls_pred, cls_gold, structure

def count_model_parameters(model):
    """
    统计并打印模型的参数量（纯 RoBERTa + GlobalPointer 架构）。
    返回可写入 train_params.json 的字典（精确个数与百万单位约数）。
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    bigru_params = 0
    if getattr(model, "bigru", None) is not None:
        bigru_params = sum(p.numel() for p in model.bigru.parameters())

    task_head_params = 0
    if hasattr(model, "q_dense") and model.q_dense is not None:
        task_head_params += sum(p.numel() for p in model.q_dense.parameters())
    if hasattr(model, "k_dense") and model.k_dense is not None:
        task_head_params += sum(p.numel() for p in model.k_dense.parameters())
    if hasattr(model, "cls_qw") and model.cls_qw is not None:
        task_head_params += sum(p.numel() for p in model.cls_qw.parameters())
    if hasattr(model, "cls_kw") and model.cls_kw is not None:
        task_head_params += sum(p.numel() for p in model.cls_kw.parameters())
    if hasattr(model, "dense") and model.dense is not None:
        task_head_params += sum(p.numel() for p in model.dense.parameters())

    print("\n" + "=" * 45)
    print("模型参数量评估指标")
    print("=" * 45)
    print(f"总参数量 (Total):             {total_params / 1e6:.4f} M")
    print(f"可训练参数量 (Trainable):      {trainable_params / 1e6:.4f} M")
    print(f"预训练底座 (Encoder):          {encoder_params / 1e6:.4f} M")
    if bigru_params > 0:
        print(f"RNN层 (BiGRU):               {bigru_params / 1e6:.4f} M")
    print(f"顶层任务头 (Task Head):        {task_head_params / 1e6:.4f} M")
    print("=" * 45 + "\n")

    return {
        "param_total": int(total_params),
        "param_trainable": int(trainable_params),
        "param_encoder": int(encoder_params),
        "param_bigru": int(bigru_params),
        "param_task_head": int(task_head_params),
        "param_total_m": round(total_params / 1e6, 4),
        "param_trainable_m": round(trainable_params / 1e6, 4),
    }

def main():
    # 设置随机种子，确保可复现
    set_seed(SEED)
    if TRAIN_ROLE not in {"experiment", DISTILLATION_TEACHER_TRAIN_ROLE}:
        raise ValueError(f"未知TRAIN_ROLE: {TRAIN_ROLE!r}")
    if USE_NESTED_STRUCTURE_DISTILLATION and TRAIN_ROLE == DISTILLATION_TEACHER_TRAIN_ROLE:
        raise ValueError("distillation_teacher必须是无KD的完整深度监督模型")
    if TRAIN_ROLE == DISTILLATION_TEACHER_TRAIN_ROLE:
        expected_teacher_train_path = resolve_data_path(DISTILLATION_TEACHER_TRAIN_FILE)
        expected_settings = {
            "gp_efficient": (GP_EFFICIENT, DISTILLATION_TEACHER_GP_EFFICIENT),
            "use_bigru": (USE_BIGRU, DISTILLATION_TEACHER_USE_BIGRU),
            "use_eight_layer_encoder": (USE_EIGHT_LAYER_ENCODER, False),
            "use_sequence_concat": (USE_SEQUENCE_CONCAT, DISTILLATION_TEACHER_USE_SEQUENCE_CONCAT),
            "use_position_shift": (USE_POSITION_SHIFT, DISTILLATION_TEACHER_USE_POSITION_SHIFT),
            "use_token_mask": (USE_TOKEN_MASK, DISTILLATION_TEACHER_USE_TOKEN_MASK),
            "token_mask_scope": (TOKEN_MASK_SCOPE, DISTILLATION_TEACHER_TOKEN_MASK_SCOPE),
            "legacy_concat_routing": (
                LEGACY_CONCAT_ROUTING,
                DISTILLATION_TEACHER_LEGACY_CONCAT_ROUTING,
            ),
            "train_file": (os.path.abspath(train_cme_path), expected_teacher_train_path),
        }
        mismatches = {k: v for k, v in expected_settings.items() if v[0] != v[1]}
        if mismatches:
            raise ValueError(
                "distillation_teacher必须为12层Efficient GP、原始Train、无KD，且"
                "Sequence Concatenation/Position Shift/Token Mask必须与声明的Teacher协议一致："
                f"{mismatches}"
            )
    existing_run_artifacts = [
        path for path in (
            TRAIN_LOG_CSV,
            TRAIN_STEP_LOG_CSV,
            TRAIN_CLASS_LOG_CSV,
            TRAIN_STRUCTURE_LOG_CSV,
            ONLINE_AUGMENTATION_LOG_CSV,
            TRAIN_PARAMS_JSON,
            BEST_MODEL_PATH,
            BEST_MODEL_PATH + ".meta.json",
            BEST_MODELS_META,
        )
        if os.path.exists(path)
    ]
    if os.path.isdir(BEST_MODELS_DIR):
        existing_run_artifacts.extend(
            os.path.join(BEST_MODELS_DIR, name)
            for name in sorted(os.listdir(BEST_MODELS_DIR))
            if name.endswith(".pth")
        )
    if existing_run_artifacts and not ALLOW_RUN_DIR_REUSE:
        raise FileExistsError(
            "当前RUN_NAME目录已有实验产物，为避免日志和top-k模型混入旧Run，已停止训练。\n"
            f"请设置新的RUN_NAME，或明确设置ALLOW_RUN_DIR_REUSE=1。已有文件: {existing_run_artifacts}"
        )
    
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_DIR, use_fast=True)

    raw_train_records = validate_cmeee_records(
        load_dataset_records(train_cme_path), source=train_cme_path
    )
    training_source_audit = audit_raw_training_source(
        raw_train_records, train_cme_path
    )
    # 审计字段记录“本次运行是否真正完成了Teacher来源契约核验”。
    # Teacher自身会在入口处核验其训练协议；KD Student则在加载Teacher元数据后
    # 才把该字段更新为True，避免此前“实际已核验但日志仍写False”的歧义。
    teacher_source_audit = {
        "distillation_teacher_source_contract_checked": (
            TRAIN_ROLE == DISTILLATION_TEACHER_TRAIN_ROLE
        ),
        "distillation_teacher_protocol": (
            (
                "12l_wokd_egp_raw_concat_legacytokenmask_no_er_position_shift"
                if DISTILLATION_TEACHER_USE_POSITION_SHIFT
                else "12l_wokd_egp_raw_concat_legacytokenmask_no_er_no_position_shift"
            )
            if TRAIN_ROLE == DISTILLATION_TEACHER_TRAIN_ROLE
            else None
        ),
    }

    train_data_pool = load_data(train_cme_path)
    if REMOVE_HOLDOUT_TEXT_OVERLAP:
        train_data_pool, holdout_decontamination = remove_holdout_text_overlap(
            train_data_pool,
            [
                eval_cme_path,
                os.path.join(DATA_DIR, TEST_FILE),
            ],
        )
    else:
        holdout_decontamination = {
            "holdout_text_decontamination_enabled": False,
            "train_rows_before_holdout_decontamination": len(train_data_pool),
            "train_rows_removed_by_exact_holdout_text": 0,
            "train_rows_after_holdout_decontamination": len(train_data_pool),
        }
    train_window_audit = audit_training_window_gold_coverage(
        train_data_pool, tokenizer, MAX_LEN, TRAIN_STRIDE
    )
    if train_window_audit["train_gold_entities_uncovered_by_sliding_windows"]:
        print(
            "[warning] 重叠滑窗后仍有 "
            f"{train_window_audit['train_gold_entities_uncovered_by_sliding_windows']} 个 Gold 无法完整容纳，涉及 "
            f"{train_window_audit['train_samples_with_uncovered_gold']} 条样本。"
        )
    augmenter = None
    if USE_SEQUENCE_CONCAT or USE_POSITION_SHIFT:
        augmenter = SequenceAugmenter(
            data_pool=train_data_pool,
            max_len=MAX_LEN,
            use_concat=USE_SEQUENCE_CONCAT,
            use_position_shift=USE_POSITION_SHIFT,
            # 槽位概率始终固定；关闭模块只把对应槽位变成原样本。
            concat_prob=CONCAT_PROB,
            position_shift_prob=POSITION_SHIFT_PROB,
            tokenizer=tokenizer,
            concat_separator=SEQUENCE_CONCAT_SEPARATOR,
            position_shift_prefixes=POSITION_SHIFT_PREFIXES,
            legacy_concat_routing=LEGACY_CONCAT_ROUTING,
            v12_student_concat_routing=V12_STUDENT_CONCAT_ROUTING,
        )
    ner_train = EntDataset(
        train_data_pool,
        tokenizer=tokenizer,
        istrain=True,
        augmenter=augmenter,
        token_mask_prob=TOKEN_MASK_PROB if USE_TOKEN_MASK else 0.0,
        token_mask_scope=TOKEN_MASK_SCOPE,
        max_len=MAX_LEN,
        train_stride=TRAIN_STRIDE,
    )
    if len(ner_train) != train_window_audit["train_windows_per_epoch"]:
        raise RuntimeError(
            "训练滑窗索引数与审计数不一致: "
            f"dataset={len(ner_train)}, audit={train_window_audit['train_windows_per_epoch']}"
        )
    # V12 Student真实代码没有给DataLoader传独立generator；其shuffle/base-seed
    # 来自set_seed后的PyTorch全局生成器。V16.3恢复该路径，但Teacher继续保留
    # V16.2已实测75.3985%的独立generator轨迹。
    train_loader_kwargs = {}
    if USE_EXPLICIT_DATALOADER_GENERATOR:
        train_loader_generator = torch.Generator()
        train_loader_generator.manual_seed(SEED)
        train_loader_kwargs["generator"] = train_loader_generator
    ner_loader_train = DataLoader(
        ner_train,
        batch_size=BATCH_SIZE,
        collate_fn=ner_train.collate,
        shuffle=True,
        num_workers=NUM_WORKERS,
        worker_init_fn=worker_init_fn,
        **train_loader_kwargs,
    )
    eval_records = validate_cmeee_records(
        load_dataset_records(eval_cme_path), source=eval_cme_path
    )

    pretrained_encoder_config = AutoConfig.from_pretrained(PRETRAINED_MODEL_DIR)
    pretrained_encoder_layers = getattr(
        pretrained_encoder_config,
        "num_hidden_layers",
        None,
    )
    if pretrained_encoder_layers is None:
        raise ValueError("预训练模型 config 不包含 num_hidden_layers")
    pretrained_encoder_layers = int(pretrained_encoder_layers)
    if pretrained_encoder_layers != EXPECTED_PRETRAINED_ENCODER_LAYERS:
        raise ValueError(
            "本实验要求 12 层 chinese-roberta-wwm-ext 底座，但当前预训练配置为 "
            f"{pretrained_encoder_layers} 层；请恢复正确底座，不能把其他深度写成12层实验"
        )
    pretrained_fingerprint = pretrained_artifact_fingerprint(
        PRETRAINED_MODEL_DIR
    )
    encoder_num_hidden_layers = (
        int(STUDENT_NUM_HIDDEN_LAYERS)
        if USE_EIGHT_LAYER_ENCODER
        else pretrained_encoder_layers
    )
    if USE_EIGHT_LAYER_ENCODER:
        if encoder_num_hidden_layers != 8:
            raise ValueError(
                "USE_EIGHT_LAYER_ENCODER=1 表示严格使用 8 层；"
                f"当前 STUDENT_NUM_HIDDEN_LAYERS={encoder_num_hidden_layers}"
            )
        # 只验证结构压缩；是否加载 Teacher 由独立 KD 开关决定。
        select_teacher_layer_indices(
            pretrained_encoder_layers,
            encoder_num_hidden_layers,
        )
    if USE_NESTED_STRUCTURE_DISTILLATION:
        if DISTILLATION_TEMPERATURE <= 0:
            raise ValueError("DISTILLATION_TEMPERATURE 必须大于 0")
        distillation_weights = {
            "span": DISTILLATION_SPAN_WEIGHT,
            "boundary": DISTILLATION_BOUNDARY_WEIGHT,
            "nested": DISTILLATION_NESTED_WEIGHT,
        }
        if any(value < 0 for value in distillation_weights.values()):
            raise ValueError(f"蒸馏损失权重不能为负: {distillation_weights}")
        if not any(value > 0 for value in distillation_weights.values()):
            raise ValueError("至少一个蒸馏损失权重必须大于 0")
        if DISTILLATION_NESTED_TOP_K <= 0:
            raise ValueError("DISTILLATION_NESTED_TOP_K 必须为正整数")
        if DISTILLATION_TEACHER_USE_BIGRU != USE_BIGRU:
            raise ValueError(
                "压缩 Teacher 与学生必须使用相同 BiGRU 设置："
                f"teacher={DISTILLATION_TEACHER_USE_BIGRU}, student={USE_BIGRU}"
            )
    encoder, pretrained_layer_mapping = build_encoder_from_pretrained(
        PRETRAINED_MODEL_DIR,
        encoder_num_hidden_layers,
    )
    model = GlobalPointer(
        encoder,
        ENT_CLS_NUM,
        GP_HEAD_SIZE,
        rnn_hidden_size=RNN_HIDDEN_SIZE,
        rnn_layers=RNN_LAYERS,
        efficient=GP_EFFICIENT,
        use_bigru=USE_BIGRU,
        sep_token_id=tokenizer.sep_token_id,
    ).to(device) #9个实体类型

    distillation_teacher = None
    distillation_teacher_device = None
    distillation_info = {
        "use_eight_layer_encoder": USE_EIGHT_LAYER_ENCODER,
        "use_nested_structure_distillation": USE_NESTED_STRUCTURE_DISTILLATION,
        "pretrained_encoder_num_hidden_layers": pretrained_encoder_layers,
        "encoder_num_hidden_layers": encoder_num_hidden_layers,
        "pretrained_layer_mapping": pretrained_layer_mapping,
        "student_depth_and_kd_switches_independent": True,
        "distillation_teacher_checkpoint": None,
        "distillation_teacher_checkpoint_sha256": None,
        "distillation_teacher_meta_sha256": None,
        "distillation_teacher_train_params": None,
        "distillation_teacher_train_params_sha256": None,
        "distillation_teacher_encoder_num_hidden_layers": None,
        "distillation_teacher_source_role": None,
        "distillation_teacher_reuses_paper_baseline": False,
        "distillation_teacher_layer_mapping": None,
        "distillation_teacher_weight_initialization_applied": False,
        "distillation_student_initialization_policy": (
            "pretrained_even_layer_mapping"
            if USE_EIGHT_LAYER_ENCODER
            else "full_pretrained_encoder"
        ),
        "distillation_temperature": (
            DISTILLATION_TEMPERATURE
            if USE_NESTED_STRUCTURE_DISTILLATION
            else None
        ),
        "distillation_span_weight": (
            DISTILLATION_SPAN_WEIGHT
            if USE_NESTED_STRUCTURE_DISTILLATION
            else 0.0
        ),
        "distillation_boundary_weight": (
            DISTILLATION_BOUNDARY_WEIGHT
            if USE_NESTED_STRUCTURE_DISTILLATION
            else 0.0
        ),
        "distillation_nested_weight": (
            DISTILLATION_NESTED_WEIGHT
            if USE_NESTED_STRUCTURE_DISTILLATION
            else 0.0
        ),
        "distillation_nested_top_k": (
            DISTILLATION_NESTED_TOP_K
            if USE_NESTED_STRUCTURE_DISTILLATION
            else None
        ),
        "distillation_span_loss": "masked_span_distribution_kl",
        "distillation_boundary_loss": "masked_start_end_distribution_kl",
        "distillation_nested_loss": "strict_containment_joint_confidence_mse_gold_union_teacher_topk",
        "distillation_uses_confidence_threshold": False,
        "distillation_adversarial_branch_policy": "clean_branch_only; adversarial_branch_remains_supervised",
    }
    if USE_NESTED_STRUCTURE_DISTILLATION:
        (
            distillation_teacher_meta,
            distillation_teacher_sha256,
            distillation_teacher_meta_sha256,
            distillation_teacher_train_params_path,
            distillation_teacher_train_params_sha256,
        ) = validate_distillation_teacher_metadata(
            DISTILLATION_TEACHER_CHECKPOINT
        )
        # validate_distillation_teacher_metadata 已逐项核验数据集、12L/EGP/无KD、
        # Teacher训练数据、seed、增强配置、完整epoch、预训练指纹及权重哈希。
        # 只有成功返回后才记录 contract_checked=True。
        teacher_source_audit.update({
            "distillation_teacher_source_contract_checked": True,
            "distillation_teacher_protocol": distillation_teacher_meta.get(
                "validated_teacher_protocol"
            ),
        })
        distillation_teacher_device = torch.device(
            DISTILLATION_TEACHER_DEVICE
        )
        _, distillation_teacher = load_prediction_model(
            weights_path=DISTILLATION_TEACHER_CHECKPOINT,
            target_device=distillation_teacher_device,
            efficient=DISTILLATION_TEACHER_GP_EFFICIENT,
            use_bigru=DISTILLATION_TEACHER_USE_BIGRU,
            encoder_num_hidden_layers=distillation_teacher_meta.get(
                "encoder_num_hidden_layers"
            ),
        )
        _, teacher_layers = _transformer_parts(distillation_teacher.encoder)
        teacher_encoder_layers = len(teacher_layers)
        if teacher_encoder_layers != pretrained_encoder_layers:
            raise ValueError(
                "压缩 Teacher 必须是当前预训练底座的完整深度模型："
                f"teacher={teacher_encoder_layers}, pretrained={pretrained_encoder_layers}"
            )
        teacher_task_head_isomorphic = (
            DISTILLATION_TEACHER_GP_EFFICIENT == GP_EFFICIENT
            and DISTILLATION_TEACHER_USE_BIGRU == USE_BIGRU
        )
        if encoder_num_hidden_layers < teacher_encoder_layers:
            layer_mapping = initialize_student_from_teacher(
                model,
                distillation_teacher,
                copy_task_head=teacher_task_head_isomorphic,
            )
            teacher_weight_initialization_applied = True
            student_initialization_policy = (
                "trained_teacher_embeddings_even_layers_and_task_head"
                if teacher_task_head_isomorphic
                else "trained_teacher_embeddings_even_layers_only_cross_head_kd"
            )
        elif encoder_num_hidden_layers == teacher_encoder_layers:
            # 两个开关真正独立：允许 12层+KD。此时若把完整 Teacher 权重
            # 原样复制给学生，初始蒸馏误差接近 0；因此保留学生的预训练初始化，
            # 仅让训练中的 Teacher soft targets 提供监督。
            layer_mapping = list(range(teacher_encoder_layers))
            teacher_weight_initialization_applied = False
            student_initialization_policy = (
                "full_pretrained_encoder_without_teacher_weight_copy"
            )
        else:
            raise ValueError(
                "学生编码器深度不能超过 Teacher："
                f"student={encoder_num_hidden_layers}, teacher={teacher_encoder_layers}"
            )
        distillation_teacher.requires_grad_(False)
        distillation_teacher.eval()
        distillation_info.update({
            "distillation_teacher_checkpoint": os.path.abspath(
                DISTILLATION_TEACHER_CHECKPOINT
            ),
            "distillation_teacher_checkpoint_sha256": distillation_teacher_sha256,
            "distillation_teacher_meta_sha256": distillation_teacher_meta_sha256,
            "distillation_teacher_train_params": os.path.abspath(
                distillation_teacher_train_params_path
            ),
            "distillation_teacher_train_params_sha256": (
                distillation_teacher_train_params_sha256
            ),
            "distillation_teacher_encoder_num_hidden_layers": teacher_encoder_layers,
            "distillation_teacher_source_role": distillation_teacher_meta.get(
                "validated_teacher_role", distillation_teacher_meta.get("train_role")
            ),
            "distillation_teacher_reuses_paper_baseline": bool(
                distillation_teacher_meta.get(
                    "reuses_paper_baseline_as_standard_teacher", False
                )
            ),
            "distillation_teacher_layer_mapping": layer_mapping,
            "distillation_teacher_weight_initialization_applied": (
                teacher_weight_initialization_applied
            ),
            "distillation_student_initialization_policy": (
                student_initialization_policy
            ),
            "distillation_teacher_device": str(distillation_teacher_device),
            "distillation_teacher_gp_efficient": DISTILLATION_TEACHER_GP_EFFICIENT,
            "distillation_teacher_use_bigru": DISTILLATION_TEACHER_USE_BIGRU,
            "distillation_teacher_task_head_isomorphic": (
                teacher_task_head_isomorphic
            ),
            "distillation_teacher_protocol": (
                "12l_wokd_egp_raw_concat_legacytokenmask_no_er_no_position_shift"
            ),
        })
        print(
            "Nested-structure distillation enabled: "
            f"teacher_layers={teacher_encoder_layers}, "
            f"student_layers={encoder_num_hidden_layers}, "
            f"mapping={layer_mapping}"
        )
    param_stats = count_model_parameters(model)
    if SAVE_TOPK_MODELS:
        print(
            f"Top-{TOPK_MODELS} checkpoint retention enabled: "
            f"{os.path.abspath(BEST_MODELS_DIR)}"
        )

    # 统一学习率（无 encoder / 任务头差分）
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # 学习率调度：所有数据规模均完整跑满 EPOCHS，增强组不再按原始集步数提前截断。
    natural_training_steps = len(ner_loader_train) * EPOCHS
    total_training_steps = natural_training_steps
    warmup_steps = int(WARMUP_RATIO * total_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps
    )

    # 对抗训练器（按配置启用）
    use_adversarial = ADV_TRAIN_METHOD.upper() in ['FGM', 'PGD']
    adversarial_trainer = None
    if use_adversarial:
        adversarial_trainer = AdversarialTrainer(
            model,
            method=ADV_TRAIN_METHOD,
            emb_name=ADV_EMB_NAME
        )

    # 即使首轮 F1 恰为 0，也必须保存一个可复现的最佳检查点。
    max_f = -1.0
    # 全部类别仍写入CSV；控制台Focus摘要只显示Baseline证据支持的弱势类。
    focus_types = set(FOCUS_CLASS_PRIORITY)
    save_params({
        "dataset_name": DATASET_NAME,
        "pretrained_model_dir": PRETRAINED_MODEL_DIR,
        "pretrained_artifact_sha256": pretrained_fingerprint,
        "run_name": RUN_NAME,
        "train_role": TRAIN_ROLE,
        "selection_split_role": SELECTION_SPLIT_ROLE,
        "seed": SEED,
        "train_file": train_cme_path,
        "train_file_sha256": file_sha256(train_cme_path),
        "dev_file": eval_cme_path,
        "dev_file_sha256": file_sha256(eval_cme_path),
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "max_len": MAX_LEN,
        "ent_cls_num": ENT_CLS_NUM,
        "gp_head_size": GP_HEAD_SIZE,
        "gp_efficient": GP_EFFICIENT,
        **distillation_info,
        "learning_rate": LEARNING_RATE,
        "num_workers": NUM_WORKERS,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "rnn_hidden_size": RNN_HIDDEN_SIZE,
        "rnn_layers": RNN_LAYERS,
        "use_bigru": USE_BIGRU,
        "adv_train_method": ADV_TRAIN_METHOD,
        "adv_emb_name": ADV_EMB_NAME,
        "fgm_epsilon": FGM_EPSILON,
        "pgd_epsilon": PGD_EPSILON,
        "pgd_alpha": PGD_ALPHA,
        "pgd_steps": PGD_STEPS,
        "pgd_start_epoch": PGD_START_EPOCH,
        "use_early_stopping": False,
        "save_topk_models": SAVE_TOPK_MODELS,
        "topk_models": TOPK_MODELS if SAVE_TOPK_MODELS else 0,
        "use_sequence_concat": USE_SEQUENCE_CONCAT,
        "use_position_shift": USE_POSITION_SHIFT,
        "use_token_mask": USE_TOKEN_MASK,
        "concat_prob": CONCAT_PROB if USE_SEQUENCE_CONCAT else 0.0,
        "position_shift_prob": POSITION_SHIFT_PROB if USE_POSITION_SHIFT else 0.0,
        "position_shift_prefixes": list(POSITION_SHIFT_PREFIXES),
        "structural_augmentation_router": (
            "teacher_legacy_dedicated_concat_rng"
            if LEGACY_CONCAT_ROUTING
            else (
                "v12_student_python_global_random_concat"
                if V12_STUDENT_CONCAT_ROUTING
                else "fixed_mutually_exclusive_slots_no_probability_reallocation"
            )
        ),
        "token_mask_prob": TOKEN_MASK_PROB if USE_TOKEN_MASK else 0.0,
        "token_mask_scope": TOKEN_MASK_SCOPE,
        "legacy_concat_routing": LEGACY_CONCAT_ROUTING,
        "v12_student_concat_routing": V12_STUDENT_CONCAT_ROUTING,
        "use_explicit_dataloader_generator": USE_EXPLICIT_DATALOADER_GENERATOR,
        "sequence_concat_separator": SEQUENCE_CONCAT_SEPARATOR,
        "position_shift_constraints": "single_recognized_chinese_punctuation_prefix;all_entity_offsets_shifted;surface_label_nesting_preserved;reject_overlength_or_tokenizer_ignored",
        "position_shift_kd_policy": "skip_all_kd_losses_for_shifted_windows",
        "online_augmentation_log_csv": os.path.abspath(
            ONLINE_AUGMENTATION_LOG_CSV
        ),
        "training_budget_policy": "full_epochs",
        "natural_training_steps": natural_training_steps,
        "total_training_steps": total_training_steps,
        "decode_threshold": DECODE_THRESHOLD,
        "train_stride": TRAIN_STRIDE,
        "predict_stride": PREDICT_STRIDE,
        "sliding_window_dev_eval": USE_SLIDING_WINDOW_DEV_EVAL,
        "structure_bucket_assignment": "matched_entities_by_gold_attributes_false_positives_by_predicted_attributes",
        "python_version": platform.python_version(),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "pytorch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        **train_window_audit,
        **holdout_decontamination,
        **training_source_audit,
        **teacher_source_audit,
        **param_stats,
    }, TRAIN_PARAMS_JSON)
    checkpoint_architecture_metadata = {
        "dataset_name": DATASET_NAME,
        "pretrained_model_dir": os.path.abspath(PRETRAINED_MODEL_DIR),
        "pretrained_artifact_sha256": pretrained_fingerprint,
        "gp_efficient": GP_EFFICIENT,
        "use_bigru": USE_BIGRU,
        "ent_cls_num": ENT_CLS_NUM,
        "gp_head_size": GP_HEAD_SIZE,
        "rnn_hidden_size": RNN_HIDDEN_SIZE,
        "rnn_layers": RNN_LAYERS,
        "encoder_num_hidden_layers": encoder_num_hidden_layers,
        "pretrained_encoder_num_hidden_layers": pretrained_encoder_layers,
        "use_eight_layer_encoder": USE_EIGHT_LAYER_ENCODER,
        "use_nested_structure_distillation": USE_NESTED_STRUCTURE_DISTILLATION,
        "use_sequence_concat": USE_SEQUENCE_CONCAT,
        "use_position_shift": USE_POSITION_SHIFT,
        "position_shift_prob": POSITION_SHIFT_PROB if USE_POSITION_SHIFT else 0.0,
        "position_shift_prefixes": list(POSITION_SHIFT_PREFIXES),
        "use_token_mask": USE_TOKEN_MASK,
        "token_mask_scope": TOKEN_MASK_SCOPE,
        "legacy_concat_routing": LEGACY_CONCAT_ROUTING,
        "v12_student_concat_routing": V12_STUDENT_CONCAT_ROUTING,
        "use_explicit_dataloader_generator": USE_EXPLICIT_DATALOADER_GENERATOR,
        "pretrained_layer_mapping": pretrained_layer_mapping,
        "distillation_teacher_checkpoint_sha256": distillation_info[
            "distillation_teacher_checkpoint_sha256"
        ],
        "distillation_teacher_weight_initialization_applied": distillation_info[
            "distillation_teacher_weight_initialization_applied"
        ],
        "distillation_student_initialization_policy": distillation_info[
            "distillation_student_initialization_policy"
        ],
        "max_len": MAX_LEN,
        "decode_threshold": DECODE_THRESHOLD,
        "predict_stride": PREDICT_STRIDE,
        "train_role": TRAIN_ROLE,
        "selection_split_role": SELECTION_SPLIT_ROLE,
        "train_file": os.path.abspath(train_cme_path),
        "dev_file": os.path.abspath(eval_cme_path),
        "train_file_sha256": file_sha256(train_cme_path),
        "dev_file_sha256": file_sha256(eval_cme_path),
        **teacher_source_audit,
        **training_source_audit,
    }
    global_step = 0
    completed_epochs = 0
    online_augmentation_totals = {}
    for eo in range(EPOCHS):
        total_loss = 0.
        online_augmentation_epoch = {}
        if augmenter is not None:
            augmenter.set_epoch(eo, EPOCHS)
        pbar = tqdm(ner_loader_train, desc="Train {}/{}".format(eo+1, EPOCHS))
        for idx, batch in enumerate(pbar):
            (
                raw_text_list,
                input_ids,
                attention_mask,
                segment_ids,
                labels,
                _,
                batch_online_stats,
                batch_position_shifted,
            ) = batch
            for key, value in batch_online_stats.items():
                online_augmentation_epoch[key] = (
                    online_augmentation_epoch.get(key, 0) + int(value)
                )
                online_augmentation_totals[key] = (
                    online_augmentation_totals.get(key, 0) + int(value)
                )
            input_ids, attention_mask, segment_ids, labels = input_ids.to(device), attention_mask.to(device), segment_ids.to(device), labels.to(device)
            batch_position_shifted = batch_position_shifted.to(device)
            optimizer.zero_grad()

            # 监督损失只来自原始人工标注数据。
            logits = model(input_ids, attention_mask, segment_ids)
            supervised_loss = loss_fun(logits, labels)
            if distillation_teacher is not None:
                # Teacher训练协议不包含Position Shifting。为避免Teacher软标签
                # 与平移视图产生未经验证的分布冲突，平移窗口只接受硬标签监督。
                kd_eligible_mask = ~batch_position_shifted
                kd_batch = int(kd_eligible_mask.sum().item())
                kd_skipped = int(batch_position_shifted.sum().item())
                for stats in (online_augmentation_epoch, online_augmentation_totals):
                    stats["kd_eligible_windows"] = stats.get("kd_eligible_windows", 0) + kd_batch
                    stats["kd_skipped_position_shift_windows"] = (
                        stats.get("kd_skipped_position_shift_windows", 0)
                        + kd_skipped
                    )
                if kd_batch > 0:
                    # 无Position Shift时直接走V12的整batch KD张量路径，避免对
                    # 全True布尔mask做无意义拷贝；有Shift时才筛选supervised-only窗口。
                    if kd_skipped == 0:
                        kd_input_ids = input_ids
                        kd_attention_mask = attention_mask
                        kd_segment_ids = segment_ids
                        student_kd_logits = logits
                        kd_labels = labels
                    else:
                        kd_input_ids = input_ids[kd_eligible_mask]
                        kd_attention_mask = attention_mask[kd_eligible_mask]
                        kd_segment_ids = segment_ids[kd_eligible_mask]
                        student_kd_logits = logits[kd_eligible_mask]
                        kd_labels = labels[kd_eligible_mask]
                    with torch.no_grad():
                        teacher_logits = distillation_teacher(
                            kd_input_ids.to(distillation_teacher_device),
                            kd_attention_mask.to(distillation_teacher_device),
                            kd_segment_ids.to(distillation_teacher_device),
                        ).to(device)
                    valid_span_mask = _valid_span_mask(
                        student_kd_logits,
                        teacher_logits,
                        kd_attention_mask,
                    )
                    span_kd_loss = span_distribution_distillation_loss(
                        student_kd_logits,
                        teacher_logits,
                        valid_span_mask,
                        DISTILLATION_TEMPERATURE,
                    )
                    boundary_kd_loss = boundary_distribution_distillation_loss(
                        student_kd_logits,
                        teacher_logits,
                        valid_span_mask,
                        DISTILLATION_TEMPERATURE,
                    )
                    nested_kd_loss = nested_joint_confidence_distillation_loss(
                        student_kd_logits,
                        teacher_logits,
                        kd_labels,
                        valid_span_mask,
                        DISTILLATION_TEMPERATURE,
                        DISTILLATION_NESTED_TOP_K,
                    )
                else:
                    zero_kd = logits.sum() * 0.0
                    span_kd_loss = zero_kd
                    boundary_kd_loss = zero_kd
                    nested_kd_loss = zero_kd
                loss = (
                    supervised_loss
                    + DISTILLATION_SPAN_WEIGHT * span_kd_loss
                    + DISTILLATION_BOUNDARY_WEIGHT * boundary_kd_loss
                    + DISTILLATION_NESTED_WEIGHT * nested_kd_loss
                )
            else:
                loss = supervised_loss
            loss.backward()

            # 对抗训练：前 PGD_START_EPOCH 个 epoch 仅标准监督，降低含噪增强数据上早期梯度紊乱
            current_use_adv = (
                use_adversarial and adversarial_trainer is not None and eo >= PGD_START_EPOCH
            )
            if current_use_adv:
                if ADV_TRAIN_METHOD.upper() == 'FGM':
                    adversarial_trainer.attack(epsilon=FGM_EPSILON)
                    logits_adv = model(input_ids, attention_mask, segment_ids)
                    loss_adv = loss_fun(logits_adv, labels)
                    loss_adv.backward()
                    adversarial_trainer.restore()
                elif ADV_TRAIN_METHOD.upper() == 'PGD':
                    # 步骤1: 备份干净样本的真实梯度 g_clean
                    grad_backup = {}
                    for name, param in model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            grad_backup[name] = param.grad.clone()

                    # 步骤2: K 步迭代寻找最坏扰动
                    for t in range(PGD_STEPS):
                        adversarial_trainer.attack(
                            epsilon=PGD_EPSILON,
                            alpha=PGD_ALPHA,
                            is_first_attack=(t == 0)
                        )

                        if t != PGD_STEPS - 1:
                            model.zero_grad()
                        else:
                            # 最后一步恢复干净梯度，再叠加最终对抗梯度。
                            for name, param in model.named_parameters():
                                if param.requires_grad and name in grad_backup:
                                    param.grad = grad_backup[name].clone()

                        logits_adv = model(input_ids, attention_mask, segment_ids)
                        loss_adv_temp = loss_fun(logits_adv, labels)
                        loss_adv_temp.backward()

                    # 步骤3: 恢复原始 Embedding 权重
                    adversarial_trainer.restore()

            # PGD/对抗训练可能放大梯度尺度，做梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            global_step += 1
            total_loss+=loss.item()

            avg_loss = total_loss / (idx + 1)
            pbar.set_postfix({"loss": "{:.6f}".format(avg_loss)})
            # 实时写入step日志
            step_kd_components = {}
            if distillation_teacher is not None:
                step_kd_components = {
                    "supervised_loss": supervised_loss.item(),
                    "span_kd_loss": span_kd_loss.item(),
                    "boundary_kd_loss": boundary_kd_loss.item(),
                    "nested_kd_loss": nested_kd_loss.item(),
                    "kd_eligible_windows": kd_batch,
                    "kd_skipped_position_shift_windows": kd_skipped,
                }
            append_step_log(
                TRAIN_STEP_LOG_CSV,
                eo,
                idx + 1,
                loss.item(),
                avg_loss,
                **step_kd_components,
            )

        append_online_augmentation_epoch_log(
            ONLINE_AUGMENTATION_LOG_CSV,
            eo,
            online_augmentation_epoch,
        )

        with torch.no_grad():
            model.eval()
            (
                avg_f1,
                avg_precision,
                avg_recall,
                cls_tp,
                cls_pred,
                cls_gold,
                structure_counts,
            ) = evaluate_dev_with_prediction_pipeline(model, tokenizer, eval_records)

            append_epoch_log(
                TRAIN_LOG_CSV,
                eo,
                avg_loss,
                avg_f1,
                avg_precision,
                avg_recall,
                total_params=param_stats["param_total"],
                trainable_params=param_stats["param_trainable"],
            )

            # 逐类别指标
            class_rows = []
            focus_metrics = []
            for cls_id in range(ENT_CLS_NUM):
                cls_name = id2ent.get(cls_id, str(cls_id))
                p = cls_tp[cls_id] / cls_pred[cls_id] if cls_pred[cls_id] > 0 else 0.0
                r = cls_tp[cls_id] / cls_gold[cls_id] if cls_gold[cls_id] > 0 else 0.0
                f1 = 2 * cls_tp[cls_id] / (cls_pred[cls_id] + cls_gold[cls_id]) if (cls_pred[cls_id] + cls_gold[cls_id]) > 0 else 0.0
                class_rows.append({
                    "epoch": eo,
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "f1": f1,
                    "precision": p,
                    "recall": r,
                    "tp": int(cls_tp[cls_id]),
                    "pred": int(cls_pred[cls_id]),
                    "gold": int(cls_gold[cls_id]),
                })
                if cls_name in focus_types:
                    focus_metrics.append((cls_name, f1, p, r))
            append_class_epoch_log(TRAIN_CLASS_LOG_CSV, class_rows)

            structure_rows = []
            for bucket, counts in structure_counts.items():
                tp_n, pred_n, gold_n = counts["tp"], counts["pred"], counts["gold"]
                p_n = tp_n / pred_n if pred_n else 0.0
                r_n = tp_n / gold_n if gold_n else 0.0
                f1_n = 2 * tp_n / (pred_n + gold_n) if pred_n + gold_n else 0.0
                structure_rows.append({
                    "epoch": eo,
                    "bucket": bucket,
                    "f1": f1_n,
                    "precision": p_n,
                    "recall": r_n,
                    "tp": int(tp_n),
                    "pred": int(pred_n),
                    "gold": int(gold_n),
                })
            append_structure_epoch_log(TRAIN_STRUCTURE_LOG_CSV, structure_rows)

            is_best = avg_f1 > max_f
            if is_best:
                save_best_model(model, BEST_MODEL_PATH)
                best_meta = dict(checkpoint_architecture_metadata)
                best_meta.update({
                    "checkpoint": os.path.abspath(BEST_MODEL_PATH),
                    "epoch": eo,
                    "dev_f1": avg_f1,
                    "dev_precision": avg_precision,
                    "dev_recall": avg_recall,
                    "checkpoint_sha256": file_sha256(BEST_MODEL_PATH),
                })
                save_params(best_meta, BEST_MODEL_PATH + ".meta.json")
                max_f = avg_f1
            # 统一格式输出当前轮指标与历史最佳F1
            print("Epoch {} -> F1: {:.6f}  Precision: {:.6f}  Recall: {:.6f}  Best_F1: {:.6f}".format(
                eo, avg_f1, avg_precision, avg_recall, max_f
            ))
            print(
                f"Dev evaluation pipeline: {'sliding-window' if USE_SLIDING_WINDOW_DEV_EVAL else 'single-window'} "
                f"(stride={PREDICT_STRIDE if USE_SLIDING_WINDOW_DEV_EVAL else 0}, character-level exact match)."
            )
            if focus_metrics:
                focus_text = " | ".join(
                    [f"{name}:F1={f1:.4f},P={p:.4f},R={r:.4f}" for name, f1, p, r in focus_metrics]
                )
                print(f"Focus classes -> {focus_text}")
            model.train()

            if SAVE_TOPK_MODELS:
                save_topk_model(
                    model,
                    avg_f1,
                    eo,
                    BEST_MODELS_DIR,
                    BEST_MODELS_META,
                    k=TOPK_MODELS,
                    checkpoint_metadata=checkpoint_architecture_metadata,
                )
            completed_epochs = eo + 1

    # 将实际完成口径写回参数文件；Top-3开启时必须有真实权重、sidecar和清单。
    if SAVE_TOPK_MODELS:
        topk_verification = verify_topk_models(
            BEST_MODELS_DIR,
            BEST_MODELS_META,
            TOPK_MODELS,
            completed_epochs,
        )
        print(
            f"Top-{TOPK_MODELS} checkpoints verified: "
            f"count={topk_verification['topk_checkpoint_count']}, "
            f"directory={topk_verification['topk_checkpoint_dir']}"
        )
    else:
        topk_verification = {
            "topk_checkpoint_verification_passed": False,
            "topk_checkpoint_count": 0,
            "topk_checkpoint_dir": None,
            "topk_checkpoint_manifest": None,
            "topk_checkpoint_files": [],
        }
    with open(TRAIN_PARAMS_JSON, "r", encoding="utf-8") as f:
        final_params = json.load(f)
    final_params.update({
        "actual_completed_epochs": completed_epochs,
        "actual_optimizer_steps": global_step,
        "actual_data_passes": global_step / len(ner_loader_train) if len(ner_loader_train) else 0.0,
        "completed_full_configured_epochs": completed_epochs == EPOCHS,
        "online_augmentation_actual_totals": summarize_online_augmentation(
            online_augmentation_totals
        ),
        "online_augmentation_probability_is_attempt_probability": True,
        "online_augmentation_log_csv": os.path.abspath(
            ONLINE_AUGMENTATION_LOG_CSV
        ),
        **topk_verification,
    })
    save_params(final_params, TRAIN_PARAMS_JSON)

    try:
        from plot_metrics import plot_curves
        plot_curves()
    except ImportError:
        print("plot_metrics.py 未提供，跳过训练曲线绘制；模型与日志已正常保存。")


if __name__ == '__main__':
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
    # main返回后训练模型、优化器和最后一个batch已离开局部作用域，再清理缓存，
    # 避免自动效率子进程与训练过程同时占用显存。
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    run_automatic_efficiency_benchmark()
