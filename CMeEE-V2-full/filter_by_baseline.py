# -*- coding: utf-8 -*-
"""
基线模型（Teacher）质检：仅保留 LLM 标注 + 一致性区间过滤（不做 Teacher 伪标签写入）。

1) 实体列表始终使用 LLM 输出的 Gold；Teacher 仅用于一致性打分与拒收低质量句，不把预测并入 entities。
2) 一致性：pred vs LLM Gold；固定 TEACHER_CONSISTENCY_METRIC=exact（精确跨度 F1）；默认 TEACHER_CONSISTENCY_MIN=0.45；augment_mode=mutation 时 poison 下限默认 TEACHER_CONSISTENCY_MIN_MUTATION=0.40；
   trivial：仅当 TEACHER_CONSISTENCY_MAX<1.0 时启用（score>=MAX 拒收）；默认 MAX=1.0 表示关闭 trivial，避免 LOCAL/BOUNDARY 实体未改时 Teacher F1=1 被误杀。
   augment_mode=mutation 且原句实体≥3 时，诱变后实体数须≥ceil(source×MUTATION_ENTITY_MIN_RATIO)（默认 0.6）；
   augment_mode=negative 且 Gold 无实体：仍用 Teacher 预测；若预测到至少一个实体则拒收（漏标专名毒样本），可用 NEGATIVE_TEACHER_ENTITY_FILTER=0 关闭。

输入 datasets/augmented_incremental.jsonl；输出 datasets/augmented_filtered_by_baseline.jsonl（供 smart_sample 使用）。

前置：Teacher 必须为 RoBERTa + Standard GlobalPointer，在 Train-core 上训练并用
internal calibration 选模；`TRAIN_ROLE=baseline_teacher` 的训练入口只会在完整跑满
10 轮后把最优权重与 `.meta.json` 发布到 config.BASELINE_FILTER_MODEL_PATH。
为防止 Teacher 被增强模型污染，缺失时不再回退到普通 BEST_MODEL_PATH。
"""
import math
import os
import json
import hashlib
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone

import jsonlines
from tqdm import tqdm

from config import (
    DATASET_NAME,
    DATA_DIR,
    MAX_LEN,
    BASELINE_FILTER_MODEL_PATH,
    BASELINE_TEACHER_GP_EFFICIENT,
    BASELINE_TEACHER_USE_BIGRU,
    BASELINE_TEACHER_TRAIN_ROLE,
    BASELINE_TEACHER_SELECTION_SPLIT_ROLE,
    EPOCHS,
    PREDICT_STRIDE,
    EXPECTED_OFFLINE_LLM_MODEL,
    EXPECTED_OFFLINE_PROMPT_VERSION,
    EXPECTED_OFFLINE_PIPELINE_VERSION,
    OFFLINE_AUGMENTATION_RAW_TRAIN_FILE,
)
from predict_CME import load_model, predict_batch


def resolve_baseline_checkpoint():
    """只接受专用纯训练集基线，避免误用已被增强训练覆盖的 BEST_MODEL_PATH。"""
    if os.path.isfile(BASELINE_FILTER_MODEL_PATH):
        meta_path = BASELINE_FILTER_MODEL_PATH + ".meta.json"
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(
                f"Teacher 检查点缺少架构元数据: {meta_path}"
            )
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        expected = {
            "dataset_name": DATASET_NAME,
            "gp_efficient": BASELINE_TEACHER_GP_EFFICIENT,
            "use_bigru": BASELINE_TEACHER_USE_BIGRU,
            "train_role": BASELINE_TEACHER_TRAIN_ROLE,
            "selection_split_role": BASELINE_TEACHER_SELECTION_SPLIT_ROLE,
            "configured_epochs": EPOCHS,
            "actual_completed_epochs": EPOCHS,
            "completed_full_configured_epochs": True,
            "teacher_checkpoint_finalized_after_full_training": True,
            "baseline_teacher_source_purity_checked": True,
            "baseline_teacher_rows_with_augmentation_provenance": 0,
            "baseline_teacher_calibration_is_distinct_from_official_dev": True,
            "baseline_teacher_split_prepared": True,
            "baseline_teacher_train_calibration_text_overlap": 0,
            "baseline_teacher_holdout_exclusion_verified": True,
            "use_eight_layer_encoder": False,
            "use_nested_structure_distillation": False,
            "use_sequence_concat": False,
            "use_position_shift": False,
            "use_token_mask": False,
            "augment_synthetic_online": False,
        }
        mismatches = {
            key: (meta.get(key), value)
            for key, value in expected.items()
            if meta.get(key) != value
        }
        if mismatches:
            raise ValueError(
                "Teacher 必须使用 RoBERTa + Standard GlobalPointer，"
                "并且由 Train-core / internal calibration 协议产生；"
                f"当前检查点元数据不符: {mismatches}"
            )
        for evidence_key in (
            "baseline_teacher_split_manifest_sha256",
            "baseline_teacher_raw_train_sha256",
            "baseline_teacher_train_core_sha256",
            "baseline_teacher_calibration_sha256",
        ):
            if not isinstance(meta.get(evidence_key), str) or not meta[evidence_key]:
                raise ValueError(
                    f"Teacher 元数据缺少拆分证据字段: {evidence_key}；"
                    "请用当前版本重新训练11号筛选Teacher"
                )
        claimed_hash = meta.get("checkpoint_sha256")
        actual_hash = file_sha256(BASELINE_FILTER_MODEL_PATH)
        if not isinstance(claimed_hash, str) or claimed_hash != actual_hash:
            raise ValueError(
                "Teacher 权重与元数据哈希不一致，可能是中断发布、手工替换或错配："
                f"meta={claimed_hash}, actual={actual_hash}"
            )
        return BASELINE_FILTER_MODEL_PATH
    raise FileNotFoundError(
        f"未找到专用 Teacher 权重：{BASELINE_FILTER_MODEL_PATH}\n"
        "请先按 Train-core / internal-calibration 协议训练 Standard GP Teacher，"
        "并将对应最佳权重与元数据保存为 baseline_best_model.pth(.meta.json)。"
    )


FILTER_BATCH_SIZE = max(1, int(os.environ.get("FILTER_BATCH_SIZE", "16")))
TEACHER_CONSISTENCY_FILTER = os.environ.get("TEACHER_CONSISTENCY_FILTER", "1") != "0"
TEACHER_CONSISTENCY_MIN = float(os.environ.get("TEACHER_CONSISTENCY_MIN", "0.45"))
TEACHER_CONSISTENCY_MIN_MUTATION = float(os.environ.get("TEACHER_CONSISTENCY_MIN_MUTATION", "0.40"))
# trivial：score>=MAX 拒收；默认 1.0 表示不启用（LOCAL/BOUNDARY 常得 F1=1 仍保留）。若需筛「过于简单」样本可设 0.99 等。
TEACHER_CONSISTENCY_MAX = float(os.environ.get("TEACHER_CONSISTENCY_MAX", "1.0"))
# mutation：诱变后实体条数不得低于原句条数×该比例（且原句至少 3 个实体时才启用）
MUTATION_ENTITY_MIN_RATIO = float(os.environ.get("MUTATION_ENTITY_MIN_RATIO", "0.6"))
# 固定 exact：(type,start,end) 精确 multiset F1；overlap 已弃用（易放过松散边界）
TEACHER_CONSISTENCY_METRIC = "exact"
# 无实体医学负例：LLM 漏标专名时 Gold 仍为空；用 Teacher 是否检出实体拦截（默认开）
NEGATIVE_TEACHER_ENTITY_FILTER = os.environ.get("NEGATIVE_TEACHER_ENTITY_FILTER", "1") != "0"


def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def multiset_exact_entity_f1(pred_entities, gold_entities):
    """
    基于 (type, start_idx, end_idx) 多重集合的微平均 F1（精确跨度，不含重叠近似）。
    空对空：视为完全一致，返回 1.0（可由一致性区间过滤为「无增益」样本）。
    """
    def sig(e):
        return (e["type"], int(e["start_idx"]), int(e["end_idx"]))

    pc = Counter(sig(p) for p in pred_entities)
    gc = Counter(sig(g) for g in gold_entities)
    if not pc and not gc:
        return 1.0
    if not pc or not gc:
        return 0.0
    tp = sum((pc & gc).values())
    p_den = sum(pc.values())
    r_den = sum(gc.values())
    p = tp / p_den if p_den else 0.0
    r = tp / r_den if r_den else 0.0
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def multiset_overlap_entity_f1(pred_entities, gold_entities):
    """
    宽松一致性：同类型且半开区间有交集的 pred/gold 贪心配对算 TP，再算实体级 micro-F1。
    缓解「Teacher 与 LLM 仅差一个字符边界」导致精确 F1 归零的问题。
    """
    preds = list(pred_entities)
    golds = list(gold_entities)
    if not preds and not golds:
        return 1.0
    if not preds or not golds:
        return 0.0

    pair_scores = []
    for gi, g in enumerate(golds):
        for pi, p in enumerate(preds):
            if p["type"] != g["type"]:
                continue
            os_ = max(int(p["start_idx"]), int(g["start_idx"]))
            oe = min(int(p["end_idx"]), int(g["end_idx"]))
            if os_ < oe:
                inter = oe - os_
                pair_scores.append((inter, gi, pi))

    pair_scores.sort(reverse=True)
    matched_g = set()
    matched_p = set()
    tp = 0
    for inter, gi, pi in pair_scores:
        if gi in matched_g or pi in matched_p:
            continue
        matched_g.add(gi)
        matched_p.add(pi)
        tp += 1

    fp = len(preds) - len(matched_p)
    fn = len(golds) - len(matched_g)
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom else 0.0


def teacher_consistency_score(pred_entities, gold_entities):
    if TEACHER_CONSISTENCY_METRIC == "exact":
        return multiset_exact_entity_f1(pred_entities, gold_entities)
    return multiset_overlap_entity_f1(pred_entities, gold_entities)


def summarize_scores(values):
    """记录Teacher分数分布，使确认偏差可观察，而不是只报告最终通过数。"""
    if not values:
        return {"count": 0}
    ordered = sorted(float(x) for x in values)

    def percentile(q):
        idx = round(q * (len(ordered) - 1))
        return ordered[max(0, min(len(ordered) - 1, idx))]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p25": percentile(0.25),
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "p75": percentile(0.75),
        "max": ordered[-1],
    }


def validate_augmentation_provenance(augmented_data):
    """保证一次 Teacher 过滤只处理同一可复现 LLM 增强协议。"""
    required = ("dataset_name", "source_id", "llm_model", "prompt_version", "pipeline_version")
    missing = [
        index
        for index, item in enumerate(augmented_data)
        if any(item.get(key) is None for key in required)
    ]
    if missing:
        raise ValueError(
            "Teacher 输入缺少来源/版本元数据，无法证明同一增强协议: "
            f"rows={missing[:20]}"
        )
    wrong_dataset = [
        index for index, item in enumerate(augmented_data)
        if item.get("dataset_name") != DATASET_NAME
    ]
    if wrong_dataset:
        raise ValueError(
            f"Teacher输入混入非{DATASET_NAME}样本: rows={wrong_dataset[:20]}"
        )
    invalid = [
        index
        for index, item in enumerate(augmented_data)
        if not isinstance(item.get("source_id"), int)
        or isinstance(item.get("source_id"), bool)
        or any(
            not isinstance(item.get(key), str) or not item[key].strip()
            for key in ("llm_model", "prompt_version", "pipeline_version")
        )
    ]
    if invalid:
        raise ValueError(
            "Teacher 输入的来源/版本元数据类型非法: "
            f"rows={invalid[:20]}"
        )
    signatures = {
        (item["llm_model"], item["prompt_version"], item["pipeline_version"])
        for item in augmented_data
    }
    if len(signatures) > 1:
        raise ValueError(
            "Teacher 输入混入多个模型/提示词/流水线版本，拒绝合并: "
            f"{sorted(signatures)}"
        )
    if signatures:
        actual = next(iter(signatures))
        expected = (
            EXPECTED_OFFLINE_LLM_MODEL,
            EXPECTED_OFFLINE_PROMPT_VERSION,
            EXPECTED_OFFLINE_PIPELINE_VERSION,
        )
        if actual != expected:
            raise ValueError(
                "Teacher输入虽然内部版本一致，但与当前代码锁定协议不一致："
                f"actual={actual}, expected={expected}"
            )
    return signatures


def validate_augmentation_input_schema(augmented_data):
    """拒绝绕过最新版 LLM 解析/跨度校验的手工混入行。"""
    allowed_modes = {"local", "boundary", "mutation", "negative"}
    allowed_types = {"bod", "dis", "sym", "mic", "pro", "ite", "dep", "dru", "equ"}
    errors = []
    for index, item in enumerate(augmented_data):
        text = item.get("text")
        entities = item.get("entities")
        mode = item.get("augment_mode")
        source_id = item.get("source_id")
        if (
            isinstance(source_id, bool)
            or not isinstance(source_id, int)
            or source_id < 0
        ):
            errors.append((index, "invalid_source_id"))
            continue
        if not isinstance(text, str) or not text:
            errors.append((index, "invalid_text"))
            continue
        if mode not in allowed_modes:
            errors.append((index, "invalid_augment_mode"))
            continue
        if not isinstance(entities, list):
            errors.append((index, "entities_not_list"))
            continue
        if mode == "negative" and entities:
            errors.append((index, "negative_has_entities"))
            continue
        if mode != "negative" and (
            not entities
            or not isinstance(item.get("source_text"), str)
            or not item["source_text"]
        ):
            errors.append((index, "positive_missing_entities_or_source_text"))
            continue
        if mode != "negative":
            source_similarity = item.get("source_similarity")
            if (
                isinstance(source_similarity, bool)
                or not isinstance(source_similarity, (int, float))
                or not 0.0 <= float(source_similarity) <= 1.0
            ):
                errors.append((index, "invalid_source_similarity"))
                continue
        if mode == "mutation":
            replacements = item.get("mutation_replacements")
            if not isinstance(replacements, list) or len(replacements) != 1:
                errors.append((index, "mutation_requires_one_replacement_record"))
                continue
            replacement = replacements[0]
            allowed = replacement.get("allowed_replacements")
            selected = replacement.get("replacement")
            if (
                not isinstance(allowed, list)
                or not allowed
                or not isinstance(selected, str)
                or selected not in allowed
            ):
                errors.append((index, "mutation_selection_not_in_allowed_candidates"))
                continue
        for ent in entities:
            if not isinstance(ent, dict):
                errors.append((index, "entity_not_object"))
                break
            start, end = ent.get("start_idx"), ent.get("end_idx")
            if (
                isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, int)
                or not isinstance(end, int)
                or not (0 <= start < end <= len(text))
                or ent.get("type") not in allowed_types
            ):
                errors.append((index, "invalid_entity_span_or_type"))
                break
            value = ent.get("entity", ent.get("value"))
            if value is not None and text[start:end] != value:
                errors.append((index, "entity_value_mismatch"))
                break
    if errors:
        raise ValueError(
            "Teacher 输入不符合最新版增强样本结构，拒绝过滤: "
            f"{errors[:20]}"
        )


def validate_source_alignment(augmented_data, source_records):
    """核对 source_id 与原 Train 正文，防止手工合并后来源错位。"""
    errors = []
    for index, item in enumerate(augmented_data):
        source_id = item["source_id"]
        if not (0 <= source_id < len(source_records)):
            errors.append((index, "source_id_out_of_range"))
            continue
        if (
            item.get("augment_mode") != "negative"
            and item.get("source_text") != source_records[source_id].get("text")
        ):
            errors.append((index, "source_text_mismatch"))
            continue
        if item.get("augment_mode") == "mutation":
            replacement = item["mutation_replacements"][0]
            try:
                start = int(replacement["start_idx"])
                end = int(replacement["end_idx"])
                selected = replacement["replacement"]
            except (KeyError, TypeError, ValueError):
                errors.append((index, "invalid_mutation_replacement_record"))
                continue
            source_text = source_records[source_id].get("text", "")
            if not (
                isinstance(selected, str)
                and 0 <= start < end <= len(source_text)
                and item.get("text")
                == source_text[:start] + selected + source_text[end:]
            ):
                errors.append((index, "mutation_changed_context_outside_leaf"))
    if errors:
        raise ValueError(
            "Teacher 输入与原 Train 来源索引不一致，拒绝过滤: "
            f"{errors[:20]}"
        )


def validate_augmentation_run_manifest(input_file, source_file, augmented_data):
    """建立augment_cmeee输出与Teacher过滤输入的文件哈希链。"""
    manifest_path = os.path.join(
        os.path.dirname(os.path.abspath(input_file)),
        "augmentation_run_manifest.json",
    )
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            "缺少augmentation_run_manifest.json，无法证明输入由当前增强Run产生："
            f"{manifest_path}"
        )
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    expected_provenance = {
        "dataset_name": DATASET_NAME,
        "llm_model": EXPECTED_OFFLINE_LLM_MODEL,
        "prompt_version": EXPECTED_OFFLINE_PROMPT_VERSION,
        "pipeline_version": EXPECTED_OFFLINE_PIPELINE_VERSION,
    }
    actual_provenance = {
        key: manifest.get(key) for key in expected_provenance
    }
    mismatches = {}
    checks = {
        "incremental_file_sha256": (
            manifest.get("incremental_file_sha256"), file_sha256(input_file)
        ),
        "train_file_sha256": (
            manifest.get("train_file_sha256"), file_sha256(source_file)
        ),
        "accepted_total_in_incremental": (
            manifest.get("accepted_total_in_incremental"), len(augmented_data)
        ),
        "augmentation_provenance": (actual_provenance, expected_provenance),
    }
    for key, (actual, expected) in checks.items():
        if actual != expected:
            mismatches[key] = (actual, expected)
    if mismatches:
        raise ValueError(
            "增强Run清单与Teacher过滤输入不一致，拒绝继续："
            f"{mismatches}"
        )
    return manifest_path


def main():
    input_file = os.path.join(DATA_DIR, "augmented_incremental.jsonl")
    source_file = (
        OFFLINE_AUGMENTATION_RAW_TRAIN_FILE
        if os.path.isabs(OFFLINE_AUGMENTATION_RAW_TRAIN_FILE)
        else os.path.join(DATA_DIR, OFFLINE_AUGMENTATION_RAW_TRAIN_FILE)
    )
    output_file = os.path.join(DATA_DIR, "augmented_filtered_by_baseline.jsonl")
    reject_file = os.path.join(DATA_DIR, "augmented_rejected_by_baseline.jsonl")
    manifest_file = os.path.join(DATA_DIR, "baseline_filter_manifest.json")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"找不到输入文件: {input_file}")
    if not os.path.exists(source_file):
        raise FileNotFoundError(f"找不到增强来源文件: {source_file}")
    with open(source_file, "r", encoding="utf-8") as f:
        source_records = json.load(f)

    augmented_data = []
    with jsonlines.open(input_file) as reader:
        for obj in reader:
            augmented_data.append(obj)
    provenance_signatures = validate_augmentation_provenance(augmented_data)
    validate_augmentation_input_schema(augmented_data)
    validate_source_alignment(augmented_data, source_records)
    augmentation_manifest_path = validate_augmentation_run_manifest(
        input_file, source_file, augmented_data
    )
    augmented_data.sort(key=lambda x: (x["source_id"], x.get("text", "")))
    source_counts = Counter(
        int(item["source_id"])
        for item in augmented_data
        if item.get("source_id") is not None
    )
    duplicate_source_ids = sorted(
        source_id for source_id, count in source_counts.items() if count > 1
    )
    if duplicate_source_ids:
        raise ValueError(
            "Teacher 输入中同一 source_id 出现多条增强，拒绝静默改变样本权重: "
            f"{duplicate_source_ids[:20]}"
        )

    # 先完成廉价的文件/协议/来源校验，再加载大模型；输入非法时不占用 GPU。
    ckpt_path = resolve_baseline_checkpoint()
    print("正在加载基线模型权重进行验证...")
    tokenizer, model = load_model(
        weights_path=ckpt_path,
        efficient=BASELINE_TEACHER_GP_EFFICIENT,
        use_bigru=BASELINE_TEACHER_USE_BIGRU,
    )

    accepted_data = []
    rejected_data = []

    print(
        f"开始过滤 {len(augmented_data)} 条增强数据（batch_size={FILTER_BATCH_SIZE}，"
        f"teacher_consistency={TEACHER_CONSISTENCY_FILTER} "
        f"poison_min 普通={TEACHER_CONSISTENCY_MIN} / mutation={TEACHER_CONSISTENCY_MIN_MUTATION}，"
        f"trivial_max={TEACHER_CONSISTENCY_MAX}{'（已关闭）' if TEACHER_CONSISTENCY_MAX >= 1.0 else ''}，metric={TEACHER_CONSISTENCY_METRIC}）..."
    )

    texts = [item["text"] for item in augmented_data]
    all_pred_entities = predict_batch(
        texts,
        tokenizer,
        model,
        max_len=MAX_LEN,
        batch_size=FILTER_BATCH_SIZE,
        stride=PREDICT_STRIDE,
    )

    consistency_reject = {"trivial": 0, "poison": 0}
    mutation_collapse_reject = 0
    negative_teacher_entity_reject = 0
    teacher_scores_by_mode = defaultdict(list)

    for item, pred_entities in tqdm(
        zip(augmented_data, all_pred_entities),
        total=len(augmented_data),
        desc="Teacher filtering",
    ):
        llm_entities = item.get("entities", [])

        item = dict(item)
        item["entities"] = list(llm_entities)

        # 对全部输入先统一打分，再执行结构或阈值拒收；清单中的 before_threshold
        # 分布因此包含随后被 Mutation 结构规则拒绝的样本。
        score = teacher_consistency_score(pred_entities, llm_entities)
        item["teacher_consistency_f1"] = score
        item["teacher_consistency_metric"] = TEACHER_CONSISTENCY_METRIC
        teacher_scores_by_mode[item.get("augment_mode") or "unknown"].append(score)

        # 无实体负例若 Teacher 仍检出实体，视为漏标毒样本（默认识别即拒收）
        if not llm_entities:
            if (
                NEGATIVE_TEACHER_ENTITY_FILTER
                and item.get("augment_mode") == "negative"
                and pred_entities
            ):
                row = dict(item)
                row["reject_reason"] = "negative_teacher_detected_entities"
                row["baseline_pred_entities"] = pred_entities
                rejected_data.append(row)
                negative_teacher_entity_reject += 1
                continue
            accepted_data.append(item)
            continue

        # MUTATION：实体总数断崖式下降（嵌套被扁平）则拒收，避免 Teacher F1=1 仍放行毒样本
        if item.get("augment_mode") == "mutation":
            src_n = item.get("source_entity_total")
            if src_n is not None and src_n >= 3:
                min_keep = max(2, math.ceil(float(src_n) * MUTATION_ENTITY_MIN_RATIO))
                if len(llm_entities) < min_keep:
                    row = dict(item)
                    row["reject_reason"] = "mutation_entity_count_collapse"
                    row["source_entity_total"] = src_n
                    row["post_entity_total"] = len(llm_entities)
                    row["min_keep_threshold"] = min_keep
                    rejected_data.append(row)
                    mutation_collapse_reject += 1
                    continue

        if TEACHER_CONSISTENCY_FILTER:
            cur_min = (
                TEACHER_CONSISTENCY_MIN_MUTATION
                if item.get("augment_mode") == "mutation"
                else TEACHER_CONSISTENCY_MIN
            )
            if score < cur_min:
                row = dict(item)
                row["reject_reason"] = "teacher_consistency_poison"
                row["teacher_consistency_f1"] = score
                row["teacher_consistency_metric"] = TEACHER_CONSISTENCY_METRIC
                row["baseline_pred_entities"] = pred_entities
                rejected_data.append(row)
                consistency_reject["poison"] += 1
                continue
            if TEACHER_CONSISTENCY_MAX < 1.0 and score >= TEACHER_CONSISTENCY_MAX:
                row = dict(item)
                row["reject_reason"] = "teacher_consistency_trivial"
                row["teacher_consistency_f1"] = score
                row["teacher_consistency_metric"] = TEACHER_CONSISTENCY_METRIC
                row["baseline_pred_entities"] = pred_entities
                rejected_data.append(row)
                consistency_reject["trivial"] += 1
                continue

        accepted_data.append(item)

    with jsonlines.open(output_file, mode="w") as writer:
        for row in accepted_data:
            writer.write(row)

    with jsonlines.open(reject_file, mode="w") as writer:
        for row in rejected_data:
            writer.write(row)

    print("\n过滤完成！")
    print(f"原始输入: {len(augmented_data)} 条")
    print(f"通过验证 (干净数据): {len(accepted_data)} 条")
    print(f"被拒总计: {len(rejected_data)} 条")
    if mutation_collapse_reject:
        print(f"MUTATION 实体数断崖拒收: {mutation_collapse_reject} 条")
    if negative_teacher_entity_reject:
        print(f"NEGATIVE Teacher 检出实体（漏标拒收）: {negative_teacher_entity_reject} 条")
    if TEACHER_CONSISTENCY_FILTER:
        print(
            f"一致性拒绝: trivial(score>={TEACHER_CONSISTENCY_MAX}，仅当 max<1 启用)={consistency_reject['trivial']}, "
            f"poison(过低)={consistency_reject['poison']}"
        )
    print(f"有效数据: {output_file}")
    print(f"被拒样本: {reject_file}")

    manifest = {
        "dataset_name": DATASET_NAME,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "teacher_checkpoint": os.path.abspath(ckpt_path),
        "teacher_checkpoint_sha256": file_sha256(ckpt_path),
        "teacher_gp_efficient": BASELINE_TEACHER_GP_EFFICIENT,
        "teacher_use_bigru": BASELINE_TEACHER_USE_BIGRU,
        "input_file": os.path.abspath(input_file),
        "input_file_sha256": file_sha256(input_file),
        "augmentation_run_manifest": os.path.abspath(augmentation_manifest_path),
        "augmentation_run_manifest_sha256": file_sha256(augmentation_manifest_path),
        "source_file": os.path.abspath(source_file),
        "source_file_sha256": file_sha256(source_file),
        "output_file": os.path.abspath(output_file),
        "output_file_sha256": file_sha256(output_file),
        "reject_file": os.path.abspath(reject_file),
        "teacher_metric": TEACHER_CONSISTENCY_METRIC,
        "teacher_consistency_filter": TEACHER_CONSISTENCY_FILTER,
        "teacher_consistency_min": TEACHER_CONSISTENCY_MIN,
        "teacher_consistency_min_mutation": TEACHER_CONSISTENCY_MIN_MUTATION,
        "teacher_consistency_max": TEACHER_CONSISTENCY_MAX,
        "negative_teacher_entity_filter": NEGATIVE_TEACHER_ENTITY_FILTER,
        "mutation_entity_min_ratio": MUTATION_ENTITY_MIN_RATIO,
        "predict_stride": PREDICT_STRIDE,
        "input_count": len(augmented_data),
        "augmentation_provenance": (
            {
                "llm_model": next(iter(provenance_signatures))[0],
                "prompt_version": next(iter(provenance_signatures))[1],
                "pipeline_version": next(iter(provenance_signatures))[2],
            }
            if provenance_signatures
            else None
        ),
        "accepted_count": len(accepted_data),
        "rejected_count": len(rejected_data),
        "accepted_mode_counts": dict(Counter(x.get("augment_mode") or "unknown" for x in accepted_data)),
        "rejected_mode_counts": dict(Counter(x.get("augment_mode") or "unknown" for x in rejected_data)),
        "reject_reason_counts": dict(Counter(x.get("reject_reason") or "unknown" for x in rejected_data)),
        "teacher_score_summary_by_mode_before_threshold": {
            mode: summarize_scores(scores)
            for mode, scores in sorted(teacher_scores_by_mode.items())
        },
        "teacher_filter_interpretation": (
            "Teacher agreement is a conservative quality gate and may preferentially "
            "remove valid hard examples; use the per-mode score distributions when "
            "discussing confirmation bias."
        ),
    }
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"过滤清单: {manifest_file}")


if __name__ == "__main__":
    main()
