# -*- coding: utf-8 -*-
"""Strict post-selection evaluation on the labeled ChiNesE Test split.

Protocol contract:
1) checkpoint selection must already have been completed on official_dev;
2) this script loads only config.BEST_MODEL_PATH and never scans/ranks Top-K by Test;
3) the configured decode threshold is reused unchanged;
4) Test metrics are report-only and must not feed back into training/hyperparameters.
"""
import csv
import hashlib
import json
import os

from config import (
    DATASET_NAME,
    DATA_DIR,
    TEST_FILE,
    TRAIN_FILE,
    BEST_MODEL_PATH,
    RUN_OUTPUT_DIR,
    RUN_NAME,
    SEED,
    MAX_LEN,
    PREDICT_STRIDE,
    DECODE_THRESHOLD,
)
from data_loader import ent2id, validate_cmeee_records, load_dataset_records

def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _signature_set(entities):
    out = set()
    for entity in entities:
        start = int(entity["start_idx"])
        end = int(entity["end_idx"])
        entity_type = entity["type"]
        out.add((entity_type, start, end))
    return out


def _safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def _metrics(tp, fp, fn):
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    return precision, recall, f1


def evaluate_records(records, predictions):
    if len(records) != len(predictions):
        raise ValueError(
            f"Prediction count mismatch: gold_rows={len(records)}, pred_rows={len(predictions)}"
        )

    per_class = {
        entity_type: {"tp": 0, "fp": 0, "fn": 0, "gold": 0, "pred": 0}
        for entity_type in ent2id
    }
    total_tp = total_fp = total_fn = total_gold = total_pred = 0

    for row, predicted_entities in zip(records, predictions):
        gold = _signature_set(row["entities"])
        pred = _signature_set(predicted_entities)
        tp_set = gold & pred
        fp_set = pred - gold
        fn_set = gold - pred

        total_tp += len(tp_set)
        total_fp += len(fp_set)
        total_fn += len(fn_set)
        total_gold += len(gold)
        total_pred += len(pred)

        for entity_type in ent2id:
            gold_c = {x for x in gold if x[0] == entity_type}
            pred_c = {x for x in pred if x[0] == entity_type}
            per_class[entity_type]["tp"] += len(gold_c & pred_c)
            per_class[entity_type]["fp"] += len(pred_c - gold_c)
            per_class[entity_type]["fn"] += len(gold_c - pred_c)
            per_class[entity_type]["gold"] += len(gold_c)
            per_class[entity_type]["pred"] += len(pred_c)

    precision, recall, f1 = _metrics(total_tp, total_fp, total_fn)
    class_rows = []
    for entity_type in sorted(ent2id, key=ent2id.get):
        counts = per_class[entity_type]
        p, r, c_f1 = _metrics(counts["tp"], counts["fp"], counts["fn"])
        class_rows.append({
            "class_id": int(ent2id[entity_type]),
            "class_name": entity_type,
            "precision": p,
            "recall": r,
            "f1": c_f1,
            "tp": int(counts["tp"]),
            "fp": int(counts["fp"]),
            "fn": int(counts["fn"]),
            "pred": int(counts["pred"]),
            "gold": int(counts["gold"]),
        })

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(total_tp),
        "fp": int(total_fp),
        "fn": int(total_fn),
        "pred": int(total_pred),
        "gold": int(total_gold),
    }, class_rows


def validate_post_selection_contract(test_path):
    if not os.path.isfile(BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Best checkpoint not found: {BEST_MODEL_PATH}. Train/select on Dev first."
        )
    meta_path = BEST_MODEL_PATH + ".meta.json"
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Checkpoint metadata not found: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)

    if meta.get("dataset_name") != DATASET_NAME:
        raise RuntimeError(
            f"Checkpoint dataset mismatch: {meta.get('dataset_name')!r} != {DATASET_NAME!r}"
        )
    if str(meta.get("selection_split_role", "")).lower() != "official_dev":
        raise RuntimeError(
            "Refusing Test evaluation because checkpoint was not explicitly selected on official_dev: "
            f"selection_split_role={meta.get('selection_split_role')!r}"
        )
    dev_file = os.path.abspath(str(meta.get("dev_file", "")))
    if dev_file and dev_file == os.path.abspath(test_path):
        raise RuntimeError("Refusing Test evaluation: checkpoint dev_file points to Test")

    recorded_hash = meta.get("checkpoint_sha256")
    actual_hash = file_sha256(BEST_MODEL_PATH)
    if recorded_hash and recorded_hash != actual_hash:
        raise RuntimeError(
            f"Checkpoint SHA256 mismatch: meta={recorded_hash}, actual={actual_hash}"
        )
    return meta, meta_path, actual_hash


def main():
    # Lazy import keeps metric/audit helpers usable without loading the DL stack.
    from predict_CME import set_seed, load_model, predict_batch, entities_to_chinese_label

    if DATASET_NAME != "ChiNesE":
        raise RuntimeError(f"This evaluator is ChiNesE-only, got {DATASET_NAME!r}")

    test_path = os.path.join(DATA_DIR, TEST_FILE)
    train_path = os.path.join(DATA_DIR, TRAIN_FILE)
    records = validate_cmeee_records(load_dataset_records(test_path), source=test_path)
    if not records:
        raise ValueError("ChiNesE Test is empty")
    observed_splits = {row.get("dataset_split") for row in records}
    if observed_splits != {"test"}:
        raise RuntimeError(
            f"Expected only dataset_split='test', observed={sorted(map(str, observed_splits))}"
        )

    checkpoint_meta, meta_path, checkpoint_hash = validate_post_selection_contract(test_path)

    # Informational overlap audit only. Training already performs holdout-text decontamination;
    # Test is never used to alter the model here.
    train_test_exact_text_overlap = None
    if os.path.isfile(train_path):
        train_records = validate_cmeee_records(
            load_dataset_records(train_path), source=train_path
        )
        train_texts = {row["text"] for row in train_records}
        test_texts = {row["text"] for row in records}
        train_test_exact_text_overlap = len(train_texts & test_texts)

    set_seed(SEED)
    tokenizer, model = load_model()
    predictions = predict_batch(
        [row["text"] for row in records],
        tokenizer,
        model,
        max_len=MAX_LEN,
        batch_size=8,
        threshold=DECODE_THRESHOLD,
        stride=PREDICT_STRIDE,
    )
    overall, class_rows = evaluate_records(records, predictions)

    test_eval_dir = os.path.join(RUN_OUTPUT_DIR, "test_eval")
    os.makedirs(test_eval_dir, exist_ok=True)
    metrics_path = os.path.join(test_eval_dir, "ChiNesE_test_metrics.json")
    class_path = os.path.join(test_eval_dir, "ChiNesE_test_class_metrics.csv")
    predictions_path = os.path.join(test_eval_dir, "ChiNesE_test_predictions.json")
    native_predictions_path = os.path.join(
        test_eval_dir, "ChiNesE_test_predictions.jsonl"
    )

    payload = {
        "protocol": "post_selection_final_test_evaluation_v1",
        "dataset_name": DATASET_NAME,
        "split": "test",
        "run_name": RUN_NAME,
        "seed": int(SEED),
        "test_rows": len(records),
        "test_file": os.path.abspath(test_path),
        "test_file_sha256": file_sha256(test_path),
        "checkpoint": os.path.abspath(BEST_MODEL_PATH),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_meta": os.path.abspath(meta_path),
        "checkpoint_meta_sha256": file_sha256(meta_path),
        "checkpoint_selection_split_role": checkpoint_meta.get("selection_split_role"),
        "checkpoint_dev_file": checkpoint_meta.get("dev_file"),
        "checkpoint_dev_f1": checkpoint_meta.get("dev_f1"),
        "decode_threshold": float(DECODE_THRESHOLD),
        "predict_stride": int(PREDICT_STRIDE),
        "test_time_augmentation": False,
        "test_used_for_checkpoint_selection": False,
        "test_used_for_hyperparameter_tuning": False,
        "train_test_exact_text_overlap": train_test_exact_text_overlap,
        **overall,
        "note": (
            "Report-only Test evaluation after Dev-based checkpoint/model selection. "
            "Do not use these Test metrics to choose checkpoints, thresholds, augmentation settings, "
            "KD weights, seeds, or ablation configurations."
        ),
    }
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    with open(class_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "class_id", "class_name", "precision", "recall", "f1",
                "tp", "fp", "fn", "pred", "gold",
            ],
        )
        writer.writeheader()
        writer.writerows(class_rows)

    serialized_predictions = []
    with open(native_predictions_path, "w", encoding="utf-8") as native_handle:
        for row, predicted_entities in zip(records, predictions):
            record_id = row.get("source_record_id")
            serialized_predictions.append({
                "dataset_name": DATASET_NAME,
                "dataset_split": "test",
                "source_record_id": record_id,
                "text": row["text"],
                "gold_entities": row["entities"],
                "pred_entities": predicted_entities,
            })
            native_handle.write(json.dumps({
                "id": record_id,
                "text": row["text"],
                "label": entities_to_chinese_label(row["text"], predicted_entities),
            }, ensure_ascii=False) + "\n")
    with open(predictions_path, "w", encoding="utf-8") as handle:
        json.dump(serialized_predictions, handle, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("ChiNesE final Test evaluation (report-only; Dev-selected checkpoint)")
    print(f"P={overall['precision']*100:.4f} R={overall['recall']*100:.4f} F1={overall['f1']*100:.4f}")
    print(f"TP={overall['tp']} FP={overall['fp']} FN={overall['fn']} Gold={overall['gold']}")
    print(f"Metrics: {metrics_path}")
    print(f"Per-class: {class_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
