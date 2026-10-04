# -*- coding: utf-8 -*-
import os
import json
import hashlib
import random
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer
from GlobalPointer import GlobalPointer
from config import (
    DATASET_NAME,
    PRETRAINED_MODEL_DIR, BEST_MODEL_PATH, DEVICE, DATA_DIR,
    TEST_FILE, MAX_LEN, GP_EFFICIENT, ENT_CLS_NUM,
    GP_HEAD_SIZE, RNN_HIDDEN_SIZE, RNN_LAYERS, USE_BIGRU,
    DECODE_THRESHOLD, PREDICT_STRIDE, SEED,
)
from data_loader import ent2id, validate_cmeee_records, load_dataset_records

# 确保输出目录存在
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

device = DEVICE
id2ent = {v: k for k, v in ent2id.items()}


def _file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pretrained_artifact_fingerprint(model_dir):
    fingerprints = {}
    for filename in (
        "config.json",
        "vocab.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ):
        path = os.path.join(model_dir, filename)
        if os.path.isfile(path):
            fingerprints[filename] = _file_sha256(path)
    return fingerprints


def set_seed(seed=42):
    """
    设置随机种子，确保预测结果可复现
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed} for reproducibility")


def load_model(
    weights_path=None,
    target_device=None,
    efficient=None,
    use_bigru=None,
    encoder_num_hidden_layers=None,
):
    """
    weights_path: 若为 None，则加载 config.BEST_MODEL_PATH；
    基线过滤脚本应传入 config.BASELINE_FILTER_MODEL_PATH（或等价纯净基线 ckpt）。
    """
    ckpt = weights_path if weights_path is not None else BEST_MODEL_PATH
    load_device = torch.device(target_device) if target_device is not None else device
    efficient_flag = GP_EFFICIENT if efficient is None else bool(efficient)
    bigru_flag = USE_BIGRU if use_bigru is None else bool(use_bigru)
    meta_path = ckpt + ".meta.json"
    meta = {}
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        expected = {
            "dataset_name": DATASET_NAME,
            "gp_efficient": efficient_flag,
            "use_bigru": bigru_flag,
            "ent_cls_num": ENT_CLS_NUM,
            "gp_head_size": GP_HEAD_SIZE,
            "rnn_hidden_size": RNN_HIDDEN_SIZE,
            "rnn_layers": RNN_LAYERS,
        }
        mismatches = {
            key: (meta.get(key), value)
            for key, value in expected.items()
            if meta.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Checkpoint architecture metadata mismatch: {mismatches}")
    else:
        raise FileNotFoundError(
            f"ChiNesE checkpoint必须带同名元数据以核对数据集/架构: {meta_path}"
        )

    checkpoint_pretrained_fingerprint = meta.get("pretrained_artifact_sha256")
    if checkpoint_pretrained_fingerprint is not None:
        current_pretrained_fingerprint = _pretrained_artifact_fingerprint(
            PRETRAINED_MODEL_DIR
        )
        if checkpoint_pretrained_fingerprint != current_pretrained_fingerprint:
            raise ValueError(
                "Checkpoint 与当前预训练配置/词表指纹不一致："
                f"checkpoint={checkpoint_pretrained_fingerprint}, "
                f"current={current_pretrained_fingerprint}"
            )

    meta_encoder_layers = meta.get("encoder_num_hidden_layers")
    if (
        encoder_num_hidden_layers is not None
        and meta_encoder_layers is not None
        and int(encoder_num_hidden_layers) != int(meta_encoder_layers)
    ):
        raise ValueError(
            "Checkpoint encoder depth metadata mismatch: "
            f"meta={meta_encoder_layers}, requested={encoder_num_hidden_layers}"
        )
    resolved_encoder_layers = (
        int(encoder_num_hidden_layers)
        if encoder_num_hidden_layers is not None
        else int(meta_encoder_layers) if meta_encoder_layers is not None else None
    )
    if resolved_encoder_layers is not None and resolved_encoder_layers <= 0:
        raise ValueError(
            f"encoder_num_hidden_layers 必须为正整数，当前为 {resolved_encoder_layers}"
        )
    print(f"Loading tokenizer from {PRETRAINED_MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_DIR, use_fast=True)

    print("Loading model structure...")
    encoder_config = AutoConfig.from_pretrained(PRETRAINED_MODEL_DIR)
    pretrained_layers = getattr(encoder_config, "num_hidden_layers", None)
    if resolved_encoder_layers is not None:
        if pretrained_layers is None:
            raise ValueError("预训练模型 config 不包含 num_hidden_layers，无法重建学生编码器")
        if resolved_encoder_layers > int(pretrained_layers):
            raise ValueError(
                "检查点编码器层数超过预训练模型层数："
                f"checkpoint={resolved_encoder_layers}, pretrained={pretrained_layers}"
            )
        encoder_config.num_hidden_layers = resolved_encoder_layers
    encoder = AutoModel.from_pretrained(
        PRETRAINED_MODEL_DIR,
        config=encoder_config,
    )
    model = GlobalPointer(
        encoder,
        ENT_CLS_NUM,
        GP_HEAD_SIZE,
        rnn_hidden_size=RNN_HIDDEN_SIZE,
        rnn_layers=RNN_LAYERS,
        efficient=efficient_flag,
        use_bigru=bigru_flag,
        sep_token_id=tokenizer.sep_token_id,
    ).to(load_device)

    print(f"Loading weights from {ckpt}...")
    # 增加 map_location 防止 GPU/CPU 不匹配报错
    state_dict = torch.load(ckpt, map_location=load_device)
    model.load_state_dict(state_dict)
    model.eval()
    return tokenizer, model


def _entities_from_scores(text, scores, offset_mapping, threshold=DECODE_THRESHOLD, debug=False):
    """
    scores: (ent_type, seq, seq)
    offset_mapping: numpy (seq_len, 2)，与 tokenizer 输出一致
    """
    pred_indices = np.where(scores > threshold)
    entities = []

    if debug and len(pred_indices[0]) == 0:
        print(f" [Debug] No entities found. Max score: {np.max(scores)}")
        return []

    for label_idx, start_token, end_token in zip(*pred_indices):
        start_char_span = offset_mapping[start_token]
        end_char_span = offset_mapping[end_token]

        if start_char_span[0] == 0 and start_char_span[1] == 0:
            continue
        if end_char_span[0] == 0 and end_char_span[1] == 0:
            continue

        char_start = start_char_span[0]
        char_end = end_char_span[1]
        extracted_text = text[char_start:char_end]

        entities.append({
            "start_idx": int(char_start),
            "end_idx": int(char_end),
            "type": id2ent[label_idx],
            "entity": extracted_text,
        })

        if debug:
            print(
                f" [Debug] Found: {extracted_text} ({id2ent[label_idx]}) "
                f"Score: {scores[label_idx, start_token, end_token]:.4f}"
            )

    return entities


def predict_batch(
    texts,
    tokenizer,
    model,
    max_len=256,
    batch_size=8,
    threshold=DECODE_THRESHOLD,
    stride=PREDICT_STRIDE,
    debug=False,
):
    """批量滑窗推理；重叠窗口预测按字符 span 去重。"""
    if not texts:
        return []
    all_entities = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        use_overflow = bool(stride and stride > 0)
        safe_stride = min(int(stride or 0), max(0, max_len - 3))
        inputs = tokenizer(
            batch_texts,
            max_length=max_len,
            truncation=True,
            padding=True,
            return_offsets_mapping=True,
            return_overflowing_tokens=use_overflow,
            stride=safe_stride if use_overflow else 0,
            return_tensors="pt",
        )
        offset_mapping = inputs.pop("offset_mapping")
        overflow_mapping = inputs.pop("overflow_to_sample_mapping", None)
        if overflow_mapping is None:
            overflow_mapping = torch.arange(len(batch_texts), dtype=torch.long)

        model_device = next(model.parameters()).device
        input_ids = inputs["input_ids"].to(model_device)
        attention_mask = inputs["attention_mask"].to(model_device)
        token_type_ids = inputs.get("token_type_ids")
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        else:
            token_type_ids = token_type_ids.to(model_device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask, token_type_ids)

        scores_batch = logits.cpu().numpy()
        merged = [dict() for _ in batch_texts]
        n_chunks = input_ids.size(0)
        for b in range(n_chunks):
            sample_idx = int(overflow_mapping[b])
            text = batch_texts[sample_idx]
            scores = scores_batch[b]
            om = offset_mapping[b].cpu().numpy()
            dbg = debug if len(batch_texts) == 1 else False
            entities = _entities_from_scores(
                text,
                scores,
                om,
                threshold=threshold,
                debug=dbg,
            )
            for ent in entities:
                key = (ent["type"], ent["start_idx"], ent["end_idx"])
                merged[sample_idx][key] = ent
        all_entities.extend(
            sorted(x.values(), key=lambda e: (e["start_idx"], e["end_idx"], e["type"]))
            for x in merged
        )
    return all_entities


def predict_one(
    text,
    tokenizer,
    model,
    max_len=256,
    threshold=DECODE_THRESHOLD,
    stride=PREDICT_STRIDE,
    debug=False,
):
    return predict_batch(
        [text],
        tokenizer,
        model,
        max_len=max_len,
        batch_size=1,
        threshold=threshold,
        stride=stride,
        debug=debug,
    )[0]


def entities_to_chinese_label(text, entities):
    """转回本次 ChiNesE 附件的 label 字典和闭区间 [start,end]。"""
    label = {}
    seen = set()
    for entity in sorted(
        entities,
        key=lambda item: (
            int(item["start_idx"]), int(item["end_idx"]), item["type"]
        ),
    ):
        start = int(entity["start_idx"])
        end = int(entity["end_idx"])
        entity_type = entity["type"]
        signature = (start, end, entity_type)
        if signature in seen:
            continue
        seen.add(signature)
        if not 0 <= start < end <= len(text) or entity_type not in ent2id:
            raise ValueError(f"无法转为ChiNesE格式的预测实体: {entity!r}")
        surface = text[start:end]
        label.setdefault(entity_type, {}).setdefault(surface, []).append(
            [start, end - 1]
        )
    return label


if __name__ == '__main__':
    # 设置随机种子，确保预测结果可复现
    set_seed(SEED)
    
    tokenizer, model = load_model()

    # 读取测试文件
    test_path = os.path.join(DATA_DIR, TEST_FILE)
    print(f"Reading data from {test_path}")
    data = validate_cmeee_records(load_dataset_records(test_path), source=test_path)

    results = []
    print("Starting prediction...")

    # 仅用于调试：先测前5条，看看有没有输出
    print("-" * 30)
    print("Debug Check (First 3 samples):")
    for i in range(min(3, len(data))):
        print(f"Text: {data[i]['text'][:30]}...")
        ents = predict_one(data[i]['text'], tokenizer, model, max_len=MAX_LEN, debug=True)
        print(f"Result: {ents}")
        print("-" * 30)

    # 全量批量预测；避免逐条调用导致GPU利用率过低。
    all_ents = predict_batch(
        [d["text"] for d in data],
        tokenizer,
        model,
        max_len=MAX_LEN,
        batch_size=8,
        stride=PREDICT_STRIDE,
    )
    submission_rows = []
    seen_record_ids = set()
    for row_index, (d, ents) in enumerate(
        tqdm(zip(data, all_ents), total=len(data), desc="Formatting predictions")
    ):
        record_id = d.get("source_record_id")
        if (
            isinstance(record_id, bool)
            or not isinstance(record_id, (int, str))
            or (isinstance(record_id, str) and not record_id.strip())
        ):
            raise ValueError(
                f"ChiNesE Test第{row_index}行缺少有效source_record_id；"
                "请先用prepare_chinese_dataset.py转换原始new_test.json"
            )
        if record_id in seen_record_ids:
            raise ValueError(
                f"ChiNesE Test第{row_index}行出现重复source_record_id: {record_id!r}"
            )
        seen_record_ids.add(record_id)
        results.append({
            "dataset_name": DATASET_NAME,
            "source_record_id": record_id,
            "text": d['text'],
            "entities": ents
        })
        submission_rows.append({
            "id": record_id,
            "text": d["text"],
            "label": entities_to_chinese_label(d["text"], ents),
        })

    out_path = os.path.join(DATA_DIR, 'ChiNesE_predict_result.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    native_path = os.path.join(DATA_DIR, 'ChiNesE_test_predictions.jsonl')
    with open(native_path, 'w', encoding='utf-8') as f:
        for row in submission_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Done. Saved to {out_path}")
    print(f"ChiNesE-format JSONL saved to {native_path}")
