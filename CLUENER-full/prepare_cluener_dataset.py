# -*- coding: utf-8 -*-
"""把用户提供的CLUENER JSONL闭区间标注转换为本项目统一的半开区间JSON。"""
import argparse
import collections
import hashlib
import json
import os
from datetime import datetime, timezone

DATASET_NAME = "CLUENER"
ENTITY_TYPES = (
    "address", "book", "company", "game", "government",
    "movie", "name", "organization", "position", "scene",
)


def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} 不是合法JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} 顶层必须是object")
            rows.append(row)
    return rows


def _structure_counts(entities):
    nested_pairs = crossing_pairs = 0
    for index, first in enumerate(entities):
        s1, e1 = first["start_idx"], first["end_idx"]
        for second in entities[index + 1:]:
            s2, e2 = second["start_idx"], second["end_idx"]
            if (
                ((s1 <= s2 and e2 <= e1) or (s2 <= s1 and e1 <= e2))
                and (s1, e1) != (s2, e2)
            ):
                nested_pairs += 1
            elif max(s1, s2) < min(e1, e2):
                crossing_pairs += 1
    return nested_pairs, crossing_pairs


def convert_rows(rows, split, source_path):
    converted = []
    type_counts = collections.Counter()
    stats = collections.Counter()
    text_to_signatures = collections.defaultdict(set)
    seen_source_record_ids = set()

    for row_index, row in enumerate(rows):
        text = row.get("text")
        labels = row.get("label")
        if not isinstance(text, str) or not text:
            raise ValueError(f"{source_path}: row={row_index} 必须含非空字符串text")
        if split in {"train", "dev"} and not isinstance(labels, dict):
            raise ValueError(
                f"{source_path}: row={row_index} 的{split}记录必须含object label"
            )
        if labels is None:
            labels = {}
        if not isinstance(labels, dict):
            raise ValueError(f"{source_path}: row={row_index} 的label必须是object或缺省")

        entities = []
        seen = set()
        for entity_type, mentions in labels.items():
            if entity_type not in ENTITY_TYPES:
                raise ValueError(
                    f"{source_path}: row={row_index} 出现未知类型 {entity_type!r}"
                )
            if not isinstance(mentions, dict):
                raise ValueError(
                    f"{source_path}: row={row_index}, type={entity_type} 不是object"
                )
            for label_key, locations in mentions.items():
                if not isinstance(label_key, str) or not isinstance(locations, list):
                    raise ValueError(
                        f"{source_path}: row={row_index}, type={entity_type} 的mention格式非法"
                    )
                for location in locations:
                    if (
                        not isinstance(location, list)
                        or len(location) != 2
                        or any(isinstance(value, bool) or not isinstance(value, int) for value in location)
                    ):
                        raise ValueError(
                            f"{source_path}: row={row_index} 的闭区间非法: {location!r}"
                        )
                    start, inclusive_end = location
                    end = inclusive_end + 1
                    if not 0 <= start < end <= len(text):
                        raise ValueError(
                            f"{source_path}: row={row_index} 越界: {location!r}, text_len={len(text)}"
                        )
                    signature = (start, end, entity_type)
                    if signature in seen:
                        stats["duplicate_entity_annotations_removed"] += 1
                        continue
                    seen.add(signature)
                    surface = text[start:end]
                    entity = {
                        "start_idx": start,
                        "end_idx": end,
                        "type": entity_type,
                        "entity": surface,
                    }
                    # 模型边界以text切片为准；若后续换用的CLUENER副本出现
                    # mention键与切片不一致，只保留原键供审计，不静默改跨度。
                    if surface != label_key:
                        entity["source_label_key"] = label_key
                        stats["label_key_surface_mismatches"] += 1
                    entities.append(entity)
                    type_counts[entity_type] += 1

        entities.sort(
            key=lambda item: (
                int(item["start_idx"]),
                int(item["end_idx"]),
                item["type"],
            )
        )
        nested_pairs, crossing_pairs = _structure_counts(entities)
        stats["nested_pairs"] += nested_pairs
        stats["crossing_pairs"] += crossing_pairs
        stats["rows_with_nesting"] += int(nested_pairs > 0)
        stats["rows_with_crossing_overlap"] += int(crossing_pairs > 0)
        stats["rows_without_entities"] += int(not entities)

        if split == "test" and "id" not in row:
            raise ValueError(
                f"{source_path}: row={row_index} 的Test记录缺少附件要求的id"
            )
        source_record_id = row.get("id", row_index)
        if (
            isinstance(source_record_id, bool)
            or not isinstance(source_record_id, (int, str))
            or (isinstance(source_record_id, str) and not source_record_id.strip())
        ):
            raise ValueError(
                f"{source_path}: row={row_index} 的id必须为非空字符串或整数"
            )
        if source_record_id in seen_source_record_ids:
            raise ValueError(
                f"{source_path}: row={row_index} 出现重复id {source_record_id!r}"
            )
        seen_source_record_ids.add(source_record_id)

        output = {
            "dataset_name": DATASET_NAME,
            "dataset_split": split,
            "source_record_id": source_record_id,
            "is_labeled": bool("label" in row),
            "text": text,
            "entities": entities,
        }
        converted.append(output)
        text_to_signatures[text].add(
            tuple(
                (item["start_idx"], item["end_idx"], item["type"])
                for item in entities
            )
        )

    duplicate_groups = collections.Counter(row["text"] for row in converted)
    stats["duplicate_text_extra_rows"] = sum(
        count - 1 for count in duplicate_groups.values() if count > 1
    )
    stats["duplicate_text_groups"] = sum(count > 1 for count in duplicate_groups.values())
    stats["duplicate_text_inconsistent_label_groups"] = sum(
        len(signatures) > 1 for signatures in text_to_signatures.values()
    )
    stats["rows"] = len(converted)
    stats["entities"] = sum(type_counts.values())
    stats["unique_source_record_ids"] = len(seen_source_record_ids)
    return converted, {
        **dict(stats),
        "entity_type_counts": dict(sorted(type_counts.items())),
        "minimum_text_length": min((len(row["text"]) for row in converted), default=0),
        "maximum_text_length": max((len(row["text"]) for row in converted), default=0),
    }


def write_json_atomic(path, value):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=os.path.join(project_dir, "datasets", "raw_cluener"))
    parser.add_argument("--output-dir", default=os.path.join(project_dir, "datasets"))
    parser.add_argument("--train", default="train.json")
    parser.add_argument("--dev", default="dev.json")
    parser.add_argument("--test", default="test.json")
    args = parser.parse_args()

    source_names = {"train": args.train, "dev": args.dev, "test": args.test}
    output_names = {
        "train": "CLUENER_train.json",
        "dev": "CLUENER_dev.json",
        "test": "CLUENER_test.json",
    }
    converted = {}
    split_stats = {}
    source_hashes = {}
    output_hashes = {}
    for split in ("train", "dev", "test"):
        source_path = os.path.abspath(os.path.join(args.raw_dir, source_names[split]))
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"缺少CLUENER原始文件: {source_path}")
        source_rows = read_jsonl(source_path)
        converted[split], split_stats[split] = convert_rows(source_rows, split, source_path)
        source_hashes[split] = file_sha256(source_path)
        output_path = os.path.abspath(os.path.join(args.output_dir, output_names[split]))
        write_json_atomic(output_path, converted[split])
        output_hashes[split] = file_sha256(output_path)

    split_texts = {
        split: {row["text"] for row in rows}
        for split, rows in converted.items()
    }
    overlaps = {
        "train_dev_exact_text_overlap": len(split_texts["train"] & split_texts["dev"]),
        "train_test_exact_text_overlap": len(split_texts["train"] & split_texts["test"]),
        "dev_test_exact_text_overlap": len(split_texts["dev"] & split_texts["test"]),
    }
    manifest = {
        "dataset_name": DATASET_NAME,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_format": "CLUENER JSONL; Train/Dev use inclusive [start,end] spans; Test is unlabeled",
        "output_format": "JSON list with half-open [start_idx,end_idx) character spans",
        "entity_types": list(ENTITY_TYPES),
        "surface_policy": "text_slice_is_authoritative_label_key_is_audit_only",
        "duplicate_text_policy": "preserve_official_rows_and_report_conflicts",
        "source_file_sha256": source_hashes,
        "output_file_sha256": output_hashes,
        "split_stats": split_stats,
        "split_exact_text_overlap": overlaps,
    }
    manifest_path = os.path.join(args.output_dir, "CLUENER_conversion_manifest.json")
    write_json_atomic(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
