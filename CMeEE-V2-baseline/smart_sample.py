import difflib
import json
import os
import random
import re
import hashlib
from collections import Counter, deque
from datetime import datetime, timezone

from config import (
    DATASET_NAME,
    DATA_DIR,
    EXPECTED_OFFLINE_LLM_MODEL,
    EXPECTED_OFFLINE_PROMPT_VERSION,
    EXPECTED_OFFLINE_PIPELINE_VERSION,
    OFFLINE_AUGMENTATION_ALLOWED_MODES,
)
from dataset_policy import (
    DATASET_NAME as POLICY_DATASET_NAME,
    FOCUS_CLASS_SELECTION_POLICY,
    FOCUS_CLASS_PRIORITY,
)

if DATASET_NAME != POLICY_DATASET_NAME:
    raise RuntimeError(
        f"数据集策略与config不一致: policy={POLICY_DATASET_NAME}, config={DATASET_NAME}"
    )

try:
    import jsonlines
except ImportError:
    jsonlines = None


HTML_TAG_RE = re.compile(r"</?\s*[A-Za-z][A-Za-z0-9_:-]*[^>]*>")
HTML_TAG_TOKEN_RE = re.compile(r"<\s*/?\s*([A-Za-z][A-Za-z0-9_:-]*)\b[^>]*>")
BROKEN_HTML_FRAGMENT_RE = re.compile(
    r"(?<![</A-Za-z0-9_])(?:sub|sup)\s*>|<\s*/?\s*(?:sub|sup)\b[^>]*$",
    re.IGNORECASE,
)
INCOMPLETE_HTML_TAG_RE = re.compile(r"<\s*/?\s*[A-Za-z][A-Za-z0-9_:-]*[^>]*$")
VOID_HTML_TAGS = frozenset({"br"})
# 与 augment_cmeee 一致：保留 CMeEE 数据集中常见的合法 HTML，仅拦截非白名单标签
ALLOWED_HTML_TAGS = frozenset({"sup", "sub", "br", "p", "span", "div", "b", "i", "strong", "em"})
# 这里的 TARGET 表示“现有 Baseline Dev 结果支持的难类”，不是简单的
# 低频类。用户提供的 Baseline 类别日志中，ite/equ/sym/pro/bod 的 F1
# 约为 0.575/0.602/0.658/0.715/0.724；dep/mic 约为 0.843/0.797。
# 默认 target_ratio=None，因此只做 mode×难类×嵌套分层和多标签主层归属，
# 不会强制人工过采样配额。
TARGET_CLASS_PRIORITY = FOCUS_CLASS_PRIORITY
TARGET_CLASSES = set(TARGET_CLASS_PRIORITY)
DEFAULT_MAX_AUGMENT_TEXT_LEN = int(os.environ.get("AUGMENT_MAX_TEXT_LEN", "120"))


def has_illegal_hallucinated_tags(text):
    """合法实体标签应已剥离；其余角标标签须在白名单内。"""
    for match in HTML_TAG_TOKEN_RE.finditer(text):
        tag_name = match.group(1).lower()
        if tag_name not in ALLOWED_HTML_TAGS:
            return True
    return False


def read_jsonl(path):
    if jsonlines is not None:
        with jsonlines.open(path) as reader:
            return list(reader)

    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_items_sha256(items):
    payload = json.dumps(
        items,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _quality_float(item, key):
    value = item.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key}缺失或不是数值")
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{key}必须位于[0,1]，实际={value}")
    return value


def augmentation_utility(item):
    """固定、无需Dev调参的主质量信号：Teacher实体一致性。"""
    teacher_score = _quality_float(item, "teacher_consistency_f1")
    # source_similarity仍必须存在且有效，但不再与Teacher分数相乘；相乘会
    # 在Teacher同样可靠时系统性奖励接近复制原句的样本。
    _quality_float(item, "source_similarity")
    return teacher_score


def quality_sort_key(item, seed):
    teacher_score = _quality_float(item, "teacher_consistency_f1")
    similarity = _quality_float(item, "source_similarity")
    tie_payload = (
        f"{int(seed)}\0{item.get('source_id')}\0{item.get('text', '')}"
    ).encode("utf-8")
    tie = hashlib.sha256(tie_payload).hexdigest()
    # 先保证Teacher一致性；一致性相同时优先选择与来源差异更大的样本。
    # 使用字典序而非加权和，避免引入需要Dev搜索的新权重。
    return (-teacher_score, similarity, tie)


def entity_value(item, ent):
    start, end = ent["start_idx"], ent["end_idx"]
    return item["text"][start:end]


def entity_fingerprint_set(item):
    """
    无序实体指纹集合；value 经 strip，减轻 CMeEE 首尾空格与 Teacher 伪标签导致的指纹错位。
    """
    return set(
        (ent["type"], entity_value(item, ent).strip())
        for ent in item.get("entities", [])
    )


def has_bad_span(item):
    text = item.get("text", "")
    for ent in item.get("entities", []):
        start, end = ent.get("start_idx"), ent.get("end_idx")
        label_value = ent.get("entity", ent.get("value"))
        if not isinstance(start, int) or not isinstance(end, int):
            return True
        if start < 0 or end <= start or end > len(text):
            return True
        if label_value is not None and text[start:end] != label_value:
            return True
    return False


def normalize_item(item):
    text = item["text"]
    entities = []
    for ent in item.get("entities", []):
        start, end = ent["start_idx"], ent["end_idx"]
        entities.append({
            "start_idx": start,
            "end_idx": end,
            "type": ent["type"],
            "entity": text[start:end],
        })
    out = {
        "dataset_name": item.get("dataset_name"),
        "text": text,
        "entities": entities,
        "augment_mode": item.get("augment_mode"),
    }
    if item.get("source_text") is not None:
        out["source_text"] = item["source_text"]
    if item.get("source_entity_total") is not None:
        out["source_entity_total"] = item["source_entity_total"]
    for key in (
        "llm_model",
        "prompt_version",
        "pipeline_version",
        "source_id",
        "teacher_consistency_f1",
        "teacher_consistency_metric",
        "source_similarity",
        "mutation_replacements",
    ):
        if item.get(key) is not None:
            out[key] = item[key]
    return out


def source_similarity(item, source_candidates):
    if not source_candidates:
        return 0.0
    return max(
        difflib.SequenceMatcher(None, item["text"], source["text"]).ratio()
        for source in source_candidates
    )


def boundary_only_insertion_is_valid(source_item, generated_item):
    source_text = source_item["text"]
    generated_text = generated_item["text"]
    delta = len(generated_text) - len(source_text)
    if delta < 1 or delta > 3:
        return False
    candidate_positions = {
        int(pos)
        for ent in source_item.get("entities", [])
        if ent.get("type") in {"ite", "equ"}
        for pos in (ent["start_idx"], ent["end_idx"])
    }
    valid_positions = {
        pos
        for pos in candidate_positions
        if not any(
            int(other["start_idx"]) < pos < int(other["end_idx"])
            for other in source_item.get("entities", [])
        )
    }
    if not valid_positions:
        return False
    inserted_total = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
        None, source_text, generated_text, autojunk=False
    ).get_opcodes():
        if tag == "equal":
            continue
        if tag != "insert" or i1 != i2 or i1 not in valid_positions:
            return False
        inserted = generated_text[j1:j2]
        if not inserted or inserted.isspace():
            return False
        inserted_total += len(inserted)
    return inserted_total == delta


def has_target_class(item):
    return any(ent.get("type") in TARGET_CLASSES for ent in item.get("entities", []))


def is_nested_sample(item):
    """样本中只要存在一对严格包含实体，就视为嵌套样本。"""
    entities = item.get("entities", [])
    for i, outer in enumerate(entities):
        os_, oe_ = int(outer["start_idx"]), int(outer["end_idx"])
        for j, inner in enumerate(entities):
            if i == j:
                continue
            is_, ie_ = int(inner["start_idx"]), int(inner["end_idx"])
            if os_ <= is_ and ie_ <= oe_ and (os_, oe_) != (is_, ie_):
                return True
    return False


def primary_target_class(item):
    """多标签样本只分配到一个确定性主层，防止分层抽样时重复计数。"""
    present = {e.get("type") for e in item.get("entities", [])}
    for cls in TARGET_CLASS_PRIORITY:
        if cls in present:
            return cls
    return "normal"


def sample_stratum_key(item):
    return (
        item.get("augment_mode") or "unknown",
        primary_target_class(item),
        "nested" if is_nested_sample(item) else "non_nested",
    )


def nested_stratified_order(items, seed):
    """
    生成唯一的全局顺序，再由前缀构造不同规模数据集。
    各层内部先按Teacher一致性降序，再在相同Teacher分数内按
    source_similarity升序；层间按最终层大小的最大欠额交错，使每个前缀
    同时保持结构分布。这样不降低Teacher门禁，又不再奖励近似复制原句。
    """
    if not items:
        return []
    groups = {}
    for item in items:
        groups.setdefault(sample_stratum_key(item), []).append(item)

    for key in sorted(groups):
        groups[key].sort(key=lambda item: quality_sort_key(item, seed))

    total = len(items)
    group_sizes = {key: len(values) for key, values in groups.items()}
    groups = {key: deque(values) for key, values in groups.items()}
    selected_counts = Counter()
    ordered = []
    for prefix_size in range(1, total + 1):
        available = [key for key, values in groups.items() if values]
        chosen = max(
            available,
            key=lambda key: (
                prefix_size * group_sizes[key] / total - selected_counts[key],
                repr(key),
            ),
        )
        ordered.append(groups[chosen].popleft())
        selected_counts[chosen] += 1
    return ordered


def stratum_counts(items):
    return dict(Counter("|".join(sample_stratum_key(x)) for x in items))


def has_malformed_html_tag(text):
    if BROKEN_HTML_FRAGMENT_RE.search(text) or INCOMPLETE_HTML_TAG_RE.search(text):
        return True

    tags = []
    for match in HTML_TAG_TOKEN_RE.finditer(text):
        raw = match.group(0)
        tag = match.group(1).lower()
        stripped = raw.strip()
        is_close = stripped.startswith("</")
        is_self_closing = stripped.endswith("/>") or tag in VOID_HTML_TAGS
        tags.append((tag, is_close, is_self_closing))

    if not tags:
        return False

    stack = []
    for tag, is_close, is_self_closing in tags:
        if is_self_closing:
            if is_close:
                return True
            continue
        if not is_close:
            stack.append(tag)
        elif not stack or stack.pop() != tag:
            return True
    return bool(stack)


def dedupe_items(items):
    seen = {}
    output = []
    for item in items:
        key = item["text"]
        signature = tuple(sorted(
            (int(e["start_idx"]), int(e["end_idx"]), e["type"])
            for e in item.get("entities", [])
        ))
        if key in seen:
            if seen[key] != signature:
                raise ValueError(
                    "发现同文异标的增强样本，已停止抽样以避免静默保留错误标签："
                    f"{key[:80]}"
                )
            continue
        seen[key] = signature
        output.append(item)
    return output


def sample_balanced_targets(pool, n, used_texts, seed=42):
    """显式旧配额模式：类间轮询，类内按固定质量顺序排序。"""
    if n <= 0:
        return []
    class_pools = {
        cls: [x for x in pool if any(e.get("type") == cls for e in x.get("entities", []))]
        for cls in sorted(TARGET_CLASSES)
    }
    for candidates in class_pools.values():
        candidates.sort(key=lambda item: quality_sort_key(item, seed), reverse=True)

    selected = []
    while len(selected) < n:
        progressed = False
        for cls in sorted(TARGET_CLASSES):
            candidates = class_pools[cls]
            while candidates and candidates[-1]["text"] in used_texts:
                candidates.pop()
            if not candidates:
                continue
            item = candidates.pop()
            selected.append(item)
            used_texts.add(item["text"])
            progressed = True
            if len(selected) >= n:
                break
        if not progressed:
            break

    if len(selected) < n:
        remaining = sorted(
            (x for x in pool if x["text"] not in used_texts),
            key=lambda item: quality_sort_key(item, seed),
        )
        fill_n = min(n - len(selected), len(remaining))
        fill = remaining[:fill_n]
        selected.extend(fill)
        used_texts.update(x["text"] for x in fill)
    return selected


def sample_without_overlap(pool, n, used_texts, seed=42):
    """正文去重后按固定质量顺序取前n条，而非随机丢弃质量信号。"""
    candidates = sorted(
        (item for item in pool if item["text"] not in used_texts),
        key=lambda item: quality_sort_key(item, seed),
    )
    if n <= 0:
        return []
    if n > len(candidates):
        raise ValueError(f"需要抽取 {n} 条，但候选池只有 {len(candidates)} 条")
    sampled = candidates[:n]
    used_texts.update(item["text"] for item in sampled)
    return sampled


def smart_sample(
    augmented_file,
    original_file,
    output_file,
    sample_size=1500,
    seed=42,
    target_ratio=None,
    min_similarity=0.45,
    max_similarity=0.85,
    mutation_min_text_similarity=0.35,
    max_text_len=DEFAULT_MAX_AUGMENT_TEXT_LEN,
    max_entity_count=25,
    use_all_filtered=False,
    max_target_augmented=None,
    holdout_files=None,
    allowed_modes=None,
):
    """
    max_target_augmented：对「含 TARGET_CLASSES 的增强条数」硬性上限，用于消融时与全量组对齐重点类规模；
        未传时读环境变量 SMART_SAMPLE_MAX_TARGET_AUGMENTED（未设置或非法则不限）。
    若 target_pool 凑不齐 target_ratio 定额，缺口由 normal_pool 填补，终端会提示分布偏移风险。
    """
    if max_target_augmented is None:
        cap_raw = os.environ.get("SMART_SAMPLE_MAX_TARGET_AUGMENTED", "").strip()
        if cap_raw:
            try:
                max_target_augmented = int(cap_raw)
            except ValueError:
                max_target_augmented = None
        else:
            max_target_augmented = None
    if max_target_augmented is not None and max_target_augmented < 0:
        max_target_augmented = None

    aug_data = [normalize_item(item) for item in read_jsonl(augmented_file)]
    wrong_dataset = [
        index for index, item in enumerate(aug_data)
        if item.get("dataset_name") != DATASET_NAME
    ]
    if wrong_dataset:
        raise ValueError(
            f"增强输入混入非{DATASET_NAME}样本: rows={wrong_dataset[:20]}"
        )
    allowed_modes = tuple(allowed_modes or OFFLINE_AUGMENTATION_ALLOWED_MODES)
    if not allowed_modes:
        raise ValueError("allowed_modes 不能为空")
    unknown_modes = set(allowed_modes) - {"local", "boundary", "mutation", "negative"}
    if unknown_modes:
        raise ValueError(f"allowed_modes 含未知模式: {sorted(unknown_modes)}")

    # 建立 filter_by_baseline -> smart_sample 的哈希链，避免过滤JSONL被手工
    # 替换后仍生成看似合规的sampling manifest。
    filter_manifest_path = os.path.join(
        os.path.dirname(os.path.abspath(augmented_file)),
        "baseline_filter_manifest.json",
    )
    if not os.path.isfile(filter_manifest_path):
        raise FileNotFoundError(
            "缺少 baseline_filter_manifest.json，无法证明输入确由Teacher过滤产生："
            f"{filter_manifest_path}"
        )
    with open(filter_manifest_path, "r", encoding="utf-8") as f:
        filter_manifest = json.load(f)
    actual_augmented_hash = file_sha256(augmented_file)
    expected_provenance = {
        "llm_model": EXPECTED_OFFLINE_LLM_MODEL,
        "prompt_version": EXPECTED_OFFLINE_PROMPT_VERSION,
        "pipeline_version": EXPECTED_OFFLINE_PIPELINE_VERSION,
    }
    filter_mismatches = {}
    if filter_manifest.get("dataset_name") != DATASET_NAME:
        filter_mismatches["dataset_name"] = (
            filter_manifest.get("dataset_name"), DATASET_NAME
        )
    if filter_manifest.get("output_file_sha256") != actual_augmented_hash:
        filter_mismatches["output_file_sha256"] = (
            filter_manifest.get("output_file_sha256"),
            actual_augmented_hash,
        )
    if int(filter_manifest.get("accepted_count", -1)) != len(aug_data):
        filter_mismatches["accepted_count"] = (
            filter_manifest.get("accepted_count"),
            len(aug_data),
        )
    if filter_manifest.get("augmentation_provenance") != expected_provenance:
        filter_mismatches["augmentation_provenance"] = (
            filter_manifest.get("augmentation_provenance"),
            expected_provenance,
        )
    if filter_mismatches:
        raise ValueError(
            "Teacher过滤清单与当前增强输入不一致："
            f"{filter_mismatches}"
        )

    with open(original_file, "r", encoding="utf-8") as f:
        orig_data = json.load(f)

    requested_holdout_files = [path for path in (holdout_files or []) if path]
    missing_holdout_files = [path for path in requested_holdout_files if not os.path.isfile(path)]
    if missing_holdout_files:
        raise FileNotFoundError(
            f"指定的 holdout 文件不存在，拒绝在去污染不完整时抽样: {missing_holdout_files}"
        )
    holdout_files = requested_holdout_files
    holdout_texts = set()
    for path in holdout_files:
        with open(path, "r", encoding="utf-8") as f:
            holdout_texts.update(
                row.get("text", "") for row in json.load(f) if row.get("text")
            )

    # CMeEE-V2 附件中实测存在 1 条 train–test 完全同文。为保证Baseline与
    # mixed实验拥有完全相同的人工Train前缀，保留官方Train行；但把它排除
    # 于LLM来源候选，并拒绝source_text命中holdout的合成样本。
    original_holdout_overlap = [row for row in orig_data if row.get("text") in holdout_texts]
    augmentation_source_reference = [
        row for row in orig_data if row.get("text") not in holdout_texts
    ]

    random.seed(seed)
    orig_texts = {item["text"] for item in orig_data}
    orig_by_text = {item["text"]: item for item in augmentation_source_reference}
    orig_data_fps = [
        (item, entity_fingerprint_set(item))
        for item in augmentation_source_reference
    ]

    filtered = []
    reject_counts = {
        "bad_span": 0,
        "duplicate_original_text": 0,
        "duplicate_holdout_text": 0,
        "source_holdout_text": 0,
        "html_xml_tag": 0,
        "illegal_hallucinated_tag": 0,
        "too_short_or_too_different": 0,
        "too_similar": 0,
        "mutation_no_source_text": 0,
        "mutation_text_too_different": 0,
        "invalid_boundary_edit": 0,
        "too_long": 0,
        "too_long_relative": 0,
        "too_many_entities": 0,
        "disallowed_augment_mode": 0,
        "missing_or_invalid_quality_signal": 0,
    }

    for item in aug_data:
        if has_bad_span(item):
            reject_counts["bad_span"] += 1
            continue
        if item["text"] in orig_texts:
            reject_counts["duplicate_original_text"] += 1
            continue
        if item["text"] in holdout_texts:
            reject_counts["duplicate_holdout_text"] += 1
            continue
        if item.get("source_text") in holdout_texts:
            reject_counts["source_holdout_text"] += 1
            continue
        if has_malformed_html_tag(item["text"]):
            reject_counts["html_xml_tag"] += 1
            continue
        if has_illegal_hallucinated_tags(item["text"]):
            reject_counts["illegal_hallucinated_tag"] += 1
            continue
        if len(item["text"]) > max_text_len:
            reject_counts["too_long"] += 1
            continue
        if len(item.get("entities", [])) > max_entity_count:
            reject_counts["too_many_entities"] += 1
            continue

        # 有显式 source_text 时，始终与真实来源比较，不再从全训练集实体指纹候选中取最大相似度。
        src_text = item.get("source_text") or ""
        if src_text and len(item["text"]) > len(src_text) * 1.15:
            reject_counts["too_long_relative"] += 1
            continue

        mode = item.get("augment_mode")
        if mode not in allowed_modes:
            reject_counts["disallowed_augment_mode"] += 1
            continue
        try:
            item["sampling_utility"] = augmentation_utility(item)
        except ValueError:
            # 主实验只允许local/mutation，两者必须同时具有Teacher一致性和
            # source_similarity；缺任一项就不能执行有证据的质量排序。
            reject_counts["missing_or_invalid_quality_signal"] += 1
            continue

        # 无实体医学负例：不走子集指纹与对原训练集的相似度
        if not item.get("entities") or item.get("augment_mode") == "negative":
            filtered.append(item)
            continue

        if mode in ("local", "boundary", "mutation") and src_text:
            sim = difflib.SequenceMatcher(None, src_text, item["text"], autojunk=False).ratio()
            if mode == "mutation" and sim < mutation_min_text_similarity:
                reject_counts["mutation_text_too_different"] += 1
                continue
            if mode in ("local", "boundary") and sim < min_similarity:
                reject_counts["too_short_or_too_different"] += 1
                continue
            if item["text"] == src_text:
                reject_counts["too_similar"] += 1
                continue
            if mode == "boundary":
                source_item = orig_by_text.get(src_text)
                if source_item is None or not boundary_only_insertion_is_valid(source_item, item):
                    reject_counts["invalid_boundary_edit"] = reject_counts.get("invalid_boundary_edit", 0) + 1
                    continue
            # 合规的小范围改写允许高相似度；不再用统一0.85上限与提示词互相冲突。
            filtered.append(item)
            continue

        # 旧Mutation数据若缺失source_text则无法验证句级骨架，直接拒绝。
        if mode == "mutation":
            if not src_text:
                reject_counts["mutation_no_source_text"] += 1
                continue

        # 仅兼容缺少source_text的旧LOCAL/BOUNDARY输出：用实体指纹候选回退。
        aug_fp = entity_fingerprint_set(item)
        source_candidates = [
            orig_item
            for orig_item, orig_fp in orig_data_fps
            if orig_fp and orig_fp.issubset(aug_fp)
        ]
        sim = source_similarity(item, source_candidates)
        if sim < min_similarity:
            reject_counts["too_short_or_too_different"] += 1
            continue
        if sim > max_similarity:
            reject_counts["too_similar"] += 1
            continue
        filtered.append(item)

    filtered = dedupe_items(filtered)

    already_ordered = False
    if use_all_filtered:
        # 使用过滤、去重后的全部增强样本（确切条数 = len(filtered)），不再按 sample_size / 重点类比例抽样
        sampled_data = list(filtered)
        target_n = sum(1 for x in sampled_data if has_target_class(x))
        normal_n = len(sampled_data) - target_n
        target_pool = [item for item in filtered if has_target_class(item)]
        normal_pool = [item for item in filtered if not has_target_class(item)]
    elif target_ratio is None:
        # 默认不人为把难类提升到固定比例；上传结果不支持无条件过采样。
        # 现在保持过滤池的模式×难类×嵌套比例，
        # 每层内部按utility取优。
        sampled_data = nested_stratified_order(filtered, seed)[:sample_size]
        already_ordered = True
        target_n = sum(1 for item in sampled_data if has_target_class(item))
        normal_n = len(sampled_data) - target_n
        target_pool = [item for item in filtered if has_target_class(item)]
        normal_pool = [item for item in filtered if not has_target_class(item)]
    else:
        # 重点类（TARGET_CLASSES）共池，池内频次由过滤后数据自然决定；仅控制重点类与 normal 的配比
        target_pool = [item for item in filtered if has_target_class(item)]
        normal_pool = [item for item in filtered if not has_target_class(item)]

        target_quota = round(sample_size * target_ratio)
        target_n = min(target_quota, len(target_pool))
        target_cap = len(target_pool)
        if max_target_augmented is not None:
            target_cap = min(target_cap, max_target_augmented)
            target_n = min(target_n, target_cap)

        normal_n = min(sample_size - target_n, len(normal_pool))
        remaining = sample_size - target_n - normal_n
        if remaining > 0:
            # normal不足时可由target补齐，但绝不能突破显式硬上限。
            additional_target = min(remaining, target_cap - target_n)
            target_n += additional_target
            remaining -= additional_target
        if remaining > 0:
            print(
                f"[警告] 候选池或max_target_augmented硬上限使本次只能抽取 "
                f"{sample_size - remaining}/{sample_size} 条。"
            )
        normal_n = min(sample_size - target_n, len(normal_pool))

        if target_n < target_quota and len(target_pool) < target_quota:
            print(
                f"[警告] 重点类池小于名义定额：target_ratio 对应约 {target_quota} 条，池内仅 {len(target_pool)} 条，实际取 {target_n}。"
            )

        used_texts = set()
        sampled_data = []
        sampled_targets = sample_balanced_targets(
            target_pool, target_n, used_texts, seed=seed
        )
        target_n = len(sampled_targets)
        sampled_data.extend(sampled_targets)
        sampled_data.extend(
            sample_without_overlap(normal_pool, normal_n, used_texts, seed=seed)
        )

    # 只生成一次分层全局顺序；所有规模均取其前缀，兼顾集合包含与分布稳定。
    if not already_ordered:
        sampled_data = nested_stratified_order(sampled_data, seed)

    # === 新增：子集嵌套切分与多文件输出 ===
    # 设定你需要的增强数据梯度（0 就是原训练集，不需要单独生成）
    subset_sizes = [750, 1500, len(sampled_data)]

    base_dir = os.path.dirname(output_file)
    # 提取基础文件名，去掉可能带有的数字后缀
    base_name = (
        os.path.basename(output_file)
        .replace(".json", "")
        .replace("_1500", "")
        .replace("_3000", "")
    )

    written_sizes = set()
    subset_records = []
    previous_subset = []
    for requested_size in dict.fromkeys(subset_sizes):
        size = min(requested_size, len(sampled_data))
        if size in written_sizes:
            continue
        written_sizes.add(size)

        # 核心机制：所有规模只取同一分层顺序的前缀，禁止各规模独立重抽。
        current_subset = sampled_data[:size]
        if current_subset[: len(previous_subset)] != previous_subset:
            raise AssertionError("增强规模子集包含关系被破坏")
        mixed_data = orig_data + current_subset

        # 动态生成带有明确梯度的文件名
        current_out_path = os.path.join(base_dir, f"{base_name}_{size}.json")

        with open(current_out_path, "w", encoding="utf-8") as f:
            json.dump(mixed_data, f, ensure_ascii=False, indent=4)

        subset_record = {
            "augmentation_size": size,
            "total_training_size": len(mixed_data),
            "output_file": os.path.abspath(current_out_path),
            "output_file_sha256": file_sha256(current_out_path),
            "augmentation_subset_sha256": canonical_items_sha256(current_subset),
            "contains_previous_prefix": True,
            "allowed_augment_modes": sorted(allowed_modes),
            "previous_augmentation_size": len(previous_subset) if previous_subset else None,
            "mode_counts": dict(Counter(x.get("augment_mode") or "unknown" for x in current_subset)),
            "entity_counts": dict(Counter(e["type"] for x in current_subset for e in x.get("entities", []))),
            "target_sample_coverage": {
                cls: sum(
                    any(e.get("type") == cls for e in x.get("entities", []))
                    for x in current_subset
                )
                for cls in TARGET_CLASS_PRIORITY
            },
            "nested_sample_count": sum(is_nested_sample(x) for x in current_subset),
            "non_nested_sample_count": sum(not is_nested_sample(x) for x in current_subset),
            "primary_target_counts": dict(Counter(primary_target_class(x) for x in current_subset)),
            "stratum_counts": stratum_counts(current_subset),
            "sampling_utility": {
                "formula": "teacher_consistency_f1",
                "tie_policy": "lower_source_similarity_within_equal_teacher_score",
                "min": min((x["sampling_utility"] for x in current_subset), default=None),
                "max": max((x["sampling_utility"] for x in current_subset), default=None),
                "mean": (
                    sum(x["sampling_utility"] for x in current_subset) / len(current_subset)
                    if current_subset else None
                ),
            },
        }
        subset_records.append(subset_record)
        previous_subset = list(current_subset)

        print(
            f"已生成严格嵌套数据集: {current_out_path} "
            f"(新增增强 {len(current_subset)} 条，总计 {len(mixed_data)} 条)"
        )
    for idx, record in enumerate(subset_records):
        record["contained_in_next_augmentation_size"] = (
            subset_records[idx + 1]["augmentation_size"]
            if idx + 1 < len(subset_records)
            else None
        )
    # ==================================

    print(f"原始增强池大小: {len(aug_data)}")
    print(f"过滤后增强池大小: {len(filtered)}")
    print(f"过滤原因统计: {reject_counts}")
    print(f"过滤后增强模式分布: {dict(Counter(x.get('augment_mode') or 'unknown' for x in filtered))}")
    print(
        "最终抽样增强模式分布: "
        f"{dict(Counter(x.get('augment_mode') or 'unknown' for x in sampled_data))}"
    )
    print(
        "最终抽样实体类别增量: "
        f"{dict(Counter(e['type'] for x in sampled_data for e in x.get('entities', [])))}"
    )
    print(f"模式: {'全部过滤后样本并入训练集' if use_all_filtered else f'按 sample_size={sample_size} 分层抽样'}")
    print(f"过滤后重点类候选: {len(target_pool)}")
    print(f"过滤后普通候选: {len(normal_pool)}")
    print(f"实际并入重点类样本: {target_n}")
    print(f"实际并入普通样本: {normal_n}")
    if not use_all_filtered and sample_size > 0 and target_ratio is not None:
        print(
            f"重点类增强实际占比（相对本Run抽取的 sample_size）: {target_n / sample_size:.2%} "
            f"（target_ratio 名义值 {target_ratio:.2%}）"
        )
        nominal_target_quota = round(sample_size * target_ratio)
        if target_n < nominal_target_quota:
            fill_note = (
                "缺口已由 normal_pool 填补"
                if len(sampled_data) >= sample_size
                else "受候选池或硬上限约束，缺口未能补足"
            )
            print(
                f"[警告] 重点类增强 {target_n} 条 < 名义定额 {nominal_target_quota} 条，{fill_note}；"
                f"消融对比时请核对本条与「过滤后重点类候选」，避免把策略失效误判为模型失效。"
            )
    elif not use_all_filtered:
        print(
            "重点类不使用人工目标占比；保持过滤池自然分布，并在各结构层内按utility排序。"
        )
    print(f"文本长度上限: {max_text_len}")
    print(f"实体数量上限: {max_entity_count}")
    print(
        f"新格式LOCAL/BOUNDARY仅设最低来源相似度 {min_similarity}，MUTATION最低来源相似度 "
        f"{mutation_min_text_similarity}；不再以0.85上限误杀合规小改。"
    )
    print(f"成功抽取增强样本: {len(sampled_data)}")
    print(f"最大梯度混合训练集大小: {len(orig_data) + len(sampled_data)}")
    print(f"嵌套梯度输出: {os.path.join(base_dir, base_name + '_*.json')}")

    manifest_path = os.path.join(base_dir, f"{base_name}_sampling_manifest.json")
    manifest = {
        "dataset_name": DATASET_NAME,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "augmented_file": os.path.abspath(augmented_file),
        "augmented_file_sha256": file_sha256(augmented_file),
        "baseline_filter_manifest": os.path.abspath(filter_manifest_path),
        "baseline_filter_manifest_sha256": file_sha256(filter_manifest_path),
        "baseline_filter_output_sha256_verified": True,
        "original_file": os.path.abspath(original_file),
        "original_file_sha256": file_sha256(original_file),
        "holdout_files": [
            {"path": os.path.abspath(path), "sha256": file_sha256(path)}
            for path in holdout_files
        ],
        "holdout_unique_text_count": len(holdout_texts),
        "original_holdout_overlap_excluded_as_augmentation_sources": len(original_holdout_overlap),
        "original_training_rows_preserved": len(orig_data),
        "original_holdout_overlap_text_sha256": [
            hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
            for row in original_holdout_overlap
        ],
        "seed": seed,
        "sample_size_requested": sample_size,
        "sample_size_actual": len(sampled_data),
        "target_classes": sorted(TARGET_CLASSES),
        "target_class_policy": FOCUS_CLASS_SELECTION_POLICY,
        "target_ratio": target_ratio,
        "target_ratio_policy": (
            "natural_filtered_distribution"
            if target_ratio is None
            else "explicit_legacy_quota"
        ),
        "allowed_augment_modes": sorted(allowed_modes),
        "max_target_augmented": max_target_augmented,
        "minimum_source_similarity": {
            "local_boundary": min_similarity,
            "mutation": mutation_min_text_similarity,
        },
        "legacy_max_similarity_without_source_text": max_similarity,
        "max_text_len": max_text_len,
        "max_entity_count": max_entity_count,
        "reject_counts": reject_counts,
        "filtered_mode_counts": dict(Counter(x.get("augment_mode") or "unknown" for x in filtered)),
        "sampled_mode_counts": dict(Counter(x.get("augment_mode") or "unknown" for x in sampled_data)),
        "sampled_llm_model_counts": dict(Counter(x.get("llm_model") or "unknown" for x in sampled_data)),
        "sampled_prompt_version_counts": dict(Counter(x.get("prompt_version") or "unknown" for x in sampled_data)),
        "sampled_pipeline_version_counts": dict(Counter(x.get("pipeline_version") or "unknown" for x in sampled_data)),
        "sampled_entity_counts": dict(Counter(e["type"] for x in sampled_data for e in x.get("entities", []))),
        "sampled_target_sample_coverage": {
            cls: sum(any(e.get("type") == cls for e in x.get("entities", [])) for x in sampled_data)
            for cls in sorted(TARGET_CLASSES)
        },
        "sampling_order": {
            "method": "teacher_first_novelty_tiebreak_ranked_stratified_prefix_v3",
            "strata": ["augment_mode", "primary_target_class", "nested_status"],
            "within_stratum_rank": "teacher_consistency_f1_desc_then_source_similarity_asc",
            "sampling_order_sha256": canonical_items_sha256(sampled_data),
            "all_scales_share_one_order": True,
        },
        "subsets": subset_records,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"抽样清单: {manifest_path}")


if __name__ == "__main__":
    # 推荐流水线：augment_cmeee → filter_by_baseline（Teacher 门控 + 一致性区间）→ 本脚本混合抽样
    _aug_filtered = os.path.join(DATA_DIR, "augmented_filtered_by_baseline.jsonl")
    if not os.path.isfile(_aug_filtered):
        raise FileNotFoundError(
            f"未找到 {_aug_filtered}。请先运行 filter_by_baseline.py；"
            "勿跳过基线过滤直接用 augmented_incremental.jsonl（会绕过 Teacher–Student 防线）。"
        )
    smart_sample(
        _aug_filtered,
        os.path.join(DATA_DIR, "CMeEE-V2_train.json"),
        os.path.join(DATA_DIR, "CMeEE-V2_train_mixed.json"),
        sample_size=3000,
        seed=42,
        target_ratio=None,
        min_similarity=0.45,
        max_similarity=0.85,
        mutation_min_text_similarity=0.35,
        max_text_len=DEFAULT_MAX_AUGMENT_TEXT_LEN,
        max_entity_count=25,
        use_all_filtered=False,
        holdout_files=[
            os.path.join(DATA_DIR, "CMeEE-V2_dev.json"),
            os.path.join(DATA_DIR, "CMeEE-V2_test.json"),
        ],
    )
