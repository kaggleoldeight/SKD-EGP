# -*- coding: utf-8 -*-
"""
@Time: 2021/8/27 13:52
@Auth: Xhw
@Description: 实体识别的数据载入器
"""
import json
import torch
from torch.utils.data import Dataset
import numpy as np

ent2id = {"bod": 0, "dis": 1, "sym": 2, "mic": 3, "pro": 4, "ite": 5, "dep": 6, "dru": 7, "equ": 8}
id2ent = {}
for k, v in ent2id.items(): id2ent[v] = k


class DataItem(list):
    """保持旧版list接口，并携带逐样本在线增强审计。"""

    def __init__(
        self,
        values,
        online_aug_trace=None,
        token_mask_protected_char_spans=None,
    ):
        super().__init__(values)
        self.online_aug_trace = dict(online_aug_trace or {})
        # 保留旧参数只为兼容既有DataItem构造调用；V16不再对某一类实体做
        # 特殊Mask保护，而是在collate中统一保护所有Gold实体token。
        self.token_mask_protected_char_spans = tuple(
            token_mask_protected_char_spans or ()
        )


def validate_cmeee_records(records, source="<memory>"):
    """严格校验半开区间 CMeEE JSON，禁止训练时静默丢弃损坏标注。"""
    if not isinstance(records, list):
        raise ValueError(f"{source}: JSON 顶层必须是 list")
    for row_index, row in enumerate(records):
        if not isinstance(row, dict):
            raise ValueError(f"{source}: row={row_index} 不是 object")
        text = row.get("text")
        entities = row.get("entities")
        if not isinstance(text, str):
            raise ValueError(f"{source}: row={row_index} 的 text 不是字符串")
        if not isinstance(entities, list):
            raise ValueError(f"{source}: row={row_index} 的 entities 不是 list")
        for entity_index, entity in enumerate(entities):
            if not isinstance(entity, dict):
                raise ValueError(
                    f"{source}: row={row_index}, entity={entity_index} 不是 object"
                )
            start = entity.get("start_idx")
            end = entity.get("end_idx")
            label = entity.get("type")
            if (
                not isinstance(start, int)
                or isinstance(start, bool)
                or not isinstance(end, int)
                or isinstance(end, bool)
                or not (0 <= start < end <= len(text))
            ):
                raise ValueError(
                    f"{source}: row={row_index}, entity={entity_index} 的半开区间非法: "
                    f"start={start}, end={end}, text_len={len(text)}"
                )
            if label not in ent2id:
                raise ValueError(
                    f"{source}: row={row_index}, entity={entity_index} 的类型非法: {label}"
                )
            annotated_value = entity.get("entity", entity.get("value"))
            if annotated_value is not None and annotated_value != text[start:end]:
                raise ValueError(
                    f"{source}: row={row_index}, entity={entity_index} 的实体字面与跨度不一致"
                )
    return records

def load_data(path):
    """
    加载训练数据
    
    核心修复：CMeEE-V2 数据集的 end_idx 是开区间格式（指向实体最后一个字符的后一位）
    例如："苯丙酮尿症" start_idx=22, end_idx=27（指向"症"后面的"筛"字位置）
    
    为了配合 GlobalPointer 的闭区间逻辑（end_idx 指向最后一个字符），必须减 1
    转换后：end_idx=26（指向"症"字本身）
    
    如果不进行此转换，模型会学习到错误的边界，导致预测时多包含一个字符
    """
    with open(path, "r", encoding="utf-8") as f:
        records = validate_cmeee_records(json.load(f), source=path)
    D = []
    for d in records:
        item = DataItem([d['text']])
        D.append(item)
        for e in d['entities']:
            start, end, label = e['start_idx'], e['end_idx'], e['type']
            
            # 【关键修复】必须执行：将开区间转换为闭区间
            # CMeEE-V2 的 end_idx 指向实体最后一个字符的后一位（开区间）
            # GlobalPointer 需要闭区间（end_idx 指向最后一个字符）
            # 因此必须减 1
            end = end - 1
            
            D[-1].append((start, end, ent2id[label]))
    return D

class EntDataset(Dataset):
    def __init__(
        self,
        data,
        tokenizer,
        istrain=True,
        augmenter=None,
        token_mask_prob=None,
        token_mask_scope="all_non_special_tokens",
        max_len=256,
        train_stride=0,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.istrain = istrain
        self.augmenter = augmenter
        self.max_len = max_len
        self.train_stride = max(0, int(train_stride))
        # 仅 collate 时注入 [MASK]，与标签矩阵长度一致；None/0 表示关闭（验证集请保持关闭）
        self.token_mask_prob = token_mask_prob
        self.token_mask_scope = str(token_mask_scope)
        if self.token_mask_scope not in {
            "non_entity_context_only",
            "all_non_special_tokens",
        }:
            raise ValueError(
                f"未知TOKEN_MASK_SCOPE: {self.token_mask_scope!r}"
            )
        # 训练滑窗必须在 Dataset 索引层展开，而不是在 collate 后把
        # 一个 source batch 膨胀成不受控的 window batch。这样 BATCH_SIZE
        # 始终表示模型实际接收的窗口数。
        self.window_index = None
        self.window_cache = {}
        if self.istrain and self.train_stride > 0:
            self.window_index = []
            for source_index, item in enumerate(self.data):
                encoded_windows = self.encoder(item)
                window_count = len(encoded_windows)
                if window_count <= 0:
                    raise ValueError(f"分词器未为第 {source_index} 条样本生成窗口")
                # 多窗口源文本无法通过 SequenceAugmenter 的 max_len 门禁，
                # 因此可安全缓存原始分窗；若自定义增强器真的改变了它，
                # __getitem__ 会检测到内容差异并改为重新编码。
                if window_count > 1:
                    self.window_cache[source_index] = encoded_windows
                self.window_index.extend(
                    (source_index, window_index)
                    for window_index in range(window_count)
                )

    def __len__(self):
        return len(self.window_index) if self.window_index is not None else len(self.data)

    def encoder(self, item):
        text = item[0]
        # 训练长文本与Dev/预测一样使用重叠滑窗。Fast tokenizer 返回的
        # offset_mapping 仍是原文字符坐标，因此标签无需人工平移。
        use_sliding_window = self.istrain and self.train_stride > 0
        safe_stride = min(self.train_stride, max(0, self.max_len - 3))
        encoder_txt = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            return_offsets_mapping=True,
            return_overflowing_tokens=use_sliding_window,
            stride=safe_stride if use_sliding_window else 0,
        )

        input_ids = encoder_txt["input_ids"]
        attention_mask = encoder_txt["attention_mask"]
        offset_mappings = encoder_txt["offset_mapping"]

        # return_overflowing_tokens=True 时为多窗口嵌套列表；关闭时统一包成单窗口。
        if not input_ids or not isinstance(input_ids[0], (list, tuple, np.ndarray)):
            input_ids = [input_ids]
            attention_mask = [attention_mask]
            offset_mappings = [offset_mappings]
            raw_token_type_ids = encoder_txt.get("token_type_ids")
            token_type_ids = [
                raw_token_type_ids
                if raw_token_type_ids is not None
                else [0] * len(input_ids[0])
            ]
        else:
            raw_token_type_ids = encoder_txt.get("token_type_ids")
            token_type_ids = (
                raw_token_type_ids
                if raw_token_type_ids is not None
                else [[0] * len(ids) for ids in input_ids]
            )

        return [
            (text, offsets, ids, segments, mask)
            for offsets, ids, segments, mask in zip(
                offset_mappings,
                input_ids,
                token_type_ids,
                attention_mask,
            )
        ]

    def sequence_padding(self, inputs, length=None, value=0, seq_dims=1, mode='post'):
        """Numpy函数，将序列padding到同一长度
        """
        if length is None:
            length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
        elif not hasattr(length, '__getitem__'):
            length = [length]

        slices = [np.s_[:length[i]] for i in range(seq_dims)]
        slices = tuple(slices) if len(slices) > 1 else slices[0]
        pad_width = [(0, 0) for _ in np.shape(inputs[0])]

        outputs = []
        for x in inputs:
            x = x[slices]
            for i in range(seq_dims):
                if mode == 'post':
                    pad_width[i] = (0, length[i] - np.shape(x)[i])
                elif mode == 'pre':
                    pad_width[i] = (length[i] - np.shape(x)[i], 0)
                else:
                    raise ValueError('"mode" argument must be "post" or "pre".')
            x = np.pad(x, pad_width, 'constant', constant_values=value)
            outputs.append(x)

        return np.array(outputs)

    def collate(self, examples):
        raw_text_list, batch_input_ids, batch_attention_mask, batch_labels, batch_segment_ids = [], [], [], [], []
        entity_spans_list = []
        batch_entity_token_masks = []
        batch_position_shifted = []
        online_stats = {}

        def add_stat(name, value=1):
            online_stats[name] = online_stats.get(name, 0) + int(value)

        for example in examples:
            selected_window = None
            cached_window = None
            if (
                isinstance(example, tuple)
                and len(example) in (2, 3)
                and isinstance(example[1], int)
            ):
                item, selected_window = example[0], example[1]
                if len(example) == 3:
                    cached_window = example[2]
            else:
                item = example
            for name, value in getattr(item, "online_aug_trace", {}).items():
                add_stat(name, value)
            position_shifted = bool(
                getattr(item, "online_aug_trace", {}).get(
                    "position_shift_applied", 0
                )
            )
            windows = [cached_window] if cached_window is not None else self.encoder(item)
            if not windows:
                raise ValueError("分词器未为样本生成任何训练窗口")
            if selected_window is not None and cached_window is None:
                if selected_window >= len(windows):
                    raise RuntimeError(
                        "在线增强改变了训练窗口数，无法与预计算索引对齐: "
                        f"selected={selected_window}, actual={len(windows)}"
                    )
                windows = [windows[selected_window]]
            for raw_text, token2char_span_mapping, input_ids, token_type_ids, attention_mask in windows:
                add_stat("raw_windows")

                # 只按当前 token 长度分配，避免每条短文本都先创建 9×256×256 的 float64 矩阵。
                labels = np.zeros((len(ent2id), len(input_ids), len(input_ids)), dtype=np.int8)
                entity_token_mask = np.zeros(len(input_ids), dtype=np.bool_)
                spans = []
                for start, end, label in item[1:]:
                    start_token_idx, end_token_idx = -1, -1
                    for idx, (token_start, token_end) in enumerate(token2char_span_mapping):
                        if token_start <= start < token_end:
                            start_token_idx = idx
                            break
                    for idx, (token_start, token_end) in enumerate(token2char_span_mapping):
                        if token_start <= end < token_end:
                            end_token_idx = idx
                            break
                    if start_token_idx != -1 and end_token_idx != -1 and start_token_idx <= end_token_idx:
                        labels[label, start_token_idx, end_token_idx] = 1
                        entity_token_mask[start_token_idx:end_token_idx + 1] = True
                        spans.append((start, end, label))
                raw_text_list.append(raw_text)
                batch_input_ids.append(input_ids)
                batch_segment_ids.append(token_type_ids)
                batch_attention_mask.append(attention_mask)
                batch_labels.append(labels)
                entity_spans_list.append(spans)
                batch_entity_token_masks.append(entity_token_mask)
                batch_position_shifted.append(position_shifted)
        batch_inputids = torch.tensor(self.sequence_padding(batch_input_ids)).long()
        batch_segmentids = torch.tensor(self.sequence_padding(batch_segment_ids)).long()
        batch_attentionmask = torch.tensor(self.sequence_padding(batch_attention_mask)).float()
        batch_labels = torch.tensor(self.sequence_padding(batch_labels, seq_dims=3)).long()
        batch_entity_token_mask = torch.tensor(
            self.sequence_padding(batch_entity_token_masks)
        ).bool()
        batch_position_shifted = torch.tensor(
            batch_position_shifted, dtype=torch.bool
        )

        # 动态 Token 掩蔽：只改 input_ids，长度与 GP 标签矩阵不变；与 SequenceAugmenter 无冲突
        p = self.token_mask_prob
        if p is not None and p > 0:
            tk = self.tokenizer
            cls_id = tk.cls_token_id if tk.cls_token_id is not None else 101
            sep_id = tk.sep_token_id if tk.sep_token_id is not None else 102
            pad_id = tk.pad_token_id if tk.pad_token_id is not None else 0
            mask_id = tk.mask_token_id if tk.mask_token_id is not None else 103
            base_valid_mask = (batch_inputids != cls_id) & (batch_inputids != sep_id) & (batch_inputids != pad_id)
            if self.token_mask_scope == "all_non_special_tokens":
                # KDteacher恢复历史Token Mask语义：实体与非实体内容token
                # 使用相同Mask概率；特殊符号和Padding仍不参与。
                valid_mask = base_valid_mask
                add_stat("token_mask_scope_all_non_special_windows", len(raw_text_list))
            else:
                # 保留旧V16.2作用域，仅供显式诊断；V16.3 Full使用all_non_special_tokens。
                valid_mask = base_valid_mask & ~batch_entity_token_mask
                add_stat("token_mask_scope_non_entity_windows", len(raw_text_list))
            rand_mask = torch.rand(batch_inputids.shape, dtype=torch.float32) < p
            applied_mask = valid_mask & rand_mask
            add_stat(
                "token_mask_excluded_gold_entity_tokens",
                (
                    (base_valid_mask & batch_entity_token_mask).sum().item()
                    if self.token_mask_scope == "non_entity_context_only"
                    else 0
                ),
            )
            add_stat("token_mask_eligible", valid_mask.sum().item())
            add_stat("token_masked", applied_mask.sum().item())
            add_stat(
                "entity_token_mask_eligible",
                (valid_mask & batch_entity_token_mask).sum().item(),
            )
            add_stat(
                "entity_tokens_masked",
                (applied_mask & batch_entity_token_mask).sum().item(),
            )
            batch_inputids[applied_mask] = mask_id

        return (
            raw_text_list,
            batch_inputids,
            batch_attentionmask,
            batch_segmentids,
            batch_labels,
            entity_spans_list,
            online_stats,
            batch_position_shifted,
        )

    def __getitem__(self, index):
        selected_window = None
        if self.window_index is not None:
            source_index, selected_window = self.window_index[index]
        else:
            source_index = index
        source_item = self.data[source_index]
        item = source_item
        if self.istrain and self.augmenter is not None:
            # 保留原对象身份，供SequenceAugmenter排除“与自身拼接”；增强器不会原位修改item。
            item = self.augmenter.augment_raw_item(item)
        if selected_window is None:
            return item
        cached_windows = self.window_cache.get(source_index)
        if cached_windows is not None and item == source_item:
            return item, selected_window, cached_windows[selected_window]
        return item, selected_window
