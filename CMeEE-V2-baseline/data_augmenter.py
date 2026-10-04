# -*- coding: utf-8 -*-
import random


class SequenceAugmenter:
    """Sequence Concatenation + optional entity-preserving Position Shifting.

    Three routing modes are deliberately separated:
    1) v12_student_concat_routing: reproduces the V12 Student concat decision and
       partner-selection semantics with Python's worker-seeded global ``random``.
    2) legacy_concat_routing: preserves the V16.2 12L Teacher trajectory that was
       actually re-run at 75.3985% Dev F1; it uses the dedicated ``_concat_rng``.
    3) default fixed-slot routing: retained only for backwards/diagnostic use.

    Full in V16.4 uses mode (1), all-content Token Mask, and Position Shift.
    Position Shift uses an independent RNG so it cannot advance the recovered V12
    concat random stream; the w/o Position Shift package disables only this module.
    """

    def __init__(
        self,
        data_pool,
        max_len=256,
        use_concat=True,
        use_position_shift=False,
        concat_prob=0.3,
        position_shift_prob=0.2,
        tokenizer=None,
        concat_separator="。",
        position_shift_prefixes=("，", "。", "；", "、"),
        legacy_concat_routing=False,
        v12_student_concat_routing=False,
    ):
        self.data_pool = list(data_pool)
        self.concat_pool = list(self.data_pool)
        self.max_len = int(max_len)
        self.use_concat = bool(use_concat)
        self.use_position_shift = bool(use_position_shift)
        self.concat_prob = float(concat_prob)
        self.position_shift_prob = float(position_shift_prob)
        self.tokenizer = tokenizer
        self.concat_separator = str(concat_separator)
        self.position_shift_prefixes = tuple(str(x) for x in position_shift_prefixes)
        self.legacy_concat_routing = bool(legacy_concat_routing)
        self.v12_student_concat_routing = bool(v12_student_concat_routing)

        if not 0.0 <= self.concat_prob <= 1.0:
            raise ValueError("concat_prob必须位于[0,1]")
        if not 0.0 <= self.position_shift_prob <= 1.0:
            raise ValueError("position_shift_prob必须位于[0,1]")
        if self.legacy_concat_routing and self.v12_student_concat_routing:
            raise ValueError("两种历史Concat路由不能同时启用")
        # Teacher legacy模式必须保持V16.2已经实测的无Position-Shift协议。
        if self.legacy_concat_routing and self.use_position_shift:
            raise ValueError("KDteacher legacy Concat协议禁止Position Shifting")
        if not self.v12_student_concat_routing and not self.legacy_concat_routing:
            if self.concat_prob + self.position_shift_prob > 1.0:
                raise ValueError("固定槽位结构增强概率之和不能超过1")
        if self.v12_student_concat_routing and self.use_concat and self.use_position_shift:
            if self.position_shift_prob > (1.0 - self.concat_prob):
                raise ValueError("V12恢复路由中Position Shift概率不能超过非Concat剩余概率")
        if self.use_position_shift:
            if not self.position_shift_prefixes or any(not x for x in self.position_shift_prefixes):
                raise ValueError("启用Position Shifting时前缀不能为空")

        self.current_epoch = 0
        self.total_epochs = 1
        self._router_rng = random.Random(0)
        self._concat_rng = random.Random(1)
        self._position_rng = random.Random(2)

    def seed_worker(self, worker_seed):
        """Seed dedicated streams. Global random is seeded by train_CME.worker_init_fn."""
        worker_seed = int(worker_seed) % (2 ** 32)
        self._router_rng.seed(worker_seed ^ 0x524F5554)
        self._concat_rng.seed(worker_seed ^ 0x434F4E43)
        self._position_rng.seed(worker_seed ^ 0x504F5349)

    def set_epoch(self, epoch, total_epochs):
        epoch = int(epoch)
        total_epochs = int(total_epochs)
        if total_epochs <= 0 or not 0 <= epoch < total_epochs:
            raise ValueError(f"非法epoch计划: epoch={epoch}, total_epochs={total_epochs}")
        self.current_epoch = epoch
        self.total_epochs = total_epochs

    def _token_length(self, text):
        if self.tokenizer is None:
            return len(text) + 2
        return len(self.tokenizer.encode(text, add_special_tokens=True, truncation=False))

    def _fits_model_window(self, text):
        return self._token_length(text) <= self.max_len

    def _pick_concat_partner(self, item):
        if self.v12_student_concat_routing:
            # Exact V12 code semantics: one global-random start index, then a cyclic
            # scan that excludes both the same object and identical source text.
            n = len(self.concat_pool)
            if n == 0:
                return None
            start = random.randrange(n)
            source_text = item[0]
            for offset in range(n):
                candidate = self.concat_pool[(start + offset) % n]
                if candidate is not item and candidate[0] != source_text:
                    return candidate
            return None

        if len(self.concat_pool) <= 1:
            return None
        for _ in range(10):
            candidate = self._concat_rng.choice(self.concat_pool)
            if candidate is not item:
                return candidate
        return None

    def _apply_concat(self, item, text, entities, trace):
        trace["concat_attempted"] = 1
        other_item = self._pick_concat_partner(item)
        if other_item is None:
            trace["concat_no_partner"] = 1
            return text, entities
        other_text = other_item[0]
        new_text = text + self.concat_separator + other_text
        if not self._fits_model_window(new_text):
            trace["concat_rejected_by_length"] = 1
            return text, entities
        offset = len(text) + len(self.concat_separator)
        new_entities = list(entities)
        new_entities.extend(
            (start + offset, end + offset, label)
            for start, end, label in other_item[1:]
        )
        trace["concat_applied"] = 1
        return new_text, new_entities

    def _apply_position_shift(self, text, entities, trace):
        trace["position_shift_attempted"] = 1
        prefix = self._position_rng.choice(self.position_shift_prefixes)
        new_text = prefix + text
        if not self._fits_model_window(new_text):
            trace["position_shift_rejected_by_length"] = 1
            return text, entities
        if self._token_length(new_text) <= self._token_length(text):
            trace["position_shift_tokenizer_ignored"] = 1
            return text, entities
        offset = len(prefix)
        new_entities = [(start + offset, end + offset, label) for start, end, label in entities]
        for old_entity, new_entity in zip(entities, new_entities):
            old_start, old_end, old_label = old_entity
            new_start, new_end, new_label = new_entity
            if new_label != old_label:
                raise RuntimeError("Position Shifting意外改变实体类别")
            if new_start != old_start + offset or new_end != old_end + offset:
                raise RuntimeError("Position Shifting实体坐标平移错误")
            if new_text[new_start:new_end + 1] != text[old_start:old_end + 1]:
                raise RuntimeError("Position Shifting意外改变实体表面")
        trace["position_shift_applied"] = 1
        trace["position_shift_prefix_characters"] = offset
        return new_text, new_entities

    def augment_raw_item(self, item):
        text = item[0]
        entities = list(item[1:])
        trace = {
            "augmenter_fetches": 1,
            "concat_attempted": 0,
            "concat_applied": 0,
            "concat_no_partner": 0,
            "concat_rejected_by_length": 0,
            "position_shift_attempted": 0,
            "position_shift_applied": 0,
            "position_shift_rejected_by_length": 0,
            "position_shift_tokenizer_ignored": 0,
            "position_shift_prefix_characters": 0,
            "legacy_concat_routing_fetches": int(self.legacy_concat_routing),
            "v12_student_concat_routing_fetches": int(self.v12_student_concat_routing),
        }

        if self.legacy_concat_routing:
            # Keep V16.2 Teacher behavior byte-for-byte in routing semantics.
            if self.use_concat and self._concat_rng.random() < self.concat_prob:
                text, entities = self._apply_concat(item, text, entities, trace)

        elif self.v12_student_concat_routing:
            # V12 always consumed one Python-random concat decision per fetch.
            concat_triggered = random.random() < (self.concat_prob if self.use_concat else 0.0)
            if concat_triggered:
                text, entities = self._apply_concat(item, text, entities, trace)

            # Position Shift is part of Full in V16.4. Keep it mutually exclusive
            # with Concat without consuming Python's recovered V12 concat RNG.
            if self.use_position_shift and not concat_triggered:
                remaining = (1.0 - self.concat_prob) if self.use_concat else 1.0
                conditional_p = self.position_shift_prob / remaining if remaining > 0 else 0.0
                if self._position_rng.random() < conditional_p:
                    text, entities = self._apply_position_shift(text, entities, trace)

            # V12 w/o-ER仍会执行一次 ``random.random() < 0.0`` 的ER判定。
            # 虽然永远不会替换实体，但这一步会推进Python随机流，并直接影响
            # 后续样本的Concat decision/partner轨迹；为真实复现V12必须保留。
            random.random()
            trace["v12_removed_entity_replacement_rng_consumed"] = 1

        else:
            router_draw = self._router_rng.random()
            if router_draw < self.concat_prob:
                if self.use_concat:
                    text, entities = self._apply_concat(item, text, entities, trace)
            elif router_draw < self.concat_prob + self.position_shift_prob:
                if self.use_position_shift:
                    text, entities = self._apply_position_shift(text, entities, trace)

        values = [text] + entities
        try:
            return item.__class__(values, online_aug_trace=trace)
        except TypeError:
            return values
