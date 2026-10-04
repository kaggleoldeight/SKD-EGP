# -*- coding: utf-8 -*-
"""Baseline：仅使用原始人工标注训练数据，不包含离线生成数据。"""
import os

_ROOT = os.path.dirname(os.path.abspath(__file__))
PROFILE = {
    "dataset_name": 'CMeEE-V2',
    "experiment": 'Baseline',
    "encoder_layers": 12,
    "gp_head": 'Standard GlobalPointer',
    "train_data": "raw Train",
    "sequence_concatenation": False,
    "position_shift": False,
    "token_mask": False,
    "token_mask_scope": "all_non_special_tokens",
    "legacy_concat_routing": False,
    "v12_student_concat_routing": False,
    "explicit_dataloader_generator": False,
    "knowledge_distillation": False,
    "teacher": None,
}
LOCKED_ENV = {
    'SEED': '42',
    'GP_EFFICIENT': '0',
    'USE_EIGHT_LAYER_ENCODER': '0',
    'STUDENT_NUM_HIDDEN_LAYERS': '8',
    'USE_NESTED_STRUCTURE_DISTILLATION': '0',
    'USE_SEQUENCE_CONCAT': '0',
    'USE_POSITION_SHIFT': '0',
    'USE_TOKEN_MASK': '0',
    'TOKEN_MASK_SCOPE': 'all_non_special_tokens',
    'LEGACY_CONCAT_ROUTING': '0',
    'V12_STUDENT_CONCAT_ROUTING': '0',
    'USE_EXPLICIT_DATALOADER_GENERATOR': '0',
    'TRAIN_ROLE': 'experiment',
    'SELECTION_SPLIT_ROLE': 'official_dev',
    'SAVE_TOPK_MODELS': '1',
    'AUTO_RUN_EFFICIENCY_BENCHMARK': '1',
    'DISTILLATION_TEACHER_GP_EFFICIENT': '1',
    'DISTILLATION_TEACHER_USE_BIGRU': '0',
    'DISTILLATION_TEACHER_USE_SEQUENCE_CONCAT': '1',
    'DISTILLATION_TEACHER_USE_POSITION_SHIFT': '0',
    'DISTILLATION_TEACHER_USE_TOKEN_MASK': '1',
    'DISTILLATION_TEACHER_TOKEN_MASK_SCOPE': 'all_non_special_tokens',
    'DISTILLATION_TEACHER_LEGACY_CONCAT_ROUTING': '1',
    'DISTILLATION_TEACHER_SEED': '42',
    'DISTILLATION_TEACHER_CHECKPOINT': '',
}
DEFAULT_ENV = {
    'TRAIN_FILE': 'CMeEE-V2_train.json',
    'RUN_NAME': 'baseline_12l_standardgp_raw_noda_nokd_seed42_v16_4_ps_full',
    'DISTILLATION_TEACHER_TRAIN_FILE': 'CMeEE-V2_train.json',
}

def apply_experiment_profile():
    for key, value in LOCKED_ENV.items():
        os.environ[key] = value
    for key, value in DEFAULT_ENV.items():
        os.environ.setdefault(key, value)
