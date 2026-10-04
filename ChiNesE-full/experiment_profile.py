# -*- coding: utf-8 -*-
"""Final paper experiment profile. Generated from the verified V16.7 S2 code base."""
import os

_ROOT = os.path.dirname(os.path.abspath(__file__))
PROFILE = {
    "dataset_name": 'ChiNesE',
    "experiment": 'Full S2: 8L Efficient GP + structured KD + online DA (Concat + Position Shift + Token Mask)',
    "encoder_layers": 8,
    "gp_head": 'Efficient GlobalPointer',
    "train_data": "raw Train",
    "sequence_concatenation": True,
    "position_shift": True,
    "token_mask": True,
    "token_mask_scope": "all_non_special_tokens",
    "legacy_concat_routing": False,
    "v12_student_concat_routing": True,
    "explicit_dataloader_generator": False,
    "knowledge_distillation": True,
    "teacher": '12L w/o-KD Teacher+PS: Efficient GP, raw Train, legacy Concat + Position Shift + all-content Token Mask, no Entity Replacement',
    "position_shift_kd_policy": 'same_view_kd',
}
LOCKED_ENV = {
    'SEED': '42',
    'GP_EFFICIENT': '1',
    'USE_EIGHT_LAYER_ENCODER': '1',
    'STUDENT_NUM_HIDDEN_LAYERS': '8',
    'USE_NESTED_STRUCTURE_DISTILLATION': '1',
    'USE_SEQUENCE_CONCAT': '1',
    'USE_POSITION_SHIFT': '1',
    'USE_TOKEN_MASK': '1',
    'TOKEN_MASK_SCOPE': 'all_non_special_tokens',
    'LEGACY_CONCAT_ROUTING': '0',
    'V12_STUDENT_CONCAT_ROUTING': '1',
    'USE_EXPLICIT_DATALOADER_GENERATOR': '0',
    'TRAIN_ROLE': 'experiment',
    'SELECTION_SPLIT_ROLE': 'official_dev',
    'SAVE_TOPK_MODELS': '1',
    'AUTO_RUN_EFFICIENCY_BENCHMARK': '1',
    'DISTILLATION_TEACHER_GP_EFFICIENT': '1',
    'DISTILLATION_TEACHER_USE_BIGRU': '0',
    'DISTILLATION_TEACHER_USE_SEQUENCE_CONCAT': '1',
    'DISTILLATION_TEACHER_USE_POSITION_SHIFT': '1',
    'DISTILLATION_TEACHER_USE_TOKEN_MASK': '1',
    'DISTILLATION_TEACHER_TOKEN_MASK_SCOPE': 'all_non_special_tokens',
    'DISTILLATION_TEACHER_LEGACY_CONCAT_ROUTING': '1',
    'DISTILLATION_TEACHER_SEED': '42',
    'DISTILLATION_KD_ON_POSITION_SHIFT': '1',
}
DEFAULT_ENV = {
    'TRAIN_FILE': 'ChiNesE_train.json',
    'RUN_NAME': 'full_s2_8l_egp_da_kd_seed42_final',
    'DISTILLATION_TEACHER_TRAIN_FILE': 'ChiNesE_train.json',
    'DISTILLATION_TEACHER_CHECKPOINT': os.path.join(_ROOT, 'teacher_checkpoints', 'Teacher_12L_woKD_PS', 'best_model.pth') if True else '',
}

def apply_experiment_profile():
    for key, value in LOCKED_ENV.items():
        os.environ[key] = value
    for key, value in DEFAULT_ENV.items():
        os.environ.setdefault(key, value)
