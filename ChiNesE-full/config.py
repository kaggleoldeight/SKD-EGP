# -*- coding: utf-8 -*-
"""
集中配置文件：预训练模型、数据路径与训练/预测超参
"""
import os

# 每个独立压缩包都包含自己的 experiment_profile.py。先应用实验结构锁，
# 再读取下面的统一配置，确保直接运行 train_CME.py 也不会误用其他消融开关。
from experiment_profile import PROFILE, apply_experiment_profile
apply_experiment_profile()

import torch

DATASET_NAME = 'ChiNesE'

# 路径配置
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PRETRAINED_MODEL_DIR = os.path.join(PROJECT_DIR, 'pretrained_models', 'chinese-roberta-wwm-ext')
# 本组实验把 chinese-roberta-wwm-ext 作为 12 层底座；训练入口会读取
# config.json 并显式核验，避免把被替换过的其他深度模型误写成“12层”。
EXPECTED_PRETRAINED_ENCODER_LAYERS = 12

DATA_DIR = os.path.join(PROJECT_DIR, 'datasets')
# 所有实验统一读取原始人工标注Train；数据增强仅指训练期在线增强。
TRAIN_FILE = os.environ.get('TRAIN_FILE', 'ChiNesE_train.json')
DEV_FILE = os.environ.get('DEV_FILE', 'ChiNesE_eval.json')
TEST_FILE = os.environ.get('TEST_FILE', 'ChiNesE_test.json')
# 输出
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 实验隔离：不同消融/随机种子使用独立目录，避免日志与最佳模型互相覆盖。
SEED = int(os.environ.get('SEED', '42'))
if SEED < 0:
    raise ValueError('SEED 必须为非负整数')
RUN_NAME = os.environ.get('RUN_NAME', f"chinese_egp_seed{SEED}")
TRAIN_ROLE = os.environ.get('TRAIN_ROLE', 'experiment').strip().lower()
SELECTION_SPLIT_ROLE = os.environ.get('SELECTION_SPLIT_ROLE', 'official_dev').strip().lower()
RUN_OUTPUT_DIR = os.path.join(OUTPUT_DIR, RUN_NAME)
os.makedirs(RUN_OUTPUT_DIR, exist_ok=True)
SAVE_MODEL_PATH = os.path.join(RUN_OUTPUT_DIR, 'ent_model.pth')
# 最优模型与日志/图表
BEST_MODEL_PATH = os.path.join(RUN_OUTPUT_DIR, 'best_model.pth')
BEST_MODELS_DIR = os.path.join(RUN_OUTPUT_DIR, 'best_models')
BEST_MODELS_META = os.path.join(BEST_MODELS_DIR, 'best_models.json')
TRAIN_LOG_CSV = os.path.join(RUN_OUTPUT_DIR, 'train_log.csv')
TRAIN_STEP_LOG_CSV = os.path.join(RUN_OUTPUT_DIR, 'train_step_log.csv')
TRAIN_CLASS_LOG_CSV = os.path.join(RUN_OUTPUT_DIR, 'train_class_log.csv')
TRAIN_STRUCTURE_LOG_CSV = os.path.join(RUN_OUTPUT_DIR, 'train_structure_log.csv')
TRAIN_PARAMS_JSON = os.path.join(RUN_OUTPUT_DIR, 'train_params.json')
ONLINE_AUGMENTATION_LOG_CSV = os.path.join(
    RUN_OUTPUT_DIR, 'online_augmentation_log.csv'
)
# 兼容旧版总图路径（仍保留，但默认不再使用）
METRICS_FIG_PATH = os.path.join(RUN_OUTPUT_DIR, 'metrics_curve.png')
# 分图输出路径
METRICS_F1_FIG_PATH = os.path.join(RUN_OUTPUT_DIR, 'f1_curve.png')
METRICS_PRECISION_FIG_PATH = os.path.join(RUN_OUTPUT_DIR, 'precision_curve.png')
METRICS_RECALL_FIG_PATH = os.path.join(RUN_OUTPUT_DIR, 'recall_curve.png')
METRICS_LOSS_FIG_PATH = os.path.join(RUN_OUTPUT_DIR, 'loss_curve.png')
EFFICIENCY_DIR = os.path.join(RUN_OUTPUT_DIR, 'efficiency')
# 效率目录只在实际开始基准测试时创建，避免“空目录看起来像已经评测”。
AUTO_RUN_EFFICIENCY_BENCHMARK = (
    os.environ.get('AUTO_RUN_EFFICIENCY_BENCHMARK', '1') == '1'
)
# 默认同时记录GPU与CPU；若机器没有CUDA则只测CPU。可在运行前仅覆盖设备列表，
# 例如 AUTO_EFFICIENCY_DEVICES=cuda:0，以缩短自动评测时间。
AUTO_EFFICIENCY_DEVICES = os.environ.get(
    'AUTO_EFFICIENCY_DEVICES',
    'cuda:0,cpu' if torch.cuda.is_available() else 'cpu',
).strip()
# 防止驱动、Profiler或子进程异常时永久阻塞。默认总评测最多2小时，
# 单个CPU形状最多30分钟；只限制异常挂起，不改变正常计时结果。
EFFICIENCY_BENCHMARK_TIMEOUT_SECONDS = int(os.environ.get(
    'EFFICIENCY_BENCHMARK_TIMEOUT_SECONDS', '7200'
))
EFFICIENCY_CPU_SETTING_TIMEOUT_SECONDS = int(os.environ.get(
    'EFFICIENCY_CPU_SETTING_TIMEOUT_SECONDS', '1800'
))
if EFFICIENCY_BENCHMARK_TIMEOUT_SECONDS <= 0:
    raise ValueError('EFFICIENCY_BENCHMARK_TIMEOUT_SECONDS 必须为正整数')
if EFFICIENCY_CPU_SETTING_TIMEOUT_SECONDS <= 0:
    raise ValueError('EFFICIENCY_CPU_SETTING_TIMEOUT_SECONDS 必须为正整数')
# 默认禁止复用已有实验目录，避免CSV和top-k元数据混入上一次运行。
ALLOW_RUN_DIR_REUSE = os.environ.get('ALLOW_RUN_DIR_REUSE', '0') == '1'

# 设备
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据与模型超参
MAX_LEN = 256
BATCH_SIZE = 16
NUM_WORKERS = 4  # Windows 上多进程建议适度降低
ENT_CLS_NUM = 10
GP_HEAD_SIZE = 64
# GlobalPointer efficient 头（稀疏/高效实现，训练与 predict 需一致）
GP_EFFICIENT = os.environ.get('GP_EFFICIENT', '1') == '1'
LEARNING_RATE = 2e-5
EPOCHS = 10
# 训练入口不包含早停分支，所有实验固定跑满 10 轮。

# 在线增强。V16.5继续保留V12 w/o-Entity-Replacement已验证的Student语义：
# Sequence Concatenation + all-non-special Token Mask；Position Shift由profile控制，
# Full/Full-like消融开启，09号w/o Position Shift严格关闭。
USE_SEQUENCE_CONCAT = os.environ.get('USE_SEQUENCE_CONCAT', '1') == '1'
USE_POSITION_SHIFT = os.environ.get('USE_POSITION_SHIFT', '0') == '1'
USE_TOKEN_MASK = os.environ.get('USE_TOKEN_MASK', '1') == '1'
TOKEN_MASK_SCOPE = os.environ.get(
    'TOKEN_MASK_SCOPE', 'all_non_special_tokens'
).strip()
if TOKEN_MASK_SCOPE not in {'non_entity_context_only', 'all_non_special_tokens'}:
    raise ValueError(f'未知TOKEN_MASK_SCOPE: {TOKEN_MASK_SCOPE!r}')
# 05号KDteacher继续使用V16.2已实测恢复到75.3985%的独立legacy Concat随机流。
LEGACY_CONCAT_ROUTING = os.environ.get('LEGACY_CONCAT_ROUTING', '0') == '1'
# 8L Student可显式恢复V12真实代码中的Python全局random Concat轨迹。
V12_STUDENT_CONCAT_ROUTING = (
    os.environ.get('V12_STUDENT_CONCAT_ROUTING', '0') == '1'
)
if LEGACY_CONCAT_ROUTING and V12_STUDENT_CONCAT_ROUTING:
    raise ValueError('LEGACY_CONCAT_ROUTING与V12_STUDENT_CONCAT_ROUTING不能同时开启')
# V12 Student没有给DataLoader传独立torch.Generator；恢复模式显式记录这一点。
USE_EXPLICIT_DATALOADER_GENERATOR = (
    os.environ.get('USE_EXPLICIT_DATALOADER_GENERATOR', '1') == '1'
)
CONCAT_PROB = 0.3
POSITION_SHIFT_PROB = 0.20
POSITION_SHIFT_PREFIXES = ("，", "。", "；", "、")
TOKEN_MASK_PROB = 0.10
SEQUENCE_CONCAT_SEPARATOR = "。"
# 本发布版不包含离线生成数据；PROFILE只记录结构与在线增强开关。
EXPERIMENT_PROFILE = dict(PROFILE)

# 编码器深度与知识蒸馏是两个独立开关：
# - USE_EIGHT_LAYER_ENCODER 只决定是否把完整预训练编码器均匀裁成学生深度；
# - USE_NESTED_STRUCTURE_DISTILLATION 只决定是否加载 Teacher 和加入 KD 损失。
# 因此 w/o KD 可保持与 Full 相同的 8 层结构。
USE_EIGHT_LAYER_ENCODER = (
    os.environ.get('USE_EIGHT_LAYER_ENCODER', '0') == '1'
)

# 嵌套结构感知知识蒸馏；该开关不改变编码器深度。
USE_NESTED_STRUCTURE_DISTILLATION = (
    os.environ.get('USE_NESTED_STRUCTURE_DISTILLATION', '0') == '1'
)
# 轻量学生深度；仅当 USE_EIGHT_LAYER_ENCODER=1 时生效。为使开关语义和
# 消融名称严格一致，本版训练入口要求该值为 8；保留环境变量仅用于把
# 非 8 层误配置显式报出来，而不是悄悄训练成另一种结构。
STUDENT_NUM_HIDDEN_LAYERS = int(
    os.environ.get('STUDENT_NUM_HIDDEN_LAYERS', '8')
)
# 必须显式指向上述独立强 Teacher 的 .pth；蒸馏开启而路径为空时训练会直接报错。
DISTILLATION_TEACHER_CHECKPOINT = os.environ.get(
    'DISTILLATION_TEACHER_CHECKPOINT', ''
).strip()
# 压缩Teacher固定使用原始Train：12层、无KD；其在线增强配置由元数据严格核验。
DISTILLATION_TEACHER_TRAIN_ROLE = 'distillation_teacher'
DISTILLATION_TEACHER_TRAIN_FILE = os.environ.get(
    'DISTILLATION_TEACHER_TRAIN_FILE', 'ChiNesE_train.json'
).strip()
DISTILLATION_TEACHER_GP_EFFICIENT = (
    os.environ.get('DISTILLATION_TEACHER_GP_EFFICIENT', '1') == '1'
)
DISTILLATION_TEACHER_USE_BIGRU = (
    os.environ.get('DISTILLATION_TEACHER_USE_BIGRU', '0') == '1'
)
DISTILLATION_TEACHER_USE_SEQUENCE_CONCAT = (
    os.environ.get('DISTILLATION_TEACHER_USE_SEQUENCE_CONCAT', '1') == '1'
)
DISTILLATION_TEACHER_USE_POSITION_SHIFT = (
    os.environ.get('DISTILLATION_TEACHER_USE_POSITION_SHIFT', '0') == '1'
)
DISTILLATION_KD_ON_POSITION_SHIFT = (
    os.environ.get('DISTILLATION_KD_ON_POSITION_SHIFT', '0') == '1'
)
DISTILLATION_TEACHER_USE_TOKEN_MASK = (
    os.environ.get('DISTILLATION_TEACHER_USE_TOKEN_MASK', '1') == '1'
)
DISTILLATION_TEACHER_TOKEN_MASK_SCOPE = os.environ.get(
    'DISTILLATION_TEACHER_TOKEN_MASK_SCOPE', 'all_non_special_tokens'
).strip()
if DISTILLATION_TEACHER_TOKEN_MASK_SCOPE != 'all_non_special_tokens':
    raise ValueError(
        'KDteacher必须恢复旧Token Mask语义: all_non_special_tokens'
    )
DISTILLATION_TEACHER_LEGACY_CONCAT_ROUTING = (
    os.environ.get('DISTILLATION_TEACHER_LEGACY_CONCAT_ROUTING', '1') == '1'
)
# 多随机种子实验只改变学生训练随机性；所有Full重复实验复用同一05 Teacher，
# 因此Teacher seed必须与学生SEED分离，避免seed43/44错误拒绝seed42权重。
DISTILLATION_TEACHER_SEED = int(
    os.environ.get('DISTILLATION_TEACHER_SEED', '42')
)
# 默认与学生同设备；显存不足时可显式设为 cpu（会增加训练时间）。
DISTILLATION_TEACHER_DEVICE = os.environ.get(
    'DISTILLATION_TEACHER_DEVICE', str(DEVICE)
).strip()
if DISTILLATION_KD_ON_POSITION_SHIFT:
    if not USE_NESTED_STRUCTURE_DISTILLATION:
        raise ValueError('shifted-window KD仅允许在开启知识蒸馏时使用')
    if not USE_POSITION_SHIFT:
        raise ValueError('shifted-window KD要求Student开启Position Shift')
    if not DISTILLATION_TEACHER_USE_POSITION_SHIFT:
        raise ValueError('same-view shifted-window KD要求Teacher权重来自Position Shift训练协议')
DISTILLATION_TEMPERATURE = float(
    os.environ.get('DISTILLATION_TEMPERATURE', '2.0')
)
# 总损失 = 原监督损失 + 下列三项加权损失。以下只是固定的起始配置，
# 不是由 Dev 搜索得到的最优值；训练元数据会原样记录。
DISTILLATION_SPAN_WEIGHT = float(
    os.environ.get('DISTILLATION_SPAN_WEIGHT', '0.50')
)
DISTILLATION_BOUNDARY_WEIGHT = float(
    os.environ.get('DISTILLATION_BOUNDARY_WEIGHT', '0.25')
)
DISTILLATION_NESTED_WEIGHT = float(
    os.environ.get('DISTILLATION_NESTED_WEIGHT', '0.25')
)
# 每条样本从全部实体类型的 Teacher 分布中取至多 K 个候选 span，再与 Gold 并集
# 构造严格包含对；这是数量上限，不是置信度阈值。
DISTILLATION_NESTED_TOP_K = int(
    os.environ.get('DISTILLATION_NESTED_TOP_K', '64')
)

# 所有数据规模均完整训练 EPOCHS=10 轮；训练代码不再提供固定优化器步数截断分支。

# 解码门槛仍使用 GlobalPointer 的标准 0-logit 边界；单独配置便于论文复现。
DECODE_THRESHOLD = 0.0
# 长训练文本也使用滑窗监督；窗口 offset_mapping 保持原始字符坐标。
TRAIN_STRIDE = 64
# 预测、Dev 评估与 Teacher 过滤使用同一字符坐标滑窗口径。
PREDICT_STRIDE = 64
# Dev 按与最终预测相同的字符级滑窗流程评估，避免“截断Dev选模、滑窗测试”的口径不一致。
USE_SLIDING_WINDOW_DEV_EVAL = True
# 去掉与 Dev/Test 正文完全相同的训练行，避免分割泄漏或同文异标。
REMOVE_HOLDOUT_TEXT_OVERLAP = True


# 优化器与调度器超参
# AdamW 的权重衰减，用于正则化抑制过拟合
WEIGHT_DECAY = 0.01
# 预热比例：训练前 warmup 占总步数的比例（例如 0.1 表示 10% 步数做 warmup）
WARMUP_RATIO = 0.1

# RNN 层超参
# RNN 隐藏层大小（单向），双向 RNN 的输出维度将是 RNN_HIDDEN_SIZE * 2
RNN_HIDDEN_SIZE = 256
# RNN 层数（BiLSTM 和 BiGRU 各使用此层数）
RNN_LAYERS = 1

# GlobalPointer 中是否启用 BiGRU（用于消融实验）
USE_BIGRU = False

# 本组独立实验包统一保存 Dev F1 最高的 Top-3，同时仍单独维护 best_model.pth。
SAVE_TOPK_MODELS = os.environ.get('SAVE_TOPK_MODELS', '1') == '1'
TOPK_MODELS = 3

# 对抗训练配置
# 方法：'NONE'（不使用）、'FGM' 或 'PGD'；当前暂时关闭 PGD/FGM
ADV_TRAIN_METHOD = 'NONE'
# Embedding 层参数名（HuggingFace Bert/RoBERTa 默认如下）
ADV_EMB_NAME = 'encoder.embeddings.word_embeddings'
# FGM 参数
FGM_EPSILON = 1.0
# PGD 参数
PGD_EPSILON = 0.5
PGD_ALPHA = 0.25
PGD_STEPS = 3
# 对抗训练（FGM/PGD）从该 epoch 起启用（0-based）；此前仅干净梯度，便于含噪增强数据上先拟合主分布
PGD_START_EPOCH = 3
