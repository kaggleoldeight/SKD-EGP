# -*- coding: utf-8 -*-
"""
基于阿里云百炼（通义千问，dashscope SDK）的 CMeEE-V2 离线数据扩充脚本。
API 返回内容经解析为 JSON；正例的 text 为带 XML 内联标签的句子，Python 侧剥离标签并计算索引（end_idx 开区间）。
代码保留四种策略供专门消融；主训练默认只生成并接收受控局部改写和单叶子实体诱变。
"""
import difflib
import os
import json
import random
import re
import time
import hashlib
import platform
import itertools
from collections import Counter, deque
from datetime import datetime, timezone
from enum import Enum
from tqdm import tqdm
import jsonlines
import dashscope
from dashscope import Generation
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential
from dataset_policy import (
    DATASET_NAME,
    FOCUS_CLASS_SELECTION_POLICY,
    FOCUS_ENTITY_TYPES,
)

# 百炼 API：调用时才从环境变量读取，避免导入本模块即要求联网凭据。
# 固定论文实际使用的快照，避免滚动别名 qwen-plus 随时间变化。
LLM_MODEL = "qwen-plus-2025-12-01"
RANDOM_SEED = 42
PROMPT_VERSION = "cmeee_aug_v10_2026_08_24"
PIPELINE_VERSION = "cmeee_pipeline_v13_2026_08_24"

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "datasets")
TRAIN_FILE = os.path.join(DATA_DIR, "CMeEE-V2_train.json")
HOLDOUT_FILES = (
    os.path.join(DATA_DIR, "CMeEE-V2_dev.json"),
    os.path.join(DATA_DIR, "CMeEE-V2_test.json"),
)
REMOVE_HOLDOUT_TEXT_OVERLAP = (
    os.environ.get("AUGMENT_REMOVE_HOLDOUT_TEXT_OVERLAP", "1") != "0"
)

# 线程池并发（默认 5，可通过环境变量调大/调小以适配 API 限流）
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "5"))
# 未完成任务上限，避免全量运行时一次性创建所有 Future
MAX_PENDING_FUTURES = max(MAX_WORKERS, int(os.environ.get("MAX_PENDING_FUTURES", str(MAX_WORKERS * 4))))
# 增量文件写入模式：默认覆盖，若确需续写可设置 INCREMENTAL_WRITE_MODE=a
INCREMENTAL_WRITE_MODE = os.environ.get("INCREMENTAL_WRITE_MODE", "w").lower()
if INCREMENTAL_WRITE_MODE not in {"w", "a"}:
    raise ValueError("INCREMENTAL_WRITE_MODE must be 'w' or 'a'")
# append续跑默认只重试基础设施失败；其它已记录状态视为本Run已完成，避免服务端随机性改写结果。
RESUME_RETRY_STATUSES = {
    x.strip()
    for x in os.environ.get(
        "RESUME_RETRY_STATUSES", "api_failed,worker_exception"
    ).split(",")
    if x.strip()
}
# 网络调用本身由 tenacity 重试；这里控制“API成功但内容不合格”时是否用同一提示重新生成。
LLM_CONTENT_MAX_ATTEMPTS = max(1, int(os.environ.get("LLM_CONTENT_MAX_ATTEMPTS", "2")))
CONTENT_RETRY_STATUSES = {
    "empty_response",
    "invalid_generation",
    "unchanged_generation",
    "invalid_local_context_edit",
    "invalid_boundary_edit",
    "invalid_mutation_replacement",
    "invalid_mutation_context_edit",
    "generated_exceeds_max_text_len",
    "generated_exceeds_relative_length",
    "unlabeled_medical_terms",
    "negative_contains_medical_terms",
    "negative_length_out_of_range",
}
# 自动质量门禁：静态词典扫描（find_unlabeled_medical_terms）。默认开启以降低「生造专名」污染；
# 默认开启：句中含已知医学实体词却未标注则拒收，降低漏标毒药（可用 ENABLE_UNLABELED_MEDICAL_TERM_FILTER=0 关闭）
ENABLE_UNLABELED_MEDICAL_TERM_FILTER = os.environ.get("ENABLE_UNLABELED_MEDICAL_TERM_FILTER", "1") != "0"
# 过短词容易误杀普通搭配，默认仅扫描长度 >= 3 的已知实体词
MIN_UNLABELED_TERM_LEN = int(os.environ.get("MIN_UNLABELED_TERM_LEN", "3"))
# 扫描词典最大规模；全量词典可能较大，默认保留长词优先的前 50000 个
MAX_MEDICAL_LEXICON_TERMS = int(os.environ.get("MAX_MEDICAL_LEXICON_TERMS", "50000"))
# 命中这些类型的未标注词通常风险较低，可按需加入；默认不豁免任何类型
UNLABELED_TERM_IGNORE_TYPES = set(
    t.strip() for t in os.environ.get("UNLABELED_TERM_IGNORE_TYPES", "").split(",") if t.strip()
)

# ---------- 试跑条数 ----------
# 仅对去污染后的前 N 条发起增强任务（每条仍可能因校验失败而无输出）。
# 默认全量；试跑可设置 MAX_AUGMENT_SAMPLES=200，而不再手改源代码。
_MAX_AUGMENT_SAMPLES_RAW = os.environ.get("MAX_AUGMENT_SAMPLES", "").strip()
MAX_AUGMENT_SAMPLES = (
    None if not _MAX_AUGMENT_SAMPLES_RAW else int(_MAX_AUGMENT_SAMPLES_RAW)
)
if MAX_AUGMENT_SAMPLES is not None and MAX_AUGMENT_SAMPLES <= 0:
    raise ValueError("MAX_AUGMENT_SAMPLES 必须为正整数或留空")

# 增强合并结果固定为此名（与训练 config 一致）。试跑只改 MAX_AUGMENT_SAMPLES 即可，全量前请自行备份该文件以免被覆盖。
AUGMENTED_FILE = os.path.join(DATA_DIR, "CMeEE-V2_train_augmented.json")

# ---------- 影响生成质量的主要旋钮（调参建议）----------
# 1) 主实验默认只允许 LOCAL/MUTATION 且等权路由；BOUNDARY/NEGATIVE
#    仅为兼容专门消融而保留，避免生成后被训练端固定白名单丢弃。
#    MUTATION 每条仅替换一个叶子 span；主实验实体增量上限为 0；默认校验完整的实体类型嵌套森林。
# 2) LLM_MODEL、Generation.call 的 temperature / max_tokens。
# 3) 正例不再使用统一“相似度上限”：BOUNDARY由严格插入规则校验，LOCAL拒绝原样/仅标点改写，MUTATION由强制替换校验。
# 4) P_AUG_* 均为条件路由权重而非最终比例；句中含安全 ite/equ 边界时才加入 BOUNDARY 分支。
# 5) MAX_WORKERS：并发与百炼限流匹配。

VALID_TYPES = {"bod", "dis", "sym", "mic", "pro", "ite", "dep", "dru", "equ"}
_VALID_TAGS_ALT = "|".join(sorted(VALID_TYPES))
RESIDUAL_TAG_IN_TEXT = re.compile(rf"\[\s*/?\s*({_VALID_TAGS_ALT})\s*\]", re.IGNORECASE)
# 剥离标签后的正文若仍残留同类 XML 标记，视为未净尽（可选防御）
RESIDUAL_XML_TAG_IN_TEXT = re.compile(rf"</?\s*({_VALID_TAGS_ALT})\s*>", re.IGNORECASE)
HTML_TAG_TOKEN_RE = re.compile(r"<\s*/?\s*([A-Za-z][A-Za-z0-9_:-]*)\b[^>]*>")
# CMeEE-V2 原文中允许保留的常见 HTML 小标签（合法实体 bod/dis/... 已在解析时剥离）
ALLOWED_HTML_TAGS = frozenset({"sup", "sub", "br", "p", "span", "div", "b", "i", "strong", "em"})


def has_illegal_hallucinated_tags(text: str) -> bool:
    """剥离合法 CMeEE 实体标签后，若仍存在非白名单的类 HTML 标签（如 <disease>），视为幻觉。"""
    for match in HTML_TAG_TOKEN_RE.finditer(text):
        tag_name = match.group(1).lower()
        if tag_name not in ALLOWED_HTML_TAGS:
            return True
    return False


# 裸写的sub>/sup>，或从“<sub/ <sup”开始但直到行尾没有闭合“>”的残片。
# 关键点：不能把合法闭合标签</sub>中的“sub>”再次从字母s处匹配出来。
BROKEN_HTML_FRAGMENT_RE = re.compile(
    r"(?<![</A-Za-z0-9_])(?:sub|sup)\s*>|<\s*/?\s*(?:sub|sup)\b[^>]*$",
    re.IGNORECASE,
)
INCOMPLETE_HTML_TAG_RE = re.compile(r"<\s*/?\s*[A-Za-z][A-Za-z0-9_:-]*[^>]*$")
VOID_HTML_TAGS = frozenset({"br"})
# 常见零宽 / 方向控制等不可见字符（实体 span 内不允许出现，避免标注噪声）
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]")

# 下游 smart_sample 使用同一环境变量，避免生成阶段与抽样阶段长度规则不一致。
MAX_AUGMENT_TEXT_LEN = int(os.environ.get("AUGMENT_MAX_TEXT_LEN", "120"))

# 主训练只接受 local/mutation，因此生成端默认也只路由这两种，避免付费生成
# 随后必被 smart_sample 丢弃的 boundary/negative。若做专门消融可显式覆盖。
GENERATION_ALLOWED_MODES = tuple(
    mode.strip().lower()
    for mode in os.environ.get(
        "GENERATION_ALLOWED_MODES", "local,mutation"
    ).split(",")
    if mode.strip()
)
_KNOWN_AUGMENT_MODES = {"local", "boundary", "negative", "mutation"}
if not GENERATION_ALLOWED_MODES:
    raise ValueError("GENERATION_ALLOWED_MODES 不能为空")
if set(GENERATION_ALLOWED_MODES) - _KNOWN_AUGMENT_MODES:
    raise ValueError(
        "GENERATION_ALLOWED_MODES 含未知模式: "
        f"{sorted(set(GENERATION_ALLOWED_MODES) - _KNOWN_AUGMENT_MODES)}"
    )
if len(set(GENERATION_ALLOWED_MODES)) != len(GENERATION_ALLOWED_MODES):
    raise ValueError("GENERATION_ALLOWED_MODES 不能包含重复模式")

# 策略路由权重（不是最终策略百分比）。local/mutation默认等权，取消旧版
# mutation 0.70 对 local 0.15 所形成的82.35%条件占比；这不是Dev调参，
# 而是移除无实验证据的先验偏置。
P_AUG_NEGATIVE = float(os.environ.get("P_AUG_NEGATIVE", "0.0"))
P_AUG_BOUNDARY_IF_ITE = float(os.environ.get("P_AUG_BOUNDARY_IF_ITE", "0.0"))
P_AUG_MUTATION = float(os.environ.get("P_AUG_MUTATION", "1.0"))
P_AUG_LOCAL = float(os.environ.get("P_AUG_LOCAL", "1.0"))
# stratified=权重归一化轮盘（默认，消融友好）；cascade=旧版瀑布抽样（关 MUTATION 时 LOCAL 会异常膨胀）
AUGMENT_MODE_ROUTING = os.environ.get("AUGMENT_MODE_ROUTING", "stratified").strip().lower()
# MUTATION 允许「并列拆分」等带来的实体条数增量上限（相对原句实体数）
# 主实验不允许 mutation 自由增加实体；一次只替换一个叶子实体。
MUTATION_MAX_EXTRA_ENTITIES = int(os.environ.get("MUTATION_MAX_EXTRA_ENTITIES", "0"))
# MUTATION：生成侧「外层严格包含内层」的 (type_outer,type_inner) 多重集须与原文一致，降低嵌套被展平却过检的概率（可设 0 关闭）
MUTATION_PRESERVE_CONTAINMENT_PAIRS = os.environ.get("MUTATION_PRESERVE_CONTAINMENT_PAIRS", "1") != "0"
# Mutation只在同类型且长度差不超过该值的真实实体之间替换，避免“腹→长蛋白名”等类型相同但语境失真的强制替换。
MUTATION_MAX_LENGTH_DELTA = int(os.environ.get("MUTATION_MAX_LENGTH_DELTA", "2"))
# 同类型并不等于表面形式兼容：默认要求替换前后是否含中文、拉丁字母、数字、希腊字母的模式一致，
# 只拦截明显的书写形态错配；这不是医学语义等价判定，后续仍须由 Teacher 过滤。
MUTATION_MATCH_SURFACE_PROFILE = os.environ.get("MUTATION_MATCH_SURFACE_PROFILE", "1") != "0"
# 不再从同类型的数千个真实实体中随机抽少量词。先用实体两侧的原始Train
# 上下文做确定性检索，再把前K个候选交给LLM判断医学和语法相容性。
MUTATION_CANDIDATE_COUNT = int(os.environ.get("MUTATION_CANDIDATE_COUNT", "10"))
if MUTATION_CANDIDATE_COUNT <= 0:
    raise ValueError("MUTATION_CANDIDATE_COUNT 必须为正整数")
MUTATION_CONTEXT_WINDOW = int(os.environ.get("MUTATION_CONTEXT_WINDOW", "12"))
if MUTATION_CONTEXT_WINDOW <= 0:
    raise ValueError("MUTATION_CONTEXT_WINDOW 必须为正整数")
# 仅是检索的计算预算：先用字符二元组倒排召回，再对至多200个候选执行
# SequenceMatcher精排。未用Dev F1选择该值；最终仍只输出前K项给LLM。
MUTATION_RETRIEVAL_SHORTLIST = int(
    os.environ.get("MUTATION_RETRIEVAL_SHORTLIST", "200")
)
if MUTATION_RETRIEVAL_SHORTLIST < MUTATION_CANDIDATE_COUNT:
    raise ValueError("MUTATION_RETRIEVAL_SHORTLIST 不能小于 MUTATION_CANDIDATE_COUNT")


class AugmentMode(str, Enum):
    LOCAL = "local"  # 方式1：局部模板
    BOUNDARY = "boundary"  # 方式2：边界干扰
    NEGATIVE = "negative"  # 方式3：无实体医学负例
    MUTATION = "mutation"  # 方式4：困难实体诱变（字面可替换，与 BOUNDARY 分立）


# 方式1：受控局部改写。实体、实体顺序与嵌套结构全部锁定，只允许改写
# 用户消息中列出的非实体上下文片段。
SYSTEM_PROMPT_LOCAL = f"""你是医学 NER 数据增强助手，严格遵守 CMeEE 标注规范。

用户消息提供原始句子、带字符跨度的实体锁定表、Gold XML结构和可编辑的非实体上下文片段。请完成一次**受控局部改写**。

【方式1 核心要求】
- 只能改写用户列出的非实体上下文片段；实体内部字符禁止增删改，实体出现顺序和语义位置禁止改变。允许因前文长度变化产生整体索引偏移。
- 在用户列出的可编辑上下文中改写1～3处；允许同义谓词、主动/被动表达、连接表达或局部语序变化，但必须保持原句事实、否定、程度、条件、时态、因果和比较关系不变。
- 至少一处改动应包含有实际语义的词语，不能仅作格式、空格或标点变化；不要求把改写限制为最小字面变化。
- 至少修改一个非标点、非空白字符；禁止原句照抄，也禁止只改空格或标点。
- 可编辑片段中已有的阿拉伯数字、中文数字日期/数量（如“第十二届”、“三例”）、百分比、拉丁字母串和型号必须按原顺序逐个保留；不得新增、删除、调换或改写这些事实令牌。
- 保持原句语域与文体，不得把论文、科普或说明文本强制改成病历。
- 全句总长度同时不得超过原句的 1.15 倍和 {MAX_AUGMENT_TEXT_LEN} 个字符。
- 禁止引入任何锁定表之外的具体医学实体、数值、检查结果、药物、疾病或解剖部位。

【实体铁律】
1. 锁定表中的每个实体必须且只能保留原类型、原字面和原相对顺序。
2. 实体数量、类型多重集和完整嵌套拓扑必须与Gold XML完全一致。
3. 禁止增加、删除、拆分、合并、改名或改变实体的相对语义位置，禁止给普通上下文增加标签。

【标注】
标签名仅能为 bod/dis/sym/mic/pro/ite/dep/dru/equ，须正确开闭配对，允许合法嵌套，禁止交叉错位。

【输出】
仅输出一个 JSON 对象，且键只能是 "text"；值为带XML标签的完整中文句。
【格式铁律】必须且只能输出一个合法 JSON 对象：不要任何解释性前后缀，不要 Markdown 代码块，不要以 ```json 开头；输出须可被 Python json.loads() 直接解析。"""


# 方式2：边界干扰；多样化边界词，避免对「水平」「降低」等后缀过拟合（须自洽写出铁律/标注/输出，勿写「同方式1」）
SYSTEM_PROMPT_BOUNDARY = f"""你是医学 NER 数据增强助手，严格遵守 CMeEE 标注规范。

用户消息含「原始句子」「原始标注结构（嵌套拓扑）」与「实体参考平铺 JSON」。请完成**边界干扰**增强。

【方式2 核心要求】
- 保持原始句子的语域、文体和句法复杂度，不得把论文、科普或说明文本强制改写为病历。
- 仅在 ite 或 equ 实体的紧邻左侧或右侧插入总计 1～3 个具有边界混淆性的非实体字符；所选边界不得严格位于任何其它 Gold 实体内部，以免改变外层实体字面。
- 干扰内容本身不得是 CMeEE 中需要标注的实体，也不得改变原实体字面、类型和嵌套关系。
- 除上述插入外，不得删除、替换或重写句子的任何其他部分；生成文本总长度同时不得超过原句长度的 1.15 倍和 {MAX_AUGMENT_TEXT_LEN} 个字符。

【三大铁律】
1. 实体零篡改：参考中的实体 value 必须一字不差出现在输出中。
2. 实体零遗漏：每个参考实体均须出现且用对应 XML 包裹。
3. 实体隔离：禁止新增参考列表之外的应标医学实体，也禁止给普通上下文添加 XML 标签。

【标注】
标签名仅能为 bod/dis/sym/mic/pro/ite/dep/dru/equ，须正确开闭配对，允许合法嵌套，禁止交叉错位。

【输出】
仅输出一个 JSON 对象，键只有 "text"，值为带 XML 标签的完整中文句。
【格式铁律】必须且只能输出一个合法 JSON 对象：不要任何解释性前后缀，不要 Markdown 代码块，不要以 ```json 开头；输出须可被 Python json.loads() 直接解析。"""


# 方式4：候选约束的单叶子实体诱变。候选全部来自原始Train同类型Gold，
# LLM负责在候选中选择更符合当前语境的一项，Python负责候选归属和结构校验。
SYSTEM_PROMPT_MUTATION = f"""你是医学 NER 数据增强助手，专门处理 CMeEE 数据集中的嵌套实体。

【你的任务】
每次只替换用户指定的一个叶子实体。你必须从用户提供的 allowed_replacements 中选择一个无需改动其它字符就能直接嵌入原句、且与医学语境、语法搭配和实体类型最相容的候选。

【七项铁律（必须严格遵守，否则视为失败）】
1. 剔除后缀污染：实体绝不能包含「者、部位、的、等」非医学后缀！
   ❌ 错误示范：<sym>运动抽动者</sym>，<bod>跖部位</bod>
   ✅ 正确示范：<sym>运动抽动</sym>者，<bod>跖</bod>部位
2. 候选约束：replacement必须逐字等于allowed_replacements中的一项；不得自己创造、缩写或改写候选。候选已按原始Train上下文相似度检索，但检索分数不等于医学正确性；你仍须独立判断。若所有候选都不符合当前医学语境或语法，必须拒绝本次诱变，不得勉强选择。
3. 语境优先：若某候选会改变原句事实、否定、程度、条件、时态、因果或比较关系，或必须改写上下文才能通顺，不得选择它。
4. 保持嵌套实体：须保持与参考完全一致的嵌套关系，不得新增、删除或展平任何嵌套关系。
5. 结构保真与字面分层：替换只作用于指定叶子span；原句其余字符必须逐字不变。其他非祖先实体的类型与字面必须保持不变；包含该叶子的父实体字面仅可因叶子替换被动更新，但类型、层级和兄弟顺序不得改变。禁止拆分或合并标签。
6. 文体与实体隔离：保持原句语域；除选定候选外禁止引入任何新的应标医学实体。
7. 长度上限：全句总长度同时不得超过原句的1.15倍和{MAX_AUGMENT_TEXT_LEN}个字符。

【输出格式】
仅输出一个 JSON 对象，且恰好包含两个键：
{{"replacement": "从allowed_replacements中选择的原样候选", "text": "带XML标签的新句子"}}
若没有任何语境相容候选，只能输出：
{{"replacement": "__NONE__", "text": ""}}
【格式铁律】必须且只能输出一个合法 JSON 对象：不要任何解释性前后缀，不要 Markdown 代码块，不要以 ```json 开头；输出须可被 Python json.loads() 直接解析。"""


# 方式3：无实体医学负例；不再声称仅凭提示词即可保证“困难负例”。
SYSTEM_PROMPT_NEGATIVE = """你是医学 NER 无实体负例生成助手。

【任务】
请生成 1 条中文医学相关句子（可以是学术论文结论、科普说明或笼统临床观察），内容为机制泛述、病情整体转归或一般性医学常识。
**绝不要**包含任何需要按 CMeEE 规范精细标注的具体专名（即：具体的疾病名、药物名、检验项目名、解剖部位名、微生物名、手术操作名等）。你可以使用「该指标」「相关症状」「常规治疗」「作用机制」等泛指词汇。

【硬性要求】
- 文本总长度（含标点）必须为 15～45 个字符，且只生成一句。
- 必须符合通顺的专业医学语境，不要生成强行拼凑的废话。
- 绝对不要使用任何 XML 标签。
- 句中不得出现任何可指向唯一疾病、药物、检验、解剖部位、微生物、医疗操作、科室或设备的具体专名。

【输出】
仅输出 JSON：{"text": "你的句子"} ，只有一个键 text。
【格式铁律】必须且只能输出一个合法 JSON 对象：不要任何解释性前后缀，不要 Markdown 代码块，不要以 ```json 开头；输出须可被 Python json.loads() 直接解析。"""


class DashScopeAPIError(RuntimeError):
    """DashScope 非 200 响应，携带 HTTP 状态与业务 code 便于重试判断。"""

    def __init__(self, status_code: int, code: str, message: str):
        self.status_code = status_code
        self.code = code or ""
        self.message = message or ""
        super().__init__(f"{status_code} code={self.code!r} message={self.message!r}")


def _retryable_dashscope_error(exc: BaseException) -> bool:
    if isinstance(exc, DashScopeAPIError):
        if exc.status_code in (429, 500, 502, 503, 504):
            return True
        c = exc.code or ""
        if "Throttling" in c or "RateQuota" in c or "LimitExceeded" in c:
            return True
    return isinstance(exc, (ConnectionError, TimeoutError, OSError))


@retry(
    stop=stop_after_attempt(6),
    wait=wait_random_exponential(min=1, max=60),
    retry=retry_if_exception(_retryable_dashscope_error),
)
def _generation_call(messages, temperature=0.2, max_tokens=4096):
    """原生百炼文本生成（Generation.call，result_format=message）。"""
    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("请先设置环境变量 DASHSCOPE_API_KEY")
    dashscope.api_key = api_key
    response = Generation.call(
        model=LLM_MODEL,
        messages=messages,
        result_format="message",
        temperature=temperature,
        top_p=0.9,
        max_tokens=max_tokens,
    )

    if response.status_code != 200:
        code = str(getattr(response, "code", "") or "")
        msg = getattr(response, "message", None) or str(response)
        raise DashScopeAPIError(response.status_code, code, msg)
    return response


def _strip_thinking_blocks(raw: str) -> str:
    if not raw:
        return ""
    s = raw
    for _pat in (
        r"<" + "think" + r">.*?</" + "think" + ">",
        r"<thinking>.*?</thinking>",
        r"<think>.*?</think>",
    ):
        s = re.sub(_pat, "", s, flags=re.DOTALL | re.IGNORECASE)
    return s.strip()


def build_gold_xml_structure(text: str, entities: list) -> str:
    """
    用半开区间 [start_idx, end_idx) 与 CMeEE 一致，将 Gold 实体嵌套关系编码为 XML 拓扑串，供 LOCAL/BOUNDARY/MUTATION 观察层级。
    """
    if not text or not entities:
        return text or ""
    ents = []
    for e in entities:
        try:
            s, ed = int(e["start_idx"]), int(e["end_idx"])
        except (TypeError, ValueError, KeyError):
            continue
        t = (e.get("type") or "").lower()
        if t not in VALID_TYPES or s < 0 or ed > len(text) or s >= ed:
            continue
        ents.append({"start_idx": s, "end_idx": ed, "type": t})
    if not ents:
        return text
    positions = sorted({0, len(text)} | {e["start_idx"] for e in ents} | {e["end_idx"] for e in ents})
    out = []
    cursor = 0
    for pos in positions:
        out.append(text[cursor:pos])
        for e in sorted((x for x in ents if x["end_idx"] == pos), key=lambda x: (x["end_idx"] - x["start_idx"])):
            out.append(f"</{e['type']}>")
        for e in sorted((x for x in ents if x["start_idx"] == pos), key=lambda x: -(x["end_idx"] - x["start_idx"])):
            out.append(f"<{e['type']}>")
        cursor = pos
    out.append(text[cursor:])
    return "".join(out)


def _entity_strictly_inside_span(inner_s, inner_e, outer_s, outer_e) -> bool:
    return inner_s >= outer_s and inner_e <= outer_e and (inner_s > outer_s or inner_e < outer_e)


def is_leaf_entity(ent: dict, all_entities: list) -> bool:
    """当前 span 内不包含其它实体的严格子 span 时视为叶子（可作诱变池替换字面）。"""
    try:
        es, ee = int(ent["start_idx"]), int(ent["end_idx"])
    except (TypeError, ValueError, KeyError):
        return False
    for other in all_entities:
        try:
            os_, oe_ = int(other["start_idx"]), int(other["end_idx"])
        except (TypeError, ValueError, KeyError):
            continue
        if os_ == es and oe_ == ee and other.get("type") == ent.get("type"):
            continue
        if _entity_strictly_inside_span(os_, oe_, es, ee):
            return False
    return True


def editable_context_segments(original_text: str, entities_ref: list) -> list:
    """返回不属于任何Gold实体的连续字符片段，供LOCAL提示词明确编辑边界。"""
    locked = [False] * len(original_text)
    for ent in entities_ref:
        start = ent.get("start_idx")
        end = ent.get("end_idx")
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        if not 0 <= start < end <= len(original_text):
            continue
        for index in range(start, end):
            locked[index] = True
    segments = []
    index = 0
    while index < len(original_text):
        if locked[index]:
            index += 1
            continue
        end = index + 1
        while end < len(original_text) and not locked[end]:
            end += 1
        value = original_text[index:end]
        if value:
            segments.append({"start_idx": index, "end_idx": end, "text": value})
        index = end
    return segments


def build_user_prompt(original_text: str, entities_ref: list, original_tagged_text: str) -> str:
    """给出带span的锁定表、可编辑上下文与Gold XML拓扑。"""
    ref_json = json.dumps(entities_ref, ensure_ascii=False)
    editable_json = json.dumps(
        editable_context_segments(original_text, entities_ref),
        ensure_ascii=False,
    )
    tagged = (original_tagged_text or "").strip()
    tagged_line = f"Gold XML结构（实体层级模板）：{tagged}\n" if tagged else ""
    return (
        f"原始句子：{original_text}\n"
        f"{tagged_line}"
        f"实体锁定表（原文半开区间，仅用于确认身份；改写后span可整体移动）：{ref_json}\n"
        f"允许改写的非实体上下文片段：{editable_json}\n"
        f"输出剥离XML标签后的纯文本长度不得超过：{MAX_AUGMENT_TEXT_LEN} 个字符。\n"
        "只改写1～3个允许片段；实体字面、类型、数量、顺序和Gold嵌套结构必须保持。"
    )


def filter_leaf_mutation_spans(entities_with_span):
    """
    诱变仅针对叶子 span：同类字面由池替换，外层嵌套父 span 不被整块替换，避免嵌套拓扑被「叶子池纯文本」降维。
    在叶子集合上按起点贪心选取两两无字符交集的 span，避免同一字符位多条 forced 冲突。
    """
    if not entities_with_span:
        return []
    leaves = [e for e in entities_with_span if is_leaf_entity(e, entities_with_span)]
    if not leaves:
        return []
    leaves.sort(key=lambda x: (x["start_idx"], -(x["end_idx"] - x["start_idx"])))
    picked = []
    for ent in leaves:
        s, e = ent["start_idx"], ent["end_idx"]
        if any(max(s, p["start_idx"]) < min(e, p["end_idx"]) for p in picked):
            continue
        picked.append(ent)
    return picked


def _entity_surface_profile(value: str) -> tuple:
    """实体的粗粒度书写形态；不声称可判断医学语义相容性。"""
    value = value or ""
    return (
        bool(re.search(r"[\u4e00-\u9fff]", value)),
        bool(re.search(r"[A-Za-z]", value)),
        bool(re.search(r"\d", value)),
        bool(re.search(r"[\u0370-\u03ff]", value)),
    )


def _mutation_candidate_is_compatible(original: str, candidate: str) -> bool:
    if candidate == original:
        return False
    if abs(len(candidate) - len(original)) > MUTATION_MAX_LENGTH_DELTA:
        return False
    if MUTATION_MATCH_SURFACE_PROFILE:
        return _entity_surface_profile(candidate) == _entity_surface_profile(original)
    return True


def _mutation_candidate_value(candidate) -> str:
    """兼容V9.3结构化候选和旧测试中的纯字符串候选。"""
    if isinstance(candidate, dict):
        return str(candidate.get("value", "")).strip()
    return str(candidate or "").strip()


def _mutation_pool_records(pool_t) -> list:
    """V9.3池为records+倒排索引；兼容旧测试直接传list。"""
    if isinstance(pool_t, dict):
        records = pool_t.get("records") or []
        return records if isinstance(records, list) else []
    return pool_t if isinstance(pool_t, list) else []


def _masked_entity_context(text: str, start: int, end: int) -> str:
    """固定窗口上下文；中心实体统一掩码，避免字面本身主导检索。"""
    if not isinstance(text, str) or not 0 <= int(start) < int(end) <= len(text):
        return ""
    start, end = int(start), int(end)
    left = text[max(0, start - MUTATION_CONTEXT_WINDOW):start]
    right = text[end:min(len(text), end + MUTATION_CONTEXT_WINDOW)]
    return f"{left}<ENTITY>{right}"


def _context_bigrams(context: str) -> set:
    """实体左右分别取字符二元组，禁止二元组跨过掩码中心。"""
    grams = set()
    for side in str(context or "").split("<ENTITY>"):
        compact = re.sub(r"\s+", "", side)
        grams.update(compact[i:i + 2] for i in range(max(0, len(compact) - 1)))
    return {gram for gram in grams if len(gram) == 2}


def _retrieve_mutation_candidates(pool_t, target_context: str, original: str) -> list:
    """倒排召回+固定预算；无倒排索引时回退到完整兼容池。"""
    records = _mutation_pool_records(pool_t)
    compatible_indices = [
        index
        for index, record in enumerate(records)
        if _mutation_candidate_is_compatible(
            original, _mutation_candidate_value(record)
        )
    ]
    if not isinstance(pool_t, dict) or not pool_t.get("bigram_index"):
        return [records[index] for index in compatible_indices]

    compatible_set = set(compatible_indices)
    overlap_counts = Counter()
    for gram in _context_bigrams(target_context):
        for index in pool_t["bigram_index"].get(gram, ()):
            if index in compatible_set:
                overlap_counts[index] += 1

    if not overlap_counts:
        # 极短/特殊上下文没有二元组命中时，候选规模通常较小；完整精排保证不静默失去可用候选。
        return [records[index] for index in compatible_indices]

    ranked_indices = sorted(
        overlap_counts,
        key=lambda index: (
            -overlap_counts[index],
            hashlib.sha256(
                f"{RANDOM_SEED}\0{target_context}\0{_mutation_candidate_value(records[index])}".encode("utf-8")
            ).hexdigest(),
        ),
    )
    shortlist_limit = min(MUTATION_RETRIEVAL_SHORTLIST, len(compatible_indices))
    if len(ranked_indices) < shortlist_limit:
        ranked_set = set(ranked_indices)
        remaining = sorted(
            (index for index in compatible_indices if index not in ranked_set),
            key=lambda index: hashlib.sha256(
                f"{RANDOM_SEED}\0{target_context}\0{_mutation_candidate_value(records[index])}".encode("utf-8")
            ).hexdigest(),
        )
        ranked_indices.extend(remaining[:shortlist_limit - len(ranked_indices)])
    return [
        records[index]
        for index in ranked_indices[:shortlist_limit]
    ]


def _mutation_candidate_contexts(candidate) -> list:
    if not isinstance(candidate, dict):
        return []
    contexts = candidate.get("contexts") or []
    return [str(value) for value in contexts if isinstance(value, str) and value]


def _best_context_match(target_context: str, candidate) -> tuple:
    """返回(相似度, 最相近Train上下文)；只用于排序，不作为医学正确性门禁。"""
    best_score, best_context = 0.0, ""
    for context in _mutation_candidate_contexts(candidate):
        score = difflib.SequenceMatcher(
            None, target_context, context, autojunk=False
        ).ratio()
        if score > best_score or (score == best_score and context < best_context):
            best_score, best_context = score, context
    return best_score, best_context


def sample_mutation_forced_replacements(
    entities_ref: list,
    real_entities_pool: dict,
    source_text: str = "",
    rng=None,
) -> list:
    """
    为一个叶子实体生成真实同类型候选集。

    候选先按掩码后的左右Train上下文相似度确定性排序，再交给LLM选择；
    上下文相似度不是医学等价判定，因此仍保留LLM拒绝和Teacher过滤。
    """
    _ = rng  # 保留调用兼容；候选排名不再依赖随机抽样。
    forced = []
    for ent in entities_ref:
        t = ent["type"]
        v = ent.get("value", "").strip()
        pool_t = real_entities_pool.get(t) or []
        target_context = _masked_entity_context(
            source_text,
            int(ent["start_idx"]),
            int(ent["end_idx"]),
        )
        ranked_by_value = {}
        for candidate in _retrieve_mutation_candidates(pool_t, target_context, v):
            value = _mutation_candidate_value(candidate)
            if not _mutation_candidate_is_compatible(v, value):
                continue
            score, evidence_context = _best_context_match(target_context, candidate)
            previous = ranked_by_value.get(value)
            current = (score, evidence_context)
            if previous is None or current > previous:
                ranked_by_value[value] = current
        if not ranked_by_value:
            # 不再退化到任意长度的同类型词；调用方应改选其它叶子或放弃本次Mutation。
            continue

        def rank_key(entry):
            value, (score, _context) = entry
            tie = hashlib.sha256(
                f"{RANDOM_SEED}\0{source_text}\0{ent['start_idx']}\0{value}".encode("utf-8")
            ).hexdigest()
            return (-score, tie, value)

        ranked = sorted(ranked_by_value.items(), key=rank_key)
        candidate_count = min(MUTATION_CANDIDATE_COUNT, len(ranked))
        selected = ranked[:candidate_count]
        allowed = [value for value, _meta in selected]
        candidate_evidence = [
            {
                "value": value,
                "train_context": context,
                "context_similarity": round(float(score), 6),
            }
            for value, (score, context) in selected
        ]
        forced.append({
            "type": t,
            "original": v,
            "allowed_replacements": allowed,
            "source_context": target_context,
            "candidate_evidence": candidate_evidence,
            "candidate_retrieval": "masked_train_context_bigram_recall_then_similarity",
            "start_idx": int(ent["start_idx"]),
            "end_idx": int(ent["end_idx"]),
        })
    return forced


def bind_selected_mutation_replacement(mutation_forced: list, selected: str):
    """将LLM选择绑定为强制替换；选择不属于候选集时fail closed。"""
    if len(mutation_forced or []) != 1 or not isinstance(selected, str):
        return None
    selected = selected.strip()
    forced = dict(mutation_forced[0])
    allowed = [str(value).strip() for value in forced.get("allowed_replacements", [])]
    if not selected or selected not in allowed:
        return None
    forced["must_use_as_replacement"] = selected
    return [forced]


def has_compatible_mutation_candidate(ent: dict, real_entities_pool: dict) -> bool:
    value = (ent.get("value") or "").strip()
    return any(
        _mutation_candidate_is_compatible(value, _mutation_candidate_value(candidate))
        for candidate in _mutation_pool_records(
            real_entities_pool.get(ent.get("type")) or []
        )
    )


def build_mutation_user_prompt(
    original_text: str, entities_ref: list, forced: list, original_tagged_text: str = ""
) -> str:
    """候选约束诱变：LLM只负责从真实Train同类型候选中选择。"""
    ref_json = json.dumps(entities_ref, ensure_ascii=False)
    forced_json = json.dumps(forced, ensure_ascii=False)
    # 勿用 original 字面作 dict 键：同句多处相同原文会被覆盖，导致与 forced_json 矛盾
    mapping_summary = [
        {
            "start_idx": int(x["start_idx"]),
            "end_idx": int(x["end_idx"]),
            "original": x["original"],
            "allowed_replacements": x["allowed_replacements"],
        }
        for x in forced
    ]
    mapping_json = json.dumps(mapping_summary, ensure_ascii=False)
    tagged = (original_tagged_text or "").strip()
    tagged_line = f"原始标注结构（Gold 嵌套拓扑）：{tagged}\n" if tagged else ""

    return (
        f"原始句子：{original_text}\n"
        f"{tagged_line}"
        f"原始实体参考：{ref_json}\n\n"
        f"【强制要求】\n"
        "只处理下面唯一一个叶子实体。先比较allowed_replacements与原句医学语境和语法搭配，"
        "并参考candidate_evidence中的原始Train上下文，再选择最相容的一项；不能自行创造替换词。"
        "context_similarity只用于候选检索排序，不代表医学等价或必须选择。\n"
        "如果全部候选都不符合医学语境或语法，按system要求输出__NONE__并放弃，禁止强行替换。\n"
        f"候选约束：{forced_json}\n"
        f"候选摘要（按原文[start_idx,end_idx)定位）：{mapping_json}\n"
        f"输出剥离XML标签后的纯文本长度不得超过：{MAX_AUGMENT_TEXT_LEN} 个字符。\n"
    )


_NEGATIVE_SEED_THEMES = (
    "学术论文式结论概述",
    "科普向通俗说明",
    "笼统临床观察",
    "疾病机制泛述",
    "病情整体转归",
    "一般医学常识",
)


def build_negative_user_prompt(rng=None) -> str:
    """无实体医学负例：随机语域倾向，不注入原句文本以免带入专名。"""
    rng = rng or random
    theme = rng.choice(_NEGATIVE_SEED_THEMES)
    return f"语域/主题倾向：{theme}。请生成 1 条符合 system 要求的无实体医学句子（无 XML 标签）。"


def call_llm_augmentation(
    mode: AugmentMode,
    original_text: str,
    entities_ref: list,
    real_entities_pool=None,
    mutation_forced=None,
    original_tagged_text: str = "",
    rng=None,
) -> str:
    if mode == AugmentMode.NEGATIVE:
        user_content = build_negative_user_prompt(rng=rng)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_NEGATIVE},
            {"role": "user", "content": user_content},
        ]
        response = _generation_call(messages, temperature=0.85, max_tokens=512)
    else:
        if mode == AugmentMode.MUTATION and mutation_forced:
            user_content = build_mutation_user_prompt(
                original_text, entities_ref, mutation_forced, original_tagged_text
            )
        else:
            user_content = build_user_prompt(original_text, entities_ref, original_tagged_text)
        if mode == AugmentMode.MUTATION:
            system = SYSTEM_PROMPT_MUTATION
            temperature, max_tokens = 0.3, 4096
        elif mode == AugmentMode.BOUNDARY:
            system = SYSTEM_PROMPT_BOUNDARY
            temperature, max_tokens = 0.7, 4096
        else:
            system = SYSTEM_PROMPT_LOCAL
            temperature, max_tokens = 0.4, 4096
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]
        response = _generation_call(messages, temperature=temperature, max_tokens=max_tokens)
    output = getattr(response, "output", None)
    choices = getattr(output, "choices", None) if output is not None else None
    if not choices:
        return ""
    choice = choices[0]
    if isinstance(choice, dict):
        msg = choice.get("message") or {}
        raw_content = (msg.get("content") if isinstance(msg, dict) else "") or ""
    else:
        msg = getattr(choice, "message", None)
        raw_content = (getattr(msg, "content", None) or "") if msg is not None else ""
    return _strip_thinking_blocks(raw_content)


def pick_augment_mode(item: dict, rng=None) -> AugmentMode:
    """
    在单条样本上抽样增强方式（BOUNDARY 与 MUTATION 由枚举区分，互不覆盖）。

    - stratified（默认）：只在 GENERATION_ALLOWED_MODES 允许且当前样本可用的
      分支中按权重归一化。主实验默认仅local/mutation。
    - cascade（AUGMENT_MODE_ROUTING=cascade）：旧版瀑布；P_AUG_MUTATION=0 时原属 MUTATION 的流量会塌陷到 LOCAL，消融勿用。

    BOUNDARY 触发：仅当句中含 ite 或 equ 时进入边界干扰候选池（sym 走 MUTATION/LOCAL；复用 P_AUG_BOUNDARY_IF_ITE 权重）。
    """
    rng = rng or random
    entities = item.get("entities", [])
    # 若目标实体嵌在另一个实体内部，在其内部边界插字会同步篡改外层实体字面，
    # 与 BOUNDARY 的“所有实体零篡改”约束冲突。这里只把至少存在一个安全边界的样本纳入候选。
    has_boundary_target = bool(_safe_boundary_positions(entities))

    allowed = set(GENERATION_ALLOWED_MODES)
    if AUGMENT_MODE_ROUTING != "cascade":
        branches = []
        if "negative" in allowed:
            branches.append((AugmentMode.NEGATIVE, max(0.0, P_AUG_NEGATIVE)))
        if "boundary" in allowed and has_boundary_target:
            branches.append((AugmentMode.BOUNDARY, max(0.0, P_AUG_BOUNDARY_IF_ITE)))
        if "mutation" in allowed and entities and P_AUG_MUTATION > 0:
            branches.append((AugmentMode.MUTATION, max(0.0, P_AUG_MUTATION)))
        if "local" in allowed and entities:
            branches.append((AugmentMode.LOCAL, max(0.0, P_AUG_LOCAL)))
        total = sum(w for _, w in branches)
        if total <= 0:
            if "negative" in allowed:
                return AugmentMode.NEGATIVE
            raise ValueError(
                "当前样本没有任何可用生成模式；主实验正例需要至少一个Gold实体"
            )
        r = rng.uniform(0.0, total)
        acc = 0.0
        for mode, w in branches:
            acc += w
            if r < acc:
                return mode
        return branches[-1][0]

    if "negative" in allowed and rng.random() < P_AUG_NEGATIVE:
        return AugmentMode.NEGATIVE
    if "boundary" in allowed and has_boundary_target and rng.random() < P_AUG_BOUNDARY_IF_ITE:
        return AugmentMode.BOUNDARY
    if "mutation" in allowed and entities and rng.random() < P_AUG_MUTATION:
        return AugmentMode.MUTATION
    if "local" in allowed and entities:
        return AugmentMode.LOCAL
    if "negative" in allowed:
        return AugmentMode.NEGATIVE
    raise ValueError("cascade路由后没有可用生成模式")


def _strip_entity_spans_inplace(text: str, entities: list) -> bool:
    """
    按跨度长度升序收紧首尾空白（先处理较短 span，利于嵌套），并写入 entity 字面。
    LLM 偶发输出 <dis> 肺炎 </dis> 时避免空格污染索引。
    """
    order = sorted(
        range(len(entities)),
        key=lambda i: (
            entities[i]["end_idx"] - entities[i]["start_idx"],
            entities[i]["start_idx"],
        ),
    )
    for i in order:
        ent = entities[i]
        s, e = ent["start_idx"], ent["end_idx"]
        raw = text[s:e]
        ls = raw.lstrip()
        lead = len(raw) - len(ls)
        rs = ls.rstrip()
        if not rs:
            return False
        trail = len(ls) - len(rs)
        ent["start_idx"] = s + lead
        ent["end_idx"] = e - trail
        ent["entity"] = rs
    return True


def parse_llm_xml_tagged_text(tagged_text: str):
    """
    解析 XML 内联标签，得到剥离标签后的 text 与实体列表（end_idx 为开区间）。
    非法结构（未闭合、闭合无匹配等）返回 None。
    """
    if not tagged_text or not tagged_text.strip():
        return None

    # 允许 LLM 偶发写出带属性的标签（如 <dis type="main">），属性在剥离时忽略
    tag_pattern = re.compile(rf"<(/)?\s*({_VALID_TAGS_ALT})\b[^>]*>", re.IGNORECASE)
    clean_text = ""
    entities = []
    stack = []
    last_idx = 0

    for match in tag_pattern.finditer(tagged_text):
        text_chunk = tagged_text[last_idx : match.start()]
        clean_text += text_chunk
        is_close = match.group(1) == "/"
        tag_type = match.group(2).lower()

        if not is_close:
            stack.append((tag_type, len(clean_text)))
        else:
            # XML 必须严格后进先出；不能向下搜索同名标签，否则交叉闭合的字符串
            # 可能在剥离后偶然形成嵌套 span 并通过后续结构校验。
            if not stack or stack[-1][0] != tag_type:
                return None
            _t, start_idx = stack.pop()
            end_idx = len(clean_text)
            entities.append({
                "start_idx": start_idx,
                "end_idx": end_idx,
                "type": tag_type,
                "entity": clean_text[start_idx:end_idx],
            })
        last_idx = match.end()

    clean_text += tagged_text[last_idx:]
    if stack:
        return None
    if not entities:
        return None
    if not _strip_entity_spans_inplace(clean_text, entities):
        return None
    return {"text": clean_text, "entities": entities}


def _safe_json_loads_object(raw: str):
    """剥离 Markdown 与首尾废话，尽量解析出单个 JSON 对象。"""
    s = (raw or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s)
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    i = s.find("{")
    j = s.rfind("}")
    if i >= 0 and j > i:
        try:
            obj = json.loads(s[i : j + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def parse_llm_response(clean_content: str, *, require_replacement=False):
    """解析正例JSON；MUTATION额外要求候选集中的replacement字段。"""
    obj = _safe_json_loads_object(clean_content)
    if not obj:
        return None
    allowed_keys = {"text", "replacement"} if require_replacement else {"text"}
    if set(obj) != allowed_keys:
        return None
    replacement = obj.get("replacement") if require_replacement else None
    if require_replacement and replacement == "__NONE__":
        if obj.get("text") != "":
            return None
        return {"mutation_abstained": True, "selected_replacement": "__NONE__"}
    tagged = obj.get("text")
    if tagged is None or not isinstance(tagged, str):
        return None
    parsed = parse_llm_xml_tagged_text(tagged)
    if parsed is None:
        return None
    if require_replacement:
        if not isinstance(replacement, str) or not replacement.strip():
            return None
        parsed["selected_replacement"] = replacement.strip()
    return parsed


def parse_llm_response_negative(clean_content: str):
    """无实体医学负例：JSON 中 text 为纯文本，无 XML 标签；entities 恒为空。"""
    obj = _safe_json_loads_object(clean_content)
    if not obj:
        return None
    text = obj.get("text")
    if text is None or not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None
    if RESIDUAL_TAG_IN_TEXT.search(text):
        return None
    if RESIDUAL_XML_TAG_IN_TEXT.search(text):
        return None
    if has_malformed_html_tag(text):
        return None
    if has_illegal_hallucinated_tags(text):
        return None
    if _ZERO_WIDTH_RE.search(text):
        return None
    return {"text": text, "entities": []}


def get_entity_value(item_text: str, ent: dict) -> str:
    value = ent.get("entity", ent.get("value"))
    if value is not None:
        return value
    start, end = ent.get("start_idx"), ent.get("end_idx")
    if isinstance(start, int) and isinstance(end, int) and 0 <= start <= end <= len(item_text):
        return item_text[start:end]
    return ""


def build_real_entities_pool(data):
    """
    从训练集按类型收集真实标注实体字面及其左右上下文；仅收录叶子 span
    （内部不含其它实体），避免嵌套整段纯文本作替换导致子类型标签丢失。
    保留所有通过结构/HTML检查的叶子实体。长度兼容性在实际抽样时逐目标判断，
    不再用“只保留长词”的池把单字实体强制替换成长术语。

    同一字面在Train中的多个上下文全部保留并去重；实际检索时取与目标上下文
    最相近的证据。池仍只来自原始Train Gold，不读取Dev/Test。
    """
    from collections import defaultdict

    by_type_value_contexts = defaultdict(lambda: defaultdict(set))

    for item in data:
        text = item.get("text", "")
        ents = [e for e in item.get("entities", []) if e.get("type") in VALID_TYPES]
        for e in ents:
            if not is_leaf_entity(e, ents):
                continue
            val = _normalize_seed_pool_value(get_entity_value(text, e))
            if not val:
                continue
            if _ZERO_WIDTH_RE.search(val):
                continue
            if has_malformed_html_tag(val) or has_illegal_hallucinated_tags(val):
                continue
            ent_type = e.get("type")
            context = _masked_entity_context(
                text,
                int(e["start_idx"]),
                int(e["end_idx"]),
            )
            if context:
                by_type_value_contexts[ent_type][val].add(context)

    pool = {}
    for t in VALID_TYPES:
        records = [
            {
                "value": value,
                "contexts": sorted(contexts),
            }
            for value, contexts in sorted(by_type_value_contexts.get(t, {}).items())
        ]
        bigram_index = defaultdict(list)
        for index, record in enumerate(records):
            grams = set()
            for context in record["contexts"]:
                grams.update(_context_bigrams(context))
            for gram in grams:
                bigram_index[gram].append(index)
        pool[t] = {
            "records": records,
            "bigram_index": {
                gram: tuple(indices)
                for gram, indices in bigram_index.items()
            },
        }
    return pool


def entity_fingerprint(entities: list, item_text: str = ""):
    return sorted(
        (e["type"], get_entity_value(item_text, e).strip()) for e in entities
    )


def has_malformed_html_tag(text: str) -> bool:
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


def _span_strictly_contains(outer: tuple, inner: tuple) -> bool:
    """outer 的字符区间真包含 inner（不等同）。"""
    o0, o1 = outer
    i0, i1 = inner
    return o0 <= i0 and i1 <= o1 and (o0, o1) != (i0, i1)


def _normalize_seed_pool_value(raw: str) -> str:
    """与 validate 诱变字面规则一致：去掉首尾空白及常见中英文标点，避免池中种子自带标点导致必失败。"""
    v = (raw or "").strip()
    v = re.sub(r"^[\s，。、；：,.;:]+|[\s，。、；：,.;:]+$", "", v)
    return v.strip()


def _cross_overlap_invalid(entities: list) -> bool:
    """若存在相交但不互相包含的跨度对，返回 True（非法）。完全相同的 (start,end) 对在 validate_and_merge 中已拦截。"""
    n = len(entities)
    for i in range(n):
        for j in range(i + 1, n):
            s1, e1 = entities[i]["start_idx"], entities[i]["end_idx"]
            s2, e2 = entities[j]["start_idx"], entities[j]["end_idx"]
            if s1 < e2 and s2 < e1:
                is_inclusive = (s1 >= s2 and e1 <= e2) or (s2 >= s1 and e2 <= e1)
                if not is_inclusive:
                    return True
    return False


def _has_duplicate_span(entities: list) -> bool:
    spans = [(int(e["start_idx"]), int(e["end_idx"])) for e in entities]
    return len(spans) != len(set(spans))


def _source_structure_supported(entities: list) -> bool:
    """XML内联标签只能无损表达互不交叉且不同span的层级结构。"""
    return not _has_duplicate_span(entities) and not _cross_overlap_invalid(entities)


def _typed_nesting_forest_signature(entities: list):
    """实体类型标注的直接包含森林；比仅统计(type_outer,type_inner)对更严格。"""
    if _has_duplicate_span(entities) or _cross_overlap_invalid(entities):
        return None
    n = len(entities)
    parents = [None] * n
    for child_idx, child in enumerate(entities):
        cs, ce = int(child["start_idx"]), int(child["end_idx"])
        candidates = []
        for parent_idx, parent in enumerate(entities):
            if parent_idx == child_idx:
                continue
            ps, pe = int(parent["start_idx"]), int(parent["end_idx"])
            if _span_strictly_contains((ps, pe), (cs, ce)):
                candidates.append((pe - ps, ps, pe, parent_idx))
        if candidates:
            parents[child_idx] = min(candidates)[-1]

    children = [[] for _ in range(n)]
    roots = []
    for idx, parent_idx in enumerate(parents):
        if parent_idx is None:
            roots.append(idx)
        else:
            children[parent_idx].append(idx)

    def node_signature(idx):
        child_sigs = [node_signature(c) for c in children[idx]]
        child_sigs.sort(key=repr)
        return (entities[idx]["type"], tuple(child_sigs))

    root_sigs = [node_signature(i) for i in roots]
    root_sigs.sort(key=repr)
    return tuple(root_sigs)


def _direct_parent_indices(entities: list):
    """返回每个节点的最近严格包含父节点；非法结构返回 None。"""
    if _has_duplicate_span(entities) or _cross_overlap_invalid(entities):
        return None
    parents = [None] * len(entities)
    for child_idx, child in enumerate(entities):
        cs, ce = int(child["start_idx"]), int(child["end_idx"])
        candidates = []
        for parent_idx, parent in enumerate(entities):
            if parent_idx == child_idx:
                continue
            ps, pe = int(parent["start_idx"]), int(parent["end_idx"])
            if _span_strictly_contains((ps, pe), (cs, ce)):
                candidates.append((pe - ps, ps, pe, parent_idx))
        if candidates:
            parents[child_idx] = min(candidates)[-1]
    return parents


def _anchored_mutation_forest_signature(
    entities: list,
    item_text: str,
    forced_index_to_id: dict,
):
    """
    构造有序、带Mutation锚点的森林：
    - 强制替换叶子用 forced_id 标记；
    - 其祖先允许字面随叶子变化，但记录其包含哪些 forced_id；
    - 其它节点必须保留 type+value；
    - 根与兄弟节点按字符位置排序，禁止把整条嵌套分支换位后蒙混过关。
    """
    parents = _direct_parent_indices(entities)
    if parents is None:
        return None

    descendant_forced_ids = [set() for _ in entities]
    for node_idx, forced_id in forced_index_to_id.items():
        parent_idx = parents[node_idx]
        while parent_idx is not None:
            descendant_forced_ids[parent_idx].add(forced_id)
            parent_idx = parents[parent_idx]

    children = [[] for _ in entities]
    roots = []
    for idx, parent_idx in enumerate(parents):
        if parent_idx is None:
            roots.append(idx)
        else:
            children[parent_idx].append(idx)

    def position_key(idx):
        e = entities[idx]
        return (int(e["start_idx"]), int(e["end_idx"]), e["type"])

    def node_signature(idx):
        ent = entities[idx]
        if idx in forced_index_to_id:
            label = ("forced", forced_index_to_id[idx], ent["type"])
        elif descendant_forced_ids[idx]:
            label = (
                "forced_ancestor",
                ent["type"],
                tuple(sorted(descendant_forced_ids[idx])),
            )
        else:
            label = (
                "fixed",
                ent["type"],
                get_entity_value(item_text, ent).strip(),
            )
        ordered_children = tuple(
            node_signature(child_idx)
            for child_idx in sorted(children[idx], key=position_key)
        )
        return (label, ordered_children)

    return tuple(node_signature(idx) for idx in sorted(roots, key=position_key))


def _mutation_replacement_is_anchored(
    original_entities: list,
    generated_entities: list,
    generated_text: str,
    mutation_forced: list,
) -> bool:
    """确认每个强制替换词仍位于原叶子的同一祖先路径和兄弟顺序中。"""
    if not mutation_forced:
        return False

    original_forced_indices = {}
    for forced_id, forced in enumerate(mutation_forced):
        span = (int(forced["start_idx"]), int(forced["end_idx"]))
        matches = [
            idx
            for idx, ent in enumerate(original_entities)
            if (int(ent["start_idx"]), int(ent["end_idx"])) == span
            and ent["type"] == forced["type"]
        ]
        if len(matches) != 1 or matches[0] in original_forced_indices:
            return False
        original_forced_indices[matches[0]] = forced_id

    expected = _anchored_mutation_forest_signature(
        original_entities, "", original_forced_indices
    )
    if expected is None:
        return False

    candidate_indices = []
    for forced in mutation_forced:
        replacement = forced["must_use_as_replacement"].strip()
        candidates = [
            idx
            for idx, ent in enumerate(generated_entities)
            if ent["type"] == forced["type"]
            and get_entity_value(generated_text, ent).strip() == replacement
        ]
        if not candidates:
            return False
        candidate_indices.append(candidates)

    # 当前主实验每条只替换一个叶子；保留有限枚举以防以后扩展为少量多叶子。
    checked = 0
    for assignment in itertools.product(*candidate_indices):
        if len(set(assignment)) != len(assignment):
            continue
        checked += 1
        if checked > 512:
            return False
        generated_forced_indices = {
            node_idx: forced_id for forced_id, node_idx in enumerate(assignment)
        }
        actual = _anchored_mutation_forest_signature(
            generated_entities, generated_text, generated_forced_indices
        )
        if actual == expected:
            return True
    return False


def _occurrence_aware_containment_signature(entities: list, item_text: str):
    """以(type,value,同字面出现序号)标识节点，防止内层实体换到另一个父实体仍过检。"""
    grouped = {}
    for idx, ent in enumerate(entities):
        value = get_entity_value(item_text, ent).strip()
        grouped.setdefault((ent["type"], value), []).append(idx)

    node_ids = {}
    for key, indices in grouped.items():
        indices.sort(key=lambda i: (int(entities[i]["start_idx"]), int(entities[i]["end_idx"])))
        for rank, idx in enumerate(indices):
            node_ids[idx] = (key[0], key[1], rank)

    edges = []
    for i, outer in enumerate(entities):
        os_, oe_ = int(outer["start_idx"]), int(outer["end_idx"])
        for j, inner in enumerate(entities):
            if i == j:
                continue
            is_, ie_ = int(inner["start_idx"]), int(inner["end_idx"])
            if _span_strictly_contains((os_, oe_), (is_, ie_)):
                edges.append((node_ids[i], node_ids[j]))
    return tuple(sorted(edges, key=repr))


def _safe_boundary_positions(original_entities: list):
    """返回不会落入任何其它实体内部、因而不会改变外层实体字面的 ite/equ 边界。"""
    candidate_positions = {
        int(pos)
        for ent in original_entities
        if ent.get("type") in {"ite", "equ"}
        for pos in (ent["start_idx"], ent["end_idx"])
    }
    return {
        pos
        for pos in candidate_positions
        if not any(
            int(other["start_idx"]) < pos < int(other["end_idx"])
            for other in original_entities
        )
    }


def _boundary_only_insertion_is_valid(original_text: str, generated_text: str, original_entities: list) -> bool:
    """BOUNDARY必须且只能在ite/equ边界插入总计1～3个非空字符。"""
    delta = len(generated_text) - len(original_text)
    if delta < 1 or delta > 3:
        return False
    valid_positions = _safe_boundary_positions(original_entities)
    if not valid_positions:
        return False

    inserted_total = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
        None, original_text, generated_text, autojunk=False
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


_MEANINGFUL_CONTEXT_CHAR_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")
_FACT_TOKEN_RE = re.compile(
    r"[A-Za-z]+(?:[-_][A-Za-z0-9]+)*"
    r"|\d+(?:[./:%+−-]\d+)*"
    r"|第?[零〇一二三四五六七八九十百千万亿两]+"
    r"(?:年|月|日|岁|届|期|次|例|人|名|个|条|项|级|型|%|％)"
)


def _edit_overlaps_entity(start: int, end: int, entities: list) -> bool:
    """判断源/目标侧差异片段是否进入实体内部；零长度插入仅禁止落在实体内部。"""
    if start == end:
        return any(int(ent["start_idx"]) < start < int(ent["end_idx"]) for ent in entities)
    return any(
        start < int(ent["end_idx"]) and int(ent["start_idx"]) < end
        for ent in entities
    )


def _local_context_edit_is_valid(
    original_text: str,
    generated_text: str,
    original_entities: list,
    generated_entities: list,
) -> bool:
    """
    LOCAL 必须只编辑实体外上下文，且差异块为 1～3 个。

    实体字面、类型、数量和嵌套由 validate_and_merge 独立校验；这里补上旧版
    缺失的“改动确实位于提示词列出的非实体区域”约束。difflib 对重复字面可能
    选择保守对齐，因此宁可拒绝边界不明确的样本，也不静默接收实体内改动。
    """
    # 提示词只能降低事实漂移概率；这里以确定性门禁保证数值、日期、百分比、
    # 拉丁字母串和型号的有序序列不发生改变。实体字面本身已由独立门禁锁定。
    if tuple(_FACT_TOKEN_RE.findall(original_text)) != tuple(
        _FACT_TOKEN_RE.findall(generated_text)
    ):
        return False

    edits = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
        None, original_text, generated_text, autojunk=False
    ).get_opcodes():
        if tag == "equal":
            continue
        if _edit_overlaps_entity(i1, i2, original_entities):
            return False
        if _edit_overlaps_entity(j1, j2, generated_entities):
            return False
        edits.append((original_text[i1:i2], generated_text[j1:j2]))

    if not 1 <= len(edits) <= 3:
        return False
    return any(
        _MEANINGFUL_CONTEXT_CHAR_RE.search(before + after)
        and re.sub(r"[\s，。！？、；：,.!?;:\-]", "", before)
        != re.sub(r"[\s，。！？、；：,.!?;:\-]", "", after)
        for before, after in edits
    )


def _mutation_is_exact_leaf_replacement(
    original_text: str,
    generated_text: str,
    mutation_forced: list,
) -> bool:
    """主协议的Mutation只能替换一个指定叶子span，源句其余字符逐字不变。"""
    if len(mutation_forced or []) != 1:
        return False
    forced = mutation_forced[0]
    start, end = int(forced["start_idx"]), int(forced["end_idx"])
    replacement = forced.get("must_use_as_replacement")
    if not isinstance(replacement, str) or not 0 <= start < end <= len(original_text):
        return False
    expected = original_text[:start] + replacement + original_text[end:]
    return generated_text == expected


def validate_and_merge(
    original_entities,
    generated_item,
    *,
    allow_entity_value_change: bool = False,
    mutation_forced=None,
):
    """
    校验生成样本。默认：参考与生成须类型+字面 multiset 完全一致（局部/边界增强）。
    allow_entity_value_change=True 且 mutation_forced 非空：强制叶子 (type, 替换字面) multiset 须被生成 multiset 覆盖；与强制 span 无嵌套包含关系的实体须保持 (type, 原字面)；
        祖先实体允许字面随子实体改写；默认 MUTATION_MAX_EXTRA_ENTITIES=0，因此实体条数与类型计数保持不变；同时保持完整类型嵌套森林。
    无 mutation_forced 时回退为类型 multiset 与原文一致。
    """
    if not generated_item or not generated_item.get("entities"):
        return False

    if RESIDUAL_TAG_IN_TEXT.search(generated_item["text"]):
        return False
    if RESIDUAL_XML_TAG_IN_TEXT.search(generated_item["text"]):
        return False
    if has_malformed_html_tag(generated_item["text"]):
        return False
    if has_illegal_hallucinated_tags(generated_item["text"]):
        return False

    ents = generated_item["entities"]
    if not _source_structure_supported(ents):
        return False

    if allow_entity_value_change:
        if mutation_forced:
            n_orig = len(original_entities)
            if len(ents) < n_orig:
                return False
            if len(ents) > n_orig + MUTATION_MAX_EXTRA_ENTITIES:
                return False
            orig_type_counter = Counter(e["type"] for e in original_entities)
            gen_type_counter = Counter(e["type"] for e in ents)
            for t, c in orig_type_counter.items():
                if gen_type_counter[t] < c:
                    return False
            for t in gen_type_counter:
                if t not in orig_type_counter:
                    return False

            forced_spans = {(int(f["start_idx"]), int(f["end_idx"])) for f in mutation_forced}
            forced_repl_counter = Counter(
                (f["type"], f["must_use_as_replacement"].strip()) for f in mutation_forced
            )
            preserve_counter = Counter()
            exempt_type_counter = Counter()
            for o in original_entities:
                sk = (int(o["start_idx"]), int(o["end_idx"]))
                t = o["type"]
                v0 = (o.get("value") or "").strip()
                if sk in forced_spans:
                    continue
                if any(_span_strictly_contains(sk, fs) for fs in forced_spans):
                    exempt_type_counter[t] += 1
                else:
                    preserve_counter[(t, v0)] += 1

            gen_pair_counter = Counter((e["type"], e["entity"].strip()) for e in ents)
            rem = Counter(gen_pair_counter)
            for key, need in forced_repl_counter.items():
                rem[key] -= need
                if rem[key] < 0:
                    return False
            for key, need in preserve_counter.items():
                rem[key] -= need
                if rem[key] < 0:
                    return False

            rem_type_counter = Counter()
            for (t, _v), c in rem.items():
                if c:
                    rem_type_counter[t] += c
            extra = len(ents) - n_orig
            exp_rem = sum(exempt_type_counter.values()) + extra
            if sum(rem_type_counter.values()) != exp_rem:
                return False
            for t, need in exempt_type_counter.items():
                if rem_type_counter[t] < need:
                    return False
            if MUTATION_PRESERVE_CONTAINMENT_PAIRS:
                if _typed_nesting_forest_signature(original_entities) != _typed_nesting_forest_signature(ents):
                    return False
            if not _mutation_replacement_is_anchored(
                original_entities,
                ents,
                generated_item["text"],
                mutation_forced,
            ):
                return False
        else:
            orig_types = sorted(e["type"] for e in original_entities)
            gen_types = sorted(e["type"] for e in ents)
            if orig_types != gen_types:
                return False
    else:
        orig_fingerprints = entity_fingerprint(original_entities)
        gen_fingerprints = entity_fingerprint(ents, generated_item["text"])
        if orig_fingerprints != gen_fingerprints:
            return False
        # LOCAL/BOUNDARY 同时比较完整类型森林和按实体字面区分的包含边，
        # 避免“包含类型对计数相同，但内层实体换了父实体”仍通过。
        if _typed_nesting_forest_signature(original_entities) != _typed_nesting_forest_signature(ents):
            return False
        if _occurrence_aware_containment_signature(original_entities, "") != _occurrence_aware_containment_signature(
            ents, generated_item["text"]
        ):
            return False

    for ent in ents:
        if ent["type"] not in VALID_TYPES:
            return False

        start, end = ent["start_idx"], ent["end_idx"]
        if start < 0 or end <= start or end > len(generated_item["text"]):
            return False
        extracted_value = generated_item["text"][start:end]
        if not allow_entity_value_change and extracted_value != get_entity_value(
            generated_item["text"], ent
        ):
            return False
        ent["entity"] = extracted_value
        ent.pop("value", None)
        if _ZERO_WIDTH_RE.search(extracted_value):
            return False
        if allow_entity_value_change:
            # 诱变字面不可首尾带空格/常见标点（易污染 GlobalPointer 边界）
            if re.search(r"^[\s，。、；：,.;:]+|[\s，。、；：,.;:]+$", extracted_value):
                return False

    span_seen = set()
    for ent in ents:
        span_key = (ent["start_idx"], ent["end_idx"])
        if span_key in span_seen:
            return False
        span_seen.add(span_key)

    if _cross_overlap_invalid(ents):
        return False

    return True


class MedicalLexiconMatcher:
    """只读 Aho-Corasick 匹配器；线程间共享时不修改内部状态。"""

    def __init__(self, lexicon, *, raw_term_count=None, removed_inconsistent_count=0):
        self.lexicon = {term: frozenset(types) for term, types in lexicon.items()}
        self.raw_term_count = (
            len(self.lexicon) if raw_term_count is None else int(raw_term_count)
        )
        self.removed_inconsistent_count = int(removed_inconsistent_count)
        self._next = [{}]
        self._fail = [0]
        self._outputs = [[]]
        for term in self.lexicon:
            state = 0
            for char in term:
                nxt = self._next[state].get(char)
                if nxt is None:
                    nxt = len(self._next)
                    self._next[state][char] = nxt
                    self._next.append({})
                    self._fail.append(0)
                    self._outputs.append([])
                state = nxt
            self._outputs[state].append(term)
        queue = deque(self._next[0].values())
        while queue:
            state = queue.popleft()
            for char, nxt in self._next[state].items():
                queue.append(nxt)
                fallback = self._fail[state]
                while fallback and char not in self._next[fallback]:
                    fallback = self._fail[fallback]
                self._fail[nxt] = self._next[fallback].get(char, 0)
                self._outputs[nxt].extend(self._outputs[self._fail[nxt]])

    def __len__(self):
        return len(self.lexicon)

    def iter_matches(self, text):
        state = 0
        for end_index, char in enumerate(text, start=1):
            while state and char not in self._next[state]:
                state = self._fail[state]
            state = self._next[state].get(char, 0)
            for term in self._outputs[state]:
                yield (
                    end_index - len(term),
                    end_index,
                    term,
                    self.lexicon[term],
                )

    def stats(self):
        return {
            "raw_term_count": self.raw_term_count,
            "removed_inconsistent_training_terms": self.removed_inconsistent_count,
            "retained_consistent_term_count": len(self.lexicon),
        }


def build_medical_lexicon(data):
    """
    仅保留在 Train 中没有任何“未标注出现”的实体词。

    CMeEE 的同一词在不同上下文中可能并不总被标注；若把这种词作为硬门禁，
    会把符合官方标注口径的增强句误判为漏标。
    """
    lexicon = {}
    for item in data:
        text = item.get("text", "")
        for ent in item.get("entities", []):
            value = get_entity_value(text, ent).strip()
            ent_type = ent.get("type")
            if len(value) < MIN_UNLABELED_TERM_LEN or ent_type not in VALID_TYPES:
                continue
            if ent_type in UNLABELED_TERM_IGNORE_TYPES:
                continue
            lexicon.setdefault(value, set()).add(ent_type)

    terms = sorted(lexicon, key=lambda x: (-len(x), x))
    if MAX_MEDICAL_LEXICON_TERMS > 0:
        terms = terms[:MAX_MEDICAL_LEXICON_TERMS]
    raw_lexicon = {term: lexicon[term] for term in terms}
    initial_matcher = MedicalLexiconMatcher(raw_lexicon)
    inconsistent_terms = set()
    for item in data:
        text = item.get("text", "")
        labeled_spans = [
            (int(e["start_idx"]), int(e["end_idx"]))
            for e in item.get("entities", [])
        ]
        for start, end, term, _types in initial_matcher.iter_matches(text):
            if term in inconsistent_terms:
                continue
            if not any(start < span_end and span_start < end for span_start, span_end in labeled_spans):
                inconsistent_terms.add(term)
    consistent_lexicon = {
        term: types
        for term, types in raw_lexicon.items()
        if term not in inconsistent_terms
    }
    return MedicalLexiconMatcher(
        consistent_lexicon,
        raw_term_count=len(raw_lexicon),
        removed_inconsistent_count=len(inconsistent_terms),
    )


def _overlap(span1, span2) -> bool:
    s1, e1 = span1
    s2, e2 = span2
    return s1 < e2 and s2 < e1


def find_unlabeled_medical_terms(item, medical_lexicon):
    if not ENABLE_UNLABELED_MEDICAL_TERM_FILTER or not medical_lexicon:
        return []

    text = item.get("text", "")
    entities = item.get("entities", [])
    labeled_spans = [(int(e["start_idx"]), int(e["end_idx"])) for e in entities]
    hits = []

    if isinstance(medical_lexicon, MedicalLexiconMatcher):
        matches = medical_lexicon.iter_matches(text)
    else:
        def iter_dict_matches():
            for term, types in medical_lexicon.items():
                if not term:
                    continue
                start = 0
                while True:
                    pos = text.find(term, start)
                    if pos < 0:
                        break
                    yield pos, pos + len(term), term, types
                    start = pos + 1

        matches = iter_dict_matches()
    for pos, end, term, types in matches:
        # 按「每次出现」判断是否与任一标注 span 重叠；禁止因该词在别处已标注就跳过全文扫描（同形二次出现漏标）
        if not any(_overlap((pos, end), span) for span in labeled_spans):
            hits.append({"term": term, "types": sorted(types), "start_idx": pos, "end_idx": end})
    return hits


def _process_single_item_once(item, medical_lexicon=None, real_entities_pool=None):
    source_id = int(item.get("_source_id", 0))
    # 与线程完成顺序无关的逐样本 RNG；固定快照保证模型版本，但不宣称 API 采样逐字确定。
    rng = random.Random(RANDOM_SEED + source_id)
    # 旧版只处理含dep/equ/mic/ite或实体数>=3的样本；上传的真实Train中
    # 该规则会跳过3111/15000条并改变来源分布。现在对全部含Gold实体的
    # 去污染Train行开放；无实体行只有显式启用negative时才可生成。
    if not item.get("entities") and "negative" not in GENERATION_ALLOWED_MODES:
        return None, "skipped_no_gold_entities", None

    source_entity_total = len(item.get("entities", []))

    simplified_entities = []
    for e in item["entities"]:
        start, end = e["start_idx"], e["end_idx"]
        simplified_entities.append(
            {
                "type": e["type"],
                "value": item["text"][start:end],
                "start_idx": start,
                "end_idx": end,
            }
        )

    full_ref = [
        {
            "type": e["type"],
            "value": item["text"][e["start_idx"] : e["end_idx"]],
            "start_idx": e["start_idx"],
            "end_idx": e["end_idx"],
        }
        for e in item["entities"]
    ]

    mode = pick_augment_mode(item, rng=rng)

    if mode != AugmentMode.NEGATIVE:
        # 与下游 smart_sample 的绝对长度/HTML门禁前置对齐，避免付费生成后必然被拒。
        if len(item["text"]) > MAX_AUGMENT_TEXT_LEN:
            return None, "source_exceeds_max_text_len", mode.value
        if has_malformed_html_tag(item["text"]):
            return None, "source_malformed_html", mode.value
        if has_illegal_hallucinated_tags(item["text"]):
            return None, "source_illegal_html_tag", mode.value
        # BOUNDARY至少插入1个字符；同时满足绝对长度与相对原文1.15倍上限才值得调用API。
        if mode == AugmentMode.BOUNDARY:
            max_boundary_len = min(
                MAX_AUGMENT_TEXT_LEN,
                int(len(item["text"]) * 1.15),
            )
            if len(item["text"]) + 1 > max_boundary_len:
                return None, "boundary_no_length_budget", mode.value

    # 负例不复用原标注结构；其余XML正例必须先确认源结构可由合法嵌套标签表达。
    if mode != AugmentMode.NEGATIVE and not _source_structure_supported(full_ref):
        return None, "unsupported_source_structure", mode.value

    if mode == AugmentMode.LOCAL and not any(
        _MEANINGFUL_CONTEXT_CHAR_RE.search(segment["text"])
        for segment in editable_context_segments(item["text"], full_ref)
    ):
        return None, "local_no_editable_context", mode.value

    original_tagged_text = build_gold_xml_structure(item["text"], item["entities"])

    mutation_forced = None
    if mode == AugmentMode.MUTATION:
        eligible = []
        if real_entities_pool and any(
            _mutation_pool_records(value)
            for value in real_entities_pool.values()
        ):
            eligible = [
                ent
                for ent in filter_leaf_mutation_spans(full_ref)
                if has_compatible_mutation_candidate(ent, real_entities_pool)
            ]
            if eligible:
                # 仅在同句存在多个合法叶子时，先从已有 Baseline
                # Dev 结果支持的弱类中选择；无弱类时仍保留全类覆盖。
                focused_eligible = [
                    ent for ent in eligible if ent.get("type") in FOCUS_ENTITY_TYPES
                ]
                selected_leaf = rng.choice(focused_eligible or eligible)
                mutation_forced = sample_mutation_forced_replacements(
                    [selected_leaf],
                    real_entities_pool,
                    source_text=item["text"],
                    rng=rng,
                )
        if not mutation_forced:
            # 无真实同类型候选时退回受控LOCAL，不浪费本条计划中的API机会。
            mode = AugmentMode.LOCAL

    time.sleep(rng.uniform(0.1, 0.8))

    mutation_prompt_ref = sorted(
        full_ref or simplified_entities,
        key=lambda x: (x.get("start_idx", -1), x.get("end_idx", -1), x["type"]),
    )

    try:
        clean = call_llm_augmentation(
            mode,
            item["text"],
            mutation_prompt_ref,
            real_entities_pool,
            mutation_forced=mutation_forced,
            original_tagged_text=original_tagged_text,
            rng=rng,
        )
    except Exception as e:
        print(f"API failed after retries: {e}")
        return None, "api_failed", mode.value

    if not clean:
        return None, "empty_response", mode.value

    if mode == AugmentMode.NEGATIVE:
        generated_item = parse_llm_response_negative(clean)
        if generated_item:
            if not (15 <= len(generated_item["text"]) <= 45):
                generated_item.update({
                    "augment_mode": mode.value,
                    "llm_model": LLM_MODEL,
                    "prompt_version": PROMPT_VERSION,
                    "pipeline_version": PIPELINE_VERSION,
                    "source_id": source_id,
                    "reject_reason": "negative_length_out_of_range",
                    "generated_text_length": len(generated_item["text"]),
                })
                return generated_item, "negative_length_out_of_range", mode.value
            generated_item["augment_mode"] = mode.value
            generated_item["dataset_name"] = DATASET_NAME
            generated_item["llm_model"] = LLM_MODEL
            generated_item["prompt_version"] = PROMPT_VERSION
            generated_item["pipeline_version"] = PIPELINE_VERSION
            generated_item["source_id"] = source_id
            unlabeled_terms = find_unlabeled_medical_terms(generated_item, medical_lexicon)
            if unlabeled_terms:
                generated_item["reject_reason"] = "negative_contains_medical_terms"
                generated_item["unlabeled_medical_terms"] = unlabeled_terms[:20]
                return generated_item, "negative_contains_medical_terms", mode.value
            return generated_item, "accepted_negative", mode.value
        return None, "invalid_generation", mode.value

    generated_item = parse_llm_response(
        clean,
        require_replacement=mode == AugmentMode.MUTATION,
    )
    effective_mutation_forced = mutation_forced
    if generated_item and mode == AugmentMode.MUTATION:
        if generated_item.get("mutation_abstained"):
            return None, "mutation_context_abstained", mode.value
        effective_mutation_forced = bind_selected_mutation_replacement(
            mutation_forced,
            generated_item.get("selected_replacement"),
        )
        if effective_mutation_forced is None:
            generated_item.update({
                "augment_mode": mode.value,
                "llm_model": LLM_MODEL,
                "prompt_version": PROMPT_VERSION,
                "pipeline_version": PIPELINE_VERSION,
                "source_id": source_id,
                "reject_reason": "mutation_replacement_not_in_allowed_candidates",
            })
            return generated_item, "invalid_mutation_replacement", mode.value
        generated_item.pop("selected_replacement", None)
    if generated_item:
        generated_len = len(generated_item["text"])
        if generated_len > MAX_AUGMENT_TEXT_LEN:
            generated_item.update({
                "augment_mode": mode.value,
                "llm_model": LLM_MODEL,
                "prompt_version": PROMPT_VERSION,
                "pipeline_version": PIPELINE_VERSION,
                "source_id": source_id,
                "reject_reason": "generated_exceeds_max_text_len",
                "generated_text_length": generated_len,
            })
            return generated_item, "generated_exceeds_max_text_len", mode.value
        if generated_len > len(item["text"]) * 1.15:
            generated_item.update({
                "augment_mode": mode.value,
                "llm_model": LLM_MODEL,
                "prompt_version": PROMPT_VERSION,
                "pipeline_version": PIPELINE_VERSION,
                "source_id": source_id,
                "reject_reason": "generated_exceeds_relative_length",
                "generated_text_length": generated_len,
                "source_text_length": len(item["text"]),
            })
            return generated_item, "generated_exceeds_relative_length", mode.value
    allow_entity_value_change = mode == AugmentMode.MUTATION
    if generated_item and validate_and_merge(
        simplified_entities,
        generated_item,
        allow_entity_value_change=allow_entity_value_change,
        mutation_forced=effective_mutation_forced,
    ):
        if generated_item["text"] == item["text"]:
            return None, "unchanged_generation", mode.value
        if mode == AugmentMode.LOCAL:
            if not _local_context_edit_is_valid(
                item["text"],
                generated_item["text"],
                simplified_entities,
                generated_item["entities"],
            ):
                return None, "invalid_local_context_edit", mode.value
        if mode == AugmentMode.BOUNDARY and not _boundary_only_insertion_is_valid(
            item["text"], generated_item["text"], simplified_entities
        ):
            return None, "invalid_boundary_edit", mode.value
        if mode == AugmentMode.MUTATION and not _mutation_is_exact_leaf_replacement(
            item["text"], generated_item["text"], effective_mutation_forced
        ):
            return None, "invalid_mutation_context_edit", mode.value

        generated_item["augment_mode"] = mode.value
        generated_item["dataset_name"] = DATASET_NAME
        generated_item["llm_model"] = LLM_MODEL
        generated_item["prompt_version"] = PROMPT_VERSION
        generated_item["pipeline_version"] = PIPELINE_VERSION
        generated_item["source_id"] = source_id
        generated_item["source_text"] = item["text"]
        generated_item["source_similarity"] = difflib.SequenceMatcher(
            None, item["text"], generated_item["text"], autojunk=False
        ).ratio()
        if mode == AugmentMode.MUTATION:
            generated_item["source_entity_total"] = source_entity_total
            generated_item["mutation_replacements"] = [
                {
                    "type": f["type"],
                    "original": f["original"],
                    "replacement": f["must_use_as_replacement"],
                    "allowed_replacements": list(f.get("allowed_replacements", [])),
                    "candidate_retrieval": f.get("candidate_retrieval"),
                    "source_context": f.get("source_context"),
                    "candidate_evidence": list(f.get("candidate_evidence", [])),
                    "start_idx": int(f["start_idx"]),
                    "end_idx": int(f["end_idx"]),
                }
                for f in (effective_mutation_forced or [])
            ]
        unlabeled_terms = find_unlabeled_medical_terms(generated_item, medical_lexicon)
        if unlabeled_terms:
            generated_item["reject_reason"] = "unlabeled_medical_terms"
            generated_item["unlabeled_medical_terms"] = unlabeled_terms[:20]
            return generated_item, "unlabeled_medical_terms", mode.value
        return generated_item, "accepted", mode.value
    return None, "invalid_generation", mode.value


def process_single_item(item, medical_lexicon=None, real_entities_pool=None):
    """对内容类失败最多重新生成一次；源数据/结构/API类失败不重复付费调用。"""
    last = (None, "invalid_generation", None)
    for _attempt in range(LLM_CONTENT_MAX_ATTEMPTS):
        last = _process_single_item_once(item, medical_lexicon, real_entities_pool)
        if last[1] not in CONTENT_RETRY_STATUSES:
            return last
    return last


def iter_bounded_results(
    executor,
    items,
    medical_lexicon=None,
    real_entities_pool=None,
    max_pending=MAX_PENDING_FUTURES,
):
    item_iter = iter(items)
    pending = {}

    def submit_next():
        try:
            item = next(item_iter)
        except StopIteration:
            return False
        future = executor.submit(
            process_single_item, item, medical_lexicon, real_entities_pool
        )
        pending[future] = int(item.get("_source_id", -1))
        return True

    for _ in range(max_pending):
        if not submit_next():
            break

    while pending:
        done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
        for future in done:
            source_id = pending.pop(future)
            try:
                result, status, mode_name = future.result()
                yield source_id, result, status, mode_name
            except Exception as exc:
                print(f"Worker failed unexpectedly: {exc}")
                yield source_id, None, "worker_exception", None
            submit_next()


def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_object_sha256(value) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_augmentation_run_config(
    train_file_sha256: str,
    holdout_files_sha256=None,
    medical_lexicon_stats=None,
) -> dict:
    """append 续跑的不可混用配置；不含时间戳和并发数等非语义字段。"""
    return {
        "schema_version": 2,
        "dataset_name": DATASET_NAME,
        "train_file_sha256": train_file_sha256,
        "remove_holdout_text_overlap": REMOVE_HOLDOUT_TEXT_OVERLAP,
        "holdout_files_sha256": dict(holdout_files_sha256 or {}),
        "llm_model": LLM_MODEL,
        "prompt_version": PROMPT_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "prompt_sha256": {
            "local": hashlib.sha256(SYSTEM_PROMPT_LOCAL.encode("utf-8")).hexdigest(),
            "boundary": hashlib.sha256(SYSTEM_PROMPT_BOUNDARY.encode("utf-8")).hexdigest(),
            "mutation": hashlib.sha256(SYSTEM_PROMPT_MUTATION.encode("utf-8")).hexdigest(),
            "negative": hashlib.sha256(SYSTEM_PROMPT_NEGATIVE.encode("utf-8")).hexdigest(),
        },
        "client_random_seed": RANDOM_SEED,
        "max_augment_samples": MAX_AUGMENT_SAMPLES,
        "generation_parameters": {
            "local": {"temperature": 0.4, "top_p": 0.9, "max_tokens": 4096},
            "boundary": {"temperature": 0.7, "top_p": 0.9, "max_tokens": 4096},
            "mutation": {"temperature": 0.3, "top_p": 0.9, "max_tokens": 4096},
            "negative": {"temperature": 0.85, "top_p": 0.9, "max_tokens": 512},
        },
        "routing": {
            "type": AUGMENT_MODE_ROUTING,
            "generation_allowed_modes": list(GENERATION_ALLOWED_MODES),
            "negative": P_AUG_NEGATIVE,
            "boundary": P_AUG_BOUNDARY_IF_ITE,
            "mutation": P_AUG_MUTATION,
            "local": P_AUG_LOCAL,
            "source_policy": "all_decontaminated_rows_with_gold_entities",
            "source_selection_policy": "focus_presence_then_seeded_hash_only_affects_max_augment_samples_prefix",
            "mutation_focus_entity_types": sorted(FOCUS_ENTITY_TYPES),
            "mutation_focus_policy": FOCUS_CLASS_SELECTION_POLICY,
        },
        "quality_control": {
            "max_augment_text_len": MAX_AUGMENT_TEXT_LEN,
            "max_relative_text_length": 1.15,
            "unlabeled_term_filter": ENABLE_UNLABELED_MEDICAL_TERM_FILTER,
            "min_unlabeled_term_len": MIN_UNLABELED_TERM_LEN,
            "max_medical_lexicon_terms": MAX_MEDICAL_LEXICON_TERMS,
            "medical_lexicon_policy": "retain_terms_without_unlabeled_training_occurrences",
            "medical_lexicon_stats": dict(medical_lexicon_stats or {}),
            "mutation_max_length_delta": MUTATION_MAX_LENGTH_DELTA,
            "mutation_match_surface_profile": MUTATION_MATCH_SURFACE_PROFILE,
            "mutation_candidate_count": MUTATION_CANDIDATE_COUNT,
            "mutation_retrieval_shortlist": MUTATION_RETRIEVAL_SHORTLIST,
            "mutation_context_window": MUTATION_CONTEXT_WINDOW,
            "mutation_candidate_selection": "masked_train_context_bigram_recall_then_similarity_rank_then_llm_selects_or_abstains_then_python_validates",
            "mutation_source_context_policy": "exactly_unchanged_outside_selected_leaf_span",
            "local_allowed_diff_blocks": [1, 3],
            "local_temperature": 0.4,
            "mutation_max_extra_entities": MUTATION_MAX_EXTRA_ENTITIES,
            "preserve_typed_nesting_forest": MUTATION_PRESERVE_CONTAINMENT_PAIRS,
            "content_max_attempts": LLM_CONTENT_MAX_ATTEMPTS,
            "negative_text_length_range_including_punctuation": [15, 45],
        },
    }


def write_json_atomic(path: str, value) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def durable_jsonlines_write(writer, value) -> None:
    """先将payload刷入磁盘，再允许写完成状态，避免崩溃后状态存在而结果缺失。"""
    writer.write(value)
    fp = getattr(writer, "_fp", None)
    if fp is not None and hasattr(fp, "flush"):
        fp.flush()
        try:
            os.fsync(fp.fileno())
        except (AttributeError, OSError):
            pass


def read_jsonl_if_exists(path, repair_truncated_tail=False):
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    nonempty_indices = [idx for idx, line in enumerate(lines) if line.strip()]
    last_nonempty_idx = nonempty_indices[-1] if nonempty_indices else -1
    rows = []
    for idx, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            if idx != last_nonempty_idx:
                raise ValueError(f"JSONL中间行损坏，拒绝静默续跑: {path}:{idx + 1}") from exc
            if not repair_truncated_tail:
                raise
            print(f"[warning] 丢弃崩溃留下的JSONL尾部残行: {path}:{idx + 1}")
            tmp_path = path + ".repair.tmp"
            with open(tmp_path, "w", encoding="utf-8") as out:
                for row in rows:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                os.fsync(out.fileno())
            os.replace(tmp_path, path)
            break
    return rows


def consolidate_accepted_rows(rows):
    """
    每个源样本只允许一条接收结果。崩溃重写造成的完全相同行可去重；
    同一 source_id 若出现不同文本或标注，拒绝静默二选一。
    """
    by_source = {}
    for row in rows:
        if row.get("source_id") is None:
            raise ValueError("接收的增强行缺少 source_id，无法保证一对一来源")
        source_id = int(row["source_id"])
        previous = by_source.get(source_id)
        if previous is None:
            by_source[source_id] = row
            continue
        if canonical_object_sha256(previous) != canonical_object_sha256(row):
            raise ValueError(
                "同一 source_id 存在冲突的多条接收结果，"
                f"拒绝静默合并: source_id={source_id}"
            )
    return [by_source[source_id] for source_id in sorted(by_source)]


def prepare_augmentation_sources(data, holdout_files):
    """按正文去污染，同时保留原训练文件中的稳定 source_id。"""
    holdout_texts = set()
    holdout_hashes = {}
    if REMOVE_HOLDOUT_TEXT_OVERLAP:
        for path in holdout_files:
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"开启了离线增强去污染，但找不到 holdout 文件: {path}"
                )
            holdout_hashes[os.path.basename(path)] = file_sha256(path)
            with open(path, "r", encoding="utf-8") as f:
                holdout_texts.update(
                    row.get("text", "")
                    for row in json.load(f)
                    if row.get("text")
                )
    source_rows = [
        (source_id, item)
        for source_id, item in enumerate(data)
        if item.get("text") not in holdout_texts
    ]
    removed = len(data) - len(source_rows)
    return source_rows, {
        "remove_holdout_text_overlap": REMOVE_HOLDOUT_TEXT_OVERLAP,
        "holdout_files_sha256": holdout_hashes,
        "holdout_unique_text_count": len(holdout_texts),
        "augmentation_source_rows_before_decontamination": len(data),
        "augmentation_source_rows_removed_by_exact_holdout_text": removed,
        "augmentation_source_rows_after_decontamination": len(source_rows),
    }


def source_selection_key(pair):
    """MAX_AUGMENT_SAMPLES的前缀优先覆盖已有 Baseline 结果支持的难类。"""
    source_id, row = pair
    has_focus = any(
        entity.get("type") in FOCUS_ENTITY_TYPES for entity in row.get("entities", [])
    )
    # 哈希只依赖固定种子与正文；source_id 仅作为同文最终 tie-breaker。
    # 这样只改变原始文件行序时，不会重新打乱 MAX_AUGMENT_SAMPLES 前缀。
    tie = hashlib.sha256(
        f"{RANDOM_SEED}\0{row.get('text', '')}".encode("utf-8")
    ).hexdigest()
    return (0 if has_focus else 1, tie, int(source_id))


def main():
    random.seed(RANDOM_SEED)
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Loading original training data from {TRAIN_FILE}...")
    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    source_rows, holdout_decontamination = prepare_augmentation_sources(
        data, HOLDOUT_FILES
    )
    source_data = [item for _source_id, item in source_rows]

    augmented_data = []
    reject_data = []
    reject_counts = {}
    status_counts_by_mode = {}
    medical_lexicon = build_medical_lexicon(source_data)
    medical_lexicon_stats = medical_lexicon.stats()
    real_entities_pool = build_real_entities_pool(source_data)
    seed_counts = {
        key: len(_mutation_pool_records(value))
        for key, value in real_entities_pool.items()
        if _mutation_pool_records(value)
    }
    incremental_file = os.path.join(DATA_DIR, "augmented_incremental.jsonl")
    rejected_file = os.path.join(DATA_DIR, "augmented_rejected.jsonl")
    status_file = os.path.join(DATA_DIR, "augmentation_status.jsonl")
    manifest_file = os.path.join(DATA_DIR, "augmentation_run_manifest.json")
    run_config_file = os.path.join(DATA_DIR, "augmentation_run_config.json")
    train_file_hash = file_sha256(TRAIN_FILE)
    run_config = build_augmentation_run_config(
        train_file_hash,
        holdout_decontamination["holdout_files_sha256"],
        medical_lexicon_stats,
    )
    run_config_envelope = {
        "run_config_sha256": canonical_object_sha256(run_config),
        "run_config": run_config,
    }

    existing_incremental_artifacts = any(
        os.path.isfile(path) and os.path.getsize(path) > 0
        for path in (incremental_file, rejected_file, status_file)
    )
    if INCREMENTAL_WRITE_MODE == "a" and existing_incremental_artifacts:
        if not os.path.isfile(run_config_file):
            raise RuntimeError(
                "检测到旧增量文件但缺少 augmentation_run_config.json；"
                "无法证明提示词/模型/训练集一致。请改用写入模式 w 开始新Run。"
            )
        with open(run_config_file, "r", encoding="utf-8") as f:
            existing_run_config = json.load(f)
        if existing_run_config != run_config_envelope:
            old_hash = (
                existing_run_config.get("run_config_sha256")
                if isinstance(existing_run_config, dict)
                else None
            )
            raise RuntimeError(
                "append续跑配置与旧Run不一致，已拒绝混写。"
                f" old={old_hash}"
                f" new={run_config_envelope['run_config_sha256']}"
            )
    else:
        write_json_atomic(run_config_file, run_config_envelope)

    completed_source_ids = set()
    if INCREMENTAL_WRITE_MODE == "a":
        augmented_data = read_jsonl_if_exists(incremental_file, repair_truncated_tail=True)
        reject_data = read_jsonl_if_exists(rejected_file, repair_truncated_tail=True)
        status_rows = read_jsonl_if_exists(status_file, repair_truncated_tail=True)
        accepted_payload_source_ids = {
            int(row["source_id"])
            for row in augmented_data
            if row.get("source_id") is not None
        }
        rejected_payload_source_ids = {
            int(row["source_id"])
            for row in reject_data
            if row.get("source_id") is not None
        }
        if status_rows:
            last_status_by_source = {}
            for row in status_rows:
                if row.get("source_id") is not None:
                    last_status_by_source[int(row["source_id"])] = row.get("status")
            for source_id, status in last_status_by_source.items():
                if status in ("accepted", "accepted_negative"):
                    if source_id in accepted_payload_source_ids:
                        completed_source_ids.add(source_id)
                    else:
                        print(
                            "[warning] 接收状态缺少对应payload，将重试: "
                            f"source_id={source_id}, status={status}"
                        )
                elif status not in RESUME_RETRY_STATUSES:
                    completed_source_ids.add(source_id)
            # payload 先持久化、状态后持久化；若两者之间崩溃，
            # payload 本身就是该 source 已完成的证据，不应再次调用 LLM。
            completed_source_ids.update(accepted_payload_source_ids)
            completed_source_ids.update(rejected_payload_source_ids)
        else:
            # 兼容没有status日志的旧版输出；只能从有内容的接收/拒绝文件恢复。
            for row in augmented_data + reject_data:
                if row.get("source_id") is not None:
                    completed_source_ids.add(int(row["source_id"]))

    scheduled_source_rows = sorted(source_rows, key=source_selection_key)
    selected_source_rows = (
        scheduled_source_rows
        if MAX_AUGMENT_SAMPLES is None
        else scheduled_source_rows[:MAX_AUGMENT_SAMPLES]
    )
    to_process = []
    for source_id, item in selected_source_rows:
        if source_id in completed_source_ids:
            continue
        row = dict(item)
        row["_source_id"] = source_id
        to_process.append(row)

    planned_mode_counts = Counter()
    for row in to_process:
        entities = row.get("entities", [])
        if not entities and "negative" not in GENERATION_ALLOWED_MODES:
            planned_mode_counts["not_routed"] += 1
            continue
        source_id = int(row["_source_id"])
        planned_mode = pick_augment_mode(
            row, rng=random.Random(RANDOM_SEED + source_id)
        )
        if planned_mode == AugmentMode.MUTATION:
            planned_ref = [
                {
                    "type": ent["type"],
                    "value": row["text"][ent["start_idx"]:ent["end_idx"]],
                    "start_idx": ent["start_idx"],
                    "end_idx": ent["end_idx"],
                }
                for ent in entities
            ]
            if not any(
                has_compatible_mutation_candidate(ent, real_entities_pool)
                for ent in filter_leaf_mutation_spans(planned_ref)
            ):
                planned_mode = AugmentMode.LOCAL
        planned_mode_counts[planned_mode.value] += 1
    planned_routed_total = sum(
        count
        for mode, count in planned_mode_counts.items()
        if mode != "not_routed"
    )
    print(
        f"Starting LLM augmentation (百炼 {LLM_MODEL}, seed={RANDOM_SEED}, prompt={PROMPT_VERSION})... "
        f"本次提交 {len(to_process)} 条（训练集共 {len(data)} 条，去污染后候选 {len(source_rows)} 条，"
        f"MAX_AUGMENT_SAMPLES={MAX_AUGMENT_SAMPLES}，"
        f"MAX_PENDING_FUTURES={MAX_PENDING_FUTURES}，INCREMENTAL_WRITE_MODE={INCREMENTAL_WRITE_MODE}，"
        f"P_AUG_NEGATIVE={P_AUG_NEGATIVE}，P_AUG_BOUNDARY_IF_ITE={P_AUG_BOUNDARY_IF_ITE}，P_AUG_MUTATION={P_AUG_MUTATION}，"
        f"P_AUG_LOCAL={P_AUG_LOCAL}，routing={AUGMENT_MODE_ROUTING}，allowed={GENERATION_ALLOWED_MODES}，"
        f"MUTATION_MAX_EXTRA_ENTITIES={MUTATION_MAX_EXTRA_ENTITIES}，"
        f"medical_lexicon={medical_lexicon_stats}，unlabeled_filter={ENABLE_UNLABELED_MEDICAL_TERM_FILTER}，"
        f"seed_entity_pool={seed_counts}，MUTATION_MAX_LENGTH_DELTA={MUTATION_MAX_LENGTH_DELTA}，"
        f"MUTATION_MATCH_SURFACE_PROFILE={MUTATION_MATCH_SURFACE_PROFILE}，"
        f"MAX_AUGMENT_TEXT_LEN={MAX_AUGMENT_TEXT_LEN}，resume_skipped={len(completed_source_ids)}，"
        f"planned_mode_counts={dict(planned_mode_counts)}）"
    )

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor, jsonlines.open(
        incremental_file, mode=INCREMENTAL_WRITE_MODE, flush=True
    ) as writer, jsonlines.open(
        rejected_file, mode=INCREMENTAL_WRITE_MODE, flush=True
    ) as reject_writer:
        with jsonlines.open(status_file, mode=INCREMENTAL_WRITE_MODE, flush=True) as status_writer:
            result_iterator = tqdm(
                iter_bounded_results(executor, to_process, medical_lexicon, real_entities_pool),
                total=len(to_process),
                desc="Augmenting",
            )
            for source_id, result, status, mode_name in result_iterator:
                reject_counts[status] = reject_counts.get(status, 0) + 1
                mode_key = mode_name or "not_routed"
                status_counts_by_mode.setdefault(mode_key, {})
                status_counts_by_mode[mode_key][status] = status_counts_by_mode[mode_key].get(status, 0) + 1
                # 有payload时必须先持久化payload，最后才写“已完成”状态。
                # 若中间崩溃，append 恢复会从payload推断该source已完成，
                # 避免对同一源样本重复调用 LLM。
                if status in ("accepted", "accepted_negative") and result:
                    out = dict(result)
                    durable_jsonlines_write(writer, out)
                    augmented_data.append(out)
                elif result:
                    durable_jsonlines_write(reject_writer, result)
                    reject_data.append(result)
                durable_jsonlines_write(status_writer, {
                    "source_id": source_id,
                    "status": status,
                    "augment_mode": mode_name,
                    "has_result_payload": bool(result),
                })

    # 一条源样本只能贡献一条接收增强；冲突时停止，不静默选一条。
    augmented_data = consolidate_accepted_rows(augmented_data)
    # 去污染只限制“哪些Train行可作为LLM来源”，不能改变官方人工Train本身。
    # 旧写法使用source_data会让与Dev/Test同文的Train行从合并文件静默消失。
    final_data = data + augmented_data

    print(f"\nAugmentation complete!")
    print(f"Original samples: {len(data)}")
    print(f"Original samples after holdout decontamination: {len(source_data)}")
    print(f"Successfully augmented samples: {len(augmented_data)}")
    print(f"Rejected samples with details: {len(reject_data)}")
    print(f"Reject/status counts: {reject_counts}")
    print(f"Total training samples: {len(final_data)}")

    with open(AUGMENTED_FILE, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)
    print(f"Saved augmented dataset to {AUGMENTED_FILE}")

    cumulative_status_rows = read_jsonl_if_exists(status_file)
    cumulative_status_counts_by_mode = {}
    for row in cumulative_status_rows:
        mode_key = row.get("augment_mode") or "not_routed"
        status_key = row.get("status") or "unknown"
        cumulative_status_counts_by_mode.setdefault(mode_key, {})
        cumulative_status_counts_by_mode[mode_key][status_key] = (
            cumulative_status_counts_by_mode[mode_key].get(status_key, 0) + 1
        )

    prompt_hashes = {
        "local": hashlib.sha256(SYSTEM_PROMPT_LOCAL.encode("utf-8")).hexdigest(),
        "boundary": hashlib.sha256(SYSTEM_PROMPT_BOUNDARY.encode("utf-8")).hexdigest(),
        "mutation": hashlib.sha256(SYSTEM_PROMPT_MUTATION.encode("utf-8")).hexdigest(),
        "negative": hashlib.sha256(SYSTEM_PROMPT_NEGATIVE.encode("utf-8")).hexdigest(),
    }
    manifest = {
        "dataset_name": DATASET_NAME,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "dashscope_sdk_version": getattr(dashscope, "__version__", None),
        "train_file": os.path.abspath(TRAIN_FILE),
        "train_file_sha256": file_sha256(TRAIN_FILE),
        **holdout_decontamination,
        "llm_model": LLM_MODEL,
        "prompt_version": PROMPT_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "prompt_sha256": prompt_hashes,
        "client_random_seed": RANDOM_SEED,
        "server_seed_passed": False,
        "generation_parameters": {
            "local": {"temperature": 0.4, "top_p": 0.9, "max_tokens": 4096},
            "boundary": {"temperature": 0.7, "top_p": 0.9, "max_tokens": 4096},
            "mutation": {"temperature": 0.3, "top_p": 0.9, "max_tokens": 4096},
            "negative": {"temperature": 0.85, "top_p": 0.9, "max_tokens": 512},
        },
        "routing": {
            "type": AUGMENT_MODE_ROUTING,
            "generation_allowed_modes": list(GENERATION_ALLOWED_MODES),
            "source_policy": "all_decontaminated_rows_with_gold_entities",
            "source_selection_policy": "focus_presence_then_seeded_hash_only_affects_max_augment_samples_prefix",
            "mutation_focus_entity_types": sorted(FOCUS_ENTITY_TYPES),
            "mutation_focus_policy": FOCUS_CLASS_SELECTION_POLICY,
            "conditional_weights_not_final_percentages": {
                "negative": P_AUG_NEGATIVE,
                "boundary": P_AUG_BOUNDARY_IF_ITE,
                "mutation": P_AUG_MUTATION,
                "local": P_AUG_LOCAL,
            },
            "planned_mode_counts_this_run": dict(planned_mode_counts),
            "planned_mode_rates_this_run": {
                mode: count / len(to_process) if to_process else 0.0
                for mode, count in planned_mode_counts.items()
            },
            "planned_mode_rates_among_routed_this_run": {
                mode: count / planned_routed_total if planned_routed_total else 0.0
                for mode, count in planned_mode_counts.items()
                if mode != "not_routed"
            },
        },
        "quality_control": {
            "unlabeled_term_filter": ENABLE_UNLABELED_MEDICAL_TERM_FILTER,
            "min_unlabeled_term_len": MIN_UNLABELED_TERM_LEN,
            "medical_lexicon_policy": "retain_terms_without_unlabeled_training_occurrences",
            "medical_lexicon_stats": medical_lexicon_stats,
            "mutation_max_length_delta": MUTATION_MAX_LENGTH_DELTA,
            "mutation_match_surface_profile": MUTATION_MATCH_SURFACE_PROFILE,
            "mutation_candidate_count": MUTATION_CANDIDATE_COUNT,
            "mutation_retrieval_shortlist": MUTATION_RETRIEVAL_SHORTLIST,
            "mutation_context_window": MUTATION_CONTEXT_WINDOW,
            "mutation_candidate_selection": "masked_train_context_bigram_recall_then_similarity_rank_then_llm_selects_or_abstains_then_python_validates",
            "mutation_source_context_policy": "exactly_unchanged_outside_selected_leaf_span",
            "mutation_surface_profile_scope": "coarse_script_and_digit_match_not_semantic_equivalence",
            "mutation_max_extra_entities": MUTATION_MAX_EXTRA_ENTITIES,
            "preserve_typed_nesting_forest": MUTATION_PRESERVE_CONTAINMENT_PAIRS,
            "preserve_anchored_mutation_path": True,
            "max_augment_text_len": MAX_AUGMENT_TEXT_LEN,
            "max_relative_text_length": 1.15,
            "local_allowed_diff_blocks": [1, 3],
            "local_temperature": 0.4,
            "negative_text_length_range_including_punctuation": [15, 45],
            "negative_generation_design": "generic_no-entity_medical_negative_not_guaranteed_hard",
        },
        "submitted_this_run": len(to_process),
        "resume_skipped_source_ids": len(completed_source_ids),
        "accepted_total_in_incremental": len(augmented_data),
        "incremental_file": os.path.abspath(incremental_file),
        "incremental_file_sha256": file_sha256(incremental_file),
        "accepted_mode_counts_total": dict(
            Counter(row.get("augment_mode") or "unknown" for row in augmented_data)
        ),
        "rejected_payload_total": len(reject_data),
        "rejected_file": os.path.abspath(rejected_file),
        "rejected_file_sha256": file_sha256(rejected_file),
        "rejected_payload_mode_counts_total": dict(
            Counter(row.get("augment_mode") or "unknown" for row in reject_data)
        ),
        "rejected_payload_reason_counts_total": dict(
            Counter(row.get("reject_reason") or "unknown" for row in reject_data)
        ),
        "cumulative_status_counts_by_mode": cumulative_status_counts_by_mode,
        "status_counts_this_run": reject_counts,
        "status_counts_by_mode_this_run": status_counts_by_mode,
        "status_file": os.path.abspath(status_file),
        "status_file_sha256": file_sha256(status_file),
        "resume_retry_statuses": sorted(RESUME_RETRY_STATUSES),
        "content_max_attempts": LLM_CONTENT_MAX_ATTEMPTS,
        "run_config_file": os.path.abspath(run_config_file),
        "run_config_file_sha256": file_sha256(run_config_file),
        "run_config_sha256": run_config_envelope["run_config_sha256"],
    }
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Saved augmentation manifest to {manifest_file}")


if __name__ == "__main__":
    main()
