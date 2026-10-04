# -*- coding: utf-8 -*-
"""
训练日志与参数保存、模型保存工具
"""
import csv
import hashlib
import json
import os
from typing import Dict, Optional


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def save_params(params: Dict, path: str) -> None:
    ensure_parent(path)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(params, f, ensure_ascii=False, indent=2)


def file_sha256(path: str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def append_epoch_log(
    csv_path: str,
    epoch: int,
    loss: float,
    f1: float,
    precision: float,
    recall: float,
    total_params: Optional[int] = None,
    trainable_params: Optional[int] = None,
) -> None:
    ensure_parent(csv_path)
    file_exists = os.path.exists(csv_path)
    include_params = total_params is not None and trainable_params is not None
    if file_exists and include_params:
        with open(csv_path, 'r', encoding='utf-8') as rf:
            first = rf.readline()
        try:
            ncols = len(next(csv.reader([first.strip()])))
        except StopIteration:
            ncols = 0
        # 已有旧版 5 列表头时继续只写 5 列，避免与历史文件列数不一致
        if ncols == 5:
            include_params = False

    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ['epoch', 'loss', 'f1', 'precision', 'recall']
            if include_params:
                header.extend(['total_params', 'trainable_params'])
            writer.writerow(header)
        row = [epoch, f'{loss:.6f}', f'{f1:.6f}', f'{precision:.6f}', f'{recall:.6f}']
        if include_params:
            row.extend([total_params, trainable_params])
        writer.writerow(row)
        f.flush()

def append_step_log(
    csv_path: str,
    epoch: int,
    step: int,
    loss: float,
    avg_loss: float,
    *,
    supervised_loss: Optional[float] = None,
    span_kd_loss: Optional[float] = None,
    boundary_kd_loss: Optional[float] = None,
    nested_kd_loss: Optional[float] = None,
    kd_eligible_windows: Optional[int] = None,
    kd_skipped_position_shift_windows: Optional[int] = None,
) -> None:
    """
    实时逐步写入训练日志（按step），包含本步loss与当前平均loss。

    当启用KD时，可同时记录本step参与KD的窗口数，以及由于
    Position Shifting而跳过KD的窗口数。
    """
    ensure_parent(csv_path)
    file_exists = os.path.exists(csv_path)
    include_kd = supervised_loss is not None
    include_kd_window_stats = (
        kd_eligible_windows is not None
        or kd_skipped_position_shift_windows is not None
    )

    if include_kd_window_stats and not include_kd:
        raise ValueError(
            "KD窗口统计只能在启用KD日志时写入（supervised_loss不能为空）"
        )
    if include_kd_window_stats and (
        kd_eligible_windows is None
        or kd_skipped_position_shift_windows is None
    ):
        raise ValueError(
            "kd_eligible_windows 与 kd_skipped_position_shift_windows 必须同时提供"
        )

    expected_header = ['epoch', 'step', 'loss', 'avg_loss']
    if include_kd:
        expected_header.extend([
            'supervised_loss',
            'span_kd_loss',
            'boundary_kd_loss',
            'nested_kd_loss',
        ])
        if include_kd_window_stats:
            expected_header.extend([
                'kd_eligible_windows',
                'kd_skipped_position_shift_windows',
            ])

    if file_exists:
        with open(csv_path, 'r', encoding='utf-8') as rf:
            existing_header = next(csv.reader([rf.readline().strip()]), [])
        if existing_header != expected_header:
            raise ValueError(
                "现有 step 日志列结构与当前 Run 配置不一致；"
                "请使用新的 RUN_NAME，或删除该 Run 的旧 step 日志后重新训练"
            )

    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(expected_header)

        row = [epoch, step, f'{loss:.6f}', f'{avg_loss:.6f}']
        if include_kd:
            row.extend([
                f'{supervised_loss:.6f}',
                f'{(span_kd_loss or 0.0):.6f}',
                f'{(boundary_kd_loss or 0.0):.6f}',
                f'{(nested_kd_loss or 0.0):.6f}',
            ])
            if include_kd_window_stats:
                row.extend([
                    int(kd_eligible_windows),
                    int(kd_skipped_position_shift_windows),
                ])

        writer.writerow(row)
        f.flush()

def append_class_epoch_log(csv_path: str, rows) -> None:
    """
    逐epoch写入按类别评估日志。
    rows: List[Dict], each includes epoch/class_id/class_name/f1/precision/recall/tp/pred/gold
    """
    ensure_parent(csv_path)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['epoch', 'class_id', 'class_name', 'f1', 'precision', 'recall', 'tp', 'pred', 'gold'])
        for row in rows:
            writer.writerow([
                row['epoch'],
                row['class_id'],
                row['class_name'],
                f"{row['f1']:.6f}",
                f"{row['precision']:.6f}",
                f"{row['recall']:.6f}",
                row['tp'],
                row['pred'],
                row['gold'],
            ])
        f.flush()


def append_structure_epoch_log(csv_path: str, rows) -> None:
    """记录嵌套角色与实体长度分桶的精确匹配P/R/F1。"""
    ensure_parent(csv_path)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['epoch', 'bucket', 'f1', 'precision', 'recall', 'tp', 'pred', 'gold'])
        for row in rows:
            writer.writerow([
                row['epoch'],
                row['bucket'],
                f"{row['f1']:.6f}",
                f"{row['precision']:.6f}",
                f"{row['recall']:.6f}",
                row['tp'],
                row['pred'],
                row['gold'],
            ])
        f.flush()


def save_best_model(model, path: str) -> None:
    import torch
    ensure_parent(path)
    tmp_path = path + '.tmp'
    torch.save(model.state_dict(), tmp_path)
    os.replace(tmp_path, path)

def save_topk_model(
    model,
    f1_score: float,
    epoch: int,
    out_dir: str,
    meta_path: str,
    k: int = 3,
    checkpoint_metadata: Optional[Dict] = None,
) -> None:
    """
    保存当前模型到out_dir，并使用meta文件维护前k个最优模型（按f1降序）。
    超出k的旧模型会被自动删除。
    """
    import torch
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError(f"Top-K 的 k 必须为正整数，当前为 {k}")
    ensure_parent(meta_path)
    os.makedirs(out_dir, exist_ok=True)

    # 生成文件名：f1保留4位小数，含epoch
    filename = f'best_f1_{f1_score:.4f}_epoch_{epoch}.pth'
    filepath = os.path.join(out_dir, filename)
    checkpoint_tmp = filepath + '.tmp'
    torch.save(model.state_dict(), checkpoint_tmp)
    os.replace(checkpoint_tmp, filepath)
    sidecar_path = filepath + '.meta.json'
    if checkpoint_metadata is not None:
        sidecar = dict(checkpoint_metadata)
        sidecar.update({
            'checkpoint': os.path.abspath(filepath),
            'epoch': int(epoch),
            'dev_f1': float(f1_score),
            'checkpoint_sha256': file_sha256(filepath),
        })
        sidecar_tmp = sidecar_path + '.tmp'
        save_params(sidecar, sidecar_tmp)
        os.replace(sidecar_tmp, sidecar_path)

    # 读取现有meta
    records = []
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                records = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Top-K 元数据损坏，拒绝静默覆盖: {meta_path}"
            ) from exc
        if not isinstance(records, list):
            raise ValueError(f"Top-K 元数据必须是列表: {meta_path}")

    # 加入新纪录并排序
    record = {"path": filepath, "f1": float(f1_score), "epoch": int(epoch)}
    if checkpoint_metadata is not None:
        record["meta_path"] = sidecar_path
    records.append(record)
    records.sort(key=lambda x: (-float(x["f1"]), int(x["epoch"])))

    # 保留前k，删除其余文件
    to_keep = records[:k]
    to_delete = records[k:]
    for item in to_delete:
        if os.path.exists(item["path"]):
            os.remove(item["path"])
        stale_sidecar = item.get("meta_path", item["path"] + '.meta.json')
        if os.path.exists(stale_sidecar):
            os.remove(stale_sidecar)

    # 写回meta
    meta_tmp = meta_path + '.tmp'
    with open(meta_tmp, 'w', encoding='utf-8') as f:
        json.dump(to_keep, f, ensure_ascii=False, indent=2)
    os.replace(meta_tmp, meta_path)


def verify_topk_models(out_dir: str, meta_path: str, k: int, completed_epochs: int) -> Dict:
    """训练结束后强制核验Top-K权重、sidecar和清单，禁止空目录静默通过。"""
    expected_count = min(int(k), int(completed_epochs))
    if expected_count <= 0:
        raise RuntimeError(
            f"Top-K 已开启但没有已完成epoch: completed_epochs={completed_epochs}"
        )
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Top-K 清单不存在: {meta_path}")
    with open(meta_path, 'r', encoding='utf-8') as f:
        records = json.load(f)
    if not isinstance(records, list) or len(records) != expected_count:
        raise RuntimeError(
            "Top-K 清单条数不符: "
            f"actual={len(records) if isinstance(records, list) else type(records)}, "
            f"expected={expected_count}, path={meta_path}"
        )
    scores = [float(item['f1']) for item in records]
    if scores != sorted(scores, reverse=True):
        raise RuntimeError(f"Top-K 清单未按F1降序排列: {scores}")

    out_dir_abs = os.path.abspath(out_dir)
    kept_paths = []
    for item in records:
        checkpoint = os.path.abspath(item['path'])
        if os.path.dirname(checkpoint) != out_dir_abs:
            raise RuntimeError(f"Top-K 权重越出当前Run目录: {checkpoint}")
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"Top-K 权重不存在: {checkpoint}")
        sidecar = os.path.abspath(
            item.get('meta_path', checkpoint + '.meta.json')
        )
        if not os.path.isfile(sidecar):
            raise FileNotFoundError(f"Top-K 权重缺少元数据: {sidecar}")
        with open(sidecar, 'r', encoding='utf-8') as f:
            sidecar_meta = json.load(f)
        actual_hash = file_sha256(checkpoint)
        if sidecar_meta.get('checkpoint_sha256') != actual_hash:
            raise RuntimeError(
                "Top-K 权重哈希与sidecar不一致: "
                f"checkpoint={checkpoint}"
            )
        kept_paths.append(checkpoint)

    actual_paths = sorted(
        os.path.abspath(os.path.join(out_dir, name))
        for name in os.listdir(out_dir)
        if name.endswith('.pth')
    )
    if actual_paths != sorted(kept_paths):
        raise RuntimeError(
            "Top-K 目录中的权重与清单不一致: "
            f"directory={actual_paths}, manifest={sorted(kept_paths)}"
        )
    return {
        'topk_checkpoint_verification_passed': True,
        'topk_checkpoint_count': expected_count,
        'topk_checkpoint_dir': out_dir_abs,
        'topk_checkpoint_manifest': os.path.abspath(meta_path),
        'topk_checkpoint_files': kept_paths,
    }
