# -*- coding: utf-8 -*-
"""HAEGP / Standard GlobalPointer inference efficiency benchmark.

Reports the metrics requested by the reviewers:
parameters, checkpoint size, profiled FLOPs estimate, analytical decoder
FLOPs, latency, throughput, CUDA peak memory and CPU process peak RSS.

Run this script separately for checkpoints trained with efficient=True/False;
do not load one architecture's checkpoint into the other architecture.
"""
import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import statistics
import sys
import time
import multiprocessing as mp
import queue as queue_module
from datetime import datetime, timezone

import torch
from torch.profiler import ProfilerActivity, profile
from transformers import AutoConfig, AutoModel, AutoTokenizer

try:
    import resource
except ImportError:  # Windows
    resource = None

from GlobalPointer import GlobalPointer
from config import (
    BEST_MODEL_PATH,
    EFFICIENCY_DIR,
    ENT_CLS_NUM,
    GP_EFFICIENT,
    GP_HEAD_SIZE,
    EFFICIENCY_CPU_SETTING_TIMEOUT_SECONDS,
    PRETRAINED_MODEL_DIR,
    RNN_HIDDEN_SIZE,
    RNN_LAYERS,
    SEED,
    USE_BIGRU,
)


def parse_int_list(raw):
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not values or any(x <= 0 for x in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pretrained_artifact_fingerprint(model_dir):
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
            fingerprints[filename] = file_sha256(path)
    return fingerprints


def parse_bool(raw):
    raw = str(raw).strip().lower()
    if raw in {"1", "true", "yes", "y"}:
        return True
    if raw in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def model_parameter_stats(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder = sum(p.numel() for p in model.encoder.parameters())
    rnn = sum(p.numel() for p in model.bigru.parameters()) if model.bigru is not None else 0
    head = total - encoder - rnn
    return {
        "parameters_total": int(total),
        "parameters_trainable": int(trainable),
        "parameters_encoder": int(encoder),
        "parameters_rnn": int(rnn),
        "parameters_task_head": int(head),
        "parameters_total_m": total / 1e6,
        "parameters_task_head_m": head / 1e6,
    }


def build_inputs(batch_size, seq_len, device):
    # IDs are valid for chinese-roberta-wwm-ext: [CLS]=101, [SEP]=102.
    input_ids = torch.full((batch_size, seq_len), 100, dtype=torch.long, device=device)
    input_ids[:, 0] = 101
    input_ids[:, -1] = 102
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.float32, device=device)
    token_type_ids = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
    return input_ids, attention_mask, token_type_ids


def percentile(values, q):
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[idx]


def analytical_decoder_ops(model, batch_size, seq_len):
    """Approximate decoder multiply-accumulates; additions/masks are excluded."""
    in_dim = model.rnn_hidden_size * 2 if model.use_bigru else model.hidden_size
    c = model.ent_type_size
    d = model.inner_dim
    b = batch_size
    n = seq_len
    if model.efficient:
        projection_macs = 2 * b * n * in_dim * d + 2 * b * n * in_dim * c
        span_macs = b * n * n * d
    else:
        projection_macs = b * n * in_dim * (2 * c * d)
        span_macs = b * c * n * n * d
    macs = projection_macs + span_macs
    return {
        "decoder_macs": int(macs),
        "decoder_gmacs": macs / 1e9,
        # One MAC is reported as two floating-point operations.
        "decoder_flops_estimate": int(2 * macs),
        "decoder_gflops_estimate": 2 * macs / 1e9,
    }


def profiled_flops(model, inputs, device):
    """PyTorch operator-coverage FLOPs estimate; unsupported operators are omitted."""
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    try:
        with profile(activities=activities, with_flops=True) as prof:
            with torch.inference_mode():
                model(*inputs)
            synchronize(device)
        total = sum(int(getattr(evt, "flops", 0) or 0) for evt in prof.key_averages())
        return total if total > 0 else None
    except Exception as exc:
        print(f"[warning] profiler FLOPs unavailable: {exc}")
        return None


def benchmark_setting(model, device, batch_size, seq_len, warmup, repeats):
    if device.type == "cuda":
        # 同一GPU进程连续测多组形状时，先清除上一组未占用缓存；否则 reserved peak
        # 会继承前一组的大块缓存，不能作为当前设置的独立显存指标。
        gc.collect()
        torch.cuda.empty_cache()
        synchronize(device)
    inputs = build_inputs(batch_size, seq_len, device)
    with torch.inference_mode():
        for _ in range(warmup):
            model(*inputs)
    synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    latencies_ms = []
    with torch.inference_mode():
        for _ in range(repeats):
            synchronize(device)
            start = time.perf_counter()
            model(*inputs)
            synchronize(device)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)

    mean_ms = statistics.fmean(latencies_ms)
    row = {
        "device": str(device),
        "batch_size": batch_size,
        "sequence_length": seq_len,
        "warmup_iterations": warmup,
        "timed_iterations": repeats,
        "forward_latency_mean_ms_batch": mean_ms,
        "forward_latency_median_ms_batch": statistics.median(latencies_ms),
        "forward_latency_p95_ms_batch": percentile(latencies_ms, 0.95),
        "forward_latency_mean_ms_sample": mean_ms / batch_size,
        "forward_throughput_samples_per_second": batch_size / (mean_ms / 1000.0),
    }
    row.update(analytical_decoder_ops(model, batch_size, seq_len))

    if device.type == "cuda":
        row["cuda_peak_memory_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 2**20
        row["cuda_peak_memory_reserved_mb"] = torch.cuda.max_memory_reserved(device) / 2**20
        row["cuda_cache_cleared_before_setting"] = True
    else:
        # Linux ru_maxrss is KiB; this is process-level peak RSS, not an isolated forward allocation.
        if resource is None:
            row["cpu_process_peak_rss_mb"] = None
        else:
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            row["cpu_process_peak_rss_mb"] = rss / (2**20 if sys.platform == "darwin" else 1024.0)

    measured_flops = profiled_flops(model, inputs, device)
    row["profiled_flops_estimate"] = measured_flops
    row["profiled_gflops_estimate"] = (
        measured_flops / 1e9 if measured_flops is not None else None
    )
    return row


def load_model(
    checkpoint,
    efficient,
    device,
    *,
    pretrained_model_dir=PRETRAINED_MODEL_DIR,
    ent_cls_num=ENT_CLS_NUM,
    head_size=GP_HEAD_SIZE,
    rnn_hidden_size=RNN_HIDDEN_SIZE,
    rnn_layers=RNN_LAYERS,
    use_bigru=USE_BIGRU,
    encoder_num_hidden_layers=None,
):
    meta_path = checkpoint + ".meta.json"
    expected = {
        "gp_efficient": efficient,
        "use_bigru": use_bigru,
        "ent_cls_num": ent_cls_num,
        "gp_head_size": head_size,
        "rnn_hidden_size": rnn_hidden_size,
        "rnn_layers": rnn_layers,
    }
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        mismatches = {
            key: (meta.get(key), value)
            for key, value in expected.items()
            if meta.get(key) is not None and meta.get(key) != value
        }
        if mismatches:
            raise ValueError(f"checkpoint architecture metadata mismatch: {mismatches}")
        meta_encoder_layers = meta.get("encoder_num_hidden_layers")
        if (
            encoder_num_hidden_layers is not None
            and meta_encoder_layers is not None
            and int(encoder_num_hidden_layers) != int(meta_encoder_layers)
        ):
            raise ValueError(
                "checkpoint encoder depth metadata mismatch: "
                f"meta={meta_encoder_layers}, requested={encoder_num_hidden_layers}"
            )
        meta_pretrained_fingerprint = meta.get("pretrained_artifact_sha256")
        if meta_pretrained_fingerprint is not None:
            current_pretrained_fingerprint = pretrained_artifact_fingerprint(
                pretrained_model_dir
            )
            if meta_pretrained_fingerprint != current_pretrained_fingerprint:
                raise ValueError(
                    "checkpoint pretrained config/tokenizer fingerprint mismatch: "
                    f"meta={meta_pretrained_fingerprint}, "
                    f"current={current_pretrained_fingerprint}"
                )
    else:
        print(f"[warning] checkpoint metadata not found: {meta_path}")

    encoder_config = AutoConfig.from_pretrained(pretrained_model_dir)
    pretrained_layers = getattr(encoder_config, "num_hidden_layers", None)
    if encoder_num_hidden_layers is not None:
        if int(encoder_num_hidden_layers) <= 0:
            raise ValueError("encoder_num_hidden_layers must be positive")
        if pretrained_layers is None:
            raise ValueError("pretrained model config has no num_hidden_layers")
        if int(encoder_num_hidden_layers) > int(pretrained_layers):
            raise ValueError(
                "checkpoint encoder depth exceeds pretrained encoder depth: "
                f"checkpoint={encoder_num_hidden_layers}, pretrained={pretrained_layers}"
            )
        encoder_config.num_hidden_layers = int(encoder_num_hidden_layers)
    encoder = AutoModel.from_pretrained(
        pretrained_model_dir,
        config=encoder_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
    model = GlobalPointer(
        encoder,
        ent_cls_num,
        head_size,
        rnn_hidden_size=rnn_hidden_size,
        rnn_layers=rnn_layers,
        efficient=efficient,
        use_bigru=use_bigru,
        sep_token_id=tokenizer.sep_token_id,
    )
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def _isolated_cpu_worker(queue, kwargs):
    try:
        torch.manual_seed(SEED)
        torch.set_num_threads(max(1, kwargs.pop("cpu_threads")))
        batch_size = kwargs.pop("batch_size")
        seq_len = kwargs.pop("seq_len")
        warmup = kwargs.pop("warmup")
        repeats = kwargs.pop("repeats")
        model = load_model(device=torch.device("cpu"), **kwargs)
        row = benchmark_setting(model, torch.device("cpu"), batch_size, seq_len, warmup, repeats)
        row["cpu_memory_isolated_process"] = True
        row["parameter_stats"] = model_parameter_stats(model)
        queue.put({"ok": True, "row": row})
    except Exception as exc:
        queue.put({"ok": False, "error": repr(exc)})


def benchmark_cpu_isolated(
    model_kwargs, batch_size, seq_len, warmup, repeats, cpu_threads,
    timeout_seconds,
):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    kwargs = dict(model_kwargs)
    kwargs.update({
        "batch_size": batch_size,
        "seq_len": seq_len,
        "warmup": warmup,
        "repeats": repeats,
        "cpu_threads": cpu_threads,
    })
    process = ctx.Process(target=_isolated_cpu_worker, args=(queue, kwargs))
    process.start()
    process.join(timeout=float(timeout_seconds))
    if process.is_alive():
        process.terminate()
        process.join(timeout=10)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(timeout=10)
        queue.close()
        raise TimeoutError(
            "CPU efficiency setting timed out: "
            f"batch={batch_size}, seq={seq_len}, timeout={timeout_seconds}s"
        )
    try:
        result = queue.get(timeout=5)
    except queue_module.Empty as exc:
        raise RuntimeError(
            f"CPU benchmark child produced no result (exit code {process.exitcode})"
        ) from exc
    finally:
        queue.close()
    if process.exitcode != 0 or not result.get("ok"):
        raise RuntimeError(result.get("error", f"CPU benchmark child exited with {process.exitcode}"))
    return result["row"]


def write_results(payload, output_prefix):
    os.makedirs(os.path.dirname(output_prefix) or ".", exist_ok=True)
    json_path = output_prefix + ".json"
    csv_path = output_prefix + ".csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    rows = payload["measurements"]
    if rows:
        columns = sorted({key for row in rows for key in row})
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    return json_path, csv_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=BEST_MODEL_PATH)
    parser.add_argument("--pretrained-model-dir", default=PRETRAINED_MODEL_DIR)
    parser.add_argument("--efficient", type=parse_bool, default=GP_EFFICIENT)
    parser.add_argument("--ent-cls-num", type=int, default=ENT_CLS_NUM)
    parser.add_argument("--head-size", type=int, default=GP_HEAD_SIZE)
    parser.add_argument("--rnn-hidden-size", type=int, default=RNN_HIDDEN_SIZE)
    parser.add_argument("--rnn-layers", type=int, default=RNN_LAYERS)
    parser.add_argument("--use-bigru", type=parse_bool, default=USE_BIGRU)
    parser.add_argument(
        "--encoder-num-hidden-layers",
        type=int,
        default=None,
        help=(
            "Override encoder depth. By default it is read from checkpoint metadata; "
            "old full-model checkpoints without metadata use the pretrained depth."
        ),
    )
    parser.add_argument("--devices", default="cuda,cpu" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-sizes", type=parse_int_list, default=[1, 16])
    parser.add_argument("--sequence-lengths", type=parse_int_list, default=[64, 128, 256])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats-gpu", type=int, default=100)
    parser.add_argument("--repeats-cpu", type=int, default=30)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument(
        "--cpu-setting-timeout-seconds",
        type=int,
        default=EFFICIENCY_CPU_SETTING_TIMEOUT_SECONDS,
    )
    parser.add_argument("--output-prefix", default=None)
    args = parser.parse_args()

    if args.cpu_setting_timeout_seconds <= 0:
        raise ValueError("--cpu-setting-timeout-seconds must be positive")

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, args.cpu_threads))

    checkpoint_meta_path = args.checkpoint + ".meta.json"
    checkpoint_meta = {}
    if os.path.isfile(checkpoint_meta_path):
        with open(checkpoint_meta_path, "r", encoding="utf-8") as f:
            checkpoint_meta = json.load(f)
    meta_encoder_layers = checkpoint_meta.get("encoder_num_hidden_layers")
    if (
        args.encoder_num_hidden_layers is not None
        and meta_encoder_layers is not None
        and int(args.encoder_num_hidden_layers) != int(meta_encoder_layers)
    ):
        raise ValueError(
            "checkpoint encoder depth metadata mismatch: "
            f"meta={meta_encoder_layers}, requested={args.encoder_num_hidden_layers}"
        )
    pretrained_config = AutoConfig.from_pretrained(args.pretrained_model_dir)
    pretrained_encoder_layers = getattr(pretrained_config, "num_hidden_layers", None)
    resolved_encoder_layers = (
        int(args.encoder_num_hidden_layers)
        if args.encoder_num_hidden_layers is not None
        else int(meta_encoder_layers)
        if meta_encoder_layers is not None
        else int(pretrained_encoder_layers)
        if pretrained_encoder_layers is not None
        else None
    )
    if resolved_encoder_layers is None or resolved_encoder_layers <= 0:
        raise ValueError("无法从参数、检查点元数据或预训练 config 确定编码器层数")
    if (
        pretrained_encoder_layers is not None
        and resolved_encoder_layers > int(pretrained_encoder_layers)
    ):
        raise ValueError(
            "encoder depth exceeds pretrained model depth: "
            f"resolved={resolved_encoder_layers}, pretrained={pretrained_encoder_layers}"
        )

    model_kwargs = {
        "checkpoint": args.checkpoint,
        "efficient": args.efficient,
        "pretrained_model_dir": args.pretrained_model_dir,
        "ent_cls_num": args.ent_cls_num,
        "head_size": args.head_size,
        "rnn_hidden_size": args.rnn_hidden_size,
        "rnn_layers": args.rnn_layers,
        "use_bigru": args.use_bigru,
        "encoder_num_hidden_layers": resolved_encoder_layers,
    }

    head_name = "efficient_gp" if args.efficient else "standard_gp"
    output_prefix = args.output_prefix or os.path.join(
        EFFICIENCY_DIR, f"efficiency_{head_name}"
    )
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": head_name,
        "architecture_settings": {
            "efficient": args.efficient,
            "ent_cls_num": args.ent_cls_num,
            "head_size": args.head_size,
            "rnn_hidden_size": args.rnn_hidden_size,
            "rnn_layers": args.rnn_layers,
            "use_bigru": args.use_bigru,
            "encoder_num_hidden_layers": resolved_encoder_layers,
            "pretrained_encoder_num_hidden_layers": pretrained_encoder_layers,
            "pretrained_model_dir": os.path.abspath(args.pretrained_model_dir),
        },
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_size_mb": os.path.getsize(args.checkpoint) / 2**20,
        "pytorch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_model": platform.processor() or platform.machine(),
        "cpu_threads": torch.get_num_threads(),
        "cpu_setting_timeout_seconds": args.cpu_setting_timeout_seconds,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "flops_note": (
            "profiled_flops_estimate only covers operators supported by torch.profiler; "
            "decoder_flops_estimate is analytical and counts one MAC as two FLOPs."
        ),
        "latency_note": (
            "forward_* metrics measure model.forward only; tokenization, host-to-device "
            "transfer, span decoding, deduplication and file I/O are excluded."
        ),
        "cpu_memory_note": (
            "Each CPU setting runs in a fresh spawned process, so cpu_process_peak_rss_mb "
            "is an isolated process peak including model loading and the measured forward."
        ),
        "measurements": [],
    }

    parameter_stats_written = False
    for raw_device in [x.strip() for x in args.devices.split(",") if x.strip()]:
        if raw_device.startswith("cuda") and not torch.cuda.is_available():
            print(f"[warning] skipping unavailable device: {raw_device}")
            continue
        device = torch.device(raw_device)
        repeats = args.repeats_gpu if device.type == "cuda" else args.repeats_cpu
        if device.type == "cpu":
            for batch_size in args.batch_sizes:
                for seq_len in args.sequence_lengths:
                    print(
                        f"benchmarking {head_name}: device={device}, "
                        f"batch={batch_size}, seq={seq_len}"
                    )
                    row = benchmark_cpu_isolated(
                        model_kwargs,
                        batch_size,
                        seq_len,
                        max(0, args.warmup),
                        max(1, repeats),
                        max(1, args.cpu_threads),
                        args.cpu_setting_timeout_seconds,
                    )
                    row["architecture"] = head_name
                    row["efficient"] = args.efficient
                    stats = row.pop("parameter_stats")
                    if not parameter_stats_written:
                        payload.update(stats)
                        parameter_stats_written = True
                    payload["measurements"].append(row)
        else:
            model = load_model(device=device, **model_kwargs)
            if not parameter_stats_written:
                payload.update(model_parameter_stats(model))
                parameter_stats_written = True
            for batch_size in args.batch_sizes:
                for seq_len in args.sequence_lengths:
                    print(
                        f"benchmarking {head_name}: device={device}, "
                        f"batch={batch_size}, seq={seq_len}"
                    )
                    row = benchmark_setting(
                        model,
                        device,
                        batch_size,
                        seq_len,
                        max(0, args.warmup),
                        max(1, repeats),
                    )
                    row["architecture"] = head_name
                    row["efficient"] = args.efficient
                    payload["measurements"].append(row)
            del model
            torch.cuda.empty_cache()

    json_path, csv_path = write_results(payload, output_prefix)
    print(f"saved: {json_path}")
    print(f"saved: {csv_path}")


if __name__ == "__main__":
    main()
