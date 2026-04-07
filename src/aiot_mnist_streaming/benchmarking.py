from __future__ import annotations

import copy
import io
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


@torch.no_grad()
def predict_per_sample(model_cpu, loader):
    model_cpu = model_cpu.eval().cpu()
    preds_all, labels_all, confs_all = [], [], []
    for x, y in loader:
        x = x.cpu()
        logits = model_cpu(x)
        probs = torch.softmax(logits, dim=1)
        confs, preds = probs.max(dim=1)
        preds_all.append(preds.cpu())
        labels_all.append(y.cpu())
        confs_all.append(confs.cpu())
    preds = torch.cat(preds_all).numpy()
    labels = torch.cat(labels_all).numpy()
    confs = torch.cat(confs_all).numpy()
    correct = (preds == labels).astype(np.int32)
    return {
        "pred": preds,
        "label": labels,
        "conf": confs,
        "correct": correct,
        "accuracy": float(correct.mean()),
    }


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def state_dict_size_mb(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return len(buf.getvalue()) / (1024 ** 2)


def estimate_sparsity(model):
    total, zeros = 0, 0
    for p in model.parameters():
        arr = p.detach().cpu()
        total += arr.numel()
        zeros += (arr == 0).sum().item()
    return zeros / total if total else 0.0


def estimate_macs(model, input_shape=(1, 1, 28, 28)):
    model = copy.deepcopy(model).cpu().eval()
    hooks = []
    macs = 0

    def conv_hook(module, inp, out):
        nonlocal macs
        x = inp[0]
        batch = x.shape[0]
        out_h, out_w = out.shape[2], out.shape[3]
        kh, kw = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
        macs += batch * out_h * out_w * module.out_channels * (module.in_channels // module.groups) * kh * kw

    def linear_hook(module, inp, out):
        nonlocal macs
        batch = inp[0].shape[0]
        macs += batch * module.in_features * module.out_features

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    with torch.no_grad():
        x = torch.randn(*input_shape)
        model(x)
    for hook in hooks:
        hook.remove()
    return int(macs)


@torch.no_grad()
def benchmark_single_sample_latencies(model_cpu, n_warmup=20, n_runs=100):
    model_cpu = model_cpu.cpu().eval()
    x = torch.randn(1, 1, 28, 28)
    for _ in range(n_warmup):
        _ = model_cpu(x)
    samples = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = model_cpu(x)
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1000.0)
    arr = np.array(samples, dtype=float)
    return {
        "latency_samples_ms": arr,
        "edge_mean_ms": float(arr.mean()),
        "edge_p50_ms": float(np.percentile(arr, 50)),
        "edge_p95_ms": float(np.percentile(arr, 95)),
        "edge_p99_ms": float(np.percentile(arr, 99)),
        "edge_std_ms": float(arr.std()),
    }


def build_model_metrics_table(model_bank, loader, quick_mode: bool, cloud_speedup: float, load_bandwidth_mb_per_ms: float = 0.22):
    static_rows = []
    predictions = {}
    latency_samples = {}
    accuracy_rows = []
    latency_rows = []
    for name, model in model_bank.items():
        metrics = {
            "model": name,
            "params": count_parameters(model),
            "state_dict_mb": state_dict_size_mb(model),
            "sparsity": estimate_sparsity(model),
            "macs_per_sample": estimate_macs(model),
        }
        static_rows.append(metrics)
        outputs = predict_per_sample(model, loader)
        predictions[name] = outputs
        accuracy_rows.append({"model": name, "accuracy": outputs["accuracy"]})
        bench = benchmark_single_sample_latencies(model, n_warmup=20 if quick_mode else 25, n_runs=80 if quick_mode else 140)
        latency_samples[name] = bench.pop("latency_samples_ms")
        latency_rows.append({"model": name, **bench})
    static_df = pd.DataFrame(static_rows)
    acc_df = pd.DataFrame(accuracy_rows)
    latency_df = pd.DataFrame(latency_rows)
    catalog = static_df.merge(acc_df, on="model").merge(latency_df, on="model")
    catalog["cloud_exec_mean_ms"] = catalog["edge_mean_ms"] / float(cloud_speedup)
    catalog["cloud_exec_p95_ms"] = catalog["edge_p95_ms"] / float(cloud_speedup)
    catalog["memory_mb"] = catalog["state_dict_mb"].clip(lower=0.05)
    catalog["load_ms"] = catalog["memory_mb"] / float(load_bandwidth_mb_per_ms)
    catalog["family"] = catalog["model"].apply(lambda s: "full" if s.startswith("full") else "tiny")
    catalog["compressed"] = catalog["model"].str.contains("pruned|quant", regex=True)
    catalog = catalog.sort_values(["edge_mean_ms", "accuracy"]).reset_index(drop=True)
    return catalog, static_df, acc_df, latency_df, predictions, latency_samples
