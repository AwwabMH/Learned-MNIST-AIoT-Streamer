from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


def clone_to_cpu(model):
    return copy.deepcopy(model).cpu().eval()


def apply_global_pruning(model_cpu, amount: float):
    params = []
    for module in model_cpu.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            params.append((module, "weight"))
    prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=float(amount))
    for module, name in params:
        prune.remove(module, name)
    return model_cpu


def apply_dynamic_quantization(model_cpu):
    return torch.ao.quantization.quantize_dynamic(model_cpu, {nn.Linear}, dtype=torch.qint8)
