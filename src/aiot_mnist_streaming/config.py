from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import random
from typing import Any

import numpy as np
import torch


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int = 128
    epochs_full: int = 3
    epochs_tiny: int = 3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    train_limit: int | None = 15000
    test_limit: int | None = 4000
    num_workers: int = 2
    use_amp: bool = True


@dataclass(slots=True)
class CompressionConfig:
    prune_full_amount: float = 0.40
    prune_tiny_amount: float = 0.30
    cloud_speedup: float = 2.5
    preview_model: str = "tiny_quant"
    confidence_threshold: float = 0.92


@dataclass(slots=True)
class RuntimeConfig:
    deadline_ms: tuple[float, ...] = (20.0,)
    stale_ms: float = 220.0
    queue_capacity: int = 32
    edge_memory_budget_mb: tuple[float, ...] = (2.5, 1.0, 0.5, 0.25)
    edge_workers: int = 1
    cloud_workers: int = 1
    model_load_bandwidth_mb_per_ms: tuple[float, ...] = (0.22, 0.10, 0.05)
    model_evict_penalty_ms: tuple[float, ...] = (0.8, 2.0, 5.0)
    queue_scan_window: int = 6


@dataclass(slots=True)
class ExperimentConfig:
    seeds: tuple[int, ...] = (11, 22)
    rates_sps: tuple[int, ...] = (100, 150, 250)
    arrival_modes: tuple[str, ...] = ("bursty",)
    samples_per_scenario: int = 700
    network_profiles: tuple[str, ...] = ("congested",)


@dataclass(slots=True)
class ProjectConfig:
    seed: int = 42
    quick_mode: bool = True
    device: str = "cpu"
    data_dir: Path = Path("data_mnist")
    output_root: Path = Path("outputs_aiot_project_v2")
    training: TrainingConfig = field(default_factory=TrainingConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    def snapshot(self) -> dict[str, Any]:
        return {
            "training": asdict(self.training),
            "compression": asdict(self.compression),
            "runtime": asdict(self.runtime),
            "experiment": asdict(self.experiment),
            "seed": self.seed,
            "quick_mode": self.quick_mode,
            "device": self.device,
            "data_dir": str(self.data_dir),
            "output_root": str(self.output_root),
        }


def detect_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_project_config(*, quick_mode: bool = True) -> ProjectConfig:
    training = TrainingConfig(
        epochs_full=3 if quick_mode else 6,
        epochs_tiny=3 if quick_mode else 5,
        train_limit=15000 if quick_mode else 40000,
        test_limit=4000 if quick_mode else None,
    )
    runtime = RuntimeConfig()
    experiment = ExperimentConfig(
        seeds=(11, 22) if quick_mode else (11, 22, 33, 44, 55),
        rates_sps=(100, 150, 250) if quick_mode else (10, 20, 35, 60, 100, 150, 250),
        arrival_modes=("bursty",) if quick_mode else ("periodic", "poisson", "bursty"),
        samples_per_scenario=700 if quick_mode else 1600,
        network_profiles=("congested",) if quick_mode else ("good", "congested"),
    )
    return ProjectConfig(
        quick_mode=quick_mode,
        device=str(detect_device()),
        training=training,
        runtime=runtime,
        experiment=experiment,
    )


def normalize_positive_sweep(values: float | tuple[float, ...] | list[float], name: str) -> tuple[float, ...]:
    if isinstance(values, (tuple, list, np.ndarray)):
        normalized = tuple(float(v) for v in values if float(v) > 0)
    else:
        normalized = (float(values),)
    if not normalized:
        raise ValueError(f"{name} must contain at least one positive value")
    return normalized


def write_config_snapshot(config: ProjectConfig, path: Path) -> None:
    path.write_text(json.dumps(config.snapshot(), indent=2), encoding="utf-8")
