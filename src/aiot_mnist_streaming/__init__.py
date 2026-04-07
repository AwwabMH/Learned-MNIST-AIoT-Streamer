"""AIoT streaming MNIST project package."""

from .config import (
    CompressionConfig,
    ExperimentConfig,
    ProjectConfig,
    RuntimeConfig,
    TrainingConfig,
    build_project_config,
    detect_device,
    set_global_seed,
)
from .models import LeNetFull, LeNetTiny

__all__ = [
    "CompressionConfig",
    "ExperimentConfig",
    "ProjectConfig",
    "RuntimeConfig",
    "TrainingConfig",
    "build_project_config",
    "detect_device",
    "set_global_seed",
    "LeNetFull",
    "LeNetTiny",
]
