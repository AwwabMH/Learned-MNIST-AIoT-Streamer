from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

try:
    from torchvision import datasets, transforms
except Exception as exc:  # pragma: no cover - import guard
    datasets = None
    transforms = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@dataclass(slots=True)
class MnistDataModule:
    data_dir: Path
    batch_size: int
    num_workers: int = 2
    train_limit: int | None = None
    test_limit: int | None = None

    def _transform(self):
        if transforms is None:
            raise RuntimeError(f"torchvision is required for MNIST loading: {_IMPORT_ERROR}")
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def build_datasets(self):
        if datasets is None:
            raise RuntimeError(f"torchvision is required for MNIST loading: {_IMPORT_ERROR}")
        transform = self._transform()
        train_ds = datasets.MNIST(root=self.data_dir, train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(root=self.data_dir, train=False, download=True, transform=transform)
        if self.train_limit is not None:
            train_ds = Subset(train_ds, list(range(self.train_limit)))
        if self.test_limit is not None:
            test_ds = Subset(test_ds, list(range(self.test_limit)))
        return train_ds, test_ds

    def build_loaders(self):
        train_ds, test_ds = self.build_datasets()
        pin_memory = torch.cuda.is_available()
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
        )
        return train_loader, test_loader
