from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class RunPaths:
    root: Path
    run_dir: Path
    figures: Path
    artifacts: Path
    checkpoints: Path

    def ensure(self) -> None:
        for path in (self.root, self.run_dir, self.figures, self.artifacts, self.checkpoints):
            path.mkdir(parents=True, exist_ok=True)


def create_run_paths(output_root: Path, run_name: str) -> RunPaths:
    run_dir = output_root / run_name
    paths = RunPaths(
        root=output_root,
        run_dir=run_dir,
        figures=run_dir / "figures",
        artifacts=run_dir / "artifacts",
        checkpoints=run_dir / "checkpoints",
    )
    paths.ensure()
    return paths
