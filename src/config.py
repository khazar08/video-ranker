from __future__ import annotations
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


class Config(dict):

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[key] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get_path(self, key: str, default: Any = None) -> Any:
        """Fetch a dotted path like 'retrieval.two_tower.dim', or `default`."""
        node: Any = self
        for part in key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node


def load_config(path: str | Path) -> Config:
    """Load a YAML config file into a `Config`."""
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Config {path} did not parse to a mapping.")
    cfg = Config(raw)
    cfg["_config_path"] = str(path)
    cfg["_config_name"] = path.stem
    return cfg


def seed_everything(seed: int) -> None:
    """Seed python, numpy, and (if available) torch for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic cudnn where relevant (no-op on CPU/MPS).
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


@dataclass
class Paths:

    dataset_dir: Path
    artifacts_dir: Path
    results_dir: Path
    processed_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.processed_dir = self.artifacts_dir / "processed"
        for d in (self.artifacts_dir, self.results_dir, self.processed_dir):
            d.mkdir(parents=True, exist_ok=True)


def resolve_paths(cfg: Config) -> Paths:
    """Build a `Paths` object from a config's `data` / output settings."""
    small = bool(cfg.get_path("data.small", False))
    folder = "ml-latest-small" if small else "ml-25m"
    dataset_dir = REPO_ROOT / "data" / folder
    artifacts_dir = REPO_ROOT / cfg.get_path("output.artifacts_dir", "artifacts")
    results_dir = REPO_ROOT / cfg.get_path("output.results_dir", "results")
    return Paths(
        dataset_dir=dataset_dir,
        artifacts_dir=artifacts_dir / cfg["_config_name"],
        results_dir=results_dir,
    )
