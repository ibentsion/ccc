"""Configuration dataclasses for the CodeComplete fine-tuning pipeline."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class StageConfig:
    """Configuration for a single curriculum learning stage."""
    stage: int
    min_lines: int
    max_lines: int


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""
    jsonl_path: str
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    stages: List[StageConfig] = field(default_factory=list)
