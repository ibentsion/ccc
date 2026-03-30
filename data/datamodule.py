"""PyTorch Lightning DataModule for curriculum learning.

Exports:
    CurriculumDataModule — LightningDataModule providing per-stage DataLoaders
"""

from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data.config import PipelineConfig
from data.dataset import load_jsonl, split_dataset
from data.curriculum_dataset import FIMDataset, HybridReplayDataset


def _collate_fn(batch: list) -> dict:
    """Collate a list of FIM sample dicts into a batched dict with list values.

    Args:
        batch: List of dicts, each with keys fim_text, prefix, suffix, middle, n_lines.

    Returns:
        Single dict with the same keys; each value is a Python list of per-sample values.
    """
    keys = ["fim_text", "prefix", "suffix", "middle", "n_lines"]
    return {k: [item[k] for item in batch] for k in keys}


class CurriculumDataModule(pl.LightningDataModule):
    """Lightning DataModule providing per-stage DataLoaders for curriculum training.

    Call setup() before calling any dataloader method. setup() loads the JSONL
    file and partitions it into train/val/test splits using the PipelineConfig.

    Args:
        config: PipelineConfig with jsonl_path, split ratios, seed, and stages list.
        batch_size: Batch size for all DataLoaders (default 8).
        collate_fn: Optional collate function; falls back to _collate_fn if None.
    """

    def __init__(self, config: PipelineConfig, batch_size: int = 8, collate_fn=None):
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        self.custom_collate_fn = collate_fn
        self.train_records = []
        self.val_records = []
        self.test_records = []

    def setup(self, stage: Optional[str] = None):
        """Load JSONL and partition into train/val/test splits."""
        records = load_jsonl(self.config.jsonl_path)
        self.train_records, self.val_records, self.test_records = split_dataset(
            records,
            train=self.config.train_ratio,
            val=self.config.val_ratio,
            test=self.config.test_ratio,
            seed=self.config.seed,
        )

    def train_dataloader(self, curriculum_stage: int = 1) -> DataLoader:
        """Return a DataLoader for training at the given curriculum stage.

        Builds a HybridReplayDataset mixing current-stage samples with prior-stage
        samples (25% replay for stages > 1, no replay for stage 1).

        Args:
            curriculum_stage: 1-indexed stage number (must be <= len(config.stages)).

        Returns:
            DataLoader with shuffle=False and custom collate_fn.
        """
        current_cfg = self.config.stages[curriculum_stage - 1]
        current_ds = FIMDataset(self.train_records, current_cfg, seed=self.config.seed)
        prior_ds_list = [
            FIMDataset(self.train_records, self.config.stages[i], seed=self.config.seed)
            for i in range(curriculum_stage - 1)
        ]
        hybrid_ds = HybridReplayDataset(current_ds, prior_ds_list, replay_ratio=0.25)
        return DataLoader(
            hybrid_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.custom_collate_fn if self.custom_collate_fn else _collate_fn,
        )

    def val_dataloader(self, curriculum_stage: int = 1) -> DataLoader:
        """Return a DataLoader for validation at the given curriculum stage.

        No replay on validation — only current-stage samples.

        Args:
            curriculum_stage: 1-indexed stage number (must be <= len(config.stages)).

        Returns:
            DataLoader with shuffle=False and custom collate_fn.
        """
        current_cfg = self.config.stages[curriculum_stage - 1]
        val_ds = FIMDataset(self.val_records, current_cfg, seed=self.config.seed)
        return DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.custom_collate_fn if self.custom_collate_fn else _collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """Return a DataLoader for test evaluation using the last stage config.

        Returns:
            DataLoader with shuffle=False and custom collate_fn.
        """
        last_cfg = self.config.stages[-1]
        test_ds = FIMDataset(self.test_records, last_cfg, seed=self.config.seed)
        return DataLoader(
            test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.custom_collate_fn if self.custom_collate_fn else _collate_fn,
        )
