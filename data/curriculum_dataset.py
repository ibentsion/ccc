"""FIM Dataset and Hybrid Replay Dataset for curriculum learning.

Exports:
    FIMDataset          — PyTorch Dataset wrapping FIM samples per stage config
    HybridReplayDataset — Dataset that mixes current-stage and prior-stage samples
"""

import random
import re

from torch.utils.data import Dataset

from data.config import StageConfig
from data.fim import create_fim_sample, GapError

# Pattern to extract code from markdown code blocks (```python ... ``` or ``` ... ```)
_CODE_FENCE_RE = re.compile(r"```(?:python)?\n(.*?)(?:\n```|$)", re.DOTALL)


def _extract_code(record: dict) -> str:
    """Extract Python source code from a JSONL record.

    The dataset stores code in the 'output' field, typically wrapped in a
    markdown code fence (```python ... ```). Falls back to 'input' if 'output'
    is absent or appears to be a plain text description.

    Args:
        record: A JSONL record dict with 'input' and/or 'output' keys.

    Returns:
        The Python code string (markdown fences stripped if present).
    """
    # Prefer 'output' field which contains the actual Python code
    raw = record.get("output", record.get("input", ""))
    match = _CODE_FENCE_RE.search(raw)
    if match:
        return match.group(1)
    return raw


class FIMDataset(Dataset):
    """Pre-generates FIM samples for a given set of records and stage config.

    During __init__, calls create_fim_sample for each record. Records where
    no valid gap exists (GapError) are silently skipped. The resulting list
    is stored in self.samples and never mutated after construction.

    Code is extracted from the 'output' field (stripping markdown fences),
    which is where the JSONL dataset stores actual Python source.

    Args:
        records: List of dicts from python-codes-25k.jsonl.
        stage_config: StageConfig specifying min_lines and max_lines for the gap.
        seed: Base random seed. Record i uses seed + i to vary n across records.
    """

    def __init__(self, records: list, stage_config: StageConfig, seed: int = 42):
        self.samples = []
        rng = random.Random(seed)
        line_choices = list(range(stage_config.min_lines, stage_config.max_lines + 1))
        for i, record in enumerate(records):
            n = rng.choice(line_choices)
            try:
                code = _extract_code(record)
                sample = create_fim_sample(code, n=n, seed=seed + i)
                self.samples.append(sample)
            except GapError:
                pass  # silently skip records with no valid gap

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class HybridReplayDataset(Dataset):
    """Mixes current-stage samples with prior-stage samples for replay.

    For Stage 1 (prior=[]): behaves as pure current dataset.
    For Stage N (prior non-empty): deterministically routes each index to either
    the current dataset or a prior-stage dataset using index-based routing:
        idx % 4 == 0 → prior (25%)
        idx % 4 != 0 → current (75%)

    The routing is deterministic — no runtime randomness — making batches
    reproducible across runs.

    Args:
        current: FIMDataset for the current training stage.
        prior: List of FIMDatasets from prior stages. Empty for Stage 1.
        replay_ratio: Nominal replay fraction (informational; actual routing
                      uses idx % 4 == 0 for a fixed 25% replay).
    """

    def __init__(self, current: FIMDataset, prior: list, replay_ratio: float = 0.25):
        self.current = current
        self.prior = prior
        self.replay_ratio = replay_ratio

    def __len__(self):
        return len(self.current)

    def __getitem__(self, idx):
        if self.prior and idx % 4 == 0:
            # Route to prior stage: pick prior dataset round-robin, then sample from it
            prior_ds = self.prior[idx % len(self.prior)]
            prior_idx = idx % len(prior_ds)
            return prior_ds[prior_idx]
        return self.current[idx % len(self.current)]
