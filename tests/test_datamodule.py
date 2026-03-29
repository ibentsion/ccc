"""Tests for data.curriculum_dataset and data.datamodule — TDD."""
import pytest


# ---------------------------------------------------------------------------
# Task 1: FIMDataset and HybridReplayDataset tests
# ---------------------------------------------------------------------------

def _make_stage1_config():
    from data.config import StageConfig
    return StageConfig(stage=1, min_lines=1, max_lines=1)


def _make_stage2_config():
    from data.config import StageConfig
    return StageConfig(stage=2, min_lines=2, max_lines=3)


def _real_records(n=50):
    """Load a small number of real JSONL records for testing."""
    from data.dataset import load_jsonl
    return load_jsonl("python-codes-25k.jsonl")[:n]


def test_fim_dataset_generates_samples():
    """FIMDataset with 10 real records and Stage 1 config yields len > 0."""
    from data.curriculum_dataset import FIMDataset
    records = _real_records(10)
    stage_cfg = _make_stage1_config()
    ds = FIMDataset(records, stage_cfg, seed=42)
    assert len(ds) > 0


def test_fim_dataset_items_have_fim_text():
    """Every item from FIMDataset has 'fim_text' starting with FIM_BEGIN."""
    from data.curriculum_dataset import FIMDataset
    from data.fim import FIM_BEGIN
    records = _real_records(20)
    stage_cfg = _make_stage1_config()
    ds = FIMDataset(records, stage_cfg, seed=42)
    assert len(ds) > 0
    for i in range(len(ds)):
        item = ds[i]
        assert "fim_text" in item, f"Item {i} missing fim_text"
        assert item["fim_text"].startswith(FIM_BEGIN), (
            f"Item {i} fim_text does not start with FIM_BEGIN"
        )


def test_fim_dataset_skips_gap_errors():
    """Records with trivial code (only 'def foo(): pass') are silently skipped."""
    from data.curriculum_dataset import FIMDataset
    from data.config import StageConfig
    # These trivial records have no eligible lines for a gap
    trivial_records = [{"input": "def foo(): pass"} for _ in range(5)]
    stage_cfg = StageConfig(stage=1, min_lines=1, max_lines=1)
    ds = FIMDataset(trivial_records, stage_cfg, seed=42)
    # All should be skipped silently, no exception raised
    assert len(ds) == 0


def test_hybrid_stage1_no_replay():
    """HybridReplayDataset with prior=[] returns only current stage items."""
    from data.curriculum_dataset import FIMDataset, HybridReplayDataset
    records = _real_records(30)
    stage_cfg = _make_stage1_config()
    ds = FIMDataset(records, stage_cfg, seed=42)
    hybrid = HybridReplayDataset(ds, prior=[], replay_ratio=0.25)
    assert len(hybrid) == len(ds)
    # All items should come from current stage (n_lines == min_lines=1)
    for i in range(min(len(hybrid), 20)):
        item = hybrid[i]
        assert item["n_lines"] == stage_cfg.min_lines


def test_hybrid_replay_ratio():
    """HybridReplayDataset with prior=[stage1_ds] yields ~25% stage1 items."""
    from data.curriculum_dataset import FIMDataset, HybridReplayDataset
    # Need enough records so stage2 (min_lines=2) produces samples — use 300
    records = _real_records(300)
    stage1_cfg = _make_stage1_config()
    stage2_cfg = _make_stage2_config()
    stage1_ds = FIMDataset(records, stage1_cfg, seed=42)
    stage2_ds = FIMDataset(records, stage2_cfg, seed=42)
    assert len(stage1_ds) > 0, "stage1_ds is empty — not enough records"
    assert len(stage2_ds) > 0, "stage2_ds is empty — not enough records for 2-line gaps"
    hybrid = HybridReplayDataset(stage2_ds, prior=[stage1_ds], replay_ratio=0.25)

    n_items = min(len(hybrid), 100)
    n_lines_list = [hybrid[i]["n_lines"] for i in range(n_items)]
    # Stage 1 items have n_lines == 1; stage 2 items have n_lines in [2,3]
    replay_count = sum(1 for n in n_lines_list if n == 1)
    replay_pct = replay_count / len(n_lines_list)
    # Expect roughly 25% (allow 10%-40% tolerance)
    assert 0.10 <= replay_pct <= 0.40, (
        f"Replay ratio out of range: {replay_pct:.2f} (expected ~0.25)"
    )


def test_hybrid_deterministic():
    """Same idx returns same item across two HybridReplayDataset instantiations."""
    from data.curriculum_dataset import FIMDataset, HybridReplayDataset
    # Use enough records so stage2 produces samples
    records = _real_records(300)
    stage1_cfg = _make_stage1_config()
    stage2_cfg = _make_stage2_config()

    def make_hybrid():
        s1 = FIMDataset(records, stage1_cfg, seed=42)
        s2 = FIMDataset(records, stage2_cfg, seed=42)
        return HybridReplayDataset(s2, prior=[s1], replay_ratio=0.25)

    hybrid1 = make_hybrid()
    hybrid2 = make_hybrid()
    for i in range(min(len(hybrid1), 20)):
        assert hybrid1[i]["fim_text"] == hybrid2[i]["fim_text"], (
            f"Non-deterministic result at idx {i}"
        )


# ---------------------------------------------------------------------------
# Task 2: CurriculumDataModule tests
# ---------------------------------------------------------------------------

def _make_pipeline_config():
    from data.config import PipelineConfig, StageConfig
    return PipelineConfig(
        jsonl_path="python-codes-25k.jsonl",
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42,
        stages=[
            StageConfig(stage=1, min_lines=1, max_lines=1),
            StageConfig(stage=2, min_lines=2, max_lines=3),
        ],
    )


def test_datamodule_setup_creates_splits():
    """After setup(), train+val+test records equals total loaded records."""
    from data.datamodule import CurriculumDataModule
    from data.dataset import load_jsonl
    cfg = _make_pipeline_config()
    dm = CurriculumDataModule(cfg)
    dm.setup()
    total_records = len(load_jsonl(cfg.jsonl_path))
    assert (
        len(dm.train_records) + len(dm.val_records) + len(dm.test_records)
        == total_records
    )


def test_train_dataloader_stage1_batch_structure():
    """First batch from train_dataloader(1) is a dict with 'fim_text' as list of strings."""
    from data.datamodule import CurriculumDataModule
    cfg = _make_pipeline_config()
    dm = CurriculumDataModule(cfg)
    dm.setup()
    loader = dm.train_dataloader(curriculum_stage=1)
    batch = next(iter(loader))
    assert isinstance(batch, dict), "Batch should be a dict"
    assert "fim_text" in batch, "Batch missing 'fim_text' key"
    assert isinstance(batch["fim_text"], list), "fim_text should be a list"
    assert len(batch["fim_text"]) > 0, "fim_text list should not be empty"
    assert all(isinstance(t, str) for t in batch["fim_text"]), (
        "All fim_text entries should be strings"
    )


def test_train_dataloader_stage2_has_replay():
    """Stage 2 DataLoader batches contain at least one item with n_lines == 1 (replay)."""
    from data.datamodule import CurriculumDataModule
    cfg = _make_pipeline_config()
    dm = CurriculumDataModule(cfg)
    dm.setup()
    loader = dm.train_dataloader(curriculum_stage=2)
    all_n = []
    for batch in loader:
        all_n.extend(batch["n_lines"])
        if len(all_n) >= 64:
            break
    replay_present = any(n == 1 for n in all_n)
    assert replay_present, (
        f"No Stage 1 replay samples found in first {len(all_n)} items from stage 2 dataloader"
    )


def test_val_dataloader_no_replay():
    """Val dataloader items all have n_lines >= stage_config.min_lines."""
    from data.datamodule import CurriculumDataModule
    cfg = _make_pipeline_config()
    dm = CurriculumDataModule(cfg)
    dm.setup()
    # Stage 2 val — no replay, so n_lines should all be >= 2
    loader = dm.val_dataloader(curriculum_stage=2)
    stage2_min = cfg.stages[1].min_lines  # 2
    all_n = []
    for batch in loader:
        all_n.extend(batch["n_lines"])
    assert all(n >= stage2_min for n in all_n), (
        f"Val dataloader has items with n_lines < {stage2_min}: {[n for n in all_n if n < stage2_min]}"
    )
