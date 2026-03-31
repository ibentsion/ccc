"""Mock-based tests for curriculum training loop (training/curriculum.py).

All ML libraries (pytorch, lightning, peft, bitsandbytes, clearml) are mocked
so these tests run without GPU or any ML packages installed.

Because curriculum.py imports everything inside main(), we inject mocks via
sys.modules and then monkey-patch source-module attributes so that the lazy
'from X import Y' statements inside main() resolve to our fakes.
"""
import sys
import types
import importlib
import unittest
from unittest.mock import MagicMock, call, patch


# ---------------------------------------------------------------------------
# Build and register mock modules BEFORE any project import
# ---------------------------------------------------------------------------

def _make_mod(name):
    return types.ModuleType(name)


def _install_mocks():
    """Install lightweight mock modules for all heavy ML dependencies."""

    # torch
    torch = _make_mod("torch")
    torch.float16 = "float16"
    tutils = _make_mod("torch.utils")
    tud = _make_mod("torch.utils.data")
    tud.DataLoader = MagicMock(name="DataLoader")
    tud.Dataset = MagicMock(name="Dataset")
    torch.utils = tutils
    tutils.data = tud
    sys.modules.setdefault("torch", torch)
    sys.modules.setdefault("torch.utils", tutils)
    sys.modules.setdefault("torch.utils.data", tud)

    # pytorch_lightning — LightningModule/DataModule need to be real classes
    # so that class definitions in data/datamodule.py and training/model.py
    # can inherit from them at module import time.
    class _LightningModule:
        pass

    class _LightningDataModule:
        pass

    pl = _make_mod("pytorch_lightning")
    pl.LightningModule = _LightningModule
    pl.LightningDataModule = _LightningDataModule
    pl.Trainer = MagicMock(name="Trainer")
    pl_loggers = _make_mod("pytorch_lightning.loggers")
    pl_loggers.TensorBoardLogger = MagicMock(name="TensorBoardLogger")
    pl.loggers = pl_loggers
    sys.modules.setdefault("pytorch_lightning", pl)
    sys.modules.setdefault("pytorch_lightning.loggers", pl_loggers)

    # transformers
    tr = _make_mod("transformers")
    tr.AutoModelForCausalLM = MagicMock(name="AutoModelForCausalLM")
    tr.AutoTokenizer = MagicMock(name="AutoTokenizer")
    tr.BitsAndBytesConfig = MagicMock(name="BitsAndBytesConfig")
    sys.modules.setdefault("transformers", tr)

    # peft
    peft = _make_mod("peft")
    peft.get_peft_model = MagicMock(name="get_peft_model")
    peft.LoraConfig = MagicMock(name="LoraConfig")
    peft.prepare_model_for_kbit_training = MagicMock(name="prepare_model_for_kbit_training")
    peft.PeftModel = MagicMock(name="PeftModel")
    sys.modules.setdefault("peft", peft)

    # bitsandbytes
    bnb = _make_mod("bitsandbytes")
    bnb_optim = _make_mod("bitsandbytes.optim")
    bnb_optim.PagedAdamW32bit = MagicMock(name="PagedAdamW32bit")
    bnb.optim = bnb_optim
    sys.modules.setdefault("bitsandbytes", bnb)
    sys.modules.setdefault("bitsandbytes.optim", bnb_optim)

    # clearml
    clearml = _make_mod("clearml")
    clearml.Task = MagicMock(name="Task")
    sys.modules.setdefault("clearml", clearml)

    return {
        "torch": torch, "pytorch_lightning": pl, "pytorch_lightning.loggers": pl_loggers,
        "transformers": tr, "peft": peft, "bitsandbytes": bnb, "clearml": clearml,
    }


_MOCKS = _install_mocks()

# Now safe to import project modules (they pick up mock bases above)
from data.config import PipelineConfig, StageConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_model_mock():
    m = MagicMock(name="model_instance")
    m.model = MagicMock()
    m.model.base_model = MagicMock()
    m.model.base_model.model = MagicMock()
    return m


def _reload_curriculum():
    mod = sys.modules.get("training.curriculum")
    if mod is not None:
        importlib.reload(mod)
        return mod
    import training.curriculum
    return training.curriculum


class TestCurriculumLoop(unittest.TestCase):
    """Tests for training/curriculum.py main()."""

    def _run(self):
        """
        Configure mocks, reload curriculum, call main(), return result dict.

        curriculum.py binds 'Task' at module level via 'from clearml import Task',
        and all ML deps lazily via imports inside main(). Strategy:
        - Reload curriculum so its module-level 'Task' picks up fresh clearml.Task
        - Then patch curriculum.Task directly to our controlled mock
        - Monkey-patch source modules for deps imported inside main()
        """
        # --- ClearML Task (curriculum binds this at module level) ---
        task_inst = MagicMock(name="task_instance")
        mock_logger = MagicMock(name="logger")
        task_inst.get_logger.return_value = mock_logger
        Task_cls = MagicMock(name="Task_class")
        Task_cls.init.return_value = task_inst
        # Set on clearml module BEFORE reload so reload picks it up
        _MOCKS["clearml"].Task = Task_cls

        curriculum = _reload_curriculum()
        # After reload, curriculum.Task is bound to clearml.Task at that moment.
        # Patch it directly on the module to ensure our mock is used.
        curriculum.Task = Task_cls

        # --- Lightning Trainer (imported inside main()) ---
        trainer_inst = MagicMock(name="trainer_instance")
        Trainer_cls = MagicMock(name="Trainer_class")
        Trainer_cls.return_value = trainer_inst
        _MOCKS["pytorch_lightning"].Trainer = Trainer_cls

        # --- PeftModel (imported inside main()) ---
        PeftModel_mock = MagicMock(name="PeftModel_mock")
        _MOCKS["peft"].PeftModel = PeftModel_mock

        # --- QLoRALightningModule (imported inside main() from training.model) ---
        model_inst = _fresh_model_mock()
        QLoRA_cls = MagicMock(name="QLoRA_class")
        QLoRA_cls.return_value = model_inst
        import training.model as tm
        orig_qlora = tm.QLoRALightningModule
        tm.QLoRALightningModule = QLoRA_cls

        # --- CurriculumDataModule (imported inside main() from data.datamodule) ---
        dm_inst = MagicMock(name="dm_instance")
        mock_dataset = [{"middle": "line1"}, {"middle": "line2"}]
        mock_val_dl = MagicMock()
        mock_val_dl.dataset = mock_dataset
        dm_inst.val_dataloader.return_value = mock_val_dl

        mock_test_dl = MagicMock()
        mock_test_dl.dataset = [{"middle": "test_line"}]
        dm_inst.test_dataloader.return_value = mock_test_dl

        DM_cls = MagicMock(name="DM_class")
        DM_cls.return_value = dm_inst
        import data.datamodule as dd
        orig_dm = dd.CurriculumDataModule
        dd.CurriculumDataModule = DM_cls

        # --- Tokenizer helpers (imported inside main() from training.tokenizer) ---
        import training.tokenizer as tt
        orig_load = tt.load_tokenizer
        orig_collator = tt.TokenizedCollator
        tt.load_tokenizer = MagicMock(name="load_tokenizer", return_value=MagicMock())
        tt.TokenizedCollator = MagicMock(name="TokenizedCollator")

        # run_eval is imported inside main() via 'from training.eval_metrics import run_eval',
        # which binds it in training.curriculum namespace at call time. Patch the source
        # module attribute so the 'from ... import' inside main() picks up the mock.
        import training.eval_metrics as tem
        orig_run_eval = getattr(tem, "run_eval", None)
        run_eval_mock = MagicMock(name="run_eval", return_value={"exact_match": 0.85, "edit_sim": 0.90})
        tem.run_eval = run_eval_mock
        try:
            curriculum.main()
        finally:
            tm.QLoRALightningModule = orig_qlora
            dd.CurriculumDataModule = orig_dm
            tt.load_tokenizer = orig_load
            tt.TokenizedCollator = orig_collator
            if orig_run_eval is not None:
                tem.run_eval = orig_run_eval

        return dict(
            task_cls=Task_cls,
            task_inst=task_inst,
            trainer_cls=Trainer_cls,
            trainer_inst=trainer_inst,
            model_cls=QLoRA_cls,
            model_inst=model_inst,
            dm_cls=DM_cls,
            dm_inst=dm_inst,
            peft=PeftModel_mock,
            run_eval_mock=run_eval_mock,
            mock_logger=mock_logger,
        )

    # ------------------------------------------------------------------
    # Test 1: Task.init called once per stage + once for eval-final = 4
    # ------------------------------------------------------------------

    def test_task_init_called_once_per_stage(self):
        r = self._run()
        # 3 stage tasks + 1 eval-final task
        self.assertEqual(r["task_cls"].init.call_count, 4)

    # ------------------------------------------------------------------
    # Test 2: Task naming convention
    # ------------------------------------------------------------------

    def test_task_naming_convention(self):
        r = self._run()
        calls = r["task_cls"].init.call_args_list
        # Python 3.6: call[0] = positional args tuple, call[1] = kwargs dict
        names = [
            c[1].get("task_name") or (c[0][1] if len(c[0]) > 1 else None)
            for c in calls
        ]
        self.assertIn("stage_1_gap_1_1", names)
        self.assertIn("stage_2_gap_2_2", names)
        self.assertIn("stage_3_gap_3_3", names)

    # ------------------------------------------------------------------
    # Test 3: task.close() called once per stage + once for eval-final = 4
    # ------------------------------------------------------------------

    def test_task_close_called_after_each_stage(self):
        r = self._run()
        # 3 stage closes + 1 eval-final close
        self.assertEqual(r["task_inst"].close.call_count, 4)

    # ------------------------------------------------------------------
    # Test 4: PeftModel.from_pretrained for stages > 1
    # ------------------------------------------------------------------

    def test_peft_from_pretrained_for_stage_gt_1(self):
        r = self._run()
        self.assertEqual(r["peft"].from_pretrained.call_count, 2)

    # ------------------------------------------------------------------
    # Test 5: Fresh Trainer created per stage
    # ------------------------------------------------------------------

    def test_fresh_trainer_per_stage(self):
        r = self._run()
        self.assertEqual(r["trainer_cls"].call_count, 3)

    # ------------------------------------------------------------------
    # Test 6: save_adapter and ClearML artifact upload per stage
    # ------------------------------------------------------------------

    def test_save_adapter_and_upload_per_stage(self):
        r = self._run()
        model_inst = r["model_inst"]
        task_inst = r["task_inst"]

        self.assertEqual(model_inst.save_adapter.call_count, 3)
        # Python 3.6: call[0] = positional args tuple
        save_paths = [c[0][0] for c in model_inst.save_adapter.call_args_list]
        self.assertTrue(any("stage_1" in p for p in save_paths))
        self.assertTrue(any("stage_2" in p for p in save_paths))
        self.assertTrue(any("stage_3" in p for p in save_paths))

        self.assertEqual(task_inst.upload_artifact.call_count, 3)
        artifact_names = [
            c[1].get("name") or c[0][0]
            for c in task_inst.upload_artifact.call_args_list
        ]
        self.assertTrue(any("stage_1" in n for n in artifact_names))
        self.assertTrue(any("stage_2" in n for n in artifact_names))
        self.assertTrue(any("stage_3" in n for n in artifact_names))

    # ------------------------------------------------------------------
    # Test 7: train_dataloader receives correct curriculum_stage (1-indexed)
    # ------------------------------------------------------------------

    def test_train_dataloader_receives_correct_curriculum_stage(self):
        r = self._run()
        dm_inst = r["dm_inst"]

        train_calls = dm_inst.train_dataloader.call_args_list
        # Python 3.6: call[0]=positional, call[1]=kwargs
        stages = [
            c[1].get("curriculum_stage") or (c[0][0] if c[0] else None)
            for c in train_calls
        ]
        self.assertEqual(sorted(stages), [1, 2, 3])

        # val_dataloader is called twice per stage: once during training, once for eval
        val_calls = dm_inst.val_dataloader.call_args_list
        val_stages = [
            c[1].get("curriculum_stage") or (c[0][0] if c[0] else None)
            for c in val_calls
        ]
        # Each stage appears twice (training + eval)
        self.assertEqual(sorted(val_stages), [1, 1, 2, 2, 3, 3])

    # ------------------------------------------------------------------
    # Test 8: run_eval called once per stage + once for final eval = 4
    # ------------------------------------------------------------------

    def test_per_stage_eval_called(self):
        r = self._run()
        # 3 per-stage evals + 1 final test eval
        self.assertEqual(r["run_eval_mock"].call_count, 4)

    # ------------------------------------------------------------------
    # Test 9: eval metrics logged to ClearML per stage
    # ------------------------------------------------------------------

    def test_eval_metrics_logged_per_stage(self):
        r = self._run()
        mock_logger = r["mock_logger"]
        scalar_calls = mock_logger.report_scalar.call_args_list
        series_values = [c[1].get("series") or (c[0][1] if len(c[0]) > 1 else None)
                         for c in scalar_calls]
        em_count = sum(1 for s in series_values if s == "eval_exact_match")
        es_count = sum(1 for s in series_values if s == "eval_edit_sim")
        # At least 3 per-stage evals x 2 metrics = 6 (plus final = 8 total)
        self.assertGreaterEqual(em_count, 3)
        self.assertGreaterEqual(es_count, 3)

    # ------------------------------------------------------------------
    # Test 10: eval-final Task created as 4th Task.init call
    # ------------------------------------------------------------------

    def test_eval_final_task_created(self):
        r = self._run()
        self.assertEqual(r["task_cls"].init.call_count, 4)
        calls = r["task_cls"].init.call_args_list
        last_call_name = calls[3][1].get("task_name") or (calls[3][0][1] if len(calls[3][0]) > 1 else None)
        self.assertEqual(last_call_name, "eval-final")

    # ------------------------------------------------------------------
    # Test 11: eval-final task is closed
    # ------------------------------------------------------------------

    def test_eval_final_task_closed(self):
        r = self._run()
        self.assertEqual(r["task_inst"].close.call_count, 4)

    # ------------------------------------------------------------------
    # Test 12: max_new_tokens scales with max_lines (50 per line)
    # ------------------------------------------------------------------

    def test_max_new_tokens_scales_with_max_lines(self):
        r = self._run()
        run_eval_mock = r["run_eval_mock"]
        # Per-stage calls: stage 1 max_lines=1 -> 50, stage 2 -> 100, stage 3 -> 150
        # Final eval: uses last stage max_lines=3 -> 150
        call_args = run_eval_mock.call_args_list
        # First 3 calls are per-stage evals
        max_new_tokens_vals = [
            c[1].get("max_new_tokens") if c[1].get("max_new_tokens") is not None
            else c[0][4] if len(c[0]) > 4 else None
            for c in call_args[:3]
        ]
        self.assertIn(50, max_new_tokens_vals)   # stage 1
        self.assertIn(100, max_new_tokens_vals)  # stage 2
        self.assertIn(150, max_new_tokens_vals)  # stage 3

    # ------------------------------------------------------------------
    # Test 13: test_dataloader called once for final eval
    # ------------------------------------------------------------------

    def test_test_dataloader_called_for_final_eval(self):
        r = self._run()
        self.assertEqual(r["dm_inst"].test_dataloader.call_count, 1)


if __name__ == "__main__":
    unittest.main()
