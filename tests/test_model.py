"""Tests for QLoRALightningModule using full mock injection of heavy ML deps."""
import sys
import types
import unittest
from unittest import mock
import importlib


def _build_fake_transformers():
    m = types.ModuleType("transformers")
    m.__spec__ = importlib.util.spec_from_loader("transformers", loader=None)

    class _BnbConfig:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    m.BitsAndBytesConfig = mock.MagicMock(side_effect=_BnbConfig)
    m.AutoModelForCausalLM = mock.MagicMock()
    return m


def _build_fake_peft():
    m = types.ModuleType("peft")
    m.__spec__ = importlib.util.spec_from_loader("peft", loader=None)
    m.LoraConfig = mock.MagicMock()
    m.prepare_model_for_kbit_training = mock.MagicMock(side_effect=lambda x: x)
    m.get_peft_model = mock.MagicMock()
    return m


def _build_fake_bnb():
    import torch

    m = types.ModuleType("bitsandbytes")
    m.__spec__ = importlib.util.spec_from_loader("bitsandbytes", loader=None)
    optim = types.ModuleType("bitsandbytes.optim")
    optim.__spec__ = importlib.util.spec_from_loader("bitsandbytes.optim", loader=None)

    class _PagedAdamW32bit(torch.optim.Optimizer):
        def __init__(self, params, lr):
            self.lr = lr
            params_list = list(params)
            super().__init__(params_list, {"lr": lr})

        def step(self, closure=None):
            pass

    optim.PagedAdamW32bit = _PagedAdamW32bit
    m.optim = optim
    sys.modules["bitsandbytes.optim"] = optim
    return m


def _build_fake_pl():
    m = types.ModuleType("pytorch_lightning")
    m.__spec__ = importlib.util.spec_from_loader("pytorch_lightning", loader=None)

    class _LightningModule:
        def __init__(self):
            self._logged = {}

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

        def parameters(self):
            import torch
            return iter([torch.nn.Parameter(torch.zeros(1))])

        def log(self, name, value, **kwargs):
            self._logged[name] = value.item() if hasattr(value, "item") else value

    m.LightningModule = _LightningModule
    return m


# Inject all mocks before any training.model import
_tf = _build_fake_transformers()
_peft = _build_fake_peft()
_bnb = _build_fake_bnb()
_pl = _build_fake_pl()

for name, mod in [
    ("transformers", _tf),
    ("peft", _peft),
    ("bitsandbytes", _bnb),
    ("pytorch_lightning", _pl),
]:
    sys.modules.pop(name, None)
    sys.modules[name] = mod

sys.modules.pop("training.model", None)

from training.config import TrainingConfig
from training.model import QLoRALightningModule


def _default_config():
    return TrainingConfig(
        model_name="test/model",
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )


class _FakeBase:
    def parameters(self):
        import torch
        p = torch.nn.Parameter(torch.zeros(1))
        return iter([p])

    def gradient_checkpointing_enable(self, **kw):
        pass


class _FakePeft:
    def __init__(self, base=None):
        self._base = base
        self._saved_path = None

    def parameters(self):
        import torch
        p = torch.nn.Parameter(torch.zeros(1))
        return iter([p])

    def __call__(self, input_ids=None, attention_mask=None, labels=None):
        import torch
        out = mock.MagicMock()
        out.loss = torch.tensor(1.5)
        return out

    def save_pretrained(self, path):
        self._saved_path = path


def _reset_mocks():
    _tf.AutoModelForCausalLM.from_pretrained.reset_mock()
    _tf.AutoModelForCausalLM.from_pretrained.side_effect = lambda *a, **kw: _FakeBase()
    _tf.BitsAndBytesConfig.reset_mock()
    _peft.prepare_model_for_kbit_training.reset_mock()
    _peft.prepare_model_for_kbit_training.side_effect = lambda x: x
    _peft.get_peft_model.reset_mock()
    _peft.get_peft_model.side_effect = lambda m, c: _FakePeft(m)
    _peft.LoraConfig.reset_mock()


def _make_module(config=None):
    _reset_mocks()
    return QLoRALightningModule(config or _default_config())


class TestInitOrder(unittest.TestCase):
    def setUp(self):
        _reset_mocks()

    def test_init_order(self):
        """from_pretrained -> prepare_model_for_kbit_training -> get_peft_model exactly."""
        order = []
        _tf.AutoModelForCausalLM.from_pretrained.side_effect = lambda *a, **kw: (order.append("from_pretrained"), _FakeBase())[1]
        _peft.prepare_model_for_kbit_training.side_effect = lambda x: (order.append("prepare_model_for_kbit_training"), x)[1]
        _peft.get_peft_model.side_effect = lambda m, c: (order.append("get_peft_model"), _FakePeft(m))[1]

        QLoRALightningModule(_default_config())

        self.assertEqual(order, ["from_pretrained", "prepare_model_for_kbit_training", "get_peft_model"])

    def test_device_map_not_auto(self):
        """device_map must be {"": 0}, not "auto"."""
        _make_module()
        kwargs = _tf.AutoModelForCausalLM.from_pretrained.call_args[1]
        self.assertEqual(kwargs.get("device_map"), {"": 0})

    def test_bnb_config_load_in_4bit(self):
        """BitsAndBytesConfig must have load_in_4bit=True."""
        _make_module()
        kwargs = _tf.BitsAndBytesConfig.call_args[1]
        self.assertTrue(kwargs.get("load_in_4bit"))

    def test_lora_all_7_target_modules(self):
        """LoraConfig gets all 7 target modules."""
        _make_module()
        kwargs = _peft.LoraConfig.call_args[1]
        self.assertEqual(len(kwargs.get("target_modules", [])), 7)


class TestTrainingStep(unittest.TestCase):
    def test_training_step_returns_loss_and_logs(self):
        import torch
        module = _make_module()
        batch = {
            "input_ids": torch.zeros(2, 8, dtype=torch.long),
            "attention_mask": torch.ones(2, 8, dtype=torch.long),
            "labels": torch.zeros(2, 8, dtype=torch.long),
        }
        loss = module.training_step(batch, 0)
        self.assertIn("train_loss", module._logged)
        self.assertAlmostEqual(float(module._logged["train_loss"]), 1.5, places=3)


class TestSaveAdapter(unittest.TestCase):
    def test_save_adapter(self):
        module = _make_module()
        module.save_adapter("/tmp/test_adapter_out")
        self.assertEqual(module.model._saved_path, "/tmp/test_adapter_out")


class TestConfigureOptimizers(unittest.TestCase):
    def test_configure_optimizers_returns_paged_adamw_and_constant_lr(self):
        import torch
        module = _make_module()
        result = module.configure_optimizers()

        self.assertIn("optimizer", result)
        self.assertIn("lr_scheduler", result)

        optimizer = result["optimizer"]
        self.assertIsInstance(optimizer, _bnb.optim.PagedAdamW32bit)
        self.assertAlmostEqual(optimizer.lr, 2e-4, places=6)

        scheduler = result["lr_scheduler"]["scheduler"]
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.ConstantLR)
        self.assertEqual(result["lr_scheduler"]["interval"], "step")
