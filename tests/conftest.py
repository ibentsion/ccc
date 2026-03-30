"""
Test isolation setup for the combined test suite.

Two test files (test_curriculum.py and test_model.py) both inject ML mocks
into sys.modules at module-level (collection time). Their mocks are mutually
incompatible:

 - test_curriculum.py uses setdefault() for mock injection → only works if the
   slot is empty; stores its mock objects in _MOCKS dict.
 - test_model.py uses direct assignment (sys.modules[name] = mod) → always
   overwrites, including the mocks test_curriculum.py just installed.

This causes test_curriculum.py's _MOCKS to point to objects that are no longer
in sys.modules by the time its tests run.

Fix strategy:
1. Pre-import data.datamodule with real DataLoader before any mock torch is
   installed (conftest.py loads before any test module is collected).
2. Pop pytorch_lightning before collection so test_curriculum.py's setdefault
   can install its mock first.
3. After all collection is done (pytest_collection_finish), re-sync
   test_curriculum.py's _MOCKS back into sys.modules (excluding torch, which
   must remain the real package for test_model.py and test_tokenizer.py).
"""
import sys


# ── Step 1: pre-import data modules with real torch ─────────────────────────
# This captures the real DataLoader in data.datamodule's namespace before
# any test file can install a fake torch.utils.data.
import data.datamodule       # noqa: F401
import data.curriculum_dataset  # noqa: F401

# ── Step 2: pop pytorch_lightning so test_curriculum.py's setdefault can win ─
sys.modules.pop("pytorch_lightning", None)
sys.modules.pop("pytorch_lightning.loggers", None)


# ── Step 3: re-sync after collection ─────────────────────────────────────────
def pytest_collection_finish(session):
    """
    After all test modules are collected (and their module-level code has run),
    ensure that test_curriculum.py's _MOCKS dict matches sys.modules.

    test_model.py's module-level code force-assigns sys.modules[pl/peft/...],
    overwriting the objects stored in test_curriculum._MOCKS.  We fix that
    by writing _MOCKS values back — but we skip torch and its submodules so
    that the real torch package is preserved for test_model.py tests.
    """
    try:
        import tests.test_curriculum as tc
        keep_real = {"torch", "torch.utils", "torch.utils.data",
                     "bitsandbytes", "bitsandbytes.optim"}
        for name, mod in tc._MOCKS.items():
            if name not in keep_real:
                sys.modules[name] = mod
    except (ImportError, AttributeError):
        pass
