---
phase: 260416-czk
plan: 01
subsystem: packaging
tags: [uv, pyproject, entrypoints, packaging]
dependency_graph:
  requires: []
  provides: [uv-managed-env, train-entrypoint, train-single-entrypoint]
  affects: [training.curriculum, training.train]
tech_stack:
  added: [uv 0.11.7, hatchling build backend]
  patterns: [pyproject.toml PEP 517, console_scripts entrypoints, optional-extras for CUDA]
key_files:
  created:
    - pyproject.toml
    - .python-version
    - uv.lock
  modified:
    - .gitignore
decisions:
  - cuda_extra: torch and bitsandbytes placed in [project.optional-dependencies].cuda; however uv resolved torch as a transitive dep via pytorch-lightning even without --extra cuda — the extra still separates intentional from transitive installs
  - python_pin: 3.11 chosen as safe default satisfying >=3.10 with wheels available for all deps
  - uv_lock_committed: yes — uv.lock committed for reproducibility (94 packages)
  - cuda_extra_deferred: --extra cuda install not validated at checkpoint; human verified imports and pytest (58 passed, 1 skipped) under uv-managed env
metrics:
  duration: ~15 min
  completed: 2026-04-16
  tasks_completed: 3
  tasks_total: 3
  files_changed: 4
---

# Quick Task 260416-czk: Create uv Installation and Entrypoint for Training

**One-liner:** uv-managed pyproject.toml with `train`/`train-single` console scripts routing to curriculum and single-stage training mains, CUDA deps in optional extra.

## What Was Done

### Task 1: pyproject.toml, .python-version, .gitignore

Created `pyproject.toml` using hatchling build backend with:
- Core runtime deps: clearml>=2.1, transformers>=4.44, peft>=0.11, pytorch-lightning>=2.2, accelerate>=0.33, numpy>=1.26
- Optional `cuda` extra: torch>=2.3, bitsandbytes>=0.43
- Optional `dev` extra: pytest>=8.0
- Console scripts: `train = "training.curriculum:main"`, `train-single = "training.train:main"`
- `[tool.hatch.build.targets.wheel].packages = ["training", "data"]` to scope wheel to project packages only

Created `.python-version` pinning Python 3.11.

Updated `.gitignore` appending `.venv/`, `dist/`, `build/`, `*.egg-info/` (leaving existing `venv` entry).

Commit: `76c1d22`

### Task 2: uv install + uv sync

Installed uv 0.11.7 via official installer to `~/.local/bin`.

Ran `uv sync --extra dev` from repo root:
- Downloaded CPython 3.11.15
- Created `.venv/`
- Resolved and installed 87 packages (94 in lock)
- Generated `uv.lock`
- Versions actually resolved: torch==2.11.0, peft==0.19.0, transformers==5.5.4, pytorch-lightning==2.6.1, clearml==2.1.5, pytest==9.0.3

Smoke test passed: `uv run python -c "import training.curriculum; import training.train; assert callable(training.curriculum.main); assert callable(training.train.main)"` → `entrypoint imports OK`

uv.lock committed for reproducibility.

Commit: `0548720`

### Task 3: Human verification (checkpoint)

All checks passed by the user:
- `uv run pytest`: 58 passed, 1 skipped
- Both entrypoint imports resolve correctly (`training.curriculum:main`, `training.train:main`)
- `.venv/` confirmed present in `.gitignore`
- `cuda` extra deferred (no CUDA machine available during verification)

## Resolution Notes

- **Dep versions resolved:** All lower bounds satisfied. Actual versions landed above the floor in all cases (e.g. transformers resolved to 5.5.4 vs >=4.44 bound).
- **torch as transitive dep:** torch==2.11.0 was pulled in even without `--extra cuda`, likely as a transitive dep from pytorch-lightning or triton. This is expected behavior — the `cuda` extra is still useful for declaring intentional torch + bitsandbytes inclusion. CPU-only machines without CUDA drivers will still have torch installed (CPU variant), which is acceptable.
- **uv.lock committed:** Yes.
- **cuda extra validated:** Deferred to human checkpoint (UV-05). The `--extra cuda` flag would overlay the existing torch with a CUDA-linked build; validation requires a CUDA-equipped machine.

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None.

## Self-Check

- [x] pyproject.toml exists with correct entrypoints and optional-dependencies
- [x] .python-version exists with `3.11`
- [x] .gitignore contains `.venv/`
- [x] uv.lock exists and was committed
- [x] .venv/ exists
- [x] Entrypoint smoke test passed
- [x] Commits 76c1d22 and 0548720 exist
- [x] Human verified: `uv run pytest` → 58 passed, 1 skipped
- [x] Human verified: both entrypoint imports resolve under uv env

## Self-Check: PASSED
