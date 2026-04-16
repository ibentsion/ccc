---
phase: 260416-czk
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - pyproject.toml
  - .gitignore
  - .python-version
autonomous: false
requirements:
  - UV-01  # pyproject.toml with uv-compatible dependency declarations
  - UV-02  # `train` entrypoint -> training.curriculum:main
  - UV-03  # `train-single` entrypoint -> training.train:main
  - UV-04  # Torch/bitsandbytes moved to optional extras (CUDA-specific)
  - UV-05  # Confirmed `uv sync` + `uv run train` works locally

must_haves:
  truths:
    - "Running `uv sync` creates a .venv and installs the project with core deps"
    - "Running `uv run train --help` (or similar import check) resolves to training.curriculum:main without ImportError"
    - "Running `uv run train-single` resolves to training.train:main"
    - "Torch and bitsandbytes are declared as optional extras (e.g. `uv sync --extra cuda`) so CPU-only environments don't break install"
    - "`uv run pytest` executes the existing test suite using the same environment"
  artifacts:
    - path: "pyproject.toml"
      provides: "Build metadata, dependencies, entrypoints"
      contains: "[project.scripts]"
    - path: ".python-version"
      provides: "Python version pin for uv"
    - path: ".gitignore"
      provides: "Ignore uv's .venv and uv.lock-adjacent artifacts we don't want to commit"
  key_links:
    - from: "pyproject.toml [project.scripts].train"
      to: "training.curriculum:main"
      via: "console_scripts entrypoint"
      pattern: "train\\s*=\\s*\"training\\.curriculum:main\""
    - from: "pyproject.toml [project.scripts].train-single"
      to: "training.train:main"
      via: "console_scripts entrypoint"
      pattern: "train-single\\s*=\\s*\"training\\.train:main\""
    - from: "pyproject.toml [project].dependencies"
      to: "runtime imports (clearml, transformers, peft, pytorch-lightning)"
      via: "uv sync resolution"
      pattern: "clearml|transformers|peft|pytorch-lightning"
---

<objective>
Migrate the project from plain venv to a uv-managed workflow with console entrypoints for training.

Purpose: Make `uv sync` + `uv run train` the single-command path to launch curriculum training, while keeping CUDA-heavy deps (torch, bitsandbytes) as opt-in extras so the project installs cleanly on machines without matching CUDA toolchains.

Output:
- pyproject.toml declaring the `training` + `data` packages, core runtime deps, dev deps, optional `cuda` extra, and console scripts
- .python-version pinning the interpreter
- .gitignore updated to exclude .venv and uv build artifacts
- Verified `uv sync` + `uv run train-single --help` (or import smoke) + `uv run pytest` all work
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@CLAUDE.md
@training/config.py
@training/curriculum.py
@training/train.py

<interfaces>
<!-- Key facts about the existing codebase that shape the pyproject. -->
<!-- The executor should NOT re-discover these. -->

Project layout (flat packages at repo root):
- training/  (with training/__init__.py, curriculum.py, train.py, model.py, eval_metrics.py, tokenizer.py, config.py)
- data/      (with data/__init__.py, datamodule.py, curriculum_dataset.py, dataset.py, fim.py, config.py)
- tests/     (with conftest.py, test_*.py)

Entrypoint functions already exist and take no args:
- `training.curriculum.main()` — multi-stage curriculum loop
- `training.train.main()` — single-stage training

Confirmed runtime deps (versions observed in current venv — pin as `>=` minors, not exact):
- torch==2.11.0
- peft==0.18.1
- bitsandbytes==0.49.2
- pytorch-lightning==2.6.1
- clearml==2.1.5
- transformers==5.4.0
- accelerate==1.13.0

Confirmed dev deps:
- pytest==9.0.2

Python: assume >=3.10 (dataclass `field(default_factory=...)` + `from __future__` not used, stdlib types are fine).
</interfaces>
</context>

<tasks>

<task type="auto" tdd="false">
  <name>Task 1: Author pyproject.toml with uv layout, deps, and entrypoints</name>
  <files>pyproject.toml, .python-version, .gitignore</files>
  <behavior>
    - `uv sync` (no extras) must succeed on a CPU-only machine: resolves core deps, skips CUDA packages.
    - `uv sync --extra cuda` must pull torch + bitsandbytes.
    - `uv sync --extra dev` must pull pytest.
    - `uv run train` must route to `training.curriculum.main`.
    - `uv run train-single` must route to `training.train.main`.
    - `.gitignore` must exclude `.venv/` and `uv.lock` is kept tracked (commit it) but `.venv/` is not.
  </behavior>
  <action>
Create `/home/ido/git/ccc/pyproject.toml` with the following structure (use Write tool; do not use heredoc):

```toml
[project]
name = "ccc"
version = "0.1.0"
description = "CodeComplete: curriculum-trained Python line-fill model with ClearML experiment tracking"
requires-python = ">=3.10"
readme = "CLAUDE.md"
dependencies = [
    "clearml>=2.1",
    "transformers>=4.44",
    "peft>=0.11",
    "pytorch-lightning>=2.2",
    "accelerate>=0.33",
    "numpy>=1.26",
]

[project.optional-dependencies]
cuda = [
    "torch>=2.3",
    "bitsandbytes>=0.43",
]
dev = [
    "pytest>=8.0",
]

[project.scripts]
train = "training.curriculum:main"
train-single = "training.train:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["training", "data"]

[tool.uv]
# Keep defaults; no index overrides needed for non-CUDA core.

[tool.pytest.ini_options]
testpaths = ["tests"]
```

Notes for the executor:
- Per UV-04, keep torch and bitsandbytes in `[project.optional-dependencies].cuda`. Do NOT put them in `dependencies`.
- Per UV-02 / UV-03, entrypoints MUST be exactly `training.curriculum:main` and `training.train:main` (module:function syntax).
- `transformers>=4.44` is the lower bound; the observed 5.4.0 will still resolve.
- Hatchling is used as the build backend because the repo has flat packages (no src/ layout). `[tool.hatch.build.targets.wheel].packages` explicitly enumerates them so hatch doesn't try to auto-discover and fail on stray dirs (`venv`, `.planning`, `tests`).
- Do NOT add `tests` to the wheel packages list.

Also:
1. Create `/home/ido/git/ccc/.python-version` containing a single line: `3.11` (project currently runs on a venv; 3.11 is a safe default matching the >=3.10 constraint and has wheels for all deps).
2. Update `/home/ido/git/ccc/.gitignore` by appending these lines (idempotently — check before appending):
```
.venv/
dist/
build/
*.egg-info/
```
Keep existing lines (`venv` without slash is already there for the old dir; leave it).

Do NOT modify any files under `training/` or `data/`.
  </action>
  <verify>
    <automated>test -f /home/ido/git/ccc/pyproject.toml && grep -q 'train = "training.curriculum:main"' /home/ido/git/ccc/pyproject.toml && grep -q 'train-single = "training.train:main"' /home/ido/git/ccc/pyproject.toml && grep -q '\[project.optional-dependencies\]' /home/ido/git/ccc/pyproject.toml && grep -q 'cuda = \[' /home/ido/git/ccc/pyproject.toml && test -f /home/ido/git/ccc/.python-version && grep -q '\.venv/' /home/ido/git/ccc/.gitignore</automated>
  </verify>
  <done>
    - pyproject.toml exists with `[project]`, `[project.optional-dependencies].cuda`, `[project.optional-dependencies].dev`, `[project.scripts].train`, `[project.scripts].train-single`, and `[build-system]`.
    - .python-version exists and pins a >=3.10 interpreter.
    - .gitignore excludes `.venv/`, `dist/`, `build/`, `*.egg-info/`.
    - No files under training/ or data/ were modified.
  </done>
</task>

<task type="auto" tdd="false">
  <name>Task 2: Install uv (if missing) and run `uv sync` to validate resolution</name>
  <files>uv.lock (generated)</files>
  <action>
1. Check if uv is installed: `command -v uv`. If not present, install it via the official standalone installer:
   ```
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   Then source the shell env (`source $HOME/.local/bin/env` or re-export PATH including `$HOME/.local/bin`) so `uv` is on PATH for this shell session.

2. From `/home/ido/git/ccc`, run:
   ```
   uv sync --extra dev
   ```
   This resolves the CORE deps (no CUDA) + dev extras. It must succeed on a CPU-only resolution. It will create `.venv/` and a `uv.lock` file.

3. Capture any resolution conflicts. If a dep lower bound is unsatisfiable on this machine, relax it in pyproject.toml (e.g. bump `transformers>=4.44` to `>=4.40`) and re-run. Do NOT add torch/bitsandbytes to core deps to "fix" a resolution error.

4. Do NOT run `uv sync --extra cuda` yet — that requires matching CUDA wheels and may be slow; defer to the human checkpoint if the user wants it.

5. Leave `uv.lock` in the working tree (it should be committed for reproducibility — the user can decide whether to commit).
  </action>
  <verify>
    <automated>cd /home/ido/git/ccc && command -v uv >/dev/null && test -f uv.lock && test -d .venv && uv run python -c "import training.curriculum; import training.train; assert callable(training.curriculum.main); assert callable(training.train.main); print('entrypoint imports OK')"</automated>
  </verify>
  <done>
    - `uv` binary is available on PATH.
    - `uv sync --extra dev` completed successfully.
    - `uv.lock` and `.venv/` exist at repo root.
    - `uv run python -c "import training.curriculum; import training.train"` succeeds (entrypoints are importable; does not execute training).
  </done>
</task>

<task type="checkpoint:human-verify" gate="blocking">
  <name>Task 3: Human verifies uv-driven training launch</name>
  <what-built>
    - pyproject.toml with uv layout, entrypoints (`train`, `train-single`), optional `cuda` + `dev` extras.
    - .python-version pinning Python 3.11.
    - .gitignore updated.
    - `uv sync --extra dev` run successfully; .venv + uv.lock generated.
    - Smoke-tested that `training.curriculum:main` and `training.train:main` are importable under `uv run`.
  </what-built>
  <how-to-verify>
    Run each of these from `/home/ido/git/ccc` and confirm expected behavior:

    1. `uv run pytest -x -q`
       Expected: existing test suite runs under uv's .venv. (It may fail for reasons unrelated to this plan — e.g. CUDA-gated tests — but tests that previously passed under the old venv should still pass.)

    2. `uv run python -c "from training.curriculum import main; print(main)"`
       Expected: prints `<function main at 0x...>` — confirms UV-02.

    3. `uv run python -c "from training.train import main; print(main)"`
       Expected: prints `<function main at 0x...>` — confirms UV-03.

    4. OPTIONAL (only if you have a CUDA machine ready):
       `uv sync --extra cuda` → should resolve torch + bitsandbytes.
       `uv run train-single` → should actually start training (will download the model; abort with Ctrl-C once you see stage-1 init logs).

    5. Confirm `cat .gitignore | grep .venv` shows `.venv/` is ignored.

    If any of 1-3 fail, describe the error. If 4 is deferred, say "cuda extra deferred".
  </how-to-verify>
  <resume-signal>Type "approved" or describe issues (e.g. "test_model.py failed with X", "entrypoint import raised Y").</resume-signal>
</task>

</tasks>

<verification>
- pyproject.toml is valid TOML (uv sync would fail otherwise — Task 2 covers this).
- Entrypoint strings match `training.curriculum:main` and `training.train:main` exactly (grep in Task 1 verify).
- CUDA-heavy deps are NOT in the default install path (`uv sync` without `--extra cuda` succeeds on CPU-only envs).
- No code under `training/` or `data/` was modified — this is packaging-only work.
</verification>

<success_criteria>
- `uv sync --extra dev` completes without errors from a clean checkout.
- `uv run train` and `uv run train-single` resolve to the correct `main` functions.
- `uv run pytest` uses the uv-managed environment.
- Torch + bitsandbytes install only when `--extra cuda` is passed.
- Human verifies the full flow and types "approved".
</success_criteria>

<output>
After completion, create `.planning/quick/260416-czk-create-uv-installation-and-entrypoint-fo/260416-czk-SUMMARY.md` capturing:
- Final dep version bounds chosen
- Any resolution conflicts hit and how they were resolved
- Whether `uv.lock` was committed
- Whether the `cuda` extra was validated or deferred
</output>
