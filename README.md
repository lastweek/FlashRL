# FlashRL

A learning-first reinforcement learning project for LLM post-training.

## Development

### Setup

```bash
# Source the dev environment (consolidates bytecode, sets PYTHONPATH)
source dev.sh

# Run tests
python3 -m pytest tests/ -v

# Test imports
python3 -c "from flashrl.framework import FlashRL, RunConfig"

# Browser-based viewer tests (one-time)
python3 -m playwright install chromium
```

### Project Structure

```
flashrl/
├── framework/       # Core RL training APIs
│   ├── agent/       # Agent building blocks: runtime, tools, and context
│   ├── trainer/     # Training algorithms
│   ├── rollout/     # Generation and rollout
│   ├── reward/      # Reward computation
│   ├── tools/       # Compatibility shim for relocated agent tools
│   └── models/      # Model wrappers
└── platform/        # Orchestration and scaling (coming later)
```

### Clean Bytecode

All Python bytecode is consolidated to `.cache/pycache/` instead of cluttering `__pycache__/` folders everywhere.

To clean all bytecode:
```bash
rm -rf .cache/
```

## Quick Start

### Run the Math Example

The math example now lives in its own folder with a thin training script
and a YAML config. It supports both the historical blackbox rollout path and
an explicit whitebox path built from `flashrl.framework.agent` primitives:

```bash
python3 flashrl/framework/examples/math/train.py
```

```bash
python3 flashrl/framework/examples/math/train.py --rollout-mode whitebox
```

See [flashrl/framework/examples/README.md](flashrl/framework/examples/README.md) for details.
See [flashrl/framework/agent/README.md](flashrl/framework/agent/README.md) for the agent toolbox overview.

### Run the Agent Tools Demo

The whitebox agent examples are small offline scripts that show the public
agent toolbox in increasing complexity:

```bash
python3 flashrl/framework/examples/agent-tools/run.py
```

```bash
python3 flashrl/framework/examples/agent-react/run.py
```

```bash
python3 flashrl/framework/examples/agent-dynamic-tools/run.py
```

### Run the Code Example

The first code example is a script-based Codeforces prototype with local
execution reward:

```bash
python3 flashrl/framework/examples/code-single-turn/train.py
```

### Inspect Run Artifacts

FlashRL writes machine-readable per-run artifacts under `logs/` by default:
- `events.jsonl`
- `console.log`
- `rollouts.jsonl`

`rollouts.jsonl` is a grouped prompt-major artifact. It stores shared input
messages once per prompt group, candidate completion messages separately, and
promotes common quality/performance stats into first-class fields for easier
inspection.

TensorBoard is also enabled by default for framework runs:

```bash
tensorboard --logdir logs
```

To inspect them, open the unified static viewer in Chrome or Edge:

```bash
open docs/viewer.html
```

Then switch between `Run History` and `Live Runtime` as needed. For run artifacts,
click `Open run folder` and choose the `logs/` folder.

Older `.flashrl-runs/` directories remain viewable if you already have them.

## Current Status

✅ **Step 1 Complete** - Core abstractions (data models, config, base classes)
✅ **Step 2 Complete** - GRPO trainer skeleton
✅ **Step 3 Complete** - End-to-end training examples

**What works:**
- Data models and configuration system
- Script-run example workflows and YAML-driven framework runs
- Model wrappers (Actor, Reference, Critic)
- GRPO trainer structure
- Training pipeline examples
- Whitebox agent building blocks with explicit runtimes and subprocess-backed tools

**Next steps:**
- Real rollout generation with models
- Real GRPO loss computation
- Additional agent examples and higher-level patterns built on the traced runtime
- Additional safe runtimes beyond subprocess isolation
