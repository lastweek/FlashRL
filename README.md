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
│   ├── trainer/     # Training algorithms
│   ├── rollout/     # Generation and rollout
│   ├── reward/      # Reward computation
│   ├── tools/       # Tool execution
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

### Run the Reasoning Example

The reasoning example now lives in its own folder with a thin training script
and a YAML config:

```bash
python3 -m examples.reasoning.train
```

Or run it directly from YAML:

```bash
python3 -m flashrl.framework.flashrl --config examples/reasoning/config.yaml
```

See [examples/README.md](examples/README.md) for details.

### Inspect Run Artifacts

FlashRL writes machine-readable per-run artifacts under `logs/` by default:
- `events.jsonl`
- `console.log`
- `rollouts.jsonl`

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
- YAML-driven example runs
- Model wrappers (Actor, Reference, Critic)
- GRPO trainer structure
- Training pipeline examples

**Next steps:**
- Real rollout generation with models
- Real GRPO loss computation
- Multi-turn conversations
- Tool execution
