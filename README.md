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
python3 -c "from flashrl.framework import BaseTrainer, BaseRollout"
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

### Try the Examples

**Quick test (no download):**
```bash
python3 examples/train_grpo_simple.py
```

**Full training (downloads GPT-2):**
```bash
python3 examples/train_grpo.py
```

See [examples/README.md](examples/README.md) for details.

## Current Status

✅ **Step 1 Complete** - Core abstractions (data models, config, base classes)
✅ **Step 2 Complete** - GRPO trainer skeleton
✅ **Step 3 Complete** - End-to-end training examples

**What works:**
- Data models and configuration system
- Model wrappers (Actor, Reference, Critic)
- GRPO trainer structure
- Simple rollout and reward functions
- Training pipeline examples

**Next steps:**
- Real rollout generation with models
- Real GRPO loss computation
- Multi-turn conversations
- Tool execution
