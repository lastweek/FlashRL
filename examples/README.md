# FlashRL Examples

This directory contains example scripts demonstrating how to use FlashRL.

## 🚀 Quick Start (New Simple API)

The simplest way to use FlashRL:

```python
from flashrl import FlashRL
from flashrl.framework.data_models import Prompt, RolloutOutput, RewardOutput

# Define your rollout function
def my_rollout_fn(prompts, actor):
    return actor.generate([p.text for p in prompts])

# Define your reward function
def my_reward_fn(rollout):
    return RewardOutput(reward=len(rollout.text) / 100.0)

# Create FlashRL trainer
flashrl = FlashRL(
    model="gpt2",
    rollout_fn=my_rollout_fn,
    reward_fn=my_reward_fn,
    learning_rate=1e-5,
    batch_size=4,
    max_epochs=2,
)

# Train
dataset = [Prompt(text=p) for p in my_prompts]
flashrl.train(dataset)
```

## Scripts

### train_grpo_simple.py
**Quick test - no model download required**

Tests the training pipeline structure without downloading models.
- Verifies rollout generation works
- Verifies reward computation works
- Shows GRPO advantage computation (group normalization)
- Creates training batches

**Run it:**
```bash
python3 examples/train_grpo_simple.py
```

**Expected output:**
```
✓ Pipeline test completed successfully!
The pipeline structure is working correctly!
```

---

### train_grpo.py
**Full end-to-end training with real models (NEW SIMPLIFIED API)**

Demonstrates the new clean FlashRL API. Downloads GPT-2 and runs actual GRPO training.
- **Much simpler than before** - no need to manually create models, configs, trainers
- Just provide rollout and reward functions
- FlashRL handles the rest

**Run it:**
```bash
python3 examples/train_grpo.py
```

**Note:**
- Requires internet connection (downloads GPT-2)
- Takes ~1-2 minutes to download models
- Demonstrates the new simplified API

---

### train_reasoning.py ⭐ NEW
**DeepSeek-R1 style reasoning training**

Train a model to use `<reason>` tags for step-by-step reasoning.
- Uses Qwen2.5-0.5B-Instruct (small, fast for CPU)
- Teaches model to show reasoning with `<reason>` tags
- Reward function checks for proper reasoning format
- Realistic example of reasoning RL training

**Run it:**
```bash
python3 examples/train_reasoning.py
```

**What it does:**
1. Loads Qwen2.5-0.5B-Instruct model
2. Trains on math word problems
3. Rewards model for using `<reason>` tags
4. Teaches step-by-step reasoning format

**Example prompt:**
```
Please solve this step by step. Use <reason> tags to show your reasoning.

Question: What is 15 + 27?
```

**Desired output:**
```
<reason>
I need to add 15 and 27.
15 + 27 = 42
</reason>

Answer: 42
```

**Note:**
- Requires internet connection (downloads Qwen model)
- Uses small 0.5B model for fast CPU training
- Demonstrates realistic RL training scenario

---

## Old vs New API Comparison

### Old API (Complex)
```python
# Create configs
trainer_config = TrainerConfig(learning_rate=1e-5, batch_size=4)
model_config = ModelConfig(model_name="gpt2", device=str(device))
rollout_config = RolloutConfig(max_new_tokens=32)
reward_config = RewardConfig(scale=1.0)

# Load models
actor = ActorModel(model_config)
reference = ReferenceModel(model_config)

# Create rollout and reward functions
rollout_gen = SimpleRollout(rollout_config)
reward_fn = SimpleReward(reward_config)

# Create trainer
trainer = GRPOTrainer(
    config=trainer_config,
    actor=actor,
    reference=reference,
    reward_fn=reward_fn,
    rollout_generator=rollout_gen,
)

# Train
trainer.train(dataset)
```

### New API (Simple)
```python
flashrl = FlashRL(
    model="gpt2",
    rollout_fn=my_rollout_fn,
    reward_fn=my_reward_fn,
    learning_rate=1e-5,
    batch_size=4,
)

flashrl.train(dataset)
```

Much cleaner! 🎉

---

## Pipeline Structure

All scripts demonstrate the same training pipeline:

```
1. Create FlashRL trainer with model, rollout_fn, reward_fn
2. Prepare dataset (list of Prompts)
3. Call flashrl.train(dataset)
4. Save checkpoint
```

FlashRL handles internally:
- Loading actor and reference models
- Creating GRPO trainer
- Training loop (rollout → reward → train)
- Checkpointing

---

## Next Steps

After running these examples, you can:

1. **Implement your own rollouts** - Use ActorModel to generate actual text
2. **Implement your own rewards** - Use reward models or rule-based rewards
3. **Train on real data** - Use your own datasets
4. **Add logging** - Track metrics with W&B, TensorBoard, etc.
5. **Scale up** - Move to platform layer for distributed training
