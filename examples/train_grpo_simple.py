"""Simple end-to-end test without downloading models.

This version tests the training structure without requiring model downloads.
"""

from flashrl.framework.config import (
    TrainerConfig,
    RolloutConfig,
    RewardConfig,
)
from flashrl.framework.rollout.simple import SimpleRollout
from flashrl.framework.reward.simple import SimpleReward
from flashrl.framework.data_models import Prompt, TrainingBatch
import torch


def main():
    """Test the training pipeline structure."""
    print("=" * 60)
    print("FlashRL Simple Pipeline Test (No Model Download)")
    print("=" * 60)

    # Create configs
    print("\n--- Creating configurations ---")
    trainer_config = TrainerConfig(
        learning_rate=1e-5,
        batch_size=4,
        max_epochs=2,
        kl_coefficient=0.1,
    )
    print(f"✓ Trainer config created")

    rollout_config = RolloutConfig(max_new_tokens=32)
    print(f"✓ Rollout config created")

    reward_config = RewardConfig(scale=1.0)
    print(f"✓ Reward config created")

    # Create rollout and reward functions
    print("\n--- Creating components ---")
    rollout_gen = SimpleRollout(rollout_config)
    print("✓ Rollout generator created")

    reward_fn = SimpleReward(reward_config)
    print("✓ Reward function created")

    # Create dummy data
    print("\n--- Preparing data ---")
    prompts = [
        Prompt(text="What is the capital of France?"),
        Prompt(text="Explain gravity in one sentence."),
        Prompt(text="Write a haiku about programming."),
        Prompt(text="What is 2 + 2?"),
    ]
    print(f"✓ Created {len(prompts)} prompts")

    # Test the pipeline
    print("\n--- Testing pipeline ---")

    # Generate rollouts
    rollouts = rollout_gen.generate(prompts)
    print(f"✓ Generated {len(rollouts)} rollouts")

    # Show a sample
    print(f"\n  Sample rollout:")
    print(f"    Prompt: {rollouts[0].conversation.messages[0].content}")
    print(f"    Response: {rollouts[0].text[:100]}...")

    # Compute rewards
    rewards = [reward_fn.compute(r) for r in rollouts]
    avg_reward = sum(r.reward for r in rewards) / len(rewards)
    print(f"\n✓ Computed rewards")
    print(f"  Average reward: {avg_reward:.4f}")
    print(f"  Individual rewards: {[f'{r.reward:.2f}' for r in rewards]}")

    # Test advantage computation (GRPO core!)
    print("\n--- Testing GRPO advantage computation ---")
    reward_values = torch.tensor([r.reward for r in rewards], dtype=torch.float32)
    mean = reward_values.mean()
    std = reward_values.std()
    advantages = (reward_values - mean) / (std + 1e-8)

    print(f"  Rewards: {reward_values.tolist()}")
    print(f"  Mean: {mean:.4f}, Std: {std:.4f}")
    print(f"  Advantages: {advantages.tolist()}")
    print(f"  ✓ Advantages sum to ~0: {advantages.sum().item():.6f}")

    # Create training batch
    batch = TrainingBatch(
        prompts=prompts,
        conversations=[r.conversation for r in rollouts],
        rollouts=rollouts,
        rewards=rewards,
    )
    print(f"\n✓ Created training batch")
    print(f"  Batch size: {len(batch)}")

    print("\n" + "=" * 60)
    print("✓ Pipeline test completed successfully!")
    print("=" * 60)

    print("\nThe pipeline structure is working correctly!")
    print("\nNext steps to make it fully functional:")
    print("  1. Implement real rollout generation with ActorModel")
    print("  2. Implement real GRPO loss computation")
    print("  3. Add actual model training loop")
    print("\nRun examples/train_grpo.py for the full version (downloads models).")


if __name__ == "__main__":
    main()
