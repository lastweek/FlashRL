"""Simplified GRPO training example using the new FlashRL API.

This script demonstrates how to use the clean FlashRL API for training.
You provide rollout and reward functions, FlashRL handles the rest.
"""

from flashrl import FlashRL
from flashrl.framework.data_models import (
    Prompt,
    RolloutOutput,
    RewardOutput,
    Message,
    Conversation,
)


def simple_rollout_fn(prompts: list[Prompt], actor) -> list[RolloutOutput]:
    """Generate responses using actor model.

    Args:
        prompts: List of input prompts.
        actor: Actor model to use for generation.

    Returns:
        List of rollout outputs.
    """
    # Generate responses
    texts = actor.generate([p.text for p in prompts])

    rollouts = []
    for prompt, text in zip(prompts, texts):
        conversation = Conversation(messages=[
            Message(role="user", content=prompt.text),
            Message(role="assistant", content=text),
        ])
        rollouts.append(RolloutOutput(
            text=text,
            log_prob=0.0,  # TODO: Compute actual log prob
            conversation=conversation,
        ))

    return rollouts


def simple_reward_fn(rollout: RolloutOutput) -> RewardOutput:
    """Simple reward function based on response length.

    Longer responses get higher rewards (just for demonstration).

    Args:
        rollout: Rollout output with generated text.

    Returns:
        Reward output with score.
    """
    # Simple heuristic: reward based on length (capped at 10.0)
    reward = min(len(rollout.text) / 100.0, 10.0)
    return RewardOutput(reward=reward)


def main():
    """Run simplified GRPO training."""
    print("=" * 60)
    print("FlashRL Simplified Training Example")
    print("=" * 60)

    # Create dummy dataset
    dummy_prompts = [
        "What is the capital of France?",
        "Explain gravity in one sentence.",
        "Write a haiku about programming.",
        "What is 2 + 2?",
        "Describe the color blue.",
        "What is machine learning?",
        "Tell me a joke.",
        "Explain quantum computing.",
    ]

    # Create FlashRL trainer (much simpler than before!)
    print("\nInitializing FlashRL trainer...")
    print(f"Model: gpt2")
    print(f"Prompts: {len(dummy_prompts)}")

    flashrl = FlashRL(
        model="gpt2",
        rollout_fn=simple_rollout_fn,
        reward_fn=simple_reward_fn,
        learning_rate=1e-5,
        batch_size=4,
        max_epochs=2,
    )

    # Prepare dataset
    dataset = [Prompt(text=p) for p in dummy_prompts]

    # Train
    print("\nStarting training...")
    print("This will download gpt2 if not already cached.\n")

    try:
        flashrl.train(dataset)

        # Save checkpoint
        checkpoint_path = "/tmp/flashrl_simple_checkpoint.pt"
        print(f"\nSaving checkpoint to {checkpoint_path}")
        flashrl.save_checkpoint(checkpoint_path)

        print("\n" + "=" * 60)
        print("✓ Training complete!")
        print("=" * 60)
        print("\nNotice how much simpler this is than the old API!")
        print("No need to manually create:")
        print("  - ActorModel, ReferenceModel")
        print("  - RolloutConfig, RewardConfig")
        print("  - GRPOTrainer with all dependencies")
        print("\nJust provide rollout_fn and reward_fn, and call train()!")

    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()

        print("\nNote: This script downloads 'gpt2' from HuggingFace.")
        print("If you're offline or have network issues, please:")
        print("  1. Connect to internet, or")
        print("  2. Use a local model by changing the 'model' parameter")


if __name__ == "__main__":
    main()
