"""DeepSeek-R1 style reasoning training example.

This script demonstrates training a model to use <reason> tags for
step-by-step reasoning, inspired by the DeepSeek-R1 paper.

The model learns to:
1. Use <reason> tags to show reasoning
2. Provide detailed step-by-step explanations
3. Structure answers clearly

Uses Qwen2.5-0.5B-Instruct for fast CPU training.
Defaults to 1 CPU thread; increase num_threads for faster training.
"""

import re
from flashrl import FlashRL
from flashrl.framework.data_models import (
    Prompt,
    RolloutOutput,
    RewardOutput,
    Message,
    Conversation,
)


# Reasoning prompts
REASONING_PROMPTS = [
    "Please solve this step by step. Use <reason> tags to show your reasoning.\n\nQuestion: What is 15 + 27?",
    "Please solve this step by step. Use <reason> tags to show your reasoning.\n\nQuestion: If I have 3 apples and get 5 more, how many do I have?",
    "Please solve this step by step. Use <reason> tags to show your reasoning.\n\nQuestion: What is 8 × 7?",
    "Please solve this step by step. Use <reason> tags to show your reasoning.\n\nQuestion: What is 100 - 37?",
    "Please solve this step by step. Use <reason> tags to show your reasoning.\n\nQuestion: If I divide 24 by 3, what do I get?",
    "Please solve this step by step. Use <reason> tags to show your reasoning.\n\nQuestion: What is 12 + 15 + 8?",
    "Please solve this step by step. Use <reason> tags to show your reasoning.\n\nQuestion: If I have 20 items and give away 7, how many remain?",
    "Please solve this step by step. Use <reason> tags to show your reasoning.\n\nQuestion: What is 9 × 6?",
]


def reasoning_rollout_fn(
    prompts: list[Prompt],
    actor,
) -> list[RolloutOutput]:
    """Generate reasoning responses.

    Args:
        prompts: List of input prompts.
        actor: Actor model to use for generation.

    Returns:
        List of rollout outputs with generated reasoning.
    """
    # Generate responses using actor model
    texts = actor.generate([p.text for p in prompts])

    rollouts = []
    for prompt, text in zip(prompts, texts):
        # Create conversation with user message and assistant response
        conversation = Conversation(messages=[
            Message(role="user", content=prompt.text),
            Message(role="assistant", content=text),
        ])

        rollouts.append(RolloutOutput(
            text=text,
            log_prob=0.0,  # TODO: Compute actual log prob from model
            conversation=conversation,
        ))

    return rollouts


def reasoning_reward_fn(rollout: RolloutOutput) -> RewardOutput:
    """Reward reasoning format.

    This reward function encourages the model to:
    1. Use <reason> tags
    2. Provide non-empty reasoning content
    3. Use proper closing tags
    4. Provide detailed reasoning (more content = higher reward)

    Args:
        rollout: Rollout output with generated text.

    Returns:
        Reward output with score and metadata.
    """
    text = rollout.text
    reward = 0.0

    # Check for <reason> tags
    has_open_tag = "<reason>" in text
    has_close_tag = "</reason>" in text
    has_reason = has_open_tag and has_close_tag

    if has_reason:
        # Extract reasoning content
        match = re.search(r'<reason>(.*?)</reason>', text, re.DOTALL)
        if match:
            reason_content = match.group(1).strip()
            # Reward based on reasoning length (more detail = better)
            # Cap at 10.0 reward for ~500 chars of reasoning
            reward = min(len(reason_content) / 50.0, 10.0)
        else:
            reward = 0.5  # Has tags but malformed
    else:
        reward = 0.0  # No reasoning tags

    return RewardOutput(
        reward=reward,
        metadata={
            "has_reason": has_reason,
            "has_open_tag": has_open_tag,
            "has_close_tag": has_close_tag,
        }
    )


def main():
    """Train a model to use reasoning format."""
    flashrl = FlashRL(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        rollout_fn=reasoning_rollout_fn,
        reward_fn=reasoning_reward_fn,
        learning_rate=1e-5,
        batch_size=4,
        max_epochs=3,
        # The local tutorial path keeps the frozen reference model disabled by
        # default so Mac runs stay lighter. Enable it only for KL-regularized GRPO.
        reference_enabled=False,
    )

    # Prepare dataset
    dataset = [Prompt(text=p) for p in REASONING_PROMPTS]

    try:
        flashrl.train(dataset)

        checkpoint_path = "/tmp/flashrl_reasoning_checkpoint.pt"
        flashrl.save_checkpoint(checkpoint_path)

    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()

        print("\nNote: This script requires downloading 'Qwen/Qwen2.5-0.5B-Instruct'.")
        print("If you're offline or have network issues:")
        print("  1. Connect to internet, or")
        print("  2. Use a local model by changing the 'model' parameter")


if __name__ == "__main__":
    main()
