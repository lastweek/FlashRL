"""Tiny importable hooks for platform config and CRD tests."""

from __future__ import annotations

from flashrl.framework.data_models import Conversation, Message, Prompt, RewardOutput, RolloutOutput


def build_dataset() -> list[Prompt]:
    return [Prompt(text="hello"), Prompt(text="world")]


def build_rollout():
    def rollout_fn(prompts: list[Prompt], serving_backend) -> list[RolloutOutput]:
        samples = serving_backend.generate_batch([prompt.text for prompt in prompts])
        return [
            RolloutOutput(
                text=sample.text,
                log_prob=sample.log_prob,
                prompt_token_ids=sample.prompt_token_ids,
                response_token_ids=sample.response_token_ids,
                response_token_logprobs=sample.response_token_logprobs,
                conversation=Conversation(
                    messages=[
                        Message(role="user", content=prompt.text),
                        Message(role="assistant", content=sample.text),
                    ]
                ),
                metadata=dict(sample.metadata),
            )
            for prompt, sample in zip(prompts, samples, strict=True)
        ]

    return rollout_fn


def build_reward():
    def reward_fn(rollout: RolloutOutput) -> RewardOutput:
        return RewardOutput(reward=float(len(rollout.text)))

    return reward_fn
