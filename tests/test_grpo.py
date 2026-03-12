"""Algorithm-focused unit tests for grouped GRPO."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from examples.reasoning.train import REASONING_PROMPTS, reasoning_reward_fn
import flashrl.framework.flashrl as flashrl_module
from flashrl.framework import FlashRL, GrpoConfig, LoggingConfig, MetricsConfig
from flashrl.framework.config import TrainerConfig
from flashrl.framework.data_models import Conversation, Message, Prompt, RewardOutput, RolloutOutput
from flashrl.framework.reward.user_defined import UserDefinedReward
from flashrl.framework.rollout.user_defined import UserDefinedRollout
from flashrl.framework.trainer.grpo import GRPOTrainer
from tests.conftest import (
    TinyReferenceModel,
    TinyActor,
    TinyServingBackend,
    TinyTrainingBackend,
    make_rollout_fn,
    reward_fn,
)

pytestmark = pytest.mark.unit


def build_trainer(
    *,
    batch_size: int = 4,
    group_size: int = 2,
    kl_coefficient: float = 0.0,
    rollout_fn=None,
    reference=None,
    serving_backend=None,
) -> GRPOTrainer:
    """Build a small offline GRPO trainer for algorithm tests."""
    training_backend = TinyTrainingBackend(learning_rate=1e-2)
    serving_backend = serving_backend or TinyServingBackend()
    rollout = UserDefinedRollout(
        rollout_fn=rollout_fn or make_rollout_fn(response_suffix="grpo", repeat=2),
        serving_backend=serving_backend,
        config=SimpleNamespace(),
    )
    reward = UserDefinedReward(reward_fn=reward_fn, config=SimpleNamespace())
    return GRPOTrainer(
        config=TrainerConfig(batch_size=batch_size, max_epochs=1),
        grpo_config=GrpoConfig(
            group_size=group_size,
            clip_ratio=0.2,
            kl_coefficient=kl_coefficient,
        ),
        training_backend=training_backend,
        serving_backend=serving_backend,
        reference=reference,
        reward_fn=reward,
        rollout_generator=rollout,
        run_logger=None,
        metrics_sink=None,
    )


def test_grpo_batch_size_means_total_sampled_completions_per_step() -> None:
    """A grouped GRPO step should consume batch_size/group_size unique prompts."""
    prompt_batches: list[list[str]] = []
    rollout_impl = make_rollout_fn(response_suffix="batch", repeat=1)

    def rollout_fn(prompts, serving_backend):
        prompt_batches.append([prompt.text for prompt in prompts])
        return rollout_impl(prompts, serving_backend)

    trainer = build_trainer(batch_size=4, group_size=2, rollout_fn=rollout_fn)
    dataset = [Prompt(text=f"prompt {index}") for index in range(5)]

    trainer.train(dataset)

    assert trainer._prompts_per_step() == 2
    assert prompt_batches == [
        ["prompt 0", "prompt 1"],
        ["prompt 0", "prompt 1"],
        ["prompt 2", "prompt 3"],
        ["prompt 2", "prompt 3"],
        ["prompt 4"],
        ["prompt 4"],
    ]


def test_grpo_advantages_are_normalized_within_each_prompt_group() -> None:
    """Advantage normalization should be done independently for each prompt group."""
    trainer = build_trainer(batch_size=4, group_size=2)

    advantages = trainer._compute_advantages(
        [
            RewardOutput(reward=1.0),
            RewardOutput(reward=3.0),
            RewardOutput(reward=10.0),
            RewardOutput(reward=12.0),
        ],
        prompt_count=2,
        group_size=2,
    )

    assert advantages.tolist() == pytest.approx([-1.0, 1.0, -1.0, 1.0])


def test_grpo_compute_advantages_rejects_mismatched_group_shape() -> None:
    """Reward count must match prompt_count * group_size for grouped GRPO."""
    trainer = build_trainer(batch_size=4, group_size=2)

    with pytest.raises(ValueError, match="prompt_count \\* group_size"):
        trainer._compute_advantages(
            [RewardOutput(reward=1.0), RewardOutput(reward=3.0), RewardOutput(reward=10.0)],
            prompt_count=2,
            group_size=2,
        )


def test_grpo_assemble_loss_uses_response_only_grpo_terms() -> None:
    """Prompt-token-only logit changes should not affect the response-only GRPO loss."""
    reference = TinyReferenceModel()
    trainer = build_trainer(batch_size=2, group_size=2, kl_coefficient=0.3, reference=reference)

    prompts = [Prompt(text="prompt 0"), Prompt(text="prompt 1")]
    grouped_rollouts = make_rollout_fn(response_suffix="loss", repeat=2)(
        prompts,
        trainer.serving_backend,
    )
    rollouts = grouped_rollouts

    actor = trainer.training_backend.actor
    input_ids, attention_mask, prompt_lengths, _, rollout_response_log_probs = trainer._prepare_inputs(
        SimpleNamespace(rollouts=rollouts),
        actor,
        actor.device,
    )

    actor_logits = actor.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).logits
    with torch.no_grad():
        ref_logits = reference.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits

    advantages = torch.tensor([1.25, -0.5], dtype=torch.float32)

    loss_no_ref, policy_no_ref, kl_no_ref, response_tokens_total = trainer._assemble_loss(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_lengths=prompt_lengths,
        actor_logits=actor_logits,
        ref_logits=None,
        rollout_response_log_probs=rollout_response_log_probs,
        advantages=advantages,
        kl_coefficient=trainer.grpo_config.kl_coefficient,
        clip_ratio=trainer.grpo_config.clip_ratio,
    )

    assert loss_no_ref.item() == pytest.approx(policy_no_ref.item())
    assert kl_no_ref.item() == pytest.approx(0.0)
    assert response_tokens_total == sum(len(sample) for sample in rollout_response_log_probs)

    prompt_only_mutated_logits = actor_logits.clone()
    for index, prompt_length in enumerate(prompt_lengths.tolist()):
        prompt_positions = max(int(prompt_length) - 1, 0)
        if prompt_positions == 0:
            continue
        prompt_only_mutated_logits[index, :prompt_positions, 0] += 25.0

    mutated_loss, mutated_policy, mutated_kl, _ = trainer._assemble_loss(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_lengths=prompt_lengths,
        actor_logits=prompt_only_mutated_logits,
        ref_logits=None,
        rollout_response_log_probs=rollout_response_log_probs,
        advantages=advantages,
        kl_coefficient=trainer.grpo_config.kl_coefficient,
        clip_ratio=trainer.grpo_config.clip_ratio,
    )

    assert mutated_loss.item() == pytest.approx(loss_no_ref.item())
    assert mutated_policy.item() == pytest.approx(policy_no_ref.item())
    assert mutated_kl.item() == pytest.approx(0.0)

    loss_with_ref, policy_with_ref, kl_with_ref, _ = trainer._assemble_loss(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_lengths=prompt_lengths,
        actor_logits=actor_logits,
        ref_logits=ref_logits,
        rollout_response_log_probs=rollout_response_log_probs,
        advantages=advantages,
        kl_coefficient=trainer.grpo_config.kl_coefficient,
        clip_ratio=trainer.grpo_config.clip_ratio,
    )

    expected_total = (
        policy_with_ref.item() + trainer.grpo_config.kl_coefficient * kl_with_ref.item()
    )
    assert loss_with_ref.item() == pytest.approx(expected_total)
    assert kl_with_ref.item() >= 0.0


def test_grpo_rollout_response_log_probs_align_with_response_tokens() -> None:
    """Stored rollout log-probs should line up exactly with each sampled response length."""
    trainer = build_trainer(batch_size=2, group_size=2)
    prompts = [Prompt(text="prompt 0"), Prompt(text="prompt 1")]
    grouped_rollouts = make_rollout_fn(response_suffix="align", repeat=2)(
        prompts,
        trainer.serving_backend,
    )
    rollouts = grouped_rollouts
    rollout_response_log_probs = [rollout.response_token_logprobs for rollout in rollouts]

    input_ids, attention_mask, prompt_lengths, _, _ = trainer._prepare_inputs(
        SimpleNamespace(rollouts=rollouts),
        trainer.training_backend.actor,
        trainer.training_backend.actor.device,
    )
    actor_logits = trainer.training_backend.actor.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).logits

    _, _, _, response_tokens_total = trainer._assemble_loss(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_lengths=prompt_lengths,
        actor_logits=actor_logits,
        ref_logits=None,
        rollout_response_log_probs=rollout_response_log_probs,
        advantages=torch.tensor([1.0, -1.0], dtype=torch.float32),
        kl_coefficient=0.0,
        clip_ratio=trainer.grpo_config.clip_ratio,
    )

    assert response_tokens_total == sum(len(sample) for sample in rollout_response_log_probs)

    with pytest.raises(ValueError, match="response-token count"):
        trainer._assemble_loss(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_lengths=prompt_lengths,
            actor_logits=actor_logits,
            ref_logits=None,
            rollout_response_log_probs=[sample[:-1] for sample in rollout_response_log_probs],
            advantages=torch.tensor([1.0, -1.0], dtype=torch.float32),
            kl_coefficient=0.0,
            clip_ratio=trainer.grpo_config.clip_ratio,
        )


def test_grpo_zero_advantages_with_zero_kl_produce_zero_loss() -> None:
    """Flat rewards with beta=0 should collapse to zero policy and total loss."""
    trainer = build_trainer(batch_size=2, group_size=2, kl_coefficient=0.0)
    prompts = [Prompt(text="prompt 0"), Prompt(text="prompt 1")]
    rollouts = make_rollout_fn(response_suffix="zero", repeat=2)(
        prompts,
        trainer.serving_backend,
    )

    input_ids, attention_mask, prompt_lengths, _, rollout_response_log_probs = trainer._prepare_inputs(
        SimpleNamespace(rollouts=rollouts),
        trainer.training_backend.actor,
        trainer.training_backend.actor.device,
    )
    actor_logits = trainer.training_backend.actor.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).logits

    loss, policy_loss, kl_divergence, _ = trainer._assemble_loss(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_lengths=prompt_lengths,
        actor_logits=actor_logits,
        ref_logits=None,
        rollout_response_log_probs=rollout_response_log_probs,
        advantages=torch.zeros(2, dtype=torch.float32),
        kl_coefficient=0.0,
        clip_ratio=trainer.grpo_config.clip_ratio,
    )

    assert policy_loss.item() == pytest.approx(0.0)
    assert kl_divergence.item() == pytest.approx(0.0)
    assert loss.item() == pytest.approx(0.0)


def test_reasoning_example_rewards_create_non_zero_group_advantages() -> None:
    """The example reward should separate GRPO candidates within one prompt group."""
    trainer = build_trainer(batch_size=4, group_size=4)
    prompt_text = REASONING_PROMPTS[0]
    responses = [
        "<reason>Add 15 and 27 to get 42 in the final line.</reason>\n42",
        "<reason>Add 15 and 27 to get 41 in the final line.</reason>\n41",
        "42",
        "<reason>Add 15 and 27 and stop early\n42",
    ]
    rewards = [
        reasoning_reward_fn(
            RolloutOutput(
                text=response,
                log_prob=-0.1,
                prompt_token_ids=[1, 2, 3],
                response_token_ids=[4, 5, 6],
                response_token_logprobs=[-0.1, -0.1, -0.1],
                conversation=Conversation(
                    messages=[
                        Message(role="user", content=prompt_text),
                        Message(role="assistant", content=response),
                    ]
                ),
            )
        )
        for response in responses
    ]

    reward_values = [reward.reward for reward in rewards]
    advantages = trainer._compute_advantages(rewards, prompt_count=1, group_size=4)

    assert any(value > 0.0 for value in reward_values)
    assert any(abs(float(value)) > 1e-6 for value in advantages.tolist())
    assert sum(advantages.tolist()) == pytest.approx(0.0, abs=1e-6)


def test_grpo_prepare_inputs_reuses_rollout_token_ids_without_tokenizer_calls() -> None:
    """Input preparation should use rollout token ids directly and avoid tokenizer calls."""
    trainer = build_trainer(batch_size=4, group_size=2)
    prompts = [Prompt(text="prompt 0"), Prompt(text="prompt 1")]
    _, rollouts, _, _ = trainer.rollout_generator.generate_grouped(prompts, group_size=2)
    actor = trainer.training_backend.actor
    actor.tokenizer.calls.clear()

    input_ids, attention_mask, prompt_lengths, full_lengths, rollout_response_log_probs = trainer._prepare_inputs(
        SimpleNamespace(rollouts=rollouts),
        actor,
        actor.device,
    )

    assert actor.tokenizer.calls == []
    assert input_ids.shape[0] == 4
    assert attention_mask.sum(dim=1).tolist() == full_lengths
    assert prompt_lengths.tolist() == [len(rollout.prompt_token_ids) for rollout in rollouts]
    assert rollout_response_log_probs == [rollout.response_token_logprobs for rollout in rollouts]


def test_grpo_installs_and_clears_serving_debug_context_per_step() -> None:
    """Serving live-rollout debug hooks should be installed for rollout and cleared afterward."""

    class DebugActor(TinyActor):
        def __init__(self) -> None:
            super().__init__(bias_shift=0.1)
            self.debug_events: list[tuple[str, object]] = []

        def set_live_rollout_debug(self, callback, context) -> None:
            self.debug_events.append(("set", dict(context)))

        def set_live_rollout_candidate_index(self, candidate_index: int | None) -> None:
            self.debug_events.append(("candidate", candidate_index))

        def clear_live_rollout_debug(self) -> None:
            self.debug_events.append(("clear", None))

    class DebugServingBackend:
        def __init__(self) -> None:
            self.config = SimpleNamespace(debug_live_rollout=True)
            self._actor = DebugActor()
            self.device = self._actor.device
            self.generation_defaults: dict[str, object] = {}

        def generate(self, prompts: list[str], **kwargs):
            return self._actor.generate(prompts, **kwargs)

        def generate_batch(self, prompts: list[str], **kwargs):
            return self._actor.generate_batch(prompts, **kwargs)

        def generate_grouped(self, prompts: list[str], group_size: int, **kwargs):
            return self._actor.generate_grouped(prompts, group_size, **kwargs)

        def set_generation_defaults(self, **kwargs) -> None:
            self.generation_defaults = dict(kwargs)
            self._actor.set_generation_defaults(**kwargs)

        def sync_from_training_actor(self, training_actor) -> None:
            self._actor.model.load_state_dict(training_actor.model.state_dict())

        def set_live_rollout_debug(self, callback, context) -> None:
            self._actor.set_live_rollout_debug(callback, context)

        def set_live_rollout_candidate_index(self, candidate_index: int | None) -> None:
            self._actor.set_live_rollout_candidate_index(candidate_index)

        def clear_live_rollout_debug(self) -> None:
            self._actor.clear_live_rollout_debug()

        def close(self) -> None:
            return None

    serving_backend = DebugServingBackend()
    trainer = build_trainer(
        batch_size=2,
        group_size=2,
        serving_backend=serving_backend,
    )

    trainer.train([Prompt(text="prompt 0"), Prompt(text="prompt 1")])

    assert serving_backend._actor.debug_events[0][0] == "set"
    assert ("candidate", 0) in serving_backend._actor.debug_events
    assert ("candidate", 1) in serving_backend._actor.debug_events
    assert serving_backend._actor.debug_events[-1] == ("clear", None)


def test_flashrl_rejects_batch_size_not_divisible_by_group_size(tmp_path) -> None:
    """FlashRL should fail fast when grouped GRPO cannot form full prompt groups."""
    with pytest.raises(ValueError, match="divisible by grpo.group_size"):
        FlashRL(
            model="fake/model",
            rollout_fn=make_rollout_fn(response_suffix="invalid", repeat=1),
            reward_fn=reward_fn,
            batch_size=3,
            max_epochs=1,
            grpo_config=GrpoConfig(group_size=2),
            logging_config=LoggingConfig(log_dir=tmp_path, console=False, file=True),
            metrics_config=MetricsConfig(enabled=False),
        )
