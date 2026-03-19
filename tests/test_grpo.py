"""Algorithm-focused unit tests for grouped GRPO."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

import flashrl.framework.flashrl as flashrl_module
from flashrl.framework import FlashRL, GrpoConfig, LoggingConfig, MetricsConfig
from flashrl.framework.config import ServingConfig, TrainerConfig, TrainingConfig
from flashrl.framework.data_models import (
    Conversation,
    LearnerBatch,
    Message,
    Prompt,
    RewardOutput,
    RolloutOutput,
    WeightVersionInfo,
)
from flashrl.framework.reward.user_defined import UserDefinedReward
from flashrl.framework.rollout.function import FunctionRolloutGenerator
from flashrl.framework.trainer.grpo.trainer import GRPOTrainer
from flashrl.framework.trainer.grpo.grpo_helpers import compute_advantages, prompt_batch_size
from flashrl.framework.trainer.grpo.loss_variants import assemble_grpo_loss
from flashrl.framework.data_models import LearnerBatch
from tests.conftest import (
    TinyReferenceModel,
    TinyActor,
    TinyServingBackend,
    TinyTrainingBackend,
    make_rollout_fn,
    reward_fn,
)

pytestmark = pytest.mark.unit


def load_script_module(module_name: str, relative_path: str):
    """Load one hyphen-folder script as a normal Python module for tests."""
    module_path = Path(relative_path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


reasoning_math = load_script_module(
    "flashrl_reasoning_math_train_for_grpo",
    "flashrl/framework/examples/math/train.py",
)
math_reward_fn = reasoning_math.math_reward_fn
render_math_prompt = reasoning_math.render_math_prompt


def build_trainer(
    *,
    batch_size: int = 4,
    group_size: int = 2,
    kl_coefficient: float = 0.0,
    rollout_fn=None,
    reference=None,
    serving_backend=None,
    seed: int = 42,
    shuffle_each_epoch: bool = False,
) -> GRPOTrainer:
    """Build a small offline GRPO trainer for algorithm tests."""
    training_backend = TinyTrainingBackend(
        learning_rate=1e-2,
        group_size=group_size,
    )
    serving_backend = serving_backend or TinyServingBackend()
    rollout = FunctionRolloutGenerator(
        rollout_fn=rollout_fn or make_rollout_fn(response_suffix="grpo", repeat=2),
        serving_backend=serving_backend,
        config=SimpleNamespace(),
    )
    reward = UserDefinedReward(reward_fn=reward_fn, config=SimpleNamespace())
    return GRPOTrainer(
        config=TrainerConfig(
            batch_size=batch_size,
            max_epochs=1,
            seed=seed,
            shuffle_each_epoch=shuffle_each_epoch,
        ),
        grpo_config=GrpoConfig(
            group_size=group_size,
            clip_ratio=0.2,
            kl_coefficient=kl_coefficient,
        ),
        actor_backend=training_backend,
        reference_backend=reference,
        serving_backend=serving_backend,
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

    assert prompt_batch_size(trainer.config.batch_size, trainer.grpo_config.group_size) == 2
    assert prompt_batches == [
        ["prompt 0", "prompt 1"],
        ["prompt 0", "prompt 1"],
        ["prompt 2", "prompt 3"],
        ["prompt 2", "prompt 3"],
        ["prompt 4"],
        ["prompt 4"],
    ]


def test_grpo_shuffles_dataset_each_epoch_when_enabled() -> None:
    """Trainer should reshuffle prompt order deterministically from the configured seed."""
    prompt_batches: list[list[str]] = []
    rollout_impl = make_rollout_fn(response_suffix="shuffle", repeat=1)

    def rollout_fn(prompts, serving_backend):
        prompt_batches.append([prompt.text for prompt in prompts])
        return rollout_impl(prompts, serving_backend)

    trainer = build_trainer(
        batch_size=4,
        group_size=2,
        rollout_fn=rollout_fn,
        seed=7,
        shuffle_each_epoch=True,
    )
    trainer.config.max_epochs = 2
    dataset = [Prompt(text=f"prompt {index}") for index in range(4)]

    trainer.train(dataset)

    assert prompt_batches == [
        ["prompt 3", "prompt 1"],
        ["prompt 3", "prompt 1"],
        ["prompt 0", "prompt 2"],
        ["prompt 0", "prompt 2"],
        ["prompt 0", "prompt 2"],
        ["prompt 0", "prompt 2"],
        ["prompt 3", "prompt 1"],
        ["prompt 3", "prompt 1"],
    ]


def test_grpo_advantages_are_normalized_within_each_prompt_group() -> None:
    """Advantage normalization should be done independently for each prompt group."""
    trainer = build_trainer(batch_size=4, group_size=2)

    advantages = compute_advantages(
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
        compute_advantages(
            [RewardOutput(reward=1.0), RewardOutput(reward=3.0), RewardOutput(reward=10.0)],
            prompt_count=2,
            group_size=2,
        )


def test_grpo_controller_builds_learner_batch_before_training_optimize() -> None:
    """Controller-side reward/advantage logic should hand one learner batch to training."""

    class RecordingTrainingBackend(TinyTrainingBackend):
        def __init__(self) -> None:
            super().__init__(learning_rate=1e-2, group_size=2)
            self.recorded_batches: list[LearnerBatch] = []

        def prepare_inputs(self, learner_batch: LearnerBatch):
            self.recorded_batches.append(learner_batch)
            return super().prepare_inputs(learner_batch)

    training_backend = RecordingTrainingBackend()
    serving_backend = TinyServingBackend()
    rollout = FunctionRolloutGenerator(
        rollout_fn=make_rollout_fn(response_suffix="handoff", repeat=1),
        serving_backend=serving_backend,
        config=SimpleNamespace(),
    )
    reward = UserDefinedReward(reward_fn=reward_fn, config=SimpleNamespace())
    trainer = GRPOTrainer(
        config=TrainerConfig(batch_size=4, max_epochs=1, shuffle_each_epoch=False),
        grpo_config=GrpoConfig(group_size=2, clip_ratio=0.2, kl_coefficient=0.0),
        actor_backend=training_backend,
        reference_backend=None,
        serving_backend=serving_backend,
        reward_fn=reward,
        rollout_generator=rollout,
        run_logger=None,
        metrics_sink=None,
    )

    trainer.train([Prompt(text="prompt 0"), Prompt(text="prompt 1")])

    assert len(training_backend.recorded_batches) == 1
    learner_batch = training_backend.recorded_batches[0]
    assert learner_batch.prompt_count == 2
    assert learner_batch.group_size == 2
    assert len(learner_batch.prompt_token_ids) == 4
    assert len(learner_batch.response_token_ids) == 4
    assert len(learner_batch.response_token_logprobs) == 4
    assert len(learner_batch.advantages) == 4


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
    learner_batch = LearnerBatch(
        prompt_token_ids=[[int(tid) for tid in rollout.prompt_token_ids] for rollout in rollouts],
        response_token_ids=[[int(tid) for tid in rollout.response_token_ids] for rollout in rollouts],
        response_token_logprobs=[rollout.response_token_logprobs for rollout in rollouts],
        advantages=[0.0 for _ in rollouts],
        group_size=2,
        prompt_count=2,
    )
    input_ids, attention_mask, prompt_lengths, _, rollout_response_log_probs = trainer.training_backend.prepare_inputs(
        learner_batch,
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

    loss_no_ref, policy_no_ref, kl_no_ref, response_tokens_total = assemble_grpo_loss(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_lengths=prompt_lengths,
        actor_logits=actor_logits,
        ref_logits=None,
        rollout_response_log_probs=rollout_response_log_probs,
        advantages=advantages,
        config=trainer.grpo_config,
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

    mutated_loss, mutated_policy, mutated_kl, _ = assemble_grpo_loss(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_lengths=prompt_lengths,
        actor_logits=prompt_only_mutated_logits,
        ref_logits=None,
        rollout_response_log_probs=rollout_response_log_probs,
        advantages=advantages,
        config=trainer.grpo_config,
    )

    assert mutated_loss.item() == pytest.approx(loss_no_ref.item())
    assert mutated_policy.item() == pytest.approx(policy_no_ref.item())
    assert mutated_kl.item() == pytest.approx(0.0)

    loss_with_ref, policy_with_ref, kl_with_ref, _ = assemble_grpo_loss(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_lengths=prompt_lengths,
        actor_logits=actor_logits,
        ref_logits=ref_logits,
        rollout_response_log_probs=rollout_response_log_probs,
        advantages=advantages,
        config=trainer.grpo_config,
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

    learner_batch = LearnerBatch(
        prompt_token_ids=[[int(tid) for tid in rollout.prompt_token_ids] for rollout in rollouts],
        response_token_ids=[[int(tid) for tid in rollout.response_token_ids] for rollout in rollouts],
        response_token_logprobs=rollout_response_log_probs,
        advantages=[0.0 for _ in rollouts],
        group_size=2,
        prompt_count=2,
    )
    input_ids, attention_mask, prompt_lengths, _, _ = trainer.training_backend.prepare_inputs(
        learner_batch,
    )
    actor_logits = trainer.training_backend.actor.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).logits

    _, _, _, response_tokens_total = assemble_grpo_loss(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_lengths=prompt_lengths,
        actor_logits=actor_logits,
        ref_logits=None,
        rollout_response_log_probs=rollout_response_log_probs,
        advantages=torch.tensor([1.0, -1.0], dtype=torch.float32),
        config=trainer.grpo_config,
    )

    assert response_tokens_total == sum(len(sample) for sample in rollout_response_log_probs)

    with pytest.raises(ValueError, match="response-token count"):
        assemble_grpo_loss(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_lengths=prompt_lengths,
            actor_logits=actor_logits,
            ref_logits=None,
            rollout_response_log_probs=[sample[:-1] for sample in rollout_response_log_probs],
            advantages=torch.tensor([1.0, -1.0], dtype=torch.float32),
            config=trainer.grpo_config,
        )


def test_grpo_zero_advantages_with_zero_kl_produce_zero_loss() -> None:
    """Flat rewards with beta=0 should collapse to zero policy and total loss."""
    trainer = build_trainer(batch_size=2, group_size=2, kl_coefficient=0.0)
    prompts = [Prompt(text="prompt 0"), Prompt(text="prompt 1")]
    rollouts = make_rollout_fn(response_suffix="zero", repeat=2)(
        prompts,
        trainer.serving_backend,
    )

    learner_batch = LearnerBatch(
        prompt_token_ids=[[int(tid) for tid in rollout.prompt_token_ids] for rollout in rollouts],
        response_token_ids=[[int(tid) for tid in rollout.response_token_ids] for rollout in rollouts],
        response_token_logprobs=[rollout.response_token_logprobs for rollout in rollouts],
        advantages=[0.0 for _ in rollouts],
        group_size=2,
        prompt_count=2,
    )
    input_ids, attention_mask, prompt_lengths, _, rollout_response_log_probs = trainer.training_backend.prepare_inputs(
        learner_batch,
    )
    actor_logits = trainer.training_backend.actor.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).logits

    loss, policy_loss, kl_divergence, _ = assemble_grpo_loss(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_lengths=prompt_lengths,
        actor_logits=actor_logits,
        ref_logits=None,
        rollout_response_log_probs=rollout_response_log_probs,
        advantages=torch.zeros(2, dtype=torch.float32),
        config=trainer.grpo_config,
    )

    assert policy_loss.item() == pytest.approx(0.0)
    assert kl_divergence.item() == pytest.approx(0.0)
    assert loss.item() == pytest.approx(0.0)


def test_reasoning_example_rewards_create_non_zero_group_advantages() -> None:
    """The example reward should separate GRPO candidates within one prompt group."""
    trainer = build_trainer(batch_size=4, group_size=4)
    prompt = Prompt(
        text=render_math_prompt("What is 15 + 27?"),
        metadata={
            "task_id": "gsm8k-train-000000",
            "source": "gsm8k",
            "split": "train",
            "problem": "What is 15 + 27?",
            "final_answer": "42",
            "verifier": "numeric_exact",
        },
    )
    responses = [
        "<think>Add 15 and 27 to get 42.</think><answer>42</answer>",
        "<think>Add 15 and 27 to get 41.</think><answer>41</answer>",
        "<think>Add 15 and 27 to get 42.</think><answer>42</answer>\nextra",
        "<think>Add 15 and 27 to get 42.",
    ]
    rewards = [
        math_reward_fn(
            RolloutOutput(
                text=response,
                log_prob=-0.1,
                prompt_token_ids=[1, 2, 3],
                response_token_ids=[4, 5, 6],
                response_token_logprobs=[-0.1, -0.1, -0.1],
                metadata={
                    "prompt_metadata": dict(prompt.metadata),
                    "finish_reason": "length" if response == responses[-1] else "stop",
                },
                conversation=Conversation(
                    messages=[
                        Message(role="user", content=prompt.text),
                        Message(role="assistant", content=response),
                    ]
                ),
            )
        )
        for response in responses
    ]

    reward_values = [reward.reward for reward in rewards]
    advantages = compute_advantages(rewards, prompt_count=1, group_size=4)

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

    learner_batch = LearnerBatch(
        prompt_token_ids=[[int(tid) for tid in rollout.prompt_token_ids] for rollout in rollouts],
        response_token_ids=[[int(tid) for tid in rollout.response_token_ids] for rollout in rollouts],
        response_token_logprobs=[rollout.response_token_logprobs for rollout in rollouts],
        advantages=[0.0 for _ in rollouts],
        group_size=2,
        prompt_count=2,
    )
    input_ids, attention_mask, prompt_lengths, full_lengths, rollout_response_log_probs = trainer.training_backend.prepare_inputs(
        learner_batch,
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
            self._active_weight_version = WeightVersionInfo(
                version_id=0,
                source_training_step=None,
                source_epoch=None,
                activated_at="2026-03-19T00:00:00Z",
                model_source="debug-serving://startup",
                origin="startup",
            )
            self._next_weight_version_id = 1

        def generate(self, prompts: list[str], **kwargs):
            return self._actor.generate(prompts, **kwargs)

        def generate_batch(self, prompts: list[str], **kwargs):
            return self._actor.generate_batch(prompts, **kwargs)

        def generate_grouped(self, prompts: list[str], group_size: int, **kwargs):
            return self._actor.generate_grouped(prompts, group_size, **kwargs)

        def set_generation_defaults(self, **kwargs) -> None:
            self.generation_defaults = dict(kwargs)
            self._actor.set_generation_defaults(**kwargs)

        def sync_from_training_actor(
            self,
            training_actor,
            *,
            source_training_step: int | None = None,
            source_epoch: int | None = None,
            origin: str = "sync",
        ) -> WeightVersionInfo:
            self._actor.model.load_state_dict(training_actor.model.state_dict())
            self._active_weight_version = WeightVersionInfo(
                version_id=self._next_weight_version_id,
                source_training_step=source_training_step,
                source_epoch=source_epoch,
                activated_at=f"2026-03-19T00:00:{self._next_weight_version_id:02d}Z",
                model_source=f"debug-serving://version-{self._next_weight_version_id}",
                origin=origin,
            )
            self._next_weight_version_id += 1
            return self.current_weight_version()

        def current_weight_version(self) -> WeightVersionInfo:
            return self._active_weight_version.model_copy(deep=True)

        def export_weight_version_state(self) -> dict[str, object]:
            return {
                "schema_version": 1,
                "next_version_id": self._next_weight_version_id,
            }

        def restore_weight_version_state(self, state: dict[str, object] | None) -> None:
            if not isinstance(state, dict):
                return
            next_version_id = state.get("next_version_id")
            if isinstance(next_version_id, int):
                self._next_weight_version_id = max(next_version_id, self._next_weight_version_id)

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


def test_function_rollout_generator_stamps_one_weight_version_on_all_outputs() -> None:
    """Function rollout wrapping should stamp serving provenance centrally."""
    serving_backend = TinyServingBackend()
    rollout = FunctionRolloutGenerator(
        rollout_fn=make_rollout_fn(response_suffix="weights", repeat=1),
        serving_backend=serving_backend,
        config=SimpleNamespace(),
    )

    _, rollouts, _, _ = rollout.generate_grouped(
        [Prompt(text="prompt 0"), Prompt(text="prompt 1")],
        group_size=2,
    )

    expected = serving_backend.current_weight_version().model_dump()
    assert rollouts
    assert all(sample.metadata["weight_version"] == expected for sample in rollouts)


def test_grpo_rejects_mixed_rollout_weight_versions_in_one_batch() -> None:
    """One learner batch must not mix rollout candidates from different serving versions."""
    trainer = build_trainer(batch_size=2, group_size=2)
    conversation = Conversation(
        messages=[
            Message(role="user", content="prompt"),
            Message(role="assistant", content="response"),
        ]
    )
    rollout_a = RolloutOutput(
        text="response-a",
        log_prob=-0.2,
        prompt_token_ids=[1, 2],
        response_token_ids=[3, 4],
        response_token_logprobs=[-0.1, -0.1],
        conversation=conversation,
        metadata={
            "weight_version": WeightVersionInfo(
                version_id=3,
                source_training_step=3,
                source_epoch=1,
                activated_at="2026-03-19T00:00:03Z",
                model_source="tiny-serving://version-3",
                origin="sync",
            ).model_dump()
        },
    )
    rollout_b = RolloutOutput(
        text="response-b",
        log_prob=-0.2,
        prompt_token_ids=[1, 2],
        response_token_ids=[5, 6],
        response_token_logprobs=[-0.1, -0.1],
        conversation=conversation,
        metadata={
            "weight_version": WeightVersionInfo(
                version_id=4,
                source_training_step=4,
                source_epoch=1,
                activated_at="2026-03-19T00:00:04Z",
                model_source="tiny-serving://version-4",
                origin="sync",
            ).model_dump()
        },
    )

    with pytest.raises(RuntimeError, match="mixed multiple serving weight versions"):
        trainer._build_batch_metadata([rollout_a, rollout_b])


def test_flashrl_rejects_batch_size_not_divisible_by_group_size(tmp_path) -> None:
    """FlashRL should fail fast when grouped GRPO cannot form full prompt groups."""
    with pytest.raises(ValueError, match="divisible by grpo.group_size"):
        FlashRL(
            actor_config=TrainingConfig(model_name="fake/model"),
            serving_config=ServingConfig(model_name="fake/model"),
            trainer_config=TrainerConfig(batch_size=3, max_epochs=1),
            grpo_config=GrpoConfig(group_size=2),
            rollout_fn=make_rollout_fn(response_suffix="invalid", repeat=1),
            reward_fn=reward_fn,
            logging_config=LoggingConfig(log_dir=tmp_path, console=False, file=True),
            metrics_config=MetricsConfig(enabled=False),
        )
