"""Direct unit tests for model wrappers."""

from __future__ import annotations

from types import SimpleNamespace
from queue import Queue

import pytest
import torch

import flashrl.framework.models.actor as actor_module
from flashrl.framework.config import ModelConfig, ServingConfig
from flashrl.framework.models.actor import ActorModel
from tests.conftest import TinyCausalLM, TinyTokenizer

pytestmark = pytest.mark.unit


def patch_hf_loaders(
    monkeypatch: pytest.MonkeyPatch,
    module,
    *,
    tokenizer: TinyTokenizer | None = None,
    model: TinyCausalLM | None = None,
) -> tuple[TinyCausalLM, TinyTokenizer]:
    """Patch one model module to use tiny offline HF stand-ins."""
    tiny_tokenizer = tokenizer or TinyTokenizer()
    tiny_model = model or TinyCausalLM(tiny_tokenizer.vocab_size)
    monkeypatch.setattr(module, "get_device", lambda device=None: torch.device("cpu"))
    monkeypatch.setattr(
        module,
        "AutoModelForCausalLM",
        SimpleNamespace(from_pretrained=lambda *args, **kwargs: tiny_model),
    )
    monkeypatch.setattr(
        module,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *args, **kwargs: tiny_tokenizer),
    )
    return tiny_model, tiny_tokenizer


def test_actor_model_generate_returns_completion_only_and_restores_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generation should decode only completion tokens and restore tokenizer padding."""
    tiny_model, tiny_tokenizer = patch_hf_loaders(
        monkeypatch,
        actor_module,
        tokenizer=TinyTokenizer(pad_token=None),
    )

    actor = ActorModel(ModelConfig(model_name="fake/model", device="cpu", max_length=16))
    outputs = actor.generate(["ab", "xyz"])

    assert outputs == ["decoded::30,31", "decoded::30,31"]
    assert tiny_tokenizer.calls[0]["padding_side"] == "left"
    assert actor.tokenizer.padding_side == "right"
    assert actor.tokenizer.pad_token == actor.tokenizer.eos_token
    assert actor.tokenizer.pad_token_id == actor.tokenizer.eos_token_id
    assert tiny_model.last_generate_kwargs is not None
    assert tiny_model.last_generate_kwargs["input_ids"].shape[0] == 2
    assert tiny_model.last_generate_kwargs["pad_token_id"] == actor.tokenizer.eos_token_id
    assert tiny_model.last_generate_kwargs["eos_token_id"] == actor.tokenizer.eos_token_id


def test_actor_model_generation_defaults_merge_with_call_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-call generation kwargs should override defaults while preserving other defaults."""
    tiny_model, _ = patch_hf_loaders(monkeypatch, actor_module)
    actor = ActorModel(ModelConfig(model_name="fake/model", device="cpu"))
    actor.set_generation_defaults(max_new_tokens=7, top_p=0.95, do_sample=True)

    actor.generate(["prompt"], top_p=0.7)

    assert tiny_model.last_generate_kwargs is not None
    assert tiny_model.last_generate_kwargs["max_new_tokens"] == 7
    assert tiny_model.last_generate_kwargs["top_p"] == pytest.approx(0.7)
    assert tiny_model.last_generate_kwargs["do_sample"] is True
    assert tiny_model.last_generate_kwargs["num_return_sequences"] == 1


def test_actor_model_generate_grouped_returns_prompt_major_structured_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Grouped generation should return prompt-major candidates with ids and token logprobs."""
    tiny_model, _ = patch_hf_loaders(monkeypatch, actor_module)
    actor = ActorModel(ModelConfig(model_name="fake/model", device="cpu"))

    grouped = actor.generate_grouped(["ab", "xyz"], group_size=2, temperature=0.8)

    assert len(grouped) == 2
    assert [len(candidates) for candidates in grouped] == [2, 2]
    assert tiny_model.last_generate_kwargs is not None
    assert tiny_model.last_generate_kwargs["num_return_sequences"] == 2
    assert tiny_model.last_generate_kwargs["temperature"] == pytest.approx(0.8)

    first = grouped[0][0]
    assert first.text == "decoded::30,31"
    assert first.prompt_token_ids
    assert first.response_token_ids == [30, 31]
    assert len(first.response_token_logprobs) == len(first.response_token_ids)
    assert first.log_prob == pytest.approx(sum(first.response_token_logprobs))


def test_actor_model_generate_batch_returns_structured_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Batch generation should return one structured sample per prompt."""
    tiny_model, _ = patch_hf_loaders(monkeypatch, actor_module)
    actor = ActorModel(ModelConfig(model_name="fake/model", device="cpu"))

    samples = actor.generate_batch(["ab", "xyz"], temperature=0.6)

    assert len(samples) == 2
    assert tiny_model.last_generate_kwargs is not None
    assert tiny_model.last_generate_kwargs["num_return_sequences"] == 1
    assert tiny_model.last_generate_kwargs["temperature"] == pytest.approx(0.6)
    assert all(sample.prompt_token_ids for sample in samples)
    assert all(len(sample.response_token_logprobs) == len(sample.response_token_ids) for sample in samples)


def test_actor_model_generation_metadata_marks_eos_completion_as_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Actor metadata should infer a normal stop when generation ends at EOS."""
    tiny_model, _ = patch_hf_loaders(monkeypatch, actor_module)

    actor = ActorModel(ModelConfig(model_name="fake/model", device="cpu"))
    samples = actor.generate_batch(["prompt"])

    assert samples[0].metadata["finish_reason"] == "stop"
    assert tiny_model.last_generate_kwargs is not None
    assert "stopping_criteria" not in tiny_model.last_generate_kwargs


def test_actor_model_debug_live_rollout_uses_sequential_path_and_emits_timings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Debug live rollout should switch to sequential generation and attach timing metadata."""

    class FakeStreamer:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            self._queue: Queue[object] = Queue()

        def push(self, text: str) -> None:
            self._queue.put(text)

        def end(self) -> None:
            self._queue.put(StopIteration)

        def __iter__(self):
            return self

        def __next__(self) -> str:
            item = self._queue.get(timeout=1.0)
            if item is StopIteration:
                raise StopIteration
            return str(item)

    tiny_model, _ = patch_hf_loaders(monkeypatch, actor_module)
    monkeypatch.setattr(actor_module, "TextIteratorStreamer", FakeStreamer)

    original_generate = tiny_model.generate

    def generate_with_streamer(**kwargs):
        streamer = kwargs.pop("streamer")
        streamer.push("partial ")
        streamer.push("text")
        streamer.end()
        return original_generate(**kwargs)

    tiny_model.generate = generate_with_streamer  # type: ignore[method-assign]

    perf_values = iter([10.0, 10.2, 10.8])
    monkeypatch.setattr(actor_module.time, "perf_counter", lambda: next(perf_values))

    events: list[tuple[str, dict[str, object]]] = []
    actor = ActorModel(
        ServingConfig(
            model_name="fake/model",
            device="cpu",
            debug_live_rollout=True,
        )
    )
    actor.set_live_rollout_debug(
        lambda kind, payload: events.append((kind, payload)),
        {"step": 1, "epoch": 1, "total_epochs": 1, "batch_index": 1, "batches_in_epoch": 1, "prompt_count": 1, "group_size": 1},
    )

    grouped = actor.generate_grouped(["ab"], group_size=1)

    assert len(grouped) == 1
    assert len(grouped[0]) == 1
    sample = grouped[0][0]
    assert sample.metadata["ttft_seconds"] == pytest.approx(0.2)
    assert sample.metadata["tpot_seconds"] == pytest.approx(0.6)
    assert sample.metadata["generation_seconds"] == pytest.approx(0.8)
    assert sample.metadata["response_token_count"] == len(sample.response_token_ids)
    assert tiny_model.last_generate_kwargs is not None
    assert tiny_model.last_generate_kwargs["pad_token_id"] == actor.tokenizer.pad_token_id
    assert [kind for kind, _ in events] == ["start", "chunk", "chunk", "done"]


def test_actor_model_compute_log_probs_returns_logits_on_target_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """compute_log_probs should route tensors through the configured actor device."""
    tiny_model, _ = patch_hf_loaders(monkeypatch, actor_module)
    actor = ActorModel(ModelConfig(model_name="fake/model", device="cpu"))

    logits = actor.compute_log_probs(
        input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
        attention_mask=torch.tensor([[1, 1, 1]], dtype=torch.long),
        labels=torch.tensor([[1, 2, 3]], dtype=torch.long),
    )

    assert logits.shape == (1, 3, 32)
    assert tiny_model.last_input_ids is not None
    assert tiny_model.last_input_ids.device.type == "cpu"
    assert tiny_model.last_labels is not None
    assert tiny_model.last_labels.device.type == "cpu"
