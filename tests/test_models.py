"""Direct unit tests for model wrappers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import flashrl.framework.models.actor as actor_module
import flashrl.framework.models.reference as reference_module
from flashrl.framework.config import ModelConfig
from flashrl.framework.models.actor import ActorModel
from flashrl.framework.models.reference import ReferenceModel
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
    assert tiny_model.last_generate_kwargs is not None
    assert tiny_model.last_generate_kwargs["input_ids"].shape[0] == 2


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


def test_reference_model_compute_log_probs_runs_without_grad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reference logits should be computed under no_grad regardless of outer grad state."""
    tiny_model, tiny_tokenizer = patch_hf_loaders(
        monkeypatch,
        reference_module,
        tokenizer=TinyTokenizer(pad_token=None),
    )
    reference = ReferenceModel(ModelConfig(model_name="fake/model", device="cpu"))

    with torch.enable_grad():
        logits = reference.compute_log_probs(
            input_ids=torch.tensor([[1, 2]], dtype=torch.long),
            attention_mask=torch.tensor([[1, 1]], dtype=torch.long),
            labels=torch.tensor([[1, 2]], dtype=torch.long),
        )

    assert logits.shape == (1, 2, 32)
    assert tiny_model.grad_enabled_during_forward is False
    assert tiny_tokenizer.pad_token == tiny_tokenizer.eos_token
