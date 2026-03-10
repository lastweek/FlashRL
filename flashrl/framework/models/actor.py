"""Actor model (policy model) wrapper."""

from typing import Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from flashrl.framework.config import ModelConfig
from flashrl.framework.models.device import get_device


class ActorModel:
    """Wrapper for the actor (policy) model."""

    def __init__(self, config: ModelConfig) -> None:
        """Initialize the actor model.

        Args:
            config: Model configuration.
        """
        self.config = config
        self.device = get_device(config.device)

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float32,
            device_map=None,
            trust_remote_code=config.trust_remote_code,
        )
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=config.trust_remote_code,
            padding_side="right",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompts: list[str], **kwargs: Any) -> list[str]:
        """Generate text from prompts.

        Args:
            prompts: List of input prompts.
            **kwargs: Additional generation arguments.

        Returns:
            List of generated texts.
        """
        # Use left padding for generation (required for decoder-only models)
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **kwargs)

        self.tokenizer.padding_side = original_padding_side

        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return generated_texts

    def compute_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probabilities for given inputs and labels.

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            labels: Target token IDs for computing log probs.

        Returns:
            Log probabilities tensor.
        """
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        labels = labels.to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs.logits

    def to(self, device: torch.device) -> None:
        """Move model to device.

        Args:
            device: Target device.
        """
        self.device = device
        self.model.to(device)

    def train(self) -> None:
        """Set model to training mode."""
        self.model.train()

    def eval(self) -> None:
        """Set model to evaluation mode."""
        self.model.eval()
