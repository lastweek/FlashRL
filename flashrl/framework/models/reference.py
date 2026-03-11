"""Reference model wrapper for optional KL regularization."""

from typing import Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from flashrl.framework.config import ModelConfig
from flashrl.framework.models.device import get_device


class ReferenceModel:
    """Frozen reference model used for KL-regularized GRPO."""

    def __init__(self, config: ModelConfig) -> None:
        """Initialize the reference model.

        Args:
            config: Model configuration.
        """
        self.config = config
        self.device = get_device(config.device)
        dtype = getattr(torch, config.dtype)

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            dtype=dtype,
            device_map=None,
            trust_remote_code=config.trust_remote_code,
        )
        self.model.to(self.device)
        self.model.eval()  # Always in eval mode

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=config.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def compute_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probabilities (no gradients).

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            labels: Target token IDs for computing log probs.

        Returns:
            Log probabilities tensor (no gradients).
        """
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        labels = labels.to(self.device)

        with torch.no_grad():
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
