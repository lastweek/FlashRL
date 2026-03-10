"""Critic model (value model) wrapper."""

from typing import Any
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from flashrl.framework.config import ModelConfig
from flashrl.framework.models.device import get_device


class CriticModel:
    """Wrapper for the critic (value) model."""

    def __init__(self, config: ModelConfig) -> None:
        """Initialize the critic model.

        Args:
            config: Model configuration.
        """
        self.config = config
        self.device = get_device(config.device)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=1,
            torch_dtype=torch.float32,
            device_map=None,
            trust_remote_code=config.trust_remote_code,
        )
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=config.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def compute_values(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute value estimates for given inputs.

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.

        Returns:
            Value estimates tensor.
        """
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        return outputs.logits.squeeze(-1)

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
