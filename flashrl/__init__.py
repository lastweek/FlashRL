"""FlashRL: A learning-first RL project for LLM post-training."""

__version__ = "0.0.1"

from flashrl.flashrl import FlashRL
from flashrl.framework.config import LoggingConfig

__all__ = ["FlashRL", "LoggingConfig"]
