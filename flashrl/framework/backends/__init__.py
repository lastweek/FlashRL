"""Training and serving backends."""

from flashrl.framework.backends.training import TrainingBackend
from flashrl.framework.backends.serving import ServingBackend

__all__ = ["TrainingBackend", "ServingBackend"]
