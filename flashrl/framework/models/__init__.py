"""Model wrappers and utilities."""

from flashrl.framework.models.actor import ActorModel
from flashrl.framework.models.device import get_device, set_num_threads

__all__ = [
    "ActorModel",
    "get_device",
    "set_num_threads",
]
