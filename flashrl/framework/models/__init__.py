"""Model wrappers and utilities."""

from flashrl.framework.models.actor import ActorModel
from flashrl.framework.models.device import get_device, set_num_threads
from flashrl.framework.models.reference import ReferenceModel

__all__ = [
    "ActorModel",
    "ReferenceModel",
    "get_device",
    "set_num_threads",
]
