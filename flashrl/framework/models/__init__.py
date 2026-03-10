"""Model wrappers and utilities."""

from flashrl.framework.models.actor import ActorModel
from flashrl.framework.models.critic import CriticModel
from flashrl.framework.models.device import get_device
from flashrl.framework.models.reference import ReferenceModel

__all__ = [
    "ActorModel",
    "CriticModel",
    "ReferenceModel",
    "get_device",
]
