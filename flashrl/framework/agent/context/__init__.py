"""Context-management primitives for agent rollouts."""

from flashrl.framework.agent.context.base import BaseContextManager
from flashrl.framework.agent.context.compaction import CompactionManager, CompactionPolicy
from flashrl.framework.agent.context.windowed import WindowedContextManager

__all__ = [
    "BaseContextManager",
    "CompactionManager",
    "CompactionPolicy",
    "WindowedContextManager",
]
