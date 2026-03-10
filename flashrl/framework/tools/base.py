"""Base tool executor abstraction."""

from abc import ABC, abstractmethod
from typing import Any

from flashrl.framework.data_models import ToolCall, ToolResult


class BaseToolExecutor(ABC):
    """Abstract base class for tool execution."""

    @abstractmethod
    def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call.

        Args:
            tool_call: Tool invocation request.

        Returns:
            Tool execution result.
        """
        pass

    @abstractmethod
    def execute_batch(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute multiple tool calls.

        Args:
            tool_calls: List of tool invocation requests.

        Returns:
            List of tool execution results.
        """
        pass

    @abstractmethod
    def list_available_tools(self) -> list[str]:
        """List available tools.

        Returns:
            List of tool names.
        """
        pass

    @abstractmethod
    def tool_exists(self, tool_name: str) -> bool:
        """Check if a tool exists.

        Args:
            tool_name: Name of the tool.

        Returns:
            True if tool exists, False otherwise.
        """
        pass
