from typing import Any, Dict


class AgentError(Exception):
    """Base class for other agent-related exceptions"""

    def __init__(self, message: Any) -> None:
        super().__init__(message)
        self.message = message

    def dict(self) -> Dict[str, str]:
        return {"type": self.__class__.__name__, "message": str(self.message)}
