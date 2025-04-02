import re
from abc import ABC, abstractmethod
from typing import Any, Dict

from src.chat.base import LLMClientInterface
from src.utils.load import load_prompt_from_yaml
from src.utils.logger import init_logger

logger = init_logger(name="agent")


class BaseAgent(ABC):
    """
    Base class for all types of agents that interact with LLMs
    """

    def __init__(
        self,
        model_id: str,
        client: LLMClientInterface,
        prompt_file_path: str,
        prompt_key: str = "prompt",
        max_iterations: int = 3,
        verbose: bool = False,
    ):
        self.model_id = model_id
        self.prompt_file_path = prompt_file_path
        self.prompt_key = prompt_key
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.prompt = load_prompt_from_yaml(prompt_file_path, prompt_key)
        self.client = client

    def extract_section(self, text: str, section: str) -> str:
        """Extract content within XML-style tags"""
        pattern = f"<{section}>(.*?)</{section}>"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else ""

    def _log_debug_info(self, content: str, label: str) -> None:
        """Log debug information if verbose mode is enabled"""
        if self.verbose and content:
            logger.info(f"{label}: {content}")

    @abstractmethod
    def process(self, question: str) -> Dict[str, Any]:
        """Process a user question to generate an answer"""
        pass
