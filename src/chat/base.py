from abc import abstractmethod
from typing import Dict, Optional


class LLMClientInterface:
    def __init__(self, model_id: str, api_key: Optional[str] = None):
        self.model_id = model_id
        self.api_key = api_key

    @abstractmethod
    def chat(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> str:
        """Generate a response from the LLM based on the provided prompts.

        Args:
            user_prompt: The main prompt from the user
            system_prompt: Optional system instructions
            temperature: Controls randomness (0 = deterministic)

        Returns:
            The LLM's response as a string
        """
        pass

    def get_token_usage(self) -> Dict[str, int]:
        """Get the cumulative token usage statistics.

        Returns:
            Dictionary with token usage statistics (if implemented)
        """
        return {}
