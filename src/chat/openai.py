import os
from typing import Dict, Optional

from openai import Client

from src.chat.base import LLMClientInterface
from src.utils.logger import init_token_logger


class OpenAIClient(LLMClientInterface):
    """Client for OpenAI API"""

    def __init__(
        self, model_id: str, api_key: Optional[str] = None, track_usage: bool = True
    ):
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        super().__init__(model_id=model_id, api_key=api_key)
        self.client = Client(api_key=api_key)
        self.track_usage = track_usage
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        # Initialize token usage logger if tracking is enabled
        if self.track_usage:
            self.token_logger = init_token_logger()

    def chat(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if self.track_usage:
            # Extract token usage information
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens

            # Update cumulative token usage
            self.token_usage["prompt_tokens"] += prompt_tokens
            self.token_usage["completion_tokens"] += completion_tokens
            self.token_usage["total_tokens"] += total_tokens

            # Log token usage
            self.token_logger.info(
                f"Token Usage - Model: {self.model_id}, Prompt: {prompt_tokens}, "
                f"Completion: {completion_tokens}, Total: {total_tokens}"
            )

        return str(response.choices[0].message.content.strip())

    def get_token_usage(self) -> Dict[str, int]:
        """Get the cumulative token usage statistics.

        Returns:
            Dictionary with token usage statistics
        """
        return self.token_usage.copy()
