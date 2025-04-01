from abc import abstractmethod
from typing import Optional


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
        max_tokens: Optional[int] = None,
    ) -> str:
        pass
