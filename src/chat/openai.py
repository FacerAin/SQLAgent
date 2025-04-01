import os
from typing import Optional

from openai import Client

from src.chat.base import LLMClientInterface


class OpenAIClient(LLMClientInterface):
    def __init__(self, model_id: str, api_key: Optional[str] = None):
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        super().__init__(model_id=model_id, api_key=api_key)
        self.client = Client(api_key=api_key)

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
        return str(response.choices[0].message.content.strip())
