from typing import Any, Dict, Type

from src.chat.base import LLMClientInterface
from src.chat.openai import OpenAIClient, OpenAIReasoningClient


class ChatModelFactory:
    MODEL_MAP: Dict[str, Type[LLMClientInterface]] = {
        "gpt-4o": OpenAIClient,
        "gpt-4": OpenAIClient,
        "gpt-4o-mini": OpenAIClient,
        "o3-mini": OpenAIReasoningClient,
    }

    @classmethod
    def load_model(cls, model_id: str, **kwargs: Any) -> LLMClientInterface:
        """
        Returns an instance of the model client based on the model_id.
        """
        if model_id not in cls.MODEL_MAP:
            available_models = list(cls.MODEL_MAP.keys())
            raise ValueError(
                f"Model {model_id} is not supported. Available models: {available_models}"
            )

        client_class = cls.MODEL_MAP[model_id]
        return client_class(model_id=model_id, **kwargs)
