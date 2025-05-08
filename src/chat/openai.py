import os
from typing import Any, Dict, List, Optional

from openai import Client

from src.chat.base import ChatMessage, LLMClientInterface, TokenUsage
from src.tool.base import BaseTool


class OpenAIClient(LLMClientInterface):
    """Client for OpenAI API"""

    def __init__(
        self,
        model_id: str,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ):
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        super().__init__(model_id=model_id, api_key=api_key, **kwargs)
        self.client = Client(api_key=api_key)

    def chat(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from: Optional[List[BaseTool]] = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> ChatMessage:
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            grammar=grammar,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            temperature=temperature,
            **kwargs,
        )
        response = self.client.chat.completions.create(**completion_kwargs)

        if self.track_usage:
            # Extract token usage information
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens

            # Update cumulative token usage
            self.token_usages.append(
                TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )
            )
            # Log token usage
            self.token_logger.info(
                f"Token Usage - Model: {self.model_id}, Prompt: {prompt_tokens}, "
                f"Completion: {completion_tokens}, Total: {total_tokens}"
            )

        response_message = ChatMessage.from_dict(
            response.choices[0].message.model_dump(
                include={"role", "content", "tool_calls"}
            )
        )
        return self.post_process_message(
            message=response_message, tools_to_call_from=tools_to_call_from
        )


class OpenAIReasoningClient(LLMClientInterface):
    """Client for OpenAI Reasoning API like o1, o3-mini"""

    def __init__(
        self,
        model_id: str,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ):
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        super().__init__(model_id=model_id, api_key=api_key, **kwargs)
        self.client = Client(api_key=api_key)

    def chat(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from: Optional[List[BaseTool]] = None,
        **kwargs: Any,
    ) -> ChatMessage:
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            grammar=grammar,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            **kwargs,
        )
        completion_kwargs.pop(
            "temperature", None
        )  # OpenAI's Reasoning model doesn't support temperature.
        response = self.client.chat.completions.create(**completion_kwargs)

        if self.track_usage:
            # Extract token usage information
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens

            # Update cumulative token usage
            self.token_usages.append(
                TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )
            )

            # Log token usage
            self.token_logger.info(
                f"Token Usage - Model: {self.model_id}, Prompt: {prompt_tokens}, "
                f"Completion: {completion_tokens}, Total: {total_tokens}"
            )

        response_message = ChatMessage.from_dict(
            response.choices[0].message.model_dump(
                include={"role", "content", "tool_calls"}
            )
        )
        return self.post_process_message(
            message=response_message, tools_to_call_from=tools_to_call_from
        )
