import json
import uuid
from abc import abstractmethod
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from src.tool.base import BaseTool, get_tool_json_schema
from src.utils.load import parse_json_blob


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL_CALL = "tool-call"
    TOOL_RESPONSE = "tool-response"

    @classmethod
    def roles(cls) -> list[str]:
        """Get all roles as a list of strings.
        Returns:
            List of role strings
        """
        return [r.value for r in cls]


tool_role_conversions = {
    MessageRole.TOOL_CALL: MessageRole.ASSISTANT,
    MessageRole.TOOL_RESPONSE: MessageRole.USER,
}


@dataclass
class ToolCallDefinition:
    arguments: Any
    name: str
    description: Optional[str] = None


@dataclass
class ChatMessageToolCall:
    function: ToolCallDefinition
    id: str
    type: str


def get_tool_call_from_text(
    text: str, tool_name_key: str, tool_arguments_key: str
) -> ChatMessageToolCall:
    tool_call_dictionary, _ = parse_json_blob(text)
    try:
        tool_name = tool_call_dictionary[tool_name_key]
    except Exception as e:
        raise ValueError(
            f"Key {tool_name_key=} not found in the generated tool call. Got keys: {list(tool_call_dictionary.keys())} instead"
        ) from e
    tool_arguments = tool_call_dictionary.get(tool_arguments_key, None)
    if not isinstance(tool_arguments, dict):
        tool_arguments = json.loads(tool_arguments)  # type: ignore
    return ChatMessageToolCall(
        function=ToolCallDefinition(arguments=tool_arguments, name=tool_name),
        id=str(uuid.uuid4()),
        type="function",
    )


def get_dict_from_nested_dataclasses(obj: Any, ignore_key: Optional[Any] = None) -> Any:
    def convert(obj: Any) -> Any:
        if hasattr(obj, "__dataclass_fields__"):
            return {k: convert(v) for k, v in asdict(obj).items() if k != ignore_key}
        return obj

    return convert(obj)


@dataclass
class ChatMessage:
    role: MessageRole
    content: Optional[str] = None
    tool_calls: Optional[List[ChatMessageToolCall]] = None

    def model_dump_json(self) -> str:
        return json.dumps(get_dict_from_nested_dataclasses(self, ignore_key="raw"))

    @classmethod
    def from_dict(cls, data: dict) -> "ChatMessage":
        if data.get("tool_calls"):
            tool_calls = [
                ChatMessageToolCall(
                    function=ToolCallDefinition(**tc["function"]),
                    id=tc["id"],
                    type=tc["type"],
                )
                for tc in data["tool_calls"]
            ]
            data["tool_calls"] = tool_calls
        return cls(**data)

    def dict(self) -> str:
        return json.dumps(get_dict_from_nested_dataclasses(self))


def parse_json_if_needed(arguments: Union[str, dict]) -> Union[str, dict]:
    if isinstance(arguments, dict):
        return arguments
    else:
        try:
            return json.loads(arguments)  # type: ignore
        except Exception:
            return arguments


class LLMClientInterface:
    def __init__(
        self,
        model_id: str,
        api_key: Optional[str] = None,
        tool_name_key: str = "name",
        tool_arguments_key: str = "arguments",
        **kwargs: Any,
    ):
        self.model_id = model_id
        self.api_key = api_key
        self.tool_name_key = tool_name_key
        self.tool_arguments_key = tool_arguments_key
        self.kwargs = kwargs

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from: Optional[List[BaseTool]] = None,
        **kwargs: Any,
    ) -> ChatMessage:
        """Generate a response from the LLM based on the provided prompts.

        Args:
            prompt: The main prompt from the user
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

    def _prepare_completion_kwargs(
        self,
        messages: List[ChatMessage],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from: Optional[List[BaseTool]] = None,
        **kwargs: Any,
    ) -> Dict:
        """
        Prepare parameters required for model invocation, handling parameter priorities.

        Parameter priority from high to low:
        1. Explicitly passed kwargs
        2. Specific parameters (stop_sequences, grammar, etc.)
        3. Default values in self.kwargs
        """
        # Apply role conversion for tool calls
        for message in messages:
            if message["role"] in tool_role_conversions:
                message["role"] = tool_role_conversions[message["role"]]

        # Use self.kwargs as the base configuration
        completion_kwargs = {
            **self.kwargs,
            "messages": messages,
        }

        # Handle specific parameters
        if stop_sequences is not None:
            completion_kwargs["stop"] = stop_sequences
        if grammar is not None:
            completion_kwargs["grammar"] = grammar

        # Handle tools parameter
        if tools_to_call_from:
            completion_kwargs.update(
                {
                    "tools": [
                        get_tool_json_schema(tool) for tool in tools_to_call_from
                    ],
                    "tool_choice": "required",
                }
            )

        # Finally, use the passed-in kwargs to override all settings
        completion_kwargs.update(kwargs)

        return completion_kwargs

    def post_process_message(
        self, message: ChatMessage, tools_to_call_from: Optional[List[BaseTool]]
    ) -> ChatMessage:
        message.role = MessageRole.ASSISTANT  # Overwrite role if needed
        if tools_to_call_from:
            if not message.tool_calls:
                message.tool_calls = [
                    get_tool_call_from_text(
                        message.content, self.tool_name_key, self.tool_arguments_key  # type: ignore
                    )
                ]
            for tool_call in message.tool_calls:
                tool_call.function.arguments = parse_json_if_needed(
                    tool_call.function.arguments
                )
        return message
