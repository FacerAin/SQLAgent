from dataclasses import asdict, dataclass
from typing import Any, Dict, List, TypedDict, Union

from src.agent.exceptions import AgentError
from src.chat.base import ChatMessage, MessageRole
from src.utils.load import make_json_serializable


class Message(TypedDict):
    role: MessageRole
    content: str | list[dict]


@dataclass
class ToolCall:
    name: str
    arguments: Any
    id: str

    def dict(self) -> Dict:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": make_json_serializable(self.arguments),
            },
        }


@dataclass
class MemoryStep:
    def dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_messages(self, **kwargs: Any) -> List[Message]:
        raise NotImplementedError


@dataclass
class ActionStep(MemoryStep):
    model_input_messages: List[Message] | None = None
    tool_calls: List[ToolCall] | None = None
    start_time: float | None = None
    end_time: float | None = None
    step_number: int | None = None
    error: AgentError | None = None
    duration: float | None = None
    model_output_message: ChatMessage | None = None
    model_output: str | None = None
    observations: str | None = None
    action_output: Any = None

    def dict(self) -> Dict:
        # We overwrite the method to parse the tool_calls and action_output manually
        return {
            "model_input_messages": self.model_input_messages,
            "tool_calls": (
                [tc.dict() for tc in self.tool_calls] if self.tool_calls else []
            ),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "step": self.step_number,
            "error": self.error.dict() if self.error else None,
            "duration": self.duration,
            "model_output_message": self.model_output_message,
            "model_output": self.model_output,
            "observations": self.observations,
            "action_output": make_json_serializable(self.action_output),
        }

    def to_messages(self, **kwargs: Any) -> List[Message]:
        messages = []
        if self.model_input_messages is not None:
            messages.append(
                Message(role=MessageRole.SYSTEM, content=self.model_input_messages)
            )
        if self.model_output is not None:
            messages.append(
                Message(
                    role=MessageRole.ASSISTANT,
                    content=[{"type": "text", "text": self.model_output.strip()}],
                )
            )

        if self.tool_calls is not None:
            messages.append(
                Message(
                    role=MessageRole.TOOL_CALL,
                    content=[
                        {
                            "type": "text",
                            "text": "Calling tools:\n"
                            + str([tc.dict() for tc in self.tool_calls]),
                        }
                    ],
                )
            )

        if self.observations is not None:
            messages.append(
                Message(
                    role=MessageRole.TOOL_RESPONSE,
                    content=[
                        {
                            "type": "text",
                            "text": (
                                f"Call id: {self.tool_calls[0].id}\n"
                                if self.tool_calls
                                else ""
                            )
                            + f"Observation:\n{self.observations}",
                        }
                    ],
                )
            )
        if self.error is not None:
            error_message = (
                "Error:\n"
                + str(self.error)
                + "\nNow let's retry: take care not to repeat previous errors! If you have retried several times, try a completely different approach.\n"
            )
            message_content = (
                f"Call id: {self.tool_calls[0].id}\n" if self.tool_calls else ""
            )
            message_content += error_message
            messages.append(
                Message(
                    role=MessageRole.TOOL_RESPONSE,
                    content=[{"type": "text", "text": message_content}],
                )
            )
        return messages


@dataclass
class PlanningStep(MemoryStep):
    model_input_messages: List[Message]
    model_output_message: ChatMessage
    plan: str

    def to_messages(self, **kwargs: Any) -> List[Message]:
        return [
            Message(
                role=MessageRole.ASSISTANT,
                content=[{"type": "text", "text": self.plan.strip()}],
            ),
            Message(
                role=MessageRole.USER,
                content=[{"type": "text", "text": "Now go on and execute this plan."}],
            ),
        ]


@dataclass
class TaskStep(MemoryStep):
    task: str

    def to_messages(self, **kwargs: Any) -> List[Message]:
        content = [{"type": "text", "text": f"New task:\n{self.task}"}]

        return [Message(role=MessageRole.USER, content=content)]


@dataclass
class SystemPromptStep(MemoryStep):
    system_prompt: str

    def to_messages(self, **kwargs: Any) -> List[Message]:
        return [
            Message(
                role=MessageRole.SYSTEM,
                content=[{"type": "text", "text": self.system_prompt}],
            )
        ]


@dataclass
class FinalAnswerStep(MemoryStep):
    final_answer: Any


class AgentMemory:
    def __init__(self, system_prompt: str):
        self.system_prompt = SystemPromptStep(system_prompt=system_prompt)
        self.steps: List[Union[TaskStep, ActionStep, PlanningStep]] = []

    def reset(self) -> None:
        self.steps = []

    def get_succinct_steps(self) -> list[dict]:
        return [
            {
                key: value
                for key, value in step.dict().items()
                if key != "model_input_messages"
            }
            for step in self.steps
        ]

    def get_full_steps(self) -> list[dict]:
        return [step.dict() for step in self.steps]
