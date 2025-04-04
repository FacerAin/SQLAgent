import json
import textwrap
import time
from abc import ABC
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Dict, Generator, List, Optional, TypedDict, Union

from jinja2 import StrictUndefined, Template

from src.chat.base import ChatMessage, LLMClientInterface, MessageRole
from src.tool.base import BaseTool, FinalAnswerTool
from src.utils.load import make_json_serializable
from src.utils.logger import init_logger


def populate_template(template: str, variables: Dict[str, Any]) -> str:
    compiled_template = Template(template, undefined=StrictUndefined)
    try:
        return compiled_template.render(**variables)
    except Exception as e:
        raise Exception(
            f"Error during jinja template rendering: {type(e).__name__}: {e}"
        )


class Message(TypedDict):
    role: MessageRole
    content: str | list[dict]


@dataclass
class MemoryStep:
    def dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_messages(self, **kwargs: Any) -> List[Message]:
        raise NotImplementedError


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


class AgentError(Exception):
    """Base class for other agent-related exceptions"""

    def __init__(self, message: Any) -> None:
        super().__init__(message)
        self.message = message

    def dict(self) -> Dict[str, str]:
        return {"type": self.__class__.__name__, "message": str(self.message)}


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


class PlanningPromptTemplate(TypedDict):
    """
    Prompt templates for the planning step.

    Args:
        initial_facts (`str`): Initial facts prompt.
        initial_plan (`str`): Initial plan prompt.
        update_facts_pre_messages (`str`): Update facts pre-messages prompt.
        update_facts_post_messages (`str`): Update facts post-messages prompt.
        update_plan_pre_messages (`str`): Update plan pre-messages prompt.
        update_plan_post_messages (`str`): Update plan post-messages prompt.
    """

    initial_facts: str
    initial_plan: str
    update_facts_pre_messages: str
    update_facts_post_messages: str
    update_plan_pre_messages: str
    update_plan_post_messages: str


class FinalAnswerPromptTemplate(TypedDict):
    """
    Prompt templates for the final answer.

    Args:
        pre_messages (`str`): Pre-messages prompt.
        post_messages (`str`): Post-messages prompt.
    """

    pre_messages: str
    post_messages: str


class PromptTemplates(TypedDict):
    """
    Prompt templates for the agent.

    Args:
        system_prompt (`str`): System prompt.
        planning ([`~agents.PlanningPromptTemplate`]): Planning prompt templates.
        final_answer ([`~agents.FinalAnswerPromptTemplate`]): Final answer prompt templates.
    """

    system_prompt: str
    planning: PlanningPromptTemplate
    final_answer: FinalAnswerPromptTemplate


class BaseAgent(ABC):
    """
    Base class for all types of agents that interact with LLMs
    """

    def __init__(
        self,
        client: LLMClientInterface,
        tools: List[BaseTool],
        prompt_templates: PromptTemplates,
        max_steps: int = 3,
        verbose: bool = False,
        log_to_file: bool = False,
        log_dir: str = "logs",
    ):
        self.max_steps = max_steps
        self.verbose = verbose
        self.client = client
        self.log_to_file = log_to_file
        self.log_dir = log_dir
        self.logger = init_logger(
            name="agent", log_dir=log_dir, log_to_file=log_to_file
        )
        self.task: Optional[str] = None
        self.tools = self._setup_tools(tools=tools)
        self.prompt_templates = prompt_templates
        self.system_prompt = populate_template(
            self.prompt_templates["system_prompt"],
            variables={"tools": self.tools},
        )
        self.memory = AgentMemory(self.system_prompt)
        self.input_messages = None
        self.state: Dict = {}

    def _substitute_state_variables(
        self, arguments: Union[Dict[str, str], str]
    ) -> Union[Dict[str, Any], str]:
        """Replace string values in arguments with their corresponding state values if they exist."""
        if isinstance(arguments, dict):
            return {
                key: self.state.get(value, value) if isinstance(value, str) else value
                for key, value in arguments.items()
            }
        return arguments

    def execute_tool_call(
        self, tool_name: str, arguments: Union[Dict[str, str], str]
    ) -> Any:
        """
        Execute a tool or managed agent with the provided arguments.

        The arguments are replaced with the actual values from the state if they refer to state variables.

        Args:
            tool_name (`str`): Name of the tool or managed agent to execute.
            arguments (dict[str, str] | str): Arguments passed to the tool call.
        """
        self.logger.info(f"Executing tool call: {tool_name}")
        # Check if the tool exists
        available_tools = {**self.tools}
        if tool_name not in available_tools:
            error_msg = f"Unknown tool {tool_name}, should be one of: {', '.join(available_tools)}."
            self.logger.error(error_msg)
            raise AgentError(error_msg)

        # Get the tool and substitute state variables in arguments
        tool = available_tools[tool_name]
        arguments = self._substitute_state_variables(arguments)
        self.logger.debug(f"Tool arguments after substitution: {arguments}")

        try:
            # Call tool with appropriate arguments
            self.logger.info(f"Calling tool: {tool_name}")
            self.logger.info(
                f"[Action progress] Starting execution of tool: {tool_name}"
            )
            start_time = time.time()
            result = None
            if isinstance(arguments, dict):
                result = tool(**arguments)
            elif isinstance(arguments, str):
                result = tool(arguments)
            else:
                raise TypeError(f"Unsupported arguments type: {type(arguments)}")

            execution_time = time.time() - start_time
            self.logger.info(f"Tool {tool_name} executed in {execution_time:.2f}s")
            self.logger.info(
                f"[Action progress] Completed execution of tool: {tool_name} in {execution_time:.2f}s"
            )
            return result

        except TypeError as e:
            # Handle invalid arguments
            description = getattr(tool, "description", "No description")

            error_msg = (
                f"Invalid call to tool '{tool_name}' with arguments {json.dumps(arguments)}: {e}\n"
                "You should call this tool with correct input arguments.\n"
                f"Expected inputs: {json.dumps(tool.parameters)}\n"
                f"Returns output type: {tool.output_type}\n"
                f"Tool description: '{description}'"
            )
            self.logger.error(f"Tool execution error: {error_msg}")
            self.logger.info(
                f"[Action progress] Failed execution of tool: {tool_name} due to argument error"
            )
            raise AgentError(error_msg) from e

    def step(self, memory_step: ActionStep) -> Any:
        memory_messages = self.write_memory_to_messages()
        self.logger.info("Getting model response")
        self.logger.info("[Action progress] Requesting response from model")
        try:
            model_message = self.client.chat(
                memory_messages,
                tools_to_call_from=list(self.tools.values()),
                stop_sequences=["Observation:", "Calling tools:"],
            )
            memory_step.model_output_message = model_message
            self.logger.debug(f"Model output: {model_message.content}")
            self.logger.info("[Action progress] Received response from model")
        except Exception as e:
            self.logger.error(f"Error while generating model message: {e}")
            self.logger.info("[Action progress] Failed to get response from model")
            raise AgentError(f"Error while generating model message: {e}")

        if model_message.tool_calls is None or len(model_message.tool_calls) == 0:
            self.logger.error("Model did not call any tools")
            self.logger.info(
                "[Action progress] Model response did not include any tool calls"
            )
            raise AgentError("Model did not call any tools.")

        tool_call = model_message.tool_calls[0]
        tool_name, tool_call_id = tool_call.function.name, tool_call.id
        tool_arguments = tool_call.function.arguments
        self.logger.info(f"Model called tool: {tool_name}")
        self.logger.info(f"[Action progress] Model selected tool: {tool_name}")

        memory_step.tool_calls = [
            ToolCall(name=tool_name, arguments=tool_arguments, id=tool_call_id)
        ]

        # Execute
        if tool_name == "final_answer":
            self.logger.info("Final answer tool called")
            self.logger.info("[Action progress] Processing final answer")
            if isinstance(tool_arguments, dict):
                if "answer" in tool_arguments:
                    answer = tool_arguments["answer"]
                else:
                    answer = tool_arguments
            else:
                answer = tool_arguments
            if (
                isinstance(answer, str) and answer in self.state.keys()
            ):  # if the answer is a state variable, return the value
                final_answer = self.state[answer]
            else:
                final_answer = answer

            memory_step.action_output = final_answer
            self.logger.info("Final answer generated")
            self.logger.info("[Action progress] Final answer generated successfully")
            return final_answer
        else:
            if tool_arguments is None:
                tool_arguments = {}
            self.logger.info(
                f"Executing tool: {tool_name} with arguments: {tool_arguments}"
            )
            observation = self.execute_tool_call(tool_name, tool_arguments)
            updated_information = str(observation).strip()
            memory_step.observations = updated_information
            self.logger.info(f"Tool execution completed for: {tool_name}")
            return None

    def _log_debug_info(self, content: str, label: str) -> None:
        """Log debug information if verbose mode is enabled"""
        if self.verbose and content:
            self.logger.info(f"{label}: {content}")

    def _setup_tools(self, tools: List[BaseTool]) -> Dict:
        self.logger.info("Setting up tools")
        setup_tools = {tool.name: tool for tool in tools}
        if not setup_tools:
            self.logger.info("No tools provided, using default FinalAnswerTool")
            setup_tools = {"final_answer": FinalAnswerTool()}
        else:
            self.logger.info(f"Available tools: {', '.join(setup_tools.keys())}")
        return setup_tools

    def _initialize_system_prompt(self) -> str:
        self.logger.info("Initializing system prompt")
        system_prompt = populate_template(
            self.prompt_templates["system_prompt"], variables={"tools": self.tools}
        )
        self.logger.debug("System prompt initialized")
        return system_prompt

    def _create_planning_step(self, task: str) -> PlanningStep:
        self.logger.info("Creating planning step")
        input_messages = [
            {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["planning"]["initial_plan"],
                            variables={"task": task, "tools": self.tools},
                        ),
                    }
                ],
            }
        ]

        self.logger.info("Getting plan from model")
        try:
            plan_message = self.client.chat(input_messages)
            self.logger.debug(f"Plan message received: {plan_message.content[:100]}...")
            plan = textwrap.dedent(
                f"""Here are the facts I know and the plan of action that I will follow to solve the task:\n```\n{plan_message}\n```"""
            )
            self.logger.info("Planning step created successfully")
            return PlanningStep(
                model_input_messages=input_messages,
                plan=plan,
                model_output_message=plan_message,
            )
        except Exception as e:
            self.logger.error(f"Error creating planning step: {e}")
            raise

    def _create_action_step(self, start_time: float) -> ActionStep:
        self.logger.info(f"Creating action step {self.step_num}")
        action_step = ActionStep(
            model_input_messages=self.input_messages,
            start_time=start_time,
            step_number=self.step_num,
        )
        return action_step

    def _execute_step(self, task: str, memory_step: ActionStep) -> Any:
        self.logger.info(f"Executing step {memory_step.step_number}")
        self.logger.info(f"[Action progress] Starting step {memory_step.step_number}")
        final_answer = self.step(memory_step)
        if final_answer is not None:
            self.logger.info("Step produced final answer")
            self.logger.info("[Action progress] Step produced final answer")
        else:
            self.logger.info(
                f"[Action progress] Completed step {memory_step.step_number} without final answer"
            )
        return final_answer

    def _finalize_step(self, memory_step: ActionStep, start_time: float) -> None:
        memory_step.end_time = time.time()
        memory_step.duration = memory_step.end_time - start_time
        self.logger.info(
            f"Finalized step {memory_step.step_number} (duration: {memory_step.duration:.2f}s)"
        )
        self.logger.info(
            f"[Action progress] Finalized step {memory_step.step_number} (took {memory_step.duration:.2f}s)"
        )

    def write_memory_to_messages(
        self,
    ) -> List[Dict[str, str]]:
        self.logger.debug("Writing memory to messages")
        messages = self.memory.system_prompt.to_messages()
        for memory_step in self.memory.steps:
            messages.extend(memory_step.to_messages())
        self.logger.debug(f"Generated {len(messages)} messages from memory")
        return messages

    def provide_final_answer(self, task: str) -> str:
        self.logger.info("Providing final answer")
        messages = [
            {
                "role": MessageRole.SYSTEM,
                "content": [
                    {
                        "type": "text",
                        "text": self.prompt_templates["final_answer"]["pre_messages"],
                    }
                ],
            }
        ]

        messages += self.write_memory_to_messages()[1:]

        messages += [
            {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["final_answer"]["post_messages"],
                            variables={"task": task},
                        ),
                    }
                ],
            }
        ]
        self.logger.info("Requesting final answer from model")
        try:
            final_message = self.client.chat(messages)
            self.logger.info("Final answer received from model")
            self.logger.debug(f"Final answer: {final_message.content[:100]}...")
            return final_message
        except Exception as e:
            self.logger.error(f"Error while generating final answer: {e}")
            raise AgentError(f"Error while generating final answer: {e}")

    def _handle_max_steps(self, task: str, start_time: time) -> str:
        self.logger.warning(
            f"Max steps ({self.step_num}) reached without finding answer"
        )
        self.logger.info("Providing final answer through fallback mechanism")
        final_answer = self.provide_final_answer(task)
        final_memory_step = ActionStep(
            step_number=self.step_num,
            error=AgentError("Max steps reached"),
        )
        final_memory_step.action_output = final_answer
        final_memory_step.end_time = time.time()
        final_memory_step.duration = final_memory_step.end_time - start_time
        self.memory.steps.append(final_memory_step)
        self.logger.info("Final answer generated through fallback mechanism")
        return final_answer

    def run(self, task: str, max_steps: int = 10, stream: bool = False) -> Any:
        self.logger.info(f"Starting agent run with task: {task}")
        self.task = task
        self.system_prompt = self._initialize_system_prompt()
        self.memory.system_prompt = SystemPromptStep(system_prompt=self.system_prompt)
        self.memory.steps.append(TaskStep(task=self.task))
        if stream:
            self.logger.info("Running in stream mode")
            return self._run(task=task, max_steps=max_steps)

        # Get the last step from the generator and check if it's a FinalAnswerStep
        self.logger.info(f"Running with max_steps: {max_steps}")
        last_step = deque(self._run(task=task, max_steps=max_steps), maxlen=1)[0]
        if isinstance(last_step, FinalAnswerStep):
            self.logger.info("Run completed with final answer")
            return str(last_step.final_answer)
        else:
            # If it's not a FinalAnswerStep, return some default value or error message
            self.logger.info("Run completed without final answer step")
            return (
                str(last_step.action_output)
                if hasattr(last_step, "action_output")
                and last_step.action_output is not None
                else ""
            )

    def _run(
        self, task: str, max_steps: int
    ) -> Generator[ActionStep | FinalAnswerStep, None, None]:
        final_answer = None
        self.step_num = 1
        self.logger.info(f"Starting _run with task: {task}")
        while self.step_num <= max_steps and final_answer is None:
            start_time = time.time()
            self.logger.info(f"Executing step {self.step_num}/{max_steps}")
            if self.step_num == 1:
                self.logger.info("Creating planning step")
                planning_step = self._create_planning_step(task)
                self.memory.steps.append(planning_step)
                yield planning_step
            action_step = self._create_action_step(start_time)
            try:
                self.logger.info(f"Executing step {self.step_num}")
                final_answer = self._execute_step(task, action_step)
                if final_answer is not None:
                    self.logger.info("Final answer received")
            except AgentError as e:
                self.logger.error(f"Agent error in step {self.step_num}: {e}")
                action_step.error = e
            finally:
                self._finalize_step(action_step, start_time)
                self.memory.steps.append(action_step)
                self.logger.info(
                    f"Step {self.step_num} completed in {action_step.duration:.2f}s"
                )
                self.step_num += 1
                yield action_step
        if final_answer is None:
            self.logger.info(
                "Max steps reached without final answer, handling max steps"
            )
            final_answer = self._handle_max_steps(task=task, start_time=start_time)
            yield action_step
        self.logger.info("Run completed, yielding final answer")
        yield FinalAnswerStep(final_answer=final_answer)
