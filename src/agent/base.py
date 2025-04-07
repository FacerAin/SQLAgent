import json
import logging
import textwrap
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from src.agent.exceptions import AgentError
from src.chat.base import LLMClientInterface, MessageRole
from src.memory.base import ActionStep, AgentMemory, PlanningStep
from src.prompts.base import PromptTemplates, populate_template
from src.tool.base import BaseTool, FinalAnswerTool
from src.utils.logger import init_logger


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
        logger: Optional[logging.Logger] = None,
        log_to_file: bool = False,
        log_dir: str = "logs",
    ):
        self.max_steps = max_steps
        self.client = client
        self.log_to_file = log_to_file
        self.log_dir = log_dir
        self.logger = (
            init_logger(name="agent", log_dir=log_dir, log_to_file=log_to_file)
            if logger is None
            else logger
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
        self.step_num = 1

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
            self.logger.debug(f"Plan: {plan}")
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

    @abstractmethod
    def run(self, task: str, max_steps: int = 10, stream: bool = False) -> Any:
        pass

    @abstractmethod
    def step(self, memory_step: ActionStep) -> Any:
        pass
