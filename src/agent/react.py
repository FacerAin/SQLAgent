import logging
import time
from collections import deque
from typing import Any, Generator, List, Optional

from src.agent.base import BaseAgent
from src.agent.exceptions import AgentError
from src.chat.base import LLMClientInterface
from src.memory.base import (
    ActionStep,
    FinalAnswerStep,
    PlanningStep,
    SystemPromptStep,
    TaskStep,
    ToolCall,
)
from src.prompts.base import PromptTemplates
from src.tool.base import BaseTool


class ToolReActAgent(BaseAgent):
    def __init__(
        self,
        client: LLMClientInterface,
        tools: List[BaseTool],
        prompt_templates: PromptTemplates,
        max_steps: int = 3,
        logger: Optional[logging.Logger] = None,
        log_to_file: bool = False,
        log_dir: str = "logs",
    ) -> None:
        super().__init__(
            client=client,
            tools=tools,
            prompt_templates=prompt_templates,
            max_steps=max_steps,
            logger=logger,
            log_to_file=log_to_file,
            log_dir=log_dir,
        )

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
    ) -> Generator[ActionStep | FinalAnswerStep | PlanningStep, None, None]:
        final_answer = None
        self.step_num = 1
        self.logger.info(f"Starting _run with task: {task}")
        while self.step_num <= max_steps and final_answer is None:
            start_time = time.time()
            self.logger.info(f"Executing step {self.step_num}/{max_steps}")
            if self.step_num == 1:
                planning_step = self._create_planning_step(task)
                self.memory.steps.append(planning_step)
                yield planning_step
            action_step = self._create_action_step(start_time)
            try:
                final_answer = self._execute_step(task, action_step)
                if final_answer is not None:
                    self.logger.info("Final answer received")
                    self.logger.info(f"Final answer: {final_answer}")
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
