import argparse
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv

from src.agent.base import BaseAgent
from src.agent.react import ToolReActAgent
from src.chat.base import LLMClientInterface
from src.chat.factory import ChatModelFactory
from src.database.connector import BaseDatabaseConnector, SqliteDatabaseConnector
from src.prompts.base import PromptTemplates
from src.tool.base import FinalAnswerTool, SQLTool
from src.utils.logger import init_logger

load_dotenv()


@dataclass
class QueryResult:
    """Container for query processing results."""

    # Query and response
    query: str
    answer: str

    # Performance metrics
    total_turns: int

    # Conversation history
    history: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Timestamp
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query": self.query,
            "answer": self.answer,
            "total_turns": self.total_turns,
            "history": self.history,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


class AgentContext:
    """Context manager to handle logger setup and resources for the agent."""

    SUPPORTED_AGENT_TYPES = {
        "sql_react": ["SQLTool", "FinalAnswerTool"],
    }

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.logger: Optional[logging.Logger] = None
        self.db_connector: Optional[BaseDatabaseConnector] = None
        self.client: Optional[LLMClientInterface] = None
        self.agent: Optional[BaseAgent] = None
        self.prompt_templates: Optional[PromptTemplates] = None

    def __enter__(self) -> "AgentContext":
        """Set up loggers and resources when entering the context."""
        # Initialize main logger
        self.logger = init_logger(
            name="main", log_to_file=self.args.log_to_file, log_dir=self.args.log_dir
        )

        assert self.logger is not None, "Logger initialization failed."

        # Configure agent logger based on args
        agent_logger = init_logger(
            name="agent",
            log_to_file=self.args.log_to_file,
            log_dir=self.args.log_dir,
        )
        if self.args.agent_verbose:
            agent_logger.setLevel(logging.DEBUG)
        else:
            agent_logger.setLevel(logging.ERROR)  # Only show errors from agent

        # Initialize database connector
        custom_time = None
        if (
            hasattr(self.args, "custom_time")
            and self.args.custom_time
            and self.args.custom_time != "None"
        ):
            custom_time = self.args.custom_time

        self.db_connector = SqliteDatabaseConnector(
            self.args.database, custom_time=custom_time
        )
        self.db_connector.connect()

        # Initialize model client
        self.client = ChatModelFactory.load_model(model_id=self.args.model_id)

        # Set up agent
        self._setup_agent(agent_logger)

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Close resources when exiting the context."""
        if self.db_connector:
            self.db_connector.close()
            self.logger.debug("Database connection closed")

    def _setup_agent(self, agent_logger: logging.Logger) -> None:
        """Set up the agent with appropriate tools."""
        if hasattr(self.args, "agent_type") and self.args.agent_type:
            # Setup ToolReActAgent with specified tools if agent_type is provided
            self._setup_tool_react_agent(agent_logger)

    def _setup_tool_react_agent(self, agent_logger: logging.Logger) -> None:
        """Set up a ToolReActAgent with appropriate tools."""
        # Create tool instances
        tool_instances = {
            "SQLTool": SQLTool(db_connector=self.db_connector),
            "FinalAnswerTool": FinalAnswerTool(),
        }

        # Select tools based on agent type
        if self.args.agent_type not in self.SUPPORTED_AGENT_TYPES:
            raise ValueError(f"Unsupported agent type: {self.args.agent_type}")

        tools = [
            tool_instances[tool_name]
            for tool_name in self.SUPPORTED_AGENT_TYPES[self.args.agent_type]
        ]

        # Load prompt templates
        prompt_path = Path(self.args.prompt_path)
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt_templates = yaml.safe_load(f)

        # Create agent instance
        self.agent = ToolReActAgent(
            client=self.client,
            tools=tools,
            max_steps=self.args.max_iterations,
            prompt_templates=self.prompt_templates,
            logger=agent_logger,
            log_to_file=self.args.log_to_file,
            log_dir=self.args.log_dir,
        )


class QueryProcessor:
    """Encapsulates functionality related to query processing."""

    @staticmethod
    def process_query(context: AgentContext, query: str) -> QueryResult:
        answer = context.agent.run(task=query)
        total_turns = context.agent.step_num
        history = context.agent.write_memory_to_messages()[1:]  # type: ignore

        # Create and return result
        return QueryResult(
            query=query,
            answer=answer,
            total_turns=total_turns,
            history=history,  # type: ignore
            metadata={
                "model_id": context.args.model_id,
                "agent_type": getattr(context.args, "agent_type", "sql_react"),
            },
        )


class ResultHandler:
    """Encapsulates functionality related to result processing and display."""

    @staticmethod
    def log_results(context: AgentContext, result: QueryResult) -> None:
        """Log the results of the query processing."""
        # Always show these results, regardless of verbose setting
        context.logger.info(f"Answer: {result.answer}")
        context.logger.info(f"Total Turns: {result.total_turns}")

        # Show detailed results only if verbose is enabled
        if context.args.verbose:
            context.logger.info(f"History: {result.history}")


def parse_arguments() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Process SQL query using a language model agent."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="gpt-4o-mini",
        help="The model ID to use for the client.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="what is lidocaine 5% ointment's way of ingesting it?",
        help="The query prompt to process.",
    )
    parser.add_argument(
        "--database",
        type=str,
        default="data/mimic_iii/mimic_iii.db",
        help="Path to the database file.",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=5,
        help="Maximum number of reasoning iterations for the agent",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose evaluation logging."
    )
    parser.add_argument(
        "--agent_verbose", action="store_true", help="Enable verbose agent logging."
    )
    parser.add_argument(
        "--log_to_file", action="store_true", help="Enable logging to file."
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory to store log files.",
    )
    parser.add_argument(
        "--agent_type",
        type=str,
        default="sql_react",
        help="Type of agent to use (sql_react).",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="src/prompts/react.yaml",
        help="Path to the prompt file.",
    )
    parser.add_argument(
        "--custom_time",
        type=str,
        default=None,
        help="Custom time to use for the database connection. If you don't want to use it, set it to 'None'.",
    )
    return parser.parse_args()


def main() -> None:
    """Main function to coordinate the query processing."""
    # Parse arguments
    args = parse_arguments()

    # Use context manager for the agent
    with AgentContext(args) as context:
        # Process the query
        result = QueryProcessor.process_query(context, args.query)

        # Log the results
        ResultHandler.log_results(context, result)


if __name__ == "__main__":
    main()
