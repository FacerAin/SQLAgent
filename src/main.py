import argparse
import logging
from typing import Any, Dict

from dotenv import load_dotenv

from src.agent.react import SQLReActAgent
from src.chat.factory import ChatModelFactory
from src.database.connector import SqliteDatabaseConnector
from src.utils.logger import init_logger

load_dotenv()


class AgentContext:
    """Context manager to handle logger setup and resources for the agent."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.logger = None
        self.db_connector = None
        self.client = None
        self.agent = None

    def __enter__(self) -> "AgentContext":
        """Set up loggers and resources when entering the context."""
        # Initialize main logger
        self.logger = init_logger(
            name="main", log_to_file=self.args.log_to_file, log_dir=self.args.log_dir
        )

        # Configure agent logger based on args
        agent_logger = logging.getLogger("agent")
        if self.args.agent_verbose:
            agent_logger.setLevel(logging.INFO)
        else:
            agent_logger.setLevel(logging.ERROR)  # Only show errors from agent

        # Initialize database connector
        self.db_connector = SqliteDatabaseConnector(self.args.database)
        self.db_connector.connect()

        # Initialize model client
        self.client = ChatModelFactory.load_model(
            model_id=self.args.model_id,
        )

        # Initialize agent
        self.agent = SQLReActAgent(
            db_connector=self.db_connector,
            model_id=self.args.model_id,
            client=self.client,
            prompt_file_path="src/prompts/react.yaml",
            prompt_key="prompt",
            max_iterations=self.args.max_iterations,
            verbose=self.args.agent_verbose,
        )

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Close resources when exiting the context."""
        if self.db_connector:
            self.db_connector.close()
            self.logger.debug("Database connection closed")


def parse_arguments() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Process SQL query using SQLReActAgent."
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
    return parser.parse_args()


def process_query(context: AgentContext) -> Dict[str, Any]:
    """Process the query and return the result."""
    return context.agent.process(context.args.query)


def log_results(context: AgentContext, result: Dict[str, Any]) -> None:
    """Log the results of the query processing."""
    # Always show these results, regardless of verbose setting
    context.logger.info(f"Answer: {result['answer']}")
    context.logger.info(f"SQL Query: {result['query']}")
    context.logger.info(f"Total Turns: {result['turns']}")

    # Show detailed results only if verbose is enabled
    if context.args.verbose:
        context.logger.info(f"History: {result['history']}")


def main() -> None:
    """Main function to coordinate the query processing."""
    # Parse arguments
    args = parse_arguments()

    # Use context manager for the agent
    with AgentContext(args) as context:
        # Process the query
        result = process_query(context)

        # Log the results
        log_results(context, result)


if __name__ == "__main__":
    main()
