import argparse
import logging

from dotenv import load_dotenv

from src.agent.react import SQLReActAgent
from src.chat.openai import OpenAIClient
from src.database.connector import SqliteDatabaseConnector
from src.utils.logger import init_logger

logger = init_logger(name="main")
load_dotenv()


def main() -> None:
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
    args = parser.parse_args()

    # Configure agent logger based on args
    agent_logger = logging.getLogger("agent")
    if args.agent_verbose:
        agent_logger.setLevel(logging.INFO)
    else:
        agent_logger.setLevel(logging.ERROR)  # Only show errors from agent

    db_connector = SqliteDatabaseConnector(args.database)
    db_connector.connect()
    client = OpenAIClient(model_id=args.model_id)

    agent = SQLReActAgent(
        db_connector=db_connector,
        model_id=args.model_id,
        client=client,
        prompt_file_path="src/prompts/react.yaml",
        prompt_key="prompt",
        max_iterations=args.max_iterations,
        verbose=args.agent_verbose,
    )

    result = agent.process(args.query)

    # Always show these results, regardless of verbose setting
    logger.info(f"Answer: {result['answer']}")
    logger.info(f"SQL Query: {result['query']}")
    logger.info(f"Total Turns: {result['turns']}")

    # Show detailed results only if verbose is enabled
    if args.verbose:
        logger.info(f"History: {result['history']}")

    db_connector.close()  # TODO: Refactor to use context manager


if __name__ == "__main__":
    main()
