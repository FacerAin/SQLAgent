import argparse

from dotenv import load_dotenv

from src.agent.react import SQLReActAgent
from src.chat.openai import OpenAIClient
from src.database.connector import SqliteDatabaseConnector
from src.utils.logger import init_logger

logger = init_logger()
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
    args = parser.parse_args()

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
        verbose=True,
    )

    result = agent.process(args.query)

    logger.info(f"Answer: {result['answer']}")
    logger.info(f"SQL Query: {result['query']}")
    logger.info(f"Total Turns: {result['turns']}")

    db_connector.close()  # TODO: Refactor to use context manager


if __name__ == "__main__":
    main()
