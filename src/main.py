from dotenv import load_dotenv

from src.agent.base import SQLReActAgent
from src.chat.openai import OpenAIClient
from src.database.connector import SqliteDatabaseConnector
from src.utils.logger import init_logger

logger = init_logger()
load_dotenv()


def main() -> None:
    model_id = "gpt-4o-mini"
    query = "What is the average age of patients in the database?"
    db_connector = SqliteDatabaseConnector("data/mimic_iii/mimic_iii.db")
    db_connector.connect()
    client = OpenAIClient(model_id=model_id)

    agent = SQLReActAgent(
        db_connector=db_connector,
        model_id=model_id,
        client=client,
        prompt_file_path="src/prompts/react.yaml",
        prompt_key="prompt",
        max_iterations=3,
        verbose=True,
    )

    result = agent.process(query)

    logger.info(f"Answer: {result['answer']}")
    logger.info(f"SQL Query: {result['query']}")
    logger.info(f"Total Turns: {result['turns']}")

    db_connector.close()  # TODO: Refactor to use context manager


if __name__ == "__main__":
    main()
