import re
from typing import Any, Dict, Tuple, Union

import pandas as pd

from src.chat.base import LLMClientInterface
from src.database.connector import BaseDatabaseConnector
from src.utils.load import load_prompt_from_yaml
from src.utils.logger import init_logger

logger = init_logger()


class SQLReActAgent:
    def __init__(
        self,
        db_connector: BaseDatabaseConnector,
        model_id: str,
        client: LLMClientInterface,
        prompt_file_path: str,
        prompt_key: str = "prompt",
        max_iterations: int = 3,
        verbose: bool = False,
    ):
        self.db_connector = db_connector
        self.model_id = model_id
        self.prompt_file_path = prompt_file_path
        self.prompt_key = prompt_key
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.prompt = load_prompt_from_yaml(prompt_file_path, prompt_key)
        self.client = client

    def get_db_schema(self) -> str:
        if not self.db_connector.connection:
            try:
                self.db_connector.connect()
            except Exception as e:
                raise ConnectionError(f"Failed to connect to the database: {e}")
        try:
            tables = self.db_connector.get_tables()
        except Exception as e:
            raise ConnectionError(f"Failed to retrieve tables: {e}")

        schema_info = []

        for table in tables:
            try:
                table_schema = self.db_connector.get_table_schema(table)

                columns = []
                for col_info in table_schema.get(table, []):
                    # 0: cid, 1: name, 2: type, 3: notnull, 4: dflt_value, 5: pk
                    col_name = col_info[1]
                    col_type = col_info[2]
                    is_pk = "PRIMARY KEY" if col_info[5] == 1 else ""
                    is_not_null = "NOT NULL" if col_info[3] == 1 else ""

                    column_def = f"{col_name} {col_type} {is_pk} {is_not_null}".strip()
                    columns.append(f"  {column_def}")

                table_def = f"CREATE TABLE {table} (\n" + ",\n".join(columns) + "\n);"
                schema_info.append(table_def)

            except Exception as e:
                schema_info.append(f"Error getting schema for table {table}: {str(e)}")

        return "\n\n".join(schema_info)

    def extract_section(self, text: str, section: str) -> str:
        pattern = f"<{section}>(.*?)</{section}>"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else ""

    def extract_sql_query(self, text: str) -> str:
        if not text:
            return ""

        sql_match = re.search(r"```sql\n(.*?)\n```", text, re.DOTALL)
        if sql_match:
            return sql_match.group(1).strip()
        return text.strip()

    def execute_query(self, query: str) -> Tuple[bool, Union[pd.DataFrame, str]]:
        try:
            if not self.db_connector.connection:
                self.db_connector.connect()

            results = self.db_connector.execute_query(query)

            if "error" in results.columns and len(results) > 0:
                error_message = results["error"].iloc[0]
                return False, error_message

            return True, results
        except Exception as e:
            return False, str(e)

    def process(self, question: str) -> Dict[str, Any]:
        db_schema = self.get_db_schema()
        context = self._build_initial_context(question, db_schema)

        if self.verbose:
            logger.info("Initial Context:")
            logger.info(context)

        history = []
        turns = 0
        final_answer = ""
        final_query = ""
        final_result = None

        while turns < self.max_iterations:
            turns += 1

            llm_response = self.client.chat(
                system_prompt="You are an SQL expert that follows the ReAct (Reasoning and Acting) framework to solve database questions.",
                user_prompt=context,
            )

            reasoning = self.extract_section(llm_response, "reasoning")
            sql_section = self.extract_section(llm_response, "sql")
            analysis = self.extract_section(llm_response, "analysis")
            answer = self.extract_section(llm_response, "answer")

            self._log_debug_info(reasoning, "Reasoning")

            if sql_section:
                sql_query = self.extract_sql_query(sql_section)
                final_query = sql_query

                self._log_debug_info(sql_query, "SQL Query")

                success, query_result, result_str = self._execute_and_format_query(
                    sql_query
                )

                history.append(
                    self._create_query_history_entry(
                        turns,
                        sql_query,
                        success,
                        result_str,
                    )
                )

                observation = self._format_observation(success, result_str)
                next_instruction = self._get_next_instruction(success)
                context += f"\n{llm_response}\n{observation}\n{next_instruction}"

            if answer:
                self._log_debug_info(answer, "Final Answer")
                final_answer = answer

                if analysis:
                    self._log_debug_info(analysis, "Analysis")
                    history.append(
                        {"turn": turns, "action": "analysis", "analysis": analysis}
                    )

                history.append({"turn": turns, "action": "answer", "answer": answer})
                break

        if not final_answer:
            final_answer = f"I'm sorry, I couldn't find an answer after {self.max_iterations} attempts."
            self._log_debug_info(final_answer, "Max attempts reached. Final answer")

        return {
            "question": question,
            "answer": final_answer,
            "query": final_query,
            "result": final_result,
            "turns": turns,
            "history": history,
            "success": bool(final_answer),
        }

    def _build_initial_context(self, question: str, db_schema: str) -> str:
        return f"""Question: {question}

        Database Schema:
        {db_schema}

        Start the ReAct process to answer this question. Begin with a <reasoning> tag to explain your approach, then use the <sql> tag to write the necessary SQL query.
        """

    def _log_debug_info(self, content: str, label: str) -> None:
        if self.verbose and content:
            logger.info(f"{label}: {content}")

    def _execute_and_format_query(
        self, sql_query: str
    ) -> Tuple[bool, Union[pd.DataFrame, str], str]:
        success, query_result = self.execute_query(sql_query)

        if not success:
            return False, query_result, ""

        if isinstance(query_result, pd.DataFrame):
            if not query_result.empty:
                result_str = query_result.head(10).to_string()
                if len(query_result) > 10:
                    result_str += f"\n... and {len(query_result) - 10} more rows"
            else:
                result_str = "No results"
        else:
            result_str = str(query_result)

        if self.verbose:
            logger.info("\nQuery Execution Results:")
            logger.info(result_str)

        return True, query_result, result_str

    def _create_query_history_entry(
        self, turn: int, query: str, success: bool, result_info: str
    ) -> Dict:
        if success:
            return {
                "turn": turn,
                "action": "sql_query",
                "query": query,
                "success": True,
                "result_preview": result_info[:500],
            }
        else:
            return {
                "turn": turn,
                "action": "sql_query",
                "query": query,
                "success": False,
                "error": result_info,
            }

    def _format_observation(self, success: bool, result_info: str) -> str:
        if success:
            return f"\nExcution Results: \n{result_info}\n"
        else:
            return f"\nExecution Error: \n{result_info}\n"

    def _get_next_instruction(self, success: bool) -> str:
        if success:
            return "If you need additional analysis or another query, explain using the <reasoning> tag. If you're ready to provide a final answer, use the <answer> tag."
        else:
            return "Please fix the error and try again. Write your query inside the <sql> tag."
