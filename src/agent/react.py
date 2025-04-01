import re
from typing import Any, Dict, Tuple, Union

import pandas as pd

from src.agent.base import BaseAgent
from src.chat.base import LLMClientInterface
from src.database.connector import BaseDatabaseConnector
from src.utils.logger import init_logger

logger = init_logger()


class SQLReActAgent(BaseAgent):
    """
    SQL ReAct agent that uses reasoning and SQL execution
    to answer questions about databases
    """

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
        super().__init__(
            model_id=model_id,
            client=client,
            prompt_file_path=prompt_file_path,
            prompt_key=prompt_key,
            max_iterations=max_iterations,
            verbose=verbose,
        )
        self.db_connector = db_connector

    def get_db_schema(self) -> str:
        """Retrieve and format database schema information"""
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

    def extract_sql_query(self, text: str) -> str:
        """Extract SQL query from code blocks or plain text"""
        if not text:
            return ""

        sql_match = re.search(r"```sql\n(.*?)\n```", text, re.DOTALL)
        if sql_match:
            return sql_match.group(1).strip()
        return text.strip()

    def execute_query(self, query: str) -> Tuple[bool, Union[pd.DataFrame, str]]:
        """Execute SQL query and return results or error message"""
        try:
            if not self.db_connector.connection:
                self.db_connector.connect()

            results = self.db_connector.execute_query(query)

            if (
                isinstance(results, pd.DataFrame)
                and "error" in results.columns
                and len(results) > 0
            ):
                error_message = results["error"].iloc[0]
                return False, error_message

            return True, results
        except Exception as e:
            return False, str(e)

    def _execute_and_format_query(
        self, sql_query: str
    ) -> Tuple[bool, Union[pd.DataFrame, str], str]:
        """Execute query and format results for display"""
        success, query_result = self.execute_query(sql_query)

        if not success:
            return False, query_result, str(query_result)

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

    def _build_initial_context(self, question: str, db_schema: str) -> str:
        """Build the initial prompt context with question and schema"""
        return f"""Question: {question}

        Database Schema:
        {db_schema}

        Start the ReAct process to answer this question. Begin with a <reasoning> tag to explain your approach, then use the <sql> tag to write the necessary SQL query.
        """

    def _create_query_history_entry(
        self, turn: int, query: str, success: bool, result_info: str
    ) -> Dict:
        """Create history entry for a query execution"""
        if success:
            return {
                "turn": turn,
                "action": "sql_query",
                "query": query,
                "success": True,
                "result_preview": result_info[:500] if result_info else "Empty result",
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
        """Format observation based on query execution result"""
        if success:
            return f"\nExecution Results: \n{result_info}\n"
        else:
            return f"\nExecution Error: \n{result_info}\n"

    def _get_next_instruction(self, success: bool) -> str:
        """Get next instruction based on query execution success"""
        if success:
            return "If you need additional analysis or another query, explain using the <reasoning> tag. If you're ready to provide a final answer, use the <answer> tag."
        else:
            return "Please fix the error and try again. Write your query inside the <sql> tag."

    def process(self, question: str) -> Dict[str, Any]:
        """Process a question to generate SQL queries and answers"""
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
                system_prompt=self.prompt,
                user_prompt=context,
            )

            reasoning = self.extract_section(llm_response, "reasoning")
            sql_section = self.extract_section(llm_response, "sql")
            answer = self.extract_section(llm_response, "answer")
            if reasoning:
                self._log_debug_info(reasoning, "Reasoning")
                history.append(
                    {"turn": turns, "action": "reasoning", "reasoning": reasoning}
                )

            if sql_section:
                sql_query = self.extract_sql_query(sql_section)
                final_query = sql_query

                self._log_debug_info(sql_query, "SQL Query")

                success, query_result, result_str = self._execute_and_format_query(
                    sql_query
                )

                if success:
                    final_result = query_result

                history.append(
                    self._create_query_history_entry(
                        turns,
                        sql_query,
                        success,
                        result_str,
                    )
                )

                observation = self._format_observation(success, result_str)

            # Add context for agent
            next_instruction = self._get_next_instruction(
                success if sql_section else True
            )
            observation = observation if sql_section else ""
            context += f"\n{llm_response}\n{observation}\n{next_instruction}"

            if answer:
                self._log_debug_info(answer, "Final Answer")
                final_answer = answer

                history.append({"turn": turns, "action": "answer", "answer": answer})
                break

        if not final_answer:
            final_answer = "None"
            self._log_debug_info(final_answer, "Max attempts reached. Final answer")

        return {
            "question": question,
            "answer": final_answer,
            "query": final_query,
            "result": final_result,
            "turns": turns,
            "history": history,
            "success": bool(final_answer),
            "system_prompt": self.prompt,
            "context": context,
        }
