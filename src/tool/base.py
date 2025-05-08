import json
import textwrap
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, Union

import pandas as pd

from src.context import context_sample
from src.database.connector import BaseDatabaseConnector
from src.utils.sql import extract_tables_from_query


class BaseTool(ABC):
    """
    Base class for all tools.
    """

    name: str
    description: str
    parameters: Dict[str, Dict[str, Union[str, type, bool]]]
    output_type: Any

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Forward method to be implemented by subclasses.
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the tool to a dictionary representation.
        """
        return {
            "name": self.name,
            "description": json.dumps(textwrap.dedent(self.description).strip()),
            "parameters": repr(self.parameters),
            "output_type": self.output_type,
        }

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # Handle the arguments might be passed as a single dictionary
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], dict):
            potential_kwargs = args[0]

            # If the dictionary keys match our input parameters, convert it to kwargs
            if all(key in self.parameters for key in potential_kwargs):
                args = ()
                kwargs = potential_kwargs

        outputs = self.forward(*args, **kwargs)
        return outputs

    def __repr__(self) -> str:
        """
        String representation of the tool.
        """
        return f"Tool(name={self.name}, description={self.description}, parameters={self.parameters}, output_type={self.output_type})"


class FinalAnswerTool(BaseTool):
    name = "final_answer"
    description = """
     A tool for providing the final answer to the query. Answers should be precise and concise (examples: po / True / 7 / 12.12 / 2103-12-23 07:20:00 / ceftriaxone, azithromycin, ciprofloxacin).

    Guidelines for good answers:
    - Use exact terminology as it appears in the query results
    - Keep answers brief and to the point
    - For date/time values, maintain the exact format shown in results (YYYY-MM-DD HH:MM:SS)
    - For medications or multiple items, separate with commas
    - For numeric values, maintain the same precision as shown in results

    If the question requires information not available in the database schema, respond with 'Unanswerable'.
    """
    parameters = {
        "answer": {"type": "string", "description": "The final answer string."},
        "thought": {
            "type": "string",
            "description": "Explain your reasoning for choosing this action and what you expect to accomplish with it.",
        },
    }
    output_type = str

    def __init__(
        self, use_retrieval_knowledge_only: bool = True, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        if use_retrieval_knowledge_only:
            self.description += textwrap.dedent(
                """
                Use ONLY information retrieved from the database, not your background knowledge.
                """
            )

    def forward(self, answer: str) -> Any:
        return str(answer)


class CurrentDateTool(BaseTool):
    name = "current_date"
    description = """
    A tool for providing the current date. This tool is used to get the current date in a specific format.
    You don't trust the code or the model's answer. You should use this tool to get the current date.
    The date format is 'YYYY-MM-DD HH:MM:SS'.
    """
    parameters = {
        "thought": {
            "type": "string",
            "description": "Explain your reasoning for choosing this action and what you expect to accomplish with it.",
        }
    }
    output_type = str

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, **kwargs) -> Any:
        return "2105-12-31 23:59:00"


class SQLTool(BaseTool):
    name = "sql"
    description = "A tool for executing SQL queries. The result will be returned as a string.  Maximum result set is limited to 100 rows. If the query fails, an error message will be returned."
    parameters = {
        "query": {
            "type": "string",
            "description": "The SQL query to execute.",
        },
        "thought": {
            "type": "string",
            "description": "Explain your reasoning for choosing this action and what you expect to accomplish with it.",
        },
    }
    output_type = str

    def __init__(
        self, db_connector: BaseDatabaseConnector, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.db_connector = db_connector

    def forward(self, query: str, **kwargs) -> str:
        """
        Execute a SQL query and return the result as a string.
        The result is limited to 100 rows.
        Sql syntax follows Python SQLite syntax.
        If the query fails, return an error message.
        """
        results = self.db_connector.execute_query(query)

        if isinstance(results, pd.DataFrame):
            if "error" in results.columns and len(results) > 0:
                error_message = results["error"].iloc[0]
                return f"Error: {error_message}"

            # Convert DataFrame to string
            results_str = results.head(100).to_string(index=False)
            if results.size > 100:
                results_str += "\n\nNote: Only the first 100 rows are shown."
            # If the DataFrame is empty, return a message
            if results.empty:
                return "No results found."
        elif isinstance(results, str):
            # If results is a string, return it directly
            return results

        return results_str


class OracleTableVerifierTool(BaseTool):
    name = "table_verifier"
    description = """
    A tool for verifying if the correct tables have been selected for a database query.
    This tool compares your selected tables with the tables needed for the correct query solution.
    It returns information about missing tables (that should be included) and irrelevant tables (that are unnecessary).

    IMPORTANT: You MUST use this tool at least once before calling final_answer to ensure you have selected the correct tables.

    After receiving results from this tool:
    1. If "is_missing_table" is True, explore additional tables that might be relevant to the query
    2. If "irrelevant_table_names" contains tables, consider removing them from your query planning
    3. Read the "description" field carefully for specific guidance on improving your query
    4. Use these insights to refine your SQL query before finalizing it
    5. If both checks pass (no missing or irrelevant tables), proceed with confidence in your table selection
    """
    parameters = {
        "table_list": {
            "type": "array",
            "items": {"type": "string"},  # type: ignore
            "description": "List of table names you want to verify as necessary for the query.",
        },
        "thought": {
            "type": "string",
            "description": "Explain your reasoning for selecting these tables and why you believe they're relevant to the query.",
        },
    }
    output_type = str

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, table_list: list[str], **kwargs) -> str:
        table_list = self._preprocess_table_names(table_list)
        eval_sample = context_sample.get()
        gold_sql_query = eval_sample.get("query")

        if eval_sample is None:
            return "Error: No sample provided for verification."

        if gold_sql_query is None or gold_sql_query == "null":
            return str(
                {
                    "is_missing_table": False,
                    "irrelevant_table_names": [],
                    "description": "Your table selection matches what's needed for this query. Now ensure you're using proper join conditions, filtering criteria, and aggregation methods. Don't forget to validate your results for clinical coherence. You're on the right track!",
                }
            )

        gold_tables = self._preprocess_table_names(
            extract_tables_from_query(gold_sql_query)
        )

        is_missing_table = any(list(set(gold_tables) - set(table_list)))
        irrelevant_table_names = list(set(table_list) - set(gold_tables))
        description = self._generate_description(
            is_missing_table, irrelevant_table_names
        )

        # return values
        return str(
            {
                "is_missing_table": is_missing_table,
                "irrelevant_table_names": irrelevant_table_names,
                "description": description,
            }
        )

    def _preprocess_table_names(self, table_list: list[str]) -> list[str]:
        """
        Preprocess table names to ensure they are in a consistent format.
        """
        return [table.lower() for table in table_list]

    def _generate_description(
        self, is_missing_table: bool, irrelevant_table_names: list[str]
    ) -> str:
        """
        Generate detailed natural language feedback based on table selection issues.
        """
        # Case 1: Both missing and irrelevant tables
        if is_missing_table and irrelevant_table_names:
            return "Your query needs significant restructuring. You're missing essential tables while including unnecessary ones. Consider completely rebuilding your query, focusing on the core tables needed to answer the clinical question. Try exploring additional tables that might establish relationships between data points and remove tables that don't contribute to the analysis."

        # Case 2: Missing tables only
        elif is_missing_table:
            return "Your table selection is incomplete. Your query is missing essential tables needed to properly address the question. Explore additional tables that might establish important relationships or provide necessary context for your data. Consider looking at dictionary tables if you're working with codes or identifiers, or patient-related tables if you need demographic or admission information."

        # Case 3: Irrelevant tables only
        elif irrelevant_table_names:
            return (
                "Your query includes tables that may not be necessary: "
                + ", ".join(irrelevant_table_names)
                + ". Consider whether these tables truly add value or if they might introduce incorrect relationships or unnecessary complexity. Simplifying your query by focusing only on essential tables may improve both performance and accuracy."
            )

        # Case 4: Everything looks good
        else:
            return "Your table selection matches what's needed for this query. Now ensure you're using proper join conditions, filtering criteria, and aggregation methods. Don't forget to validate your results for clinical coherence. You're on the right track!"


class PythonTool(BaseTool):
    name = "python"
    description = """
    A Python execution tool with database access for data processing.

    Features:
    - Run SQL queries: df = db_connector.execute_query('SELECT * FROM users LIMIT 10')
    - Use pandas operations on returned DataFrames (filter, group, join)
    - Perform statistical analysis and data transformation
    - No code comments needed
    - If the query fails, return an error message as a DataFrame: return pd.DataFrame({'error': [str(e)]})

    Output via print() or _result variable.
    Example: _result = {"stats": avg_values, "top_users": top_users_df}
    """
    parameters = {
        "code": {
            "type": "string",
            "description": "Python code to execute. Assign to '_result' to return values.",
        },
        "thought": {
            "type": "string",
            "description": "Explain your reasoning for choosing this action and what you expect to accomplish with it.",
        },
    }
    output_type = str

    def __init__(
        self,
        db_connector: BaseDatabaseConnector = None,
        timeout: int = 10,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.db_connector = db_connector
        self.timeout = timeout

    def forward(self, code: str, **kwargs) -> str:  # noqa: C901
        """
        Execute Python code and return the value assigned to _result.
        Handles SQLite thread limitations.
        """
        import signal
        import sys
        from io import StringIO

        import numpy as np
        import pandas as pd

        # Capture stdout
        old_stdout = sys.stdout
        redirected_output = StringIO()
        sys.stdout = redirected_output

        # Setup timeout handler
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Execution timed out after {self.timeout} seconds")

        try:
            # Set time limit
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout)

            # Execute in the current thread to avoid SQLite thread issues
            globals_dict = {
                "db_connector": self.db_connector,
                "pd": pd,
                "np": np,
                "print": print,
            }

            # Add _result variable to capture output
            local_vars = {"_result": None}

            # Execute code
            exec(code, globals_dict, local_vars)

            # Disable time limit
            signal.alarm(0)

            # Get the result
            result = local_vars.get("_result")
            output = redirected_output.getvalue()

            # Format the output
            if result is None:
                if output:
                    return f"Output:\n{output}\n\nNo _result variable was set. Assign a value to _result to return data."
                else:
                    return "No output and no _result variable was set. Use print() for output or assign a value to _result."

            # Handle different return types
            if isinstance(result, pd.DataFrame):
                # Format DataFrame
                if len(result) > 100:
                    result_str = result.head(100).to_string()
                    result_str += f"\n\n[Showing 100 of {len(result)} rows]"
                else:
                    result_str = result.to_string()

                if output:
                    return f"Output:\n{output}\n\nDataFrame Result:\n{result_str}"
                else:
                    return f"DataFrame Result:\n{result_str}"
            else:
                # For other types
                if output:
                    return f"Output:\n{output}\n\nResult: {result}"
                else:
                    return f"Result: {result}"

        except TimeoutError:
            return f"Error: Execution timed out after {self.timeout} seconds"
        except Exception as e:
            return f"Error: {str(e)}"
        finally:
            sys.stdout = old_stdout
            signal.alarm(0)


def get_tool_json_schema(tool: BaseTool) -> Dict:
    properties = deepcopy(tool.parameters)
    required = []
    for key, value in properties.items():
        if value["type"] == "any":
            value["type"] = "string"
        if not ("nullable" in value and value["nullable"]):
            required.append(key)
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }
