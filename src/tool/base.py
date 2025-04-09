import json
import textwrap
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, Union

import pandas as pd

from src.database.connector import BaseDatabaseConnector


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
    A tool for providing the final answer. Return ONLY the exact value from the SQL query without interpretation.
    If the query returns 'po', answer 'po', not 'oral'. If unanswerable, respond with 'Unanswerable'.
    """
    parameters = {
        "answer": {"type": "string", "description": "The final answer string."}
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


class SQLTool(BaseTool):
    name = "sql"
    description = "A tool for executing SQL queries. The result will be returned as a string.  Maximum result set is limited to 100 rows. If the query fails, an error message will be returned."
    parameters = {
        "query": {"type": "string", "description": "The SQL query to execute."}
    }
    output_type = str

    def __init__(
        self, db_connector: BaseDatabaseConnector, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.db_connector = db_connector

    def forward(self, query: str) -> str:
        """
        Execute a SQL query and return the result as a string.
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


class PythonTool(BaseTool):
    name = "python"
    description = """
    A Python execution tool that lets you leverage Python's data processing capabilities with database access.

    Key features:
    - The `db_connector.execute_query(query)` function executes SQL queries and returns pandas DataFrames
        Example: df = db_connector.execute_query('SELECT * FROM users LIMIT 10')
    - You can use all pandas operations on the returned DataFrames (filtering, grouping, joining, etc.)
    - Utilize Python's computational power for statistical analysis, data transformation, and visualization
    - Import and use packages like numpy, scipy, matplotlib that are available in the environment

    The tool will return the value you assign to the '_result' variable.
    Example: _result = {"stats": avg_values, "top_users": top_users_df}
    """
    parameters = {
        "code": {
            "type": "string",
            "description": "Python code to execute. Assign to '_result' to return values.",
        }
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

    def forward(self, code: str) -> str:  # noqa: C901
        """
        Execute Python code and return the value assigned to _result.
        Handles SQLite thread limitations.
        """
        import signal
        import sys
        import traceback
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
            error_trace = traceback.format_exc()
            return f"Error: {str(e)}\n\n{error_trace}"
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
