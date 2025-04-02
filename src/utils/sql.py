import re
from typing import Any, List

import sqlglot
from sqlglot import expressions as exp


def extract_tables_from_query(sql_query: str) -> List[str]:
    """
    Extract table names referenced in a SQL query using sqlglot.

    Args:
        sql_query (str): The SQL query string

    Returns:
        List[str]: List of unique table names referenced in the query
    """
    try:
        # Parse the SQL query
        parsed = sqlglot.parse_one(sql_query)

        # Find all table references
        tables = set()

        # Helper function to extract tables from parse tree
        def extract_tables_from_node(node: Any) -> None:
            if isinstance(node, exp.Table):
                tables.add(node.name)

            # Recursively process children
            for child in node.children():
                extract_tables_from_node(child)

        # Start extraction from the parsed tree
        extract_tables_from_node(parsed)

        return list(tables)
    except Exception:
        tables = set()
        matches = re.findall(
            r"(?:FROM|JOIN)\s+([a-zA-Z0-9_]+)", sql_query, re.IGNORECASE
        )
        for match in matches:
            tables.add(match.lower())
        return list(tables)


def count_tables_in_query(sql_query: str) -> int:
    """
    Count the number of unique tables referenced in a SQL query.

    Args:
        sql_query (str): The SQL query string

    Returns:
        int: Number of unique tables in the query
    """
    tables = extract_tables_from_query(sql_query)
    return len(tables)
