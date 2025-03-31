import sqlite3
from typing import Optional

import pandas as pd


class DatabaseConnector:
    def __init__(self, db_path: str) -> None:
        self.db_path: str = db_path
        self.connection: Optional[sqlite3.Connection] = None

    def _ensure_connection(self) -> None:
        if not self.connection:
            raise ConnectionError("Database connection is not established.")

    def connect(self) -> sqlite3.Connection:
        self.connection = sqlite3.connect(self.db_path)
        self._ensure_connection()
        return self.connection

    def get_tables(self) -> list:
        if not self.connection:
            raise ConnectionError("Database connection is not established.")
        cursor = self.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        return [table[0] for table in cursor.fetchall()]

    def get_table_schema(self, table_name: str) -> dict:
        if not self.connection:
            raise ConnectionError("Database connection is not established.")
        schemas = {}
        cursor = self.connection.cursor()

        if table_name:
            tables = [table_name]
        else:
            tables = self.get_tables()
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table});")
            schemas[table] = cursor.fetchall()
        return schemas

    def execute_query(self, query: str) -> pd.DataFrame:
        if not self.connection:
            raise ConnectionError("Database connection is not established.")
        try:
            return pd.read_sql_query(query, self.connection)
        except Exception as e:
            # If the query fails, return an Dataframe with an error message
            return pd.DataFrame({"error": [str(e)]})

    def close(self) -> None:
        if self.connection:
            self.connection.close()
