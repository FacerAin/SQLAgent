import sqlite3
import pandas as pd

class DatabaseConnector:
    def __init__(self, db_path):
        self.db_path = db_path
        self.connection = None

    def connect(self):
        self.connection = sqlite3.connect(self.db_path)
        return self.connection
    
    def get_tables(self):
        cursor = self.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        return [table[0] for table in cursor.fetchall()]
    
    def get_table_schema(self, table_name):
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
    
    def execute_query(self, query):
        try:
            return pd.read_sql_query(query, self.connection)
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
    def close(self):
        if self.connection:
            self.connection.close()