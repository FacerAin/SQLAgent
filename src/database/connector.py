import datetime
import os
import sqlite3
from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Optional

import pandas as pd


class BaseDatabaseConnector(ABC):
    __metaclass__ = ABCMeta

    def __init__(self, connection_string: str) -> None:
        self.connection_string: str = connection_string
        self.connection: Optional[Any] = None

    @abstractmethod
    def connect(self) -> sqlite3.Connection:
        """Establish a connection to the database."""
        pass

    @abstractmethod
    def get_tables(self) -> list:
        """Retrieve a list of tables in the database."""
        pass

    @abstractmethod
    def get_table_schema(self, table_name: str) -> dict:
        """Get the schema of a specific table."""
        pass

    @abstractmethod
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute a SQL query and return the result as a DataFrame."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the database connection."""
        pass


class SqliteDatabaseConnector(BaseDatabaseConnector):
    def __init__(self, db_path: str, custom_time: str | None = None) -> None:
        super().__init__(db_path)
        self.db_path = db_path
        self.connection: Optional[sqlite3.Connection] = None
        self.custom_time = custom_time or "2105-12-31 23:59:00"  # Set default value

    def connect(self) -> sqlite3.Connection:
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database file '{self.db_path}' not found.")
        self.connection = sqlite3.connect(self.db_path)

        # Override current_time function
        def override_current_time():
            return self.custom_time

        self.connection.create_function("current_time", 0, override_current_time)

        # Override now function
        def override_now():
            return self.custom_time

        self.connection.create_function("now", 0, override_now)

        # Handle dynamic datetime function with improved date calculations
        def custom_datetime(*args):
            # Return current time if no arguments
            if len(args) == 0:
                return self.custom_time

            # Set base time
            base_time = self.custom_time
            if len(args) >= 1:
                if args[0] != "now" and args[0] != "current_time":
                    base_time = args[0]

            try:
                # Process time calculation
                dt_obj = datetime.datetime.strptime(base_time, "%Y-%m-%d %H:%M:%S")

                # Handle various modifiers
                for modifier in args[1:]:
                    if modifier.startswith("-") or modifier.startswith("+"):
                        parts = modifier.split()
                        if len(parts) == 2:
                            try:
                                amount = int(parts[0])
                                unit = parts[1].lower()

                                if unit == "year" or unit == "years":
                                    dt_obj = dt_obj.replace(year=dt_obj.year + amount)
                                elif unit == "month" or unit == "months":
                                    # Improved month calculation with better boundary handling
                                    month = dt_obj.month - 1 + amount  # 0-based month
                                    year = dt_obj.year + month // 12
                                    month = month % 12 + 1  # back to 1-based month

                                    # Handle day overflow (e.g., Jan 31 -> Feb 28)
                                    last_day = calendar.monthrange(year, month)[1]
                                    day = min(dt_obj.day, last_day)

                                    dt_obj = dt_obj.replace(
                                        year=year, month=month, day=day
                                    )
                                elif unit == "day" or unit == "days":
                                    dt_obj = dt_obj + datetime.timedelta(days=amount)
                            except ValueError as e:
                                # Log error for debugging but return a reasonable result
                                print(f"Date calculation error: {e}")
                                # Fall back to simple calculation for robustness
                                if unit == "month" or unit == "months":
                                    dt_obj = dt_obj + datetime.timedelta(
                                        days=amount * 30
                                    )

                    elif modifier == "start of year":
                        dt_obj = dt_obj.replace(
                            month=1, day=1, hour=0, minute=0, second=0
                        )

                    elif modifier == "start of month":
                        dt_obj = dt_obj.replace(day=1, hour=0, minute=0, second=0)

                return dt_obj.strftime("%Y-%m-%d %H:%M:%S")
            except Exception as e:
                # Robust error handling to prevent function crashes
                print(f"DateTime function error: {e}")
                return base_time  # Return base time if calculation fails

        # Register custom datetime function (with variable arguments)
        self.connection.create_function("datetime", -1, custom_datetime)

        # Enhanced strftime function with better error handling
        def custom_strftime(format_str, timestring=None, *args):
            try:
                if (
                    timestring is None
                    or timestring == "now"
                    or timestring == "current_time"
                ):
                    dt_obj = datetime.datetime.strptime(
                        self.custom_time, "%Y-%m-%d %H:%M:%S"
                    )
                else:
                    # Try multiple time formats
                    formats = [
                        "%Y-%m-%d %H:%M:%S",
                        "%Y-%m-%d",
                        "%Y/%m/%d %H:%M:%S",
                        "%Y/%m/%d",
                    ]

                    success = False
                    for fmt in formats:
                        try:
                            dt_obj = datetime.datetime.strptime(timestring, fmt)
                            success = True
                            break
                        except ValueError:
                            continue

                    if not success:
                        return timestring  # Return original string if parsing fails

                # Apply additional modifiers if present
                if args:
                    # Pass to datetime function for consistent handling
                    modified_time = custom_datetime(timestring, *args)
                    dt_obj = datetime.datetime.strptime(
                        modified_time, "%Y-%m-%d %H:%M:%S"
                    )

                return dt_obj.strftime(format_str)
            except Exception as e:
                # Robust error handling
                print(f"strftime error: {e}")
                return f"{format_str}({timestring})"  # Return a reasonable fallback

        # Register custom strftime with variable arguments
        self.connection.create_function("strftime", -1, custom_strftime)

        # Add BETWEEN helper function to handle between operation more reliably
        def custom_between(value, lower, upper):
            try:
                # Parse dates if they are strings
                if (
                    isinstance(value, str)
                    and isinstance(lower, str)
                    and isinstance(upper, str)
                ):
                    dt_value = datetime.datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
                    dt_lower = datetime.datetime.strptime(lower, "%Y-%m-%d %H:%M:%S")
                    dt_upper = datetime.datetime.strptime(upper, "%Y-%m-%d %H:%M:%S")
                    return dt_lower <= dt_value <= dt_upper
                # Fall back to standard comparison for non-date values
                return lower <= value <= upper
            except Exception:
                # Fall back to string comparison if date parsing fails
                return lower <= value <= upper

        self.connection.create_function("date_between", 3, custom_between)

        # Add utility function for correct month difference calculation
        def months_between(start_date, end_date):
            try:
                start = datetime.datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
                end = datetime.datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")

                months = (end.year - start.year) * 12 + (end.month - start.month)

                # Adjust for day of month
                if end.day < start.day:
                    months -= 1

                return months
            except Exception:
                return 0

        self.connection.create_function("months_between", 2, months_between)

        # Import necessary modules
        import calendar

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
