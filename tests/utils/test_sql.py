import unittest
from unittest import TestCase

from src.utils.sql import count_tables_in_query, extract_tables_from_query


class TestTableExtraction(TestCase):
    """Tests for SQL table extraction functions"""

    def test_simple_query(self):
        """Test table extraction from a simple query"""
        query = "SELECT * FROM customers"

        tables = extract_tables_from_query(query)
        self.assertEqual(tables, ["customers"])
        self.assertEqual(count_tables_in_query(query), 1)

    def test_join_query(self):
        """Test extraction with JOIN clauses"""
        query = """
        SELECT e.name, d.department_name
        FROM employees e
        JOIN departments d ON e.dept_id = d.id
        LEFT JOIN locations l ON d.location_id = l.id
        """

        tables = extract_tables_from_query(query)
        self.assertEqual(set(tables), {"employees", "departments", "locations"})
        self.assertEqual(count_tables_in_query(query), 3)

    def test_example_from_prompt(self):
        """Test with the example query from the prompt"""
        query = """
        SELECT EXTRACT(EPOCH FROM (OUTTIME - INTIME)) / 3600 AS length_of_stay_hours
        FROM ICUSTAYS
        WHERE SUBJECT_ID = 27392
        ORDER BY INTIME
        LIMIT 1;
        """

        tables = extract_tables_from_query(query)
        self.assertEqual(tables, ["icustays"])
        self.assertEqual(count_tables_in_query(query), 1)

    def test_query_with_table_aliases(self):
        """Test with table aliases"""
        query = """
        SELECT s.student_name, c.course_name
        FROM students AS s
        JOIN enrollments e ON s.id = e.student_id
        JOIN courses c ON e.course_id = c.id
        """

        tables = extract_tables_from_query(query)
        self.assertTrue(
            all(t in tables for t in ["students", "enrollments", "courses"])
        )
        self.assertEqual(count_tables_in_query(query), 3)

    def test_union_query(self):
        """Test with UNION queries"""
        query = """
        SELECT name FROM customers
        UNION
        SELECT name FROM suppliers
        """

        tables = extract_tables_from_query(query)
        self.assertTrue(all(t in tables for t in ["customers", "suppliers"]))


if __name__ == "__main__":
    unittest.main()
