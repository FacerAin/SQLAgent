import json
import os
from typing import Any, Dict, List

from openai import OpenAI


class SimpleTableDetector:
    def __init__(self, model: str = "gpt-4o"):
        """
        Initialize the detector with OpenAI credentials.

        Args:
            model: OpenAI model to use
        """
        self.client = OpenAI()
        self.model = model

    def detect_tables(self, question: str, input_tables: List[str]) -> Dict[str, Any]:
        """
        Detect if there are missing or irrelevant tables for a given SQL question.

        Args:
            question: The natural language question to be answered
            input_tables: List of table names provided as input

        Returns:
            Dictionary with detection results (is_missing_table, irrelevant_table_names)
        """
        prompt = self._construct_prompt(question, input_tables)
        response = self._call_llm(prompt)
        result = self._parse_response(response)

        return {
            "is_missing_table": result["is_missing_table"],
            "irrelevant_table_names": result["irrelevant_table_names"],
        }

    def _construct_prompt(self, question: str, input_tables: List[str]) -> str:
        """
        Construct a prompt for the LLM to detect missing and irrelevant tables.
        """
        prompt = f"""You are an expert SQL analyzer for healthcare databases. Given a question and a list of input tables, determine:

Question: {question}

Available Input Tables: {", ".join(input_tables)}

1. If there are ANY missing tables needed to answer the question (true/false)
2. Which tables in the input list are irrelevant and not needed
3. If is_missing_table is false, there are no irrelevant_table_names, return an empty list format.
4. When is_missing_table is true, irrelevant_table_names may or may not exist.

  For tables, they contain the following information:
  (1) admissions: ROW_ID, SUBJECT_ID, HADM_ID, ADMITTIME, DISCHTIME, ADMISSION_TYPE, ADMISSION_LOCATION, DISCHARGE_LOCATION, INSURANCE, LANGUAGE, MARITAL_STATUS, ETHNICITY, AGE
  (2) chartevents: ROW_ID, SUBJECT_ID, HADM_ID, ICUSTAY_ID, ITEMID, CHARTTIME, VALUENUM, VALUEUOM
  (3) cost: ROW_ID, SUBJECT_ID, HADM_ID, EVENT_TYPE, EVENT_ID, CHARGETIME, COST
  (4) d_icd_diagnoses: ROW_ID, ICD9_CODE, SHORT_TITLE, LONG_TITLE
  (5) d_icd_procedures: ROW_ID, ICD9_CODE, SHORT_TITLE, LONG_TITLE
  (6) d_items: ROW_ID, ITEMID, LABEL, LINKSTO
  (7) d_labitems: ROW_ID, ITEMID, LABEL
  (8) dianoses_icd: ROW_ID, SUBJECT_ID, HADM_ID, ICD9_CODE, CHARTTIME
  (9) icustays: ROW_ID, SUBJECT_ID, HADM_ID, ICUSTAY_ID, FIRST_CAREUNIT, LAST_CAREUNIT, FIRST_WARDID, LAST_WARDID, INTIME, OUTTIME
  (10) inputevents_cv: ROW_ID, SUBJECT_ID, HADM_ID, ICUSTAY_ID, CHARTTIME, ITEMID, AMOUNT
  (11) labevents: ROW_ID, SUBJECT_ID, HADM_ID, ITEMID, CHARTTIME, VALUENUM, VALUEUOM
  (12) microbiologyevents: RROW_ID, SUBJECT_ID, HADM_ID, CHARTTIME, SPEC_TYPE_DESC, ORG_NAME
  (13) outputevents: ROW_ID, SUBJECT_ID, HADM_ID, ICUSTAY_ID, CHARTTIME, ITEMID, VALUE
  (14) patients: ROW_ID, SUBJECT_ID, GENDER, DOB, DOD
  (15) prescriptions: ROW_ID, SUBJECT_ID, HADM_ID, STARTDATE, ENDDATE, DRUG, DOSE_VAL_RX, DOSE_UNIT_RX, ROUTE
  (16) procedures_icd: ROW_ID, SUBJECT_ID, HADM_ID, ICD9_CODE, CHARTTIME
  (17) transfers: ROW_ID, SUBJECT_ID, HADM_ID, ICUSTAY_ID, EVENTTYPE, CAREUNIT, WARDID, INTIME, OUTTIME

Analyze the question carefully. Think about what tables would be needed to answer it.

Output your analysis in this exact JSON format only:
{{
    "is_missing_table": true/false,
    "irrelevant_table_names": ["table1", "table2"]
}}

Do not include any other text in your response besides the JSON.
"""
        return prompt

    def _call_llm(self, prompt: str) -> str:
        """
        Call the OpenAI LLM with the constructed prompt.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert SQL analyzer."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return "{}"

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse the LLM response to extract the detection results.
        """
        try:
            # Extract JSON from the response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)
            else:
                raise ValueError("No valid JSON found in response")

            # Ensure required fields exist
            if "is_missing_table" not in result:
                result["is_missing_table"] = False
            if "irrelevant_table_names" not in result:
                result["irrelevant_table_names"] = []

            return result
        except Exception:
            # Return default values if parsing fails
            return {"is_missing_table": False, "irrelevant_table_names": []}
