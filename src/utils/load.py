import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import yaml
from jinja2 import Template


def load_dataset_from_jsonl(
    file_path: str,
) -> List[Dict[str, Any]]:
    try:
        data = []
        with open(file_path, "r", encoding="utf-8") as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Error parsing JSON at line {line_num}: {e}")

        if not data:
            raise ValueError("The JSONL file is empty.")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{file_path}' was not found.")


def load_prompt_from_yaml(
    file_path: str,
    prompt_key: str = "prompt",
    template_vars: Optional[Dict[str, Any]] = None,
) -> str:
    try:
        with open(file_path, "r") as file:
            prompts = yaml.safe_load(file)
            if not prompts or prompt_key not in prompts:
                raise ValueError(
                    f"Prompt key '{prompt_key}' not found in the YAML file."
                )
            prompt_template = prompts[prompt_key]

            if not prompt_template:
                raise ValueError(f"Prompt template for key '{prompt_key}' is empty.")

            if template_vars:
                template = Template(prompt_template)
                return template.render(**template_vars)
            return str(prompt_template)

    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{file_path}' was not found.")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")


def classify_string(input_str: str) -> str:
    """
    Classifies a string as 'number', 'date', 'datetime', or 'string'.

    Args:
        input_str (str): The string to classify

    Returns:
        str: Classification as 'number', 'date', 'datetime', or 'string'
    """
    if not input_str or not isinstance(input_str, str):
        return "string"

    cleaned_str = input_str.strip()

    if re.match(r"^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$", cleaned_str):
        return "number"

    date_formats = [
        "%Y-%m-%d",  # 2023-01-30
        "%d/%m/%Y",  # 30/01/2023
        "%m/%d/%Y",  # 01/30/2023
        "%d-%m-%Y",  # 30-01-2023
        "%m-%d-%Y",  # 01-30-2023
        "%Y/%m/%d",  # 2023/01/30
        "%d.%m.%Y",  # 30.01.2023
        "%Y.%m.%d",  # 2023.01.30
        "%b %d, %Y",  # Jan 30, 2023
        "%d %b %Y",  # 30 Jan 2023
        "%B %d, %Y",  # January 30, 2023
        "%d %B %Y",  # 30 January 2023
        "%Y%m%d",  # 20230130
    ]

    datetime_formats = [
        "%Y-%m-%d %H:%M:%S",  # 2023-01-30 23:43:00
        "%d/%m/%Y %H:%M:%S",  # 30/01/2023 23:43:00
        "%Y/%m/%d %H:%M:%S",  # 2023/01/30 23:43:00
        "%Y-%m-%d %H:%M",  # 2023-01-30 23:43
        "%d/%m/%Y %H:%M",  # 30/01/2023 23:43
        "%Y/%m/%d %H:%M",  # 2023/01/30 23:43
        "%Y-%m-%dT%H:%M:%S",  # 2023-01-30T23:43:00 (ISO format)
        "%Y-%m-%dT%H:%M:%S.%f",  # 2023-01-30T23:43:00.000 (ISO with milliseconds)
        "%Y%m%d%H%M%S",  # 20230130234300
    ]

    for dt_format in datetime_formats:
        try:
            datetime.strptime(cleaned_str, dt_format)
            return "datetime"
        except ValueError:
            continue

    for date_format in date_formats:
        try:
            datetime.strptime(cleaned_str, date_format)
            return "date"
        except ValueError:
            continue

    return "string"


# Refer: https://github.com/huggingface/smolagents/blob/main/src/smolagents/utils.py#L126
def make_json_serializable(obj: Any) -> Any:
    """Recursive function to make objects JSON serializable"""
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        # Try to parse string as JSON if it looks like a JSON object/array
        if isinstance(obj, str):
            try:
                if (obj.startswith("{") and obj.endswith("}")) or (
                    obj.startswith("[") and obj.endswith("]")
                ):
                    parsed = json.loads(obj)
                    return make_json_serializable(parsed)
            except json.JSONDecodeError:
                pass
        return obj
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    elif hasattr(obj, "__dict__"):
        # For custom objects, convert their __dict__ to a serializable format
        return {
            "_type": obj.__class__.__name__,
            **{k: make_json_serializable(v) for k, v in obj.__dict__.items()},
        }
    else:
        # For any other type, convert to string
        return str(obj)


# Ref: https://github.com/huggingface/smolagents/blob/main/src/smolagents/utils.py#L152
def parse_json_blob(json_blob: str) -> Tuple[Dict[str, str], str]:
    "Extracts the JSON blob from the input and returns the JSON data and the rest of the input."
    try:
        first_accolade_index = json_blob.find("{")
        last_accolade_index = [a.start() for a in list(re.finditer("}", json_blob))][-1]
        json_data = json_blob[first_accolade_index : last_accolade_index + 1]
        json_data = json.loads(json_data, strict=False)
        return json_data, json_blob[:first_accolade_index]  # type: ignore
    except IndexError:
        logger.error("IndexError: %s", json_blob)
        raise ValueError("The JSON blob you used is invalid")
    except json.JSONDecodeError as e:
        place = e.pos
        if json_blob[place - 1 : place + 2] == "},\n":
            raise ValueError(
                "JSON is invalid: you probably tried to provide multiple tool calls in one action. PROVIDE ONLY ONE TOOL CALL."
            )
        raise ValueError(
            f"The JSON blob you used is invalid due to the following error: {e}.\n"
            f"JSON blob was: {json_blob}, decoding failed on that specific part of the blob:\n"
            f"'{json_blob[place - 4 : place + 5]}'."
        )
