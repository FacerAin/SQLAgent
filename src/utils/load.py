from typing import Any, Dict, Optional

import yaml
from jinja2 import Template


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
