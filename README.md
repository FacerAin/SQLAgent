# SQLAgent

SQLAgent is a framework for building and evaluating language model-powered SQL agents that can interact with databases through natural language. It's designed to test different agent architectures and LLM capabilities for medical database querying.


## Project Purpose

SQLAgent creates AI agents capable of:
- Converting natural language questions to SQL queries
- Executing queries against a medical database (MIMIC-III)
- Evaluating agent performance with various metrics
- Supporting different reasoning approaches (ReAct methodology)

## Key Features

- **SQL Agent Framework**: Execute natural language queries against SQL databases
- **ReAct Agent Pattern**: Step-by-step reasoning for complex queries
- **Multiple Tool Support**: SQL execution and Python code execution tools
- **Evaluation Framework**: Comprehensive metrics for agent performance assessment
- **Database Connector**: Specialized connector for SQLite with custom time handling
- **Agent Memory Management**: Track reasoning steps and maintain context
- **Confidence Checking**: Evaluate trustworthiness of agent responses with score metrics
- **LLM Verifier Tools**: Validate SQL queries and table selection before execution

## Installation

```bash
# Clone the repository
git clone https://github.com/FacerAin/SQL-R1.git
cd SQL-R1

# Install dependencies using Poetry
poetry install

# Set up environment variables
cp .env.sample .env
# Add your OpenAI API key to the .env file
```

## Requirements

- Python 3.10+
- Poetry (dependency manager)
- SQLite database (MIMIC-III)

## Usage

### Running a Single Query

```bash
# Basic query execution
./scripts/run_main.sh --query "how many times in the last year has nonexcis debridement wnd been ordered?"

# With custom model and parameters
python -m src.main --model_id "gpt-4o" --database "data/mimic_iii/mimic_iii.db" --max_iterations 5 --query "your query here" --log_to_file
```

### Quick Start (Agent Configuration)

```python
# Import necessary modules
from src.chat.factory import ChatModelFactory
from src.agent.react import ToolReActAgent
from src.tool.base import SQLTool, FinalAnswerTool, PythonTool
from src.database.connector import SqliteDatabaseConnector
import yaml
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
agent_logger = logging.getLogger("agent")

# Define database path
db_path = "data/mimic_iii/mimic_iii.db"

# Create a chat client
client = ChatModelFactory.load_model(model_id="gpt-4o")

# Create database connector
db_connector = SqliteDatabaseConnector(db_path)

# Initialize tools
# For SQL-only agent
# sql_tool = SQLTool(db_connector=db_connector)
# tools = [sql_tool, final_answer_tool]

# For Python-based agent
python_tool = PythonTool(db_connector=db_connector)
final_answer_tool = FinalAnswerTool()
tools = [python_tool, final_answer_tool]

# Load prompt templates
with open("src/prompts/react.yaml", "r", encoding="utf-8") as f:
    prompt_templates = yaml.safe_load(f)

# Create the agent
agent = ToolReActAgent(
    client=client,
    tools=tools,
    max_steps=10,
    prompt_templates=prompt_templates,
    logger=agent_logger,
)

# Run the agent
result = agent.run("How many patients with diabetes were admitted in 2012?")
print(result)
```

### Running Evaluation

```bash
# Basic evaluation
./scripts/run_evaluate.sh

# Evaluate specific agent type
./scripts/experiments/sql_react.sh

# Customize evaluation
python -m src.evaluate --model_id "gpt-4o" --dataset_path "data/test_50.jsonl" --agent_type "sql_react" --num_samples 10 --save_result

# Run confidence checking on evaluation results
./scripts/run_confidence.sh

# Custom confidence checking
python -m src.post_evaluate --input_file "results/your_results.json" --model "gpt-4.1"
```

## Project Structure

- **src/agent/**: Agent implementations (ReAct pattern)
- **src/chat/**: LLM client interfaces for different models
- **src/confidence/**: Confidence checking and trust evaluation
- **src/database/**: Database connectors (SQLite)
- **src/tool/**: Tools for agents (SQL, Python, Answer generation)
- **src/prompts/**: Prompt templates for agents
- **src/evaluation/**: Evaluation metrics and judges
- **src/utils/**: Utility functions including SQL verifiers
- **scripts/**: Run scripts for different experiments
- **data/**: Data files and sample datasets
- **results/**: Evaluation results

## Supported Agent Types

- `sql_react`: Uses ReAct methodology with SQL tool only
- `python_react`: Uses ReAct methodology with Python execution tool
- `python_sql_react`: Combined approach with both SQL and Python tools

## Advanced Features

### Confidence Checking

- **Trust Scoring**: Evaluates the reliability of agent responses on a 0-4 scale
- **Probability Distribution**: Analyzes response confidence using logprobs
- **Post-Evaluation Tool**: Batch processing for retroactive confidence scoring
- **Weighted Scoring**: Provides nuanced confidence metrics beyond binary correctness

### LLM Verifier Tools

- **Table Detection**: Identifies missing or irrelevant tables for SQL queries
- **Query Validation**: Verifies if the database schema supports answering the question
- **Pre-execution Checks**: Reduces errors by validating queries before database execution

## Examples

Query example with SQL agent:
```
Question: "What medications are prescribed for patients with hypertension?"
Thought: I need to find all medications prescribed for patients with hypertension.
Action: SQL Tool
Action Input: SELECT DISTINCT medication FROM prescriptions WHERE diagnosis='hypertension'
Observation: [List of medications returned from database]
Thought: I have the list of medications for hypertension patients.
Action: Final Answer
Action Input: Patients with hypertension are prescribed the following medications: lisinopril, hydrochlorothiazide, amlodipine, metoprolol, and losartan.
```

## Development

This project uses Poetry for dependency management and includes configurations for:
- Black (code formatting)
- isort (import sorting)
- mypy (type checking)
- flake8 (linting)
- pre-commit hooks

You can use the Makefile for common development tasks:
```bash
# Format code with black and isort
make format

# Run linting with flake8
make lint

# Run type checking with mypy
make typecheck

# Run all checks
make all

# Clean up cache files
make clean
```

Pre-commit hooks will automatically run linting and type checking before commits.

## Future Plans

- Add a dedicated reinforcement learning framework for agent training and optimization
- Improve agent reasoning capabilities with advanced prompting techniques
- Expand tool integrations for more complex database operations

## Dataset

**TBA**

## Evaluation Results

**TBA**

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Yong woo Song
```
