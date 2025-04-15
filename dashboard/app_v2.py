import glob
import json
import os
from collections import Counter
from datetime import datetime

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Medical DB Query Evaluation Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .metric-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
    }
    .metric-label {
        font-size: 14px;
        color: #555;
    }
    .question-box {
        background-color: #e6f3ff;
        border-left: 5px solid #0068c9;
        padding: 10px;
        border-radius: 3px;
        margin-bottom: 10px;
    }
    .expected-answer {
        background-color: #e6ffea;
        border-left: 5px solid #00c96d;
        padding: 10px;
        border-radius: 3px;
        margin-bottom: 10px;
    }
    .generated-answer {
        background-color: #fff5e6;
        border-left: 5px solid #ff9d00;
        padding: 10px;
        border-radius: 3px;
        margin-bottom: 10px;
    }
    .match-indicator {
        font-weight: bold;
        padding: 3px 8px;
        border-radius: 10px;
        display: inline-block;
    }
    .match-true {
        background-color: #adebad;
        color: #006600;
    }
    .match-false {
        background-color: #ffb3b3;
        color: #990000;
    }
    .turn-user-request {
        background-color: #e9ecef;
        border-left: 5px solid #6c757d;
        padding: 10px;
        border-radius: 3px;
        margin-bottom: 10px;
    }
    .turn-tool-call {
        background-color: #e0f7fa;
        border-left: 5px solid #00acc1;
        padding: 10px;
        border-radius: 3px;
        margin-bottom: 10px;
    }
    .turn-tool-response {
        background-color: #f8f9fa;
        border-left: 5px solid #28a745;
        padding: 10px;
        border-radius: 3px;
        margin-bottom: 10px;
    }
    .turn-tool-error {
        background-color: #ffebee;
        border-left: 5px solid #dc3545;
        padding: 10px;
        border-radius: 3px;
        margin-bottom: 10px;
    }
    .turn-assistant {
        background-color: #f3e5f5;
        border-left: 5px solid #9c27b0;
        padding: 10px;
        border-radius: 3px;
        margin-bottom: 10px;
    }
    .step-header {
        font-weight: bold;
        margin-bottom: 5px;
        padding: 5px;
        border-radius: 3px;
    }
    .step-error {
        background-color: rgba(255, 0, 0, 0.1);
    }
    .call-id-box {
        background-color: #f0f8ff;
        padding: 8px;
        border-radius: 5px;
        margin-bottom: 10px;
        border-left: 3px solid #4682b4;
        font-family: monospace;
    }
    .tool-call-box {
        background-color: #f5f5f5;
        padding: 10px;
        border-radius: 5px;
        font-family: monospace;
        border-left: 3px solid #9c27b0;
    }
    .observation-box {
        background-color: #f9f9f9;
        padding: 8px;
        border-radius: 5px;
        margin-top: 5px;
        margin-bottom: 10px;
        border-left: 3px solid #28a745;
        font-family: monospace;
    }
    .error-observation-box {
        background-color: #fff8f8;
        padding: 8px;
        border-radius: 5px;
        margin-top: 5px;
        margin-bottom: 10px;
        border-left: 3px solid #dc3545;
        font-family: monospace;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Function to load data
@st.cache_data
def load_data(file_path):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


# Function to get all JSON files in a directory
@st.cache_data
def get_json_files(directory_path):
    try:
        # Check if directory exists
        if not os.path.exists(directory_path):
            return []

        # Get all JSON files in the directory
        json_files = glob.glob(os.path.join(directory_path, "*.json"))
        return json_files
    except Exception as e:
        st.error(f"Error reading directory: {str(e)}")
        return []


# Sidebar - Directory path input and file selector
st.sidebar.title("📂 Data Input")

# Input for directory path
directory_path = st.sidebar.text_input(
    "Enter directory path containing JSON files",
    value="./results/04-10-analysis",  # Default directory (change as needed)
)

# Initialize data
data = None

# Get JSON files from the specified directory
json_files = get_json_files(directory_path)

if json_files:
    # Extract just the filenames for the dropdown
    filenames = [os.path.basename(f) for f in json_files]

    # Create a dictionary mapping filenames to full paths
    file_dict = dict(zip(filenames, json_files))

    # File selector dropdown
    selected_file = st.sidebar.selectbox("Select evaluation file", options=filenames)

    if selected_file:
        # Load the selected file
        file_path = file_dict[selected_file]
        data = load_data(file_path)
        st.sidebar.success(f"Loaded: {selected_file}")
else:
    st.sidebar.warning(
        f"No JSON files found in {directory_path}. Enter a valid directory path."
    )

    # Use sample data if no files are found
    st.sidebar.info("Using example data.")

    data = {
        "evaluation_history": [],  # This would be populated from the file
        "metadata": {
            "model_id": "gpt-4o-mini",
            "agent_type": "python_react",
            "database": "data/mimic_iii/mimic_iii.db",
            "dataset_path": "data/evaluation/mimic_100.jsonl",
            "num_samples": 10,
            "timestamp": "2025-04-10T17:46:30.818440",
        },
        "metrics": {
            "total_num": 10,
            "correct": 0,
            "unfinished": 0,
            "incorrect": 10,
            "sql_equality": 0,
            "llm_correct": 2,
            "sql_executable": 0,
            "norm_correct": 2,
            "exact_match_rate": 0.0,
            "normalized_match_rate": 0.2,
            "llm_match_rate": 0.2,
            "per_sample_summary": [],  # This would be populated from the file
        },
    }

# Only proceed if data is available
if data:
    # Sidebar - Show selected file info
    if "metadata" in data:
        model_id = data["metadata"].get("model_id", "N/A")
        agent_type = data["metadata"].get("agent_type", "N/A")
        timestamp = data["metadata"].get("timestamp", "N/A")

        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📄 Selected File Info")
        st.sidebar.markdown(f"**Model:** {model_id}")
        st.sidebar.markdown(f"**Agent Type:** {agent_type}")
        st.sidebar.markdown(f"**Date:** {timestamp[:10]}")

    # Sidebar navigation
    st.sidebar.markdown("---")
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "📊 Overview",
            "🔍 Sample Analysis",
        ],
    )

    # Extract evaluation history and metrics
    evaluation_history = data.get("evaluation_history", [])
    metadata = data.get("metadata", {})
    metrics = data.get("metrics", {})

    # Convert evaluation history to DataFrame for easier analysis
    if evaluation_history:
        samples_df = pd.DataFrame(
            [
                {
                    "id": sample.get("id", ""),
                    "question": sample.get("question", ""),
                    "expected_answer": sample.get("expected_answer", ""),
                    "generated_answer": sample.get("generated_answer", ""),
                    "llm_match": sample.get("sample_metrics", {}).get(
                        "llm_match", False
                    ),
                    "total_turns": sample.get("total_turns", 0),
                    "token_usages": sample.get("token_usages", []),
                }
                for sample in evaluation_history
            ]
        )

    # --- 📊 OVERVIEW PAGE ---
    if page == "📊 Overview":
        st.title("📊 Medical Database Query Evaluation Dashboard")

        # Model information and metadata
        st.header("Evaluation Setup")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Model:** {metadata.get('model_id', 'N/A')}")
        with col2:
            st.info(f"**Agent Type:** {metadata.get('agent_type', 'N/A')}")
        with col3:
            eval_date = datetime.fromisoformat(
                metadata.get("timestamp", "").replace("Z", "+00:00")
            )
            st.info(f"**Evaluation Date:** {eval_date.strftime('%Y-%m-%d')}")

        st.info(f"**Database:** {metadata.get('database', 'N/A')}")
        st.info(f"**Dataset:** {metadata.get('dataset_path', 'N/A')}")

        # Key metrics
        st.header("Key Performance Metrics")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                f"""
            <div class="metric-container">
                <div class="metric-value">{metrics.get('total_num', 0)}</div>
                <div class="metric-label">Total Samples</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
            <div class="metric-container">
                <div class="metric-value">{metrics.get('llm_match_rate', 0)*100:.1f}%</div>
                <div class="metric-label">LLM Match Rate</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                f"""
            <div class="metric-container">
                <div class="metric-value">{metrics.get('llm_correct', 0)} / {metrics.get('total_num', 0)}</div>
                <div class="metric-label">LLM Correct / Total</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Quick sample overview if samples_df exists
        if "samples_df" in locals():
            st.header("Sample Overview")

            # Create a pie chart for LLM match distribution
            llm_match_counts = samples_df["llm_match"].value_counts().reset_index()
            llm_match_counts.columns = ["LLM Match", "Count"]

            fig = px.pie(
                llm_match_counts,
                values="Count",
                names="LLM Match",
                title="LLM Match Distribution",
                color="LLM Match",
                color_discrete_map={True: "#00cc96", False: "#ef553b"},
            )
            st.plotly_chart(fig, use_container_width=True)

            # Distribution of turns per sample
            st.subheader("Distribution of Turns per Sample")
            fig = px.histogram(
                samples_df,
                x="total_turns",
                nbins=10,
                title="Distribution of Turns per Sample",
                labels={"total_turns": "Number of Turns", "count": "Number of Samples"},
                color_discrete_sequence=["#0068c9"],
            )
            st.plotly_chart(fig, use_container_width=True)

            # Samples table
            st.subheader("Samples Overview")
            st.dataframe(
                samples_df[
                    [
                        "id",
                        "question",
                        "llm_match",
                        "total_turns",
                    ]
                ],
                use_container_width=True,
            )

    # --- 🔍 SAMPLE ANALYSIS PAGE ---
    elif page == "🔍 Sample Analysis":
        st.title("🔍 Sample Analysis")

        if "samples_df" in locals() and not samples_df.empty:
            # Add filter options within the main page instead of sidebar
            st.subheader("Filter Samples")

            col1, col2 = st.columns(2)

            # LLM Match filter
            with col1:
                llm_match_filter = st.radio(
                    "Filter by LLM match:",
                    [
                        "All",
                        "Correct (LLM Match: True)",
                        "Incorrect (LLM Match: False)",
                    ],
                    index=0,
                )

            # Turn count filter
            with col2:
                use_turn_filter = st.checkbox("Filter by number of turns", value=False)
                if use_turn_filter:
                    max_turns = int(samples_df["total_turns"].max())
                    turn_filter = st.slider(
                        "Maximum number of turns:",
                        min_value=1,
                        max_value=max_turns,
                        value=max_turns,
                    )

            # Initialize keyword filter variables first
            use_keyword_filter = False
            keyword = ""

            # Apply filters to the samples dataframe
            filtered_samples = samples_df.copy()

            if llm_match_filter == "Correct (LLM Match: True)":
                filtered_samples = filtered_samples[
                    filtered_samples["llm_match"] == True
                ]
            elif llm_match_filter == "Incorrect (LLM Match: False)":
                filtered_samples = filtered_samples[
                    filtered_samples["llm_match"] == False
                ]

            if use_turn_filter:
                filtered_samples = filtered_samples[
                    filtered_samples["total_turns"] <= turn_filter
                ]

            # Display filter status
            # Update filter status display after all filters have been applied
            filter_status = []
            if llm_match_filter != "All":
                filter_status.append(llm_match_filter)
            if use_turn_filter:
                filter_status.append(f"Turns ≤ {turn_filter}")
            if use_keyword_filter and keyword:
                filter_status.append(f"Keyword: '{keyword}'")

            if filter_status:
                st.info(f"Filters applied: {', '.join(filter_status)}")
                st.write(
                    f"Showing {len(filtered_samples)} of {len(samples_df)} samples"
                )

            if filtered_samples.empty:
                st.warning(
                    "No samples match the current filters. Please adjust your filter settings."
                )
            else:
                # Sample selector with questions instead of IDs
                sample_options = filtered_samples.apply(
                    lambda row: f"{row['id']} - {row['question'][:80]}...", axis=1
                ).tolist()
                selected_option = st.selectbox(
                    "Select a sample to analyze:", sample_options
                )

                if selected_option:
                    # Extract the ID from the selected option
                    selected_id = selected_option.split(" - ")[0]

                if selected_id:
                    # Get the selected sample
                    sample = filtered_samples[
                        filtered_samples["id"] == selected_id
                    ].iloc[0]

                    # Find the corresponding sample in the original data for full details
                    full_sample = next(
                        (s for s in evaluation_history if s.get("id") == selected_id),
                        None,
                    )

                    st.header("Sample Details")

                    # Show LLM match indicator
                    st.markdown(
                        f"""
                    <div class="match-indicator {'match-true' if sample['llm_match'] else 'match-false'}">
                        {'✓' if sample['llm_match'] else '✗'} LLM Match
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    # Question and answers
                    st.markdown(
                        f"""
                    <div class="question-box">
                        <strong>Question:</strong> {sample['question']}
                    </div>

                    <div class="expected-answer">
                        <strong>Expected Answer:</strong> {sample['expected_answer']}
                    </div>

                    <div class="generated-answer">
                        <strong>Generated Answer:</strong> {sample['generated_answer']}
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    # SQL Query if available
                    if full_sample and "gold_sql_query" in full_sample:
                        st.subheader("Gold SQL Query")
                        st.code(full_sample["gold_sql_query"], language="sql")

                    # Improved turns analysis with better visualization
                    st.subheader(
                        f"Interaction Analysis ({sample['total_turns']} turns)"
                    )

                    if full_sample and "history" in full_sample:
                        # Display all interactions with improved formatting
                        for i, turn in enumerate(full_sample["history"]):
                            role = turn.get("role", "unknown")
                            content = turn.get("content", [])

                            # Determine if the content contains error
                            has_error = False
                            content_text = ""
                            call_id = ""
                            tool_call = ""

                            if content:
                                for item in content:
                                    if isinstance(item, dict) and "text" in item:
                                        item_text = item["text"]
                                        content_text += item_text + " "

                                        # Extract Call ID if present
                                        if "Call id:" in item_text:
                                            call_id = (
                                                item_text.split("Call id:")[1]
                                                .split("\n")[0]
                                                .strip()
                                            )

                                        # Extract tool call data if present
                                        if "Calling tools:" in item_text:
                                            try:
                                                # Get the text after "Calling tools:"
                                                tool_call_text = item_text.split(
                                                    "Calling tools:"
                                                )[1].strip()

                                                # Fix common escaping issues in the JSON string
                                                # Replace single quotes with double quotes for JSON parsing
                                                tool_call_text = tool_call_text.replace(
                                                    "'", '"'
                                                )
                                                # But fix double-escaped quotes in code strings
                                                tool_call_text = tool_call_text.replace(
                                                    '\\""', '\\"'
                                                )
                                                tool_call_text = tool_call_text.replace(
                                                    '\\"\\"', '\\"\\'
                                                )

                                                # Try different formats of extraction
                                                if tool_call_text.startswith(
                                                    "[{"
                                                ) and tool_call_text.endswith("}]"):
                                                    # Already in proper format
                                                    tool_call = tool_call_text
                                                elif tool_call_text.startswith(
                                                    "{"
                                                ) and tool_call_text.endswith("}"):
                                                    # Single object format
                                                    tool_call = tool_call_text
                                                else:
                                                    # Try to extract the JSON part
                                                    import re

                                                    json_match = re.search(
                                                        r"\[\s*\{.*?\}\s*\]",
                                                        tool_call_text,
                                                        re.DOTALL,
                                                    )
                                                    if json_match:
                                                        tool_call = json_match.group(0)
                                                    else:
                                                        json_match = re.search(
                                                            r"\{.*?\}",
                                                            tool_call_text,
                                                            re.DOTALL,
                                                        )
                                                        if json_match:
                                                            tool_call = (
                                                                json_match.group(0)
                                                            )
                                                        else:
                                                            tool_call = tool_call_text
                                            except Exception as e:
                                                tool_call = item_text
                                    elif isinstance(item, str):
                                        content_text += item + " "

                                has_error = "error" in content_text.lower()

                            # Improved role-based styling with more distinctions
                            if role.lower() == "user" or role.lower() == "human":
                                turn_class = "turn-user-request"
                                role_display = "USER REQUEST"
                            elif role.lower() == "assistant":
                                turn_class = "turn-assistant"
                                role_display = "ASSISTANT RESPONSE"
                            elif role.lower() == "tool-call":
                                turn_class = "turn-tool-call"
                                role_display = "TOOL CALL"
                            elif role.lower() == "tool-response":
                                # Check if this is an error response
                                if (
                                    "no output" in content_text.lower()
                                    or "error" in content_text.lower()
                                ):
                                    turn_class = "turn-tool-error"
                                    role_display = "TOOL ERROR"
                                else:
                                    turn_class = "turn-tool-response"
                                    role_display = "TOOL RESPONSE"
                            else:
                                turn_class = "turn-tool-response"
                                role_display = role.upper()

                            # Add error class if content contains error
                            step_class = "step-error" if has_error else ""

                            st.markdown(
                                f"""
                            <div class="{turn_class}">
                                <div class="step-header {step_class}">
                                    Step {i+1}: {role_display}
                                </div>
                            """,
                                unsafe_allow_html=True,
                            )

                            # Display content
                            if content:
                                # Format and display special parts like Call IDs and tool calls
                                if call_id:
                                    st.markdown(
                                        f"""
                                        <div class="call-id-box">
                                            <strong>Call ID:</strong> <code>{call_id}</code>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )

                                if tool_call and role.lower() == "tool-call":
                                    # For tool calls, display the raw content first for easier debugging
                                    with st.expander(
                                        "Show raw tool call", expanded=False
                                    ):
                                        st.text(tool_call)

                                    # Try to prettify JSON for tool calls if possible
                                    import json

                                    try:
                                        # Clean up the tool_call string before parsing
                                        cleaned_tool_call = tool_call.strip()

                                        # Try to extract Python code directly
                                        import re

                                        python_code_match = re.search(
                                            r"'code':\s*'(.*?)'(?=\s*})",
                                            cleaned_tool_call,
                                            re.DOTALL | re.MULTILINE,
                                        )
                                        if python_code_match:
                                            # We found Python code, extract it
                                            code = python_code_match.group(1)
                                            # Unescape any escaped characters
                                            code = (
                                                code.replace("\\'", "'")
                                                .replace('\\"', '"')
                                                .replace("\\n", "\n")
                                            )

                                            st.markdown(
                                                """
                                                <div class="tool-call-box">
                                                    <strong>Tool:</strong> <code>python</code>
                                                </div>
                                                """,
                                                unsafe_allow_html=True,
                                            )
                                            st.code(code, language="python")
                                        else:
                                            # Try to extract SQL query directly
                                            sql_query_match = re.search(
                                                r"'query':\s*'(.*?)'(?=\s*})",
                                                cleaned_tool_call,
                                                re.DOTALL | re.MULTILINE,
                                            )
                                            if sql_query_match:
                                                query = sql_query_match.group(1)
                                                # Unescape any escaped characters
                                                query = (
                                                    query.replace("\\'", "'")
                                                    .replace('\\"', '"')
                                                    .replace("\\n", "\n")
                                                )

                                                st.markdown(
                                                    """
                                                    <div class="tool-call-box">
                                                        <strong>Tool:</strong> <code>sql</code>
                                                    </div>
                                                    """,
                                                    unsafe_allow_html=True,
                                                )
                                                st.code(query, language="sql")
                                            else:
                                                # Extract tool name using regex
                                                tool_name_match = re.search(
                                                    r"'name':\s*'(\w+)'",
                                                    cleaned_tool_call,
                                                )
                                                if tool_name_match:
                                                    tool_name = tool_name_match.group(1)

                                                    st.markdown(
                                                        f"""
                                                        <div class="tool-call-box">
                                                            <strong>Tool:</strong> <code>{tool_name}</code>
                                                        </div>
                                                        """,
                                                        unsafe_allow_html=True,
                                                    )
                                                else:
                                                    # Fallback to just showing the call
                                                    st.markdown(
                                                        """
                                                        <div class="tool-call-box">
                                                            <strong>Tool Call</strong>
                                                        </div>
                                                        """,
                                                        unsafe_allow_html=True,
                                                    )
                                    except Exception as e:
                                        # Just show the tool call nicely formatted
                                        st.markdown(
                                            """
                                            <div class="tool-call-box">
                                                <strong>Tool Call</strong>
                                            </div>
                                            """,
                                            unsafe_allow_html=True,
                                        )

                                # For tool responses, handle "Observation:" specially
                                if (
                                    role.lower() == "tool-response"
                                    and "Observation:" in content_text
                                ):
                                    for item in content:
                                        if isinstance(item, dict) and "text" in item:
                                            text = item["text"]
                                            if "Observation:" in text:
                                                parts = text.split("Observation:", 1)
                                                # First part (if any) - typically call ID
                                                if parts[0].strip():
                                                    st.markdown(parts[0].strip())

                                                # Second part - the observation itself
                                                observation = parts[1].strip()
                                                box_class = (
                                                    "error-observation-box"
                                                    if has_error
                                                    else "observation-box"
                                                )
                                                st.markdown(
                                                    f"""
                                                    <div class="{box_class}">
                                                        <strong>Observation:</strong><br/>
                                                        {observation}
                                                    </div>
                                                    """,
                                                    unsafe_allow_html=True,
                                                )
                                else:
                                    # Display regular content
                                    for item in content:
                                        if isinstance(item, dict) and "text" in item:
                                            # Skip lines that are already handled
                                            text = item["text"]
                                            if (
                                                "Call id:" not in text
                                                and "Calling tools:" not in text
                                                and "Observation:" not in text
                                            ):
                                                st.markdown(text)
                                        elif isinstance(item, str):
                                            st.markdown(item)

                            # Close the div
                            st.markdown("</div>", unsafe_allow_html=True)

                else:
                    st.warning(
                        "No sample selected. Please select a sample from the dropdown."
                    )
        else:
            st.error("No samples available to analyze. Please check the data file.")
else:
    st.title("Medical Database Query Evaluation Dashboard")
    st.error(
        "No data available. Please specify a valid directory containing JSON evaluation files."
    )
    st.info(
        "This dashboard is designed to visualize and analyze the performance of language models on medical database queries. Specify a directory path in the sidebar to get started."
    )
