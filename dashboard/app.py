# -*- coding: utf-8 -*-
# flake8: noqa

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
            "📈 Performance Metrics",
            "🔍 Sample Analysis",
            "📝 Token Usage",
            "⏱️ Response Time Analysis",
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
                    "exact_match": sample.get("sample_metrics", {}).get(
                        "exact_match", False
                    ),
                    "normalized_match": sample.get("sample_metrics", {}).get(
                        "normalized_match", False
                    ),
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
        col1, col2, col3, col4 = st.columns(4)

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
                <div class="metric-value">{metrics.get('exact_match_rate', 0)*100:.1f}%</div>
                <div class="metric-label">Exact Match Rate</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                f"""
            <div class="metric-container">
                <div class="metric-value">{metrics.get('normalized_match_rate', 0)*100:.1f}%</div>
                <div class="metric-label">Normalized Match Rate</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col4:
            st.markdown(
                f"""
            <div class="metric-container">
                <div class="metric-value">{metrics.get('llm_match_rate', 0)*100:.1f}%</div>
                <div class="metric-label">LLM Match Rate</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Secondary metrics
        st.subheader("Secondary Metrics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(
                f"""
            <div class="metric-container">
                <div class="metric-value">{metrics.get('correct', 0)}</div>
                <div class="metric-label">Correct</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
            <div class="metric-container">
                <div class="metric-value">{metrics.get('incorrect', 0)}</div>
                <div class="metric-label">Incorrect</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                f"""
            <div class="metric-container">
                <div class="metric-value">{metrics.get('unfinished', 0)}</div>
                <div class="metric-label">Unfinished</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col4:
            st.markdown(
                f"""
            <div class="metric-container">
                <div class="metric-value">{metrics.get('sql_equality', 0)}</div>
                <div class="metric-label">SQL Equality</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Quick sample overview if samples_df exists
        if "samples_df" in locals():
            st.header("Sample Overview")
            st.dataframe(
                samples_df[
                    [
                        "id",
                        "question",
                        "exact_match",
                        "normalized_match",
                        "llm_match",
                        "total_turns",
                    ]
                ],
                use_container_width=True,
            )

    # --- 📈 PERFORMANCE METRICS PAGE ---
    elif page == "📈 Performance Metrics":
        st.title("📈 Performance Metrics")

        # Accuracy metrics visualization
        st.header("Accuracy Metrics")

        accuracy_data = pd.DataFrame(
            {
                "Metric": ["Exact Match", "Normalized Match", "LLM Match"],
                "Rate": [
                    metrics.get("exact_match_rate", 0) * 100,
                    metrics.get("normalized_match_rate", 0) * 100,
                    metrics.get("llm_match_rate", 0) * 100,
                ],
            }
        )

        fig = px.bar(
            accuracy_data,
            x="Metric",
            y="Rate",
            text="Rate",
            color="Metric",
            color_discrete_sequence=["#0068c9", "#83c9ff", "#29b5e8"],
            title="Accuracy Rates (%)",
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(
            uniformtext_minsize=8, uniformtext_mode="hide", yaxis_range=[0, 100]
        )

        st.plotly_chart(fig, use_container_width=True)

        # Comparison of metrics
        st.header("Detailed Metrics Comparison")

        col1, col2 = st.columns(2)

        with col1:
            detailed_metrics = pd.DataFrame(
                {
                    "Category": ["Correct", "Incorrect", "Unfinished"],
                    "Count": [
                        metrics.get("correct", 0),
                        metrics.get("incorrect", 0),
                        metrics.get("unfinished", 0),
                    ],
                }
            )

            fig = px.pie(
                detailed_metrics,
                values="Count",
                names="Category",
                color="Category",
                color_discrete_sequence=["#00cc96", "#ef553b", "#ab63fa"],
                title="Response Categories",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            match_types = pd.DataFrame(
                {
                    "Match Type": [
                        "Exact Match",
                        "Normalized Match",
                        "LLM Match",
                        "No Match",
                    ],
                    "Count": [
                        metrics.get("correct", 0),
                        metrics.get("norm_correct", 0) - metrics.get("correct", 0),
                        metrics.get("llm_correct", 0) - metrics.get("norm_correct", 0),
                        metrics.get("total_num", 0) - metrics.get("llm_correct", 0),
                    ],
                }
            )

            fig = px.pie(
                match_types,
                values="Count",
                names="Match Type",
                color="Match Type",
                color_discrete_sequence=["#00cc96", "#ffa15a", "#19d3f3", "#ef553b"],
                title="Match Types",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Per-sample analysis if samples_df exists
        if "samples_df" in locals():
            st.header("Per-Sample Match Analysis")

            # Calculate match counts per sample
            match_counts = samples_df.apply(
                lambda row: sum(
                    [row["exact_match"], row["normalized_match"], row["llm_match"]]
                ),
                axis=1,
            )

            # Create a new DataFrame for visualization
            match_analysis = pd.DataFrame(
                {
                    "Sample ID": samples_df["id"],
                    "Exact Match": samples_df["exact_match"],
                    "Normalized Match": samples_df["normalized_match"],
                    "LLM Match": samples_df["llm_match"],
                    "Match Count": match_counts,
                }
            )

            # Sort by match count for better visualization
            match_analysis = match_analysis.sort_values("Match Count", ascending=False)

            # Shorten sample IDs for better readability
            match_analysis["Short ID"] = match_analysis["Sample ID"].str[:8] + "..."

            # Create heatmap
            match_matrix = match_analysis.set_index("Short ID")[
                ["Exact Match", "Normalized Match", "LLM Match"]
            ]

            fig = px.imshow(
                match_matrix.T,
                color_continuous_scale=[[0, "red"], [1, "green"]],
                labels=dict(x="Sample ID", y="Match Type", color="Match"),
                title="Match Types by Sample",
                aspect="auto",
            )
            fig.update_xaxes(side="top")

            st.plotly_chart(fig, use_container_width=True)

    # --- 🔍 SAMPLE ANALYSIS PAGE ---
    elif page == "🔍 Sample Analysis":
        st.title("🔍 Sample Analysis")

        if "samples_df" in locals() and not samples_df.empty:
            # Add filter options by match type
            st.sidebar.markdown("---")
            st.sidebar.subheader("Filter Samples")
            filter_option = st.sidebar.radio(
                "Filter by match type:",
                ["All Samples", "LLM Match: Correct", "LLM Match: Incorrect"],
            )

            # Apply filters to the samples dataframe
            if filter_option == "LLM Match: Correct":
                filtered_samples = samples_df[samples_df["llm_match"] == True]
                st.info(
                    f"Showing {len(filtered_samples)} samples with correct LLM match"
                )
            elif filter_option == "LLM Match: Incorrect":
                filtered_samples = samples_df[samples_df["llm_match"] == False]
                st.info(
                    f"Showing {len(filtered_samples)} samples with incorrect LLM match"
                )
            else:
                filtered_samples = samples_df

            if filtered_samples.empty:
                st.warning(f"No samples found with the filter: {filter_option}")

            # Sample selector
            sample_ids = filtered_samples["id"].tolist()
            selected_id = st.selectbox("Select a sample to analyze:", sample_ids)

            if selected_id:
                # Add batch analysis option
                st.sidebar.markdown("---")
                st.sidebar.subheader("Batch Analysis")
                batch_analysis = st.sidebar.checkbox(
                    "Enable Batch Analysis", value=False
                )

                if batch_analysis:
                    # Number of samples to analyze in batch
                    max_samples = min(
                        5, len(filtered_samples)
                    )  # Limit to 5 samples to avoid overload
                    num_samples = st.sidebar.slider(
                        "Number of samples to analyze", 1, max_samples, 3
                    )

                    # Choose first n samples based on the filter
                    batch_samples = filtered_samples.head(num_samples)

                    st.header(f"Batch Analysis ({len(batch_samples)} samples)")
                    st.info(
                        f"Analyzing {len(batch_samples)} samples with filter: {filter_option}"
                    )

                    # Display summary of batch samples
                    summary_df = batch_samples[
                        [
                            "id",
                            "question",
                            "expected_answer",
                            "generated_answer",
                            "exact_match",
                            "normalized_match",
                            "llm_match",
                            "total_turns",
                        ]
                    ]

                    st.dataframe(summary_df, use_container_width=True)

                    # Analyze each sample in the batch
                    for idx, (_, sample) in enumerate(batch_samples.iterrows()):
                        sample_id = sample["id"]

                        # Find the corresponding sample in the original data for full details
                        full_sample = next(
                            (s for s in evaluation_history if s.get("id") == sample_id),
                            None,
                        )

                        st.markdown("---")
                        st.subheader(f"Sample {idx+1}: {sample_id}")

                        # Show match indicators
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(
                                f"""
                            <div class="match-indicator {'match-true' if sample['exact_match'] else 'match-false'}">
                                {'✓' if sample['exact_match'] else '✗'} Exact Match
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )
                        with col2:
                            st.markdown(
                                f"""
                            <div class="match-indicator {'match-true' if sample['normalized_match'] else 'match-false'}">
                                {'✓' if sample['normalized_match'] else '✗'} Normalized Match
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )
                        with col3:
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
                            st.markdown("**Gold SQL Query:**")
                            st.code(full_sample["gold_sql_query"], language="sql")

                        # Turns analysis
                        with st.expander(
                            f"Show Interaction Analysis ({sample['total_turns']} turns)",
                            expanded=False,
                        ):
                            if full_sample and "history" in full_sample:
                                # Display all interactions in a scrollable format
                                for i, turn in enumerate(full_sample["history"]):
                                    role = turn.get("role", "unknown")
                                    content = turn.get("content", [])

                                    # Format with a box around each turn
                                    st.markdown(
                                        f"""
                                    <div style="margin-bottom: 20px; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">
                                        <div style="font-weight: bold; margin-bottom: 5px; background-color: #f0f0f0; padding: 5px; border-radius: 3px;">
                                            Turn {i+1}: {role.upper()}
                                        </div>
                                    </div>
                                    """,
                                        unsafe_allow_html=True,
                                    )

                                    # Display content
                                    if content:
                                        for item in content:
                                            if (
                                                isinstance(item, dict)
                                                and "text" in item
                                            ):
                                                st.markdown(item["text"])
                                            elif isinstance(item, str):
                                                st.markdown(item)

                                    # Add a separator between turns
                                    st.markdown("---")

                else:
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

                    # Show match indicators
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(
                            f"""
                        <div class="match-indicator {'match-true' if sample['exact_match'] else 'match-false'}">
                            {'✓' if sample['exact_match'] else '✗'} Exact Match
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    with col2:
                        st.markdown(
                            f"""
                        <div class="match-indicator {'match-true' if sample['normalized_match'] else 'match-false'}">
                            {'✓' if sample['normalized_match'] else '✗'} Normalized Match
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    with col3:
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

                    # Turns analysis
                    st.subheader(
                        f"Interaction Analysis ({sample['total_turns']} turns)"
                    )

                    if full_sample and "history" in full_sample:
                        # Display all interactions in a scrollable format instead of tabs
                        for i, turn in enumerate(full_sample["history"]):
                            role = turn.get("role", "unknown")
                            content = turn.get("content", [])

                            # Format with a box around each turn
                            st.markdown(
                                f"""
                            <div style="margin-bottom: 20px; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">
                                <div style="font-weight: bold; margin-bottom: 5px; background-color: #f0f0f0; padding: 5px; border-radius: 3px;">
                                    Turn {i+1}: {role.upper()}
                                </div>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

                            # Display content
                            if content:
                                for item in content:
                                    if isinstance(item, dict) and "text" in item:
                                        st.markdown(item["text"])
                                    elif isinstance(item, str):
                                        st.markdown(item)

                            # Add a separator between turns
                            st.markdown("---")

                # Token usage analysis for this sample
                if "token_usages" in sample and sample["token_usages"]:
                    st.subheader("Token Usage")

                    token_df = pd.DataFrame(sample["token_usages"])

                    # Calculate cumulative totals
                    token_df["cumulative_prompt"] = token_df["prompt_tokens"].cumsum()
                    token_df["cumulative_completion"] = token_df[
                        "completion_tokens"
                    ].cumsum()
                    token_df["cumulative_total"] = token_df["total_tokens"].cumsum()
                    token_df["turn"] = range(1, len(token_df) + 1)

                    # Create a line chart of token usage
                    fig = px.line(
                        token_df,
                        x="turn",
                        y=["prompt_tokens", "completion_tokens", "total_tokens"],
                        title="Token Usage by Turn",
                        labels={
                            "value": "Tokens",
                            "turn": "Turn",
                            "variable": "Token Type",
                        },
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Show cumulative token usage
                    fig = px.line(
                        token_df,
                        x="turn",
                        y=[
                            "cumulative_prompt",
                            "cumulative_completion",
                            "cumulative_total",
                        ],
                        title="Cumulative Token Usage",
                        labels={
                            "value": "Tokens",
                            "turn": "Turn",
                            "variable": "Token Type",
                        },
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(
                    "No sample selected. Please select a sample from the dropdown."
                )
        else:
            st.error("No samples available to analyze. Please check the data file.")

    # --- 📝 TOKEN USAGE PAGE ---
    elif page == "📝 Token Usage":
        st.title("📝 Token Usage Analysis")

        if "samples_df" in locals() and not samples_df.empty:
            # Extract token usage data across all samples
            all_token_data = []

            for _, sample in samples_df.iterrows():
                sample_id = sample["id"]
                token_usages = sample["token_usages"]

                for i, usage in enumerate(token_usages):
                    all_token_data.append(
                        {
                            "sample_id": sample_id,
                            "turn": i + 1,
                            "prompt_tokens": usage.get("prompt_tokens", 0),
                            "completion_tokens": usage.get("completion_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0),
                        }
                    )

            token_df = pd.DataFrame(all_token_data)

            st.header("Overall Token Usage")

            # Calculate total tokens for each category
            total_prompt = token_df["prompt_tokens"].sum()
            total_completion = token_df["completion_tokens"].sum()
            total_tokens = token_df["total_tokens"].sum()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(
                    f"""
                <div class="metric-container">
                    <div class="metric-value">{total_prompt:,}</div>
                    <div class="metric-label">Total Prompt Tokens</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    f"""
                <div class="metric-container">
                    <div class="metric-value">{total_completion:,}</div>
                    <div class="metric-label">Total Completion Tokens</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            with col3:
                st.markdown(
                    f"""
                <div class="metric-container">
                    <div class="metric-value">{total_tokens:,}</div>
                    <div class="metric-label">Total Tokens</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            # Token usage distribution
            st.subheader("Token Usage Distribution")

            # Calculate average token usage per turn
            avg_token_usage = (
                token_df.groupby("turn")
                .agg(
                    {
                        "prompt_tokens": "mean",
                        "completion_tokens": "mean",
                        "total_tokens": "mean",
                    }
                )
                .reset_index()
            )

            fig = px.line(
                avg_token_usage,
                x="turn",
                y=["prompt_tokens", "completion_tokens", "total_tokens"],
                title="Average Token Usage per Turn",
                labels={"value": "Tokens", "turn": "Turn", "variable": "Token Type"},
            )
            st.plotly_chart(fig, use_container_width=True)

            # Token usage by sample
            st.subheader("Token Usage by Sample")

            # Calculate total tokens per sample
            sample_token_usage = (
                token_df.groupby("sample_id")
                .agg(
                    {
                        "prompt_tokens": "sum",
                        "completion_tokens": "sum",
                        "total_tokens": "sum",
                    }
                )
                .reset_index()
            )

            # Sort by total tokens
            sample_token_usage = sample_token_usage.sort_values(
                "total_tokens", ascending=False
            )

            fig = px.bar(
                sample_token_usage,
                x="sample_id",
                y=["prompt_tokens", "completion_tokens"],
                title="Token Usage by Sample",
                labels={
                    "value": "Tokens",
                    "sample_id": "Sample ID",
                    "variable": "Token Type",
                },
                barmode="stack",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Correlation analysis between tokens and turns
            st.subheader("Token Usage vs. Number of Turns")

            # Calculate total tokens and turns per sample
            turns_tokens = pd.DataFrame(
                {
                    "sample_id": samples_df["id"],
                    "total_turns": samples_df["total_turns"],
                    "total_tokens": sample_token_usage.set_index("sample_id")
                    .loc[samples_df["id"]]["total_tokens"]
                    .values,
                }
            )

            fig = px.scatter(
                turns_tokens,
                x="total_turns",
                y="total_tokens",
                title="Token Usage vs. Number of Turns",
                labels={"total_tokens": "Total Tokens", "total_turns": "Total Turns"},
                trendline="ols",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Token efficiency analysis
            st.subheader("Token Efficiency Analysis")

            # Calculate tokens per turn
            turns_tokens["tokens_per_turn"] = (
                turns_tokens["total_tokens"] / turns_tokens["total_turns"]
            )

            # Shorten sample IDs for better readability
            turns_tokens["short_id"] = turns_tokens["sample_id"].str[:8] + "..."

            fig = px.bar(
                turns_tokens.sort_values("tokens_per_turn", ascending=False),
                x="short_id",
                y="tokens_per_turn",
                title="Tokens per Turn by Sample",
                labels={"tokens_per_turn": "Tokens per Turn", "short_id": "Sample ID"},
                hover_data=["sample_id"],  # Show full ID on hover
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.error("No token usage data available. Please check the data file.")

    # --- ⏱️ RESPONSE TIME ANALYSIS PAGE ---
    elif page == "⏱️ Response Time Analysis":
        st.title("⏱️ Response Time Analysis")

        # Note: Response time is not directly available in the data
        # We can estimate it based on token usage and typical processing speeds
        st.info(
            "Note: Actual response time data is not available in the provided dataset. This analysis uses token counts to estimate relative processing times."
        )

        if "samples_df" in locals() and not samples_df.empty:
            # Extract token usage data for processing time estimation
            processing_data = []

            for _, sample in samples_df.iterrows():
                sample_id = sample["id"]
                token_usages = sample["token_usages"]
                total_turns = sample["total_turns"]

                # Calculate estimated processing time based on tokens
                # This is a very rough approximation
                total_prompt_tokens = sum(
                    usage.get("prompt_tokens", 0) for usage in token_usages
                )
                total_completion_tokens = sum(
                    usage.get("completion_tokens", 0) for usage in token_usages
                )

                # Assume processing speed of X tokens per second
                prompt_processing_speed = 30  # tokens per second
                completion_generation_speed = 15  # tokens per second

                estimated_prompt_time = total_prompt_tokens / prompt_processing_speed
                estimated_completion_time = (
                    total_completion_tokens / completion_generation_speed
                )
                estimated_total_time = estimated_prompt_time + estimated_completion_time

                processing_data.append(
                    {
                        "sample_id": sample_id,
                        "total_turns": total_turns,
                        "total_prompt_tokens": total_prompt_tokens,
                        "total_completion_tokens": total_completion_tokens,
                        "estimated_processing_time": estimated_total_time,
                    }
                )

            processing_df = pd.DataFrame(processing_data)

            st.header("Estimated Processing Time Analysis")

            # Display estimated processing times
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_time = processing_df["estimated_processing_time"].mean()
                st.markdown(
                    f"""
                <div class="metric-container">
                    <div class="metric-value">{avg_time:.2f}s</div>
                    <div class="metric-label">Avg. Estimated Processing Time</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            with col2:
                max_time = processing_df["estimated_processing_time"].max()
                st.markdown(
                    f"""
                <div class="metric-container">
                    <div class="metric-value">{max_time:.2f}s</div>
                    <div class="metric-label">Max Estimated Processing Time</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            with col3:
                min_time = processing_df["estimated_processing_time"].min()
                st.markdown(
                    f"""
                <div class="metric-container">
                    <div class="metric-value">{min_time:.2f}s</div>
                    <div class="metric-label">Min Estimated Processing Time</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            # Plot estimated processing times
            # Already have short_id from above
            fig = px.bar(
                processing_df.sort_values("estimated_processing_time", ascending=False),
                x="short_id",
                y="estimated_processing_time",
                title="Estimated Processing Time by Sample",
                labels={
                    "estimated_processing_time": "Estimated Time (s)",
                    "short_id": "Sample ID",
                },
                hover_data=["sample_id"],  # Show full ID on hover
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

            # Processing time vs turns
            st.subheader("Processing Time vs. Number of Turns")

            fig = px.scatter(
                processing_df,
                x="total_turns",
                y="estimated_processing_time",
                title="Processing Time vs. Number of Turns",
                labels={
                    "estimated_processing_time": "Estimated Time (s)",
                    "total_turns": "Total Turns",
                },
                trendline="ols",
                hover_data=["sample_id"],  # Show sample ID on hover
            )
            st.plotly_chart(fig, use_container_width=True)

            # Processing time efficiency
            st.subheader("Processing Time Efficiency")

            processing_df["time_per_turn"] = (
                processing_df["estimated_processing_time"]
                / processing_df["total_turns"]
            )

            # Shorten sample IDs for better readability
            processing_df["short_id"] = processing_df["sample_id"].str[:8] + "..."

            fig = px.bar(
                processing_df.sort_values("time_per_turn", ascending=False),
                x="short_id",
                y="time_per_turn",
                title="Processing Time per Turn by Sample",
                labels={"time_per_turn": "Time per Turn (s)", "short_id": "Sample ID"},
                hover_data=["sample_id"],  # Show full ID on hover
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.error("No data available for response time analysis.")
else:
    st.title("Medical Database Query Evaluation Dashboard")
    st.error(
        "No data available. Please specify a valid directory containing JSON evaluation files."
    )
    st.info(
        "This dashboard is designed to visualize and analyze the performance of language models on medical database queries. Specify a directory path in the sidebar to get started."
    )
