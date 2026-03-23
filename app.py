import os
import io
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# -----------------------------
# LOAD ENV
# -----------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="AI Data Analyst", layout="wide")

if not api_key:
    st.error("OPENAI_API_KEY not found. Put .env in the same folder as app.py.")
    st.stop()

client = OpenAI(api_key=api_key)

# -----------------------------
# PAGE TITLE
# -----------------------------
st.title("AI Data Analyst")
st.write("Upload a CSV file, explore the data, generate charts, and get AI-powered insights.")

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def load_data(uploaded_file):
    """Load CSV into a pandas DataFrame."""
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return None


def get_basic_info(df):
    """Return basic dataset information."""
    num_rows, num_cols = df.shape
    missing_values = df.isnull().sum()
    total_missing = int(missing_values.sum())

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()

    return {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "total_missing": total_missing,
        "missing_values": missing_values,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "datetime_cols": datetime_cols,
    }


def try_parse_dates(df):
    """
    Try converting object columns to datetime if possible.
    This is optional but useful for trend analysis.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                converted = pd.to_datetime(df[col], errors="coerce")
                # Only convert if enough non-null datetime values exist
                if converted.notna().sum() > 0.6 * len(df):
                    df[col] = converted
            except Exception:
                pass
    return df


def generate_numeric_summary(df, numeric_cols):
    """Create summary stats for numeric columns."""
    if not numeric_cols:
        return "No numeric columns found."

    summary = df[numeric_cols].describe().transpose()
    return summary


def generate_categorical_summary(df, categorical_cols):
    """Create summary stats for categorical columns."""
    summaries = {}
    for col in categorical_cols[:5]:
        top_values = df[col].value_counts(dropna=False).head(5)
        summaries[col] = top_values
    return summaries


def find_best_chart_columns(df):
    """
    Select reasonable default columns for charts.
    """
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()

    chart_config = {
        "bar_cat": None,
        "bar_num": None,
        "line_x": None,
        "line_y": None,
        "hist_col": None,
    }

    if categorical_cols and numeric_cols:
        chart_config["bar_cat"] = categorical_cols[0]
        chart_config["bar_num"] = numeric_cols[0]

    if datetime_cols and numeric_cols:
        chart_config["line_x"] = datetime_cols[0]
        chart_config["line_y"] = numeric_cols[0]

    if numeric_cols:
        chart_config["hist_col"] = numeric_cols[0]

    return chart_config


def plot_bar_chart(df, cat_col, num_col):
    """
    Group by category and average the numeric column.
    """
    chart_data = df.groupby(cat_col, dropna=False)[num_col].mean().sort_values(ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(8, 4))
    chart_data.plot(kind="bar", ax=ax)
    ax.set_title(f"Average {num_col} by {cat_col}")
    ax.set_ylabel(num_col)
    ax.set_xlabel(cat_col)
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)
    plt.close(fig)


def plot_line_chart(df, date_col, num_col):
    """
    Group by date and sum numeric values.
    """
    temp_df = df[[date_col, num_col]].dropna().copy()
    temp_df[date_col] = pd.to_datetime(temp_df[date_col], errors="coerce")
    temp_df = temp_df.dropna()

    if temp_df.empty:
        st.info("Not enough valid date data for line chart.")
        return

    trend_data = temp_df.groupby(date_col)[num_col].sum().sort_index()

    fig, ax = plt.subplots(figsize=(8, 4))
    trend_data.plot(ax=ax)
    ax.set_title(f"{num_col} Trend Over Time")
    ax.set_ylabel(num_col)
    ax.set_xlabel(date_col)
    st.pyplot(fig)
    plt.close(fig)


def plot_histogram(df, numeric_col):
    fig, ax = plt.subplots(figsize=(8, 4))
    df[numeric_col].dropna().plot(kind="hist", bins=20, ax=ax)
    ax.set_title(f"Distribution of {numeric_col}")
    ax.set_xlabel(numeric_col)
    st.pyplot(fig)
    plt.close(fig)


def build_dataset_summary_text(df, basic_info):
    """
    Build a compact summary for sending to the model.
    Never send huge raw CSV blindly.
    """
    lines = []
    lines.append(f"Dataset has {basic_info['num_rows']} rows and {basic_info['num_cols']} columns.")
    lines.append(f"Columns: {', '.join(df.columns.astype(str))}")
    lines.append(f"Numeric columns: {', '.join(basic_info['numeric_cols']) if basic_info['numeric_cols'] else 'None'}")
    lines.append(f"Categorical columns: {', '.join(basic_info['categorical_cols']) if basic_info['categorical_cols'] else 'None'}")
    lines.append(f"Total missing values: {basic_info['total_missing']}")

    if basic_info["numeric_cols"]:
        numeric_summary = df[basic_info["numeric_cols"]].describe().round(2)
        lines.append("\nNumeric Summary:")
        lines.append(numeric_summary.to_string())

    for col in basic_info["categorical_cols"][:3]:
        lines.append(f"\nTop values in {col}:")
        lines.append(df[col].value_counts(dropna=False).head(5).to_string())

    return "\n".join(lines)


def get_ai_insights(summary_text):
    """
    Ask the model for business insights from the compact summary.
    """
    prompt = f"""
You are a data analyst.

Below is a dataset summary:
{summary_text}

Give:
1. A short summary in simple English
2. 5 key insights
3. 3 recommendations

Keep it clear and practical.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful data analyst who explains clearly."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=600
    )
    return response.choices[0].message.content.strip()


def answer_user_question(summary_text, question):
    """
    Answer natural language questions using the dataset summary.
    """
    prompt = f"""
You are answering questions about a dataset.

Dataset summary:
{summary_text}

User question:
{question}

Answer based only on the available dataset summary.
If the answer cannot be known confidently, say that clearly.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a careful data analyst. Do not invent facts."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=400
    )
    return response.choices[0].message.content.strip()


# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)

    if df is not None:
        df = try_parse_dates(df)

        st.markdown("---")
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        # -----------------------------
        # BASIC INFO
        # -----------------------------
        basic_info = get_basic_info(df)

        st.markdown("---")
        st.subheader("Dataset Information")

        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", basic_info["num_rows"])
        col2.metric("Columns", basic_info["num_cols"])
        col3.metric("Total Missing Values", basic_info["total_missing"])

        st.write("**Column Names:**")
        st.write(list(df.columns))

        st.write("**Numeric Columns:**")
        st.write(basic_info["numeric_cols"] if basic_info["numeric_cols"] else "None")

        st.write("**Categorical Columns:**")
        st.write(basic_info["categorical_cols"] if basic_info["categorical_cols"] else "None")

        st.write("**Date/Time Columns:**")
        st.write(basic_info["datetime_cols"] if basic_info["datetime_cols"] else "None")

        st.write("**Missing Values by Column:**")
        st.dataframe(basic_info["missing_values"].reset_index().rename(columns={"index": "Column", 0: "Missing Values"}))

        # -----------------------------
        # NUMERIC SUMMARY
        # -----------------------------
        st.markdown("---")
        st.subheader("Numeric Summary")
        numeric_summary = generate_numeric_summary(df, basic_info["numeric_cols"])
        if isinstance(numeric_summary, str):
            st.info(numeric_summary)
        else:
            st.dataframe(numeric_summary)

        # -----------------------------
        # CATEGORICAL SUMMARY
        # -----------------------------
        st.markdown("---")
        st.subheader("Categorical Summary")
        cat_summary = generate_categorical_summary(df, basic_info["categorical_cols"])
        if not cat_summary:
            st.info("No categorical columns found.")
        else:
            for col_name, series in cat_summary.items():
                st.write(f"**Top values in {col_name}:**")
                st.dataframe(series.reset_index().rename(columns={"index": col_name, col_name: "Count"}))

        # -----------------------------
        # CHARTS
        # -----------------------------
        st.markdown("---")
        st.subheader("Charts")

        chart_config = find_best_chart_columns(df)

        if chart_config["bar_cat"] and chart_config["bar_num"]:
            st.write("**Bar Chart**")
            plot_bar_chart(df, chart_config["bar_cat"], chart_config["bar_num"])
        else:
            st.info("Not enough categorical + numeric data for a bar chart.")

        if chart_config["line_x"] and chart_config["line_y"]:
            st.write("**Line Chart**")
            plot_line_chart(df, chart_config["line_x"], chart_config["line_y"])
        else:
            st.info("No suitable date + numeric columns found for a line chart.")

        if chart_config["hist_col"]:
            st.write("**Histogram**")
            plot_histogram(df, chart_config["hist_col"])
        else:
            st.info("No numeric column found for histogram.")

        # -----------------------------
        # SUMMARY TEXT FOR AI
        # -----------------------------
        summary_text = build_dataset_summary_text(df, basic_info)

        # Store it for later question-answering
        st.session_state["dataset_summary_text"] = summary_text

        # -----------------------------
        # AI INSIGHTS
        # -----------------------------
        st.markdown("---")
        st.subheader("AI Insights")

        if st.button("Generate AI Insights"):
            with st.spinner("Analyzing dataset..."):
                try:
                    insights = get_ai_insights(summary_text)
                    st.session_state["ai_insights"] = insights
                    st.success("Insights generated successfully.")
                    st.write(insights)
                except Exception as e:
                    st.error(f"Failed to generate AI insights: {e}")

        if "ai_insights" in st.session_state:
            st.write(st.session_state["ai_insights"])

        # -----------------------------
        # ASK QUESTIONS
        # -----------------------------
        st.markdown("---")
        st.subheader("Ask Questions About the Dataset")

        user_question = st.text_input(
            "Ask a question",
            placeholder="e.g. Which column seems most important? What trends do you notice?"
        )

        if st.button("Answer Question") and user_question:
            with st.spinner("Thinking..."):
                try:
                    answer = answer_user_question(summary_text, user_question)
                    st.success("Answer generated.")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Failed to answer question: {e}")