from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "student_depression_dataset.csv"
TARGET_COL = "depression"


def _to_snake(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("/", " ")
    text = text.replace("?", "")
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def _clean_dataset(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df.columns = [col.strip() for col in df.columns]

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().str.strip("'").str.strip()

    if "Financial Stress" in df.columns:
        df["Financial Stress"] = df["Financial Stress"].replace("?", np.nan).astype(float)

    df = df.rename(columns={col: _to_snake(col) for col in df.columns})
    df = df.rename(
        columns={
            "have_you_ever_had_suicidal_thoughts": "suicidal_thoughts",
            "family_history_of_mental_illness": "family_history_mental_illness",
        }
    )
    return df


@st.cache_data(show_spinner=False)
def _load_dataset() -> pd.DataFrame:
    df_raw = pd.read_csv(DATASET_PATH)
    return _clean_dataset(df_raw)


def run() -> None:
    st.subheader("Exploratory Data Analysis")

    if not DATASET_PATH.exists():
        st.error(f"Dataset not found: {DATASET_PATH}")
        return

    df = _load_dataset()
    total_rows, total_cols = df.shape
    missing_values = int(df.isna().sum().sum())
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{total_rows:,}")
    c2.metric("Columns", f"{total_cols:,}")
    c3.metric("Missing Values", f"{missing_values:,}")
    c4.metric("Numeric Features", f"{len(numeric_cols):,}")

    tab_preview, tab_target, tab_numeric, tab_correlation = st.tabs(
        ["Preview", "Target", "Numeric by Target", "Correlation"]
    )

    with tab_preview:
        st.dataframe(df.head(100), use_container_width=True)
        st.dataframe(df.dtypes.astype(str).rename("dtype"))

    with tab_target:
        if TARGET_COL not in df.columns:
            st.warning(f"Target column `{TARGET_COL}` not found.")
        else:
            counts = df[TARGET_COL].value_counts().sort_index()
            props = df[TARGET_COL].value_counts(normalize=True).sort_index()
            left, right = st.columns(2)
            left.dataframe(counts.to_frame("count"), use_container_width=True)
            right.dataframe(props.to_frame("proportion"), use_container_width=True)

            fig, ax = plt.subplots(figsize=(6, 4))
            counts.plot(kind="bar", ax=ax)
            ax.set_title("Target Distribution")
            ax.set_xlabel("depression (0=No, 1=Yes)")
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=0)
            plt.tight_layout()
            st.pyplot(fig)

    with tab_numeric:
        if TARGET_COL not in df.columns:
            st.warning(f"Target column `{TARGET_COL}` not found.")
        else:
            default_num_cols = [
                "age",
                "cgpa",
                "academic_pressure",
                "work_pressure",
                "study_satisfaction",
                "job_satisfaction",
                "work_study_hours",
                "financial_stress",
            ]
            available_cols = [col for col in default_num_cols if col in df.columns]
            if not available_cols:
                st.info("No expected numeric feature found.")
            else:
                selected = st.selectbox("Select numeric feature", available_cols, index=0)
                fig, ax = plt.subplots(figsize=(8, 4))
                df[df[TARGET_COL] == 0][selected].plot(
                    kind="hist", bins=30, alpha=0.6, label="No Depression", ax=ax
                )
                df[df[TARGET_COL] == 1][selected].plot(
                    kind="hist", bins=30, alpha=0.6, label="Depression", ax=ax
                )
                ax.set_title(f"Distribution of {selected} by target")
                ax.set_xlabel(selected)
                ax.set_ylabel("Frequency")
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)

    with tab_correlation:
        if len(numeric_cols) < 2:
            st.info("Correlation needs at least two numeric columns.")
        else:
            corr = df[numeric_cols].corr(numeric_only=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            cax = ax.imshow(corr, interpolation="nearest")
            ax.set_title("Numeric Correlation Heatmap")
            ax.set_xticks(range(len(numeric_cols)))
            ax.set_yticks(range(len(numeric_cols)))
            ax.set_xticklabels(numeric_cols, rotation=90)
            ax.set_yticklabels(numeric_cols)
            fig.colorbar(cax)
            plt.tight_layout()
            st.pyplot(fig)


def render_eda() -> None:
    run()


if __name__ == "__main__":
    run()
