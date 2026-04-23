from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.joblib"


@st.cache_resource(show_spinner=False)
def _load_model() -> tuple[Any | None, str | None]:
    if not MODEL_PATH.exists():
        return None, f"Model file not found: {MODEL_PATH}"

    try:
        return joblib.load(MODEL_PATH), None
    except Exception as err:
        return None, str(err)


def _build_input_dataframe(
    age: float,
    academic_pressure: float,
    work_pressure: float,
    cgpa: float,
    study_satisfaction: float,
    job_satisfaction: float,
    work_study_hours: float,
    financial_stress: float,
    gender: str,
    city: str,
    profession: str,
    sleep_duration: str,
    dietary_habits: str,
    degree: str,
    suicidal_thoughts: str,
    family_history_mental_illness: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "age": float(age),
                "academic_pressure": float(academic_pressure),
                "work_pressure": float(work_pressure),
                "cgpa": float(cgpa),
                "study_satisfaction": float(study_satisfaction),
                "job_satisfaction": float(job_satisfaction),
                "work_study_hours": float(work_study_hours),
                "financial_stress": float(financial_stress),
                "gender": gender,
                "city": city,
                "profession": profession,
                "sleep_duration": sleep_duration,
                "dietary_habits": dietary_habits,
                "degree": degree,
                "suicidal_thoughts": suicidal_thoughts,
                "family_history_mental_illness": family_history_mental_illness,
            }
        ]
    )


def run() -> None:
    st.subheader("Prediction")

    model, load_error = _load_model()
    if model is None:
        st.error("Model failed to load.")
        if load_error:
            st.code(load_error)
        return

    with st.form("form_depression"):
        name = st.text_input("Name (optional)", value="")

        st.markdown("### Numeric Input")
        n_left, n_right = st.columns(2)
        with n_left:
            age = st.number_input("Age", min_value=10, max_value=100, value=20, step=1)
            cgpa = st.number_input(
                "CGPA", min_value=0.0, max_value=10.0, value=7.0, step=0.1
            )
            academic_pressure = st.number_input(
                "Academic Pressure (0-5)",
                min_value=0.0,
                max_value=5.0,
                value=2.0,
                step=0.5,
            )
            work_pressure = st.number_input(
                "Work Pressure (0-5)", min_value=0.0, max_value=5.0, value=1.0, step=0.5
            )
        with n_right:
            study_satisfaction = st.number_input(
                "Study Satisfaction (0-5)",
                min_value=0.0,
                max_value=5.0,
                value=3.0,
                step=0.5,
            )
            job_satisfaction = st.number_input(
                "Job Satisfaction (0-5)",
                min_value=0.0,
                max_value=5.0,
                value=3.0,
                step=0.5,
            )
            work_study_hours = st.number_input(
                "Work/Study Hours (per day)",
                min_value=0.0,
                max_value=24.0,
                value=6.0,
                step=0.5,
            )
            financial_stress = st.number_input(
                "Financial Stress (0-5)",
                min_value=0.0,
                max_value=5.0,
                value=2.0,
                step=0.5,
            )

        st.markdown("### Categorical Input")
        c_left, c_right = st.columns(2)
        with c_left:
            gender = st.selectbox("Gender", ("Male", "Female", "Other"), index=0)
            city = st.text_input("City", value="Unknown")
            profession = st.text_input("Profession", value="Student")
            sleep_duration = st.selectbox(
                "Sleep Duration",
                ("Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"),
                index=2,
            )
        with c_right:
            dietary_habits = st.selectbox(
                "Dietary Habits", ("Healthy", "Moderate", "Unhealthy"), index=1
            )
            degree = st.text_input("Degree", value="Bachelor")
            suicidal_thoughts = st.selectbox(
                "Suicidal Thoughts?", ("Yes", "No"), index=1
            )
            family_history_mental_illness = st.selectbox(
                "Family History of Mental Illness?", ("Yes", "No"), index=1
            )

        submitted = st.form_submit_button("Predict")

    data_inf = _build_input_dataframe(
        age=age,
        academic_pressure=academic_pressure,
        work_pressure=work_pressure,
        cgpa=cgpa,
        study_satisfaction=study_satisfaction,
        job_satisfaction=job_satisfaction,
        work_study_hours=work_study_hours,
        financial_stress=financial_stress,
        gender=gender,
        city=city,
        profession=profession,
        sleep_duration=sleep_duration,
        dietary_habits=dietary_habits,
        degree=degree,
        suicidal_thoughts=suicidal_thoughts,
        family_history_mental_illness=family_history_mental_illness,
    )

    st.write("Input Preview")
    st.dataframe(data_inf, use_container_width=True)

    if submitted:
        try:
            y_pred = model.predict(data_inf)[0]
            proba = model.predict_proba(data_inf)[0] if hasattr(model, "predict_proba") else None

            label = "Depression" if int(y_pred) == 1 else "No Depression"
            st.success(f"Prediction Result: {label} (class={int(y_pred)})")

            if proba is not None and len(proba) >= 2:
                st.write(f"No Depression probability: {proba[0]:.4f}")
                st.write(f"Depression probability: {proba[1]:.4f}")

            if name.strip():
                st.caption(f"Input by: {name.strip()}")
        except Exception as err:
            st.error(f"Prediction failed: {err}")


def render_prediction() -> None:
    run()


if __name__ == "__main__":
    run()
