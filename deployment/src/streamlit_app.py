import streamlit as st

from eda import render_eda
from prediction import render_prediction

st.set_page_config(
    page_title="Student Depression App",
    layout="wide",
)

st.title("Student Depression Analysis and Prediction")
st.caption("Dashboard for EDA and prediction based on the trained model.")

with st.sidebar:
    st.header("Menu")
    page = st.radio("Select page", ["EDA", "Prediction"], index=0)

if page == "EDA":
    render_eda()
else:
    render_prediction()
