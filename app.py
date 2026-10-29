from pathlib import Path
import pandas as pd
import streamlit as st

st.title("Student Performance Dashboard")

folder = Path(__file__).parent

st.header("Assignment Questions")

question_files = [
    "Q1.py",
    "Q2.py",
    "Q3.py",
    "Q4.py",
]

for filename in question_files:
    file = folder / filename
    if file.exists():
        st.subheader(filename)
        content = file.read_text(encoding="utf-8")

        if filename.endswith(".py"):
            st.code(content, language="python")
        else:
            st.markdown(content)

st.header("Student Performance Data")

csv_file = folder / "cleaned_student_performance.csv"
data = pd.read_csv(csv_file)

st.metric("Number of students", len(data))

number_columns = data.select_dtypes(include="number").columns
if len(number_columns) > 0:
    st.subheader("Average scores")
    st.bar_chart(data[number_columns].mean())

st.subheader("Student data")
st.dataframe(data)
