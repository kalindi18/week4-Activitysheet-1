import streamlit as st
import pandas as pd

st.title("Student Performance Dashboard")

data = pd.read_csv("cleaned_student_performance.csv")

st.metric("Number of students", len(data))

number_columns = data.select_dtypes(include="number").columns

if len(number_columns) > 0:
    st.subheader("Average scores")
    st.bar_chart(data[number_columns].mean())

st.subheader("Student data")
st.dataframe(data)
