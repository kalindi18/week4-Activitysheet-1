from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import sympy as sp

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="Activity Sheet Dashboard", layout="wide")
st.title("Activity Sheet Dashboard")
st.caption("Answers and charts for Questions 1–4")

q1_tab, q2_tab, q3_tab, q4_tab = st.tabs(
    ["Q1: Study hours", "Q2: Matrices", "Q3: Algebra", "Q4: Student results"]
)

with q1_tab:
    st.header("Question 1 — Study hours")
    original_hours = np.array(
        [5, 2, 8, 4, 7, 1, 5, 10, 4, 7, 3, 6, 3, 5, 2, 7, 4, 9, 5, 11]
    )
    updated_hours = original_hours.copy()
    updated_hours[2] = 12

    def stats(values):
        return {
            "Mean": np.mean(values),
            "Median": np.median(values),
            "Sample variance": np.var(values, ddof=1),
            "Sample standard deviation": np.std(values, ddof=1),
        }

    q1_results = pd.DataFrame(
        {
            "Original data": stats(original_hours),
            "After changing 8 to 12": stats(updated_hours),
        }
    ).round(2)
    st.dataframe(q1_results, use_container_width=True)

    fig, ax = plt.subplots()
    ax.hist(updated_hours, bins=np.arange(0.5, 13.5, 1), edgecolor="black")
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Study hours")
    ax.set_ylabel("Number of students")
    ax.set_title("Updated study hours")
    st.pyplot(fig)
    plt.close(fig)

with q2_tab:
    st.header("Question 2 — Matrix calculations")
    matrix_a = np.array([[2, 1, 1], [1, 3, 2], [3, 1, 2]], dtype=float)
    vector_b = np.array([10, 18, 17], dtype=float)
    names = ["x1", "x2", "x3"]

    left, right = st.columns(2)
    with left:
        st.subheader("Matrix A")
        st.dataframe(pd.DataFrame(matrix_a, columns=names), use_container_width=True)
    with right:
        st.subheader("Vector B")
        st.dataframe(pd.DataFrame({"B": vector_b}), use_container_width=True)

    determinant = np.linalg.det(matrix_a)
    inverse = np.linalg.inv(matrix_a)
    eigenvalues, eigenvectors = np.linalg.eig(matrix_a)
    solution = np.linalg.solve(matrix_a, vector_b)
    residual = matrix_a @ solution - vector_b
    residual_norm = np.linalg.norm(residual, ord=2)

    st.metric("Determinant of A", f"{determinant:.2f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Inverse of A")
        st.dataframe(
            pd.DataFrame(inverse, index=names, columns=names).round(3),
            use_container_width=True,
        )
    with right:
        st.subheader("Eigenvalues")
        st.dataframe(
            pd.DataFrame({"Eigenvalue": np.round(eigenvalues, 3)}),
            use_container_width=True,
        )

    st.subheader("Eigenvectors")
    st.caption("Each column corresponds to the eigenvalue in the same position above.")
    st.dataframe(
        pd.DataFrame(
            np.round(eigenvectors, 3),
            index=["row 1", "row 2", "row 3"],
            columns=["Eigenvector 1", "Eigenvector 2", "Eigenvector 3"],
        ),
        use_container_width=True,
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Solution to A × X = B")
        st.dataframe(
            pd.DataFrame({"Value": np.round(solution, 3)}, index=names),
            use_container_width=True,
        )
    with right:
        st.subheader("Residual")
        st.dataframe(
            pd.DataFrame({"Residual": np.round(residual, 10)}, index=["row 1", "row 2", "row 3"]),
            use_container_width=True,
        )
        st.metric("Euclidean norm of residual", f"{residual_norm:.3e}")

with q3_tab:
    st.header("Question 3 — Algebra and calculus")
    t = sp.symbols("t")
    function = t**3 - 6 * t**2 + 9 * t + 2

    st.latex(r"f(t) = " + sp.latex(function))
    st.subheader("(a) Factorisation")
    st.latex(sp.latex(sp.factor(function)))

    st.subheader("(b) Solutions")
    roots_zero = sp.solve(function, t)
    roots_two = sp.solve(function - 2, t)
    st.write("Solutions to f(t) = 0:")
    st.latex(sp.latex(sp.FiniteSet(*roots_zero)))
    st.write("Solutions to f(t) = 2:")
    st.latex(sp.latex(sp.FiniteSet(*roots_two)))

    st.subheader("(c) First and second derivatives")
    st.latex(r"f'(t) = " + sp.latex(sp.diff(function, t)))
    st.latex(r"f''(t) = " + sp.latex(sp.diff(function, t, 2)))

    st.subheader("(d) Indefinite integral")
    st.latex(
        r"\int f(t)\,dt = " + sp.latex(sp.integrate(function, t)) + r" + C"
    )

    st.subheader("(e) Definite integral from 0 to 2")
    definite_result = sp.integrate(function, (t, 0, 2))
    st.latex(
        r"\int_0^2 f(t)\,dt = " + sp.latex(definite_result)
    )

with q4_tab:
    st.header("Question 4 — Student performance")
    csv_path = BASE_DIR / "student_performance.csv"

    if not csv_path.exists():
        st.error("student_performance.csv must be in the same GitHub folder as app.py.")
    else:
        students = pd.read_csv(csv_path)
        required = {"Student", "Study Hours", "Assignment", "Exam"}
        missing = required.difference(students.columns)

        if missing:
            st.error("The CSV is missing these columns: " + ", ".join(sorted(missing)))
            st.dataframe(students.head(), use_container_width=True)
        else:
            exam_rows = students.dropna(subset=["Exam"])
            if exam_rows.empty:
                st.error("There are no exam marks to summarize.")
            else:
                average_exam = exam_rows["Exam"].mean()
                median_exam = exam_rows["Exam"].median()
                std_exam = exam_rows["Exam"].std()
                top_student = exam_rows.loc[exam_rows["Exam"].idxmax()]
                correlation = students["Study Hours"].corr(students["Exam"])

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Average exam mark", f"{average_exam:.2f}")
                c2.metric("Median exam mark", f"{median_exam:.2f}")
                c3.metric("Exam standard deviation", f"{std_exam:.2f}")
                c4.metric("Highest exam mark", f"{top_student['Exam']:.2f}")
                st.write(f"Student with the highest mark: **{top_student['Student']}**")
                st.write(f"Correlation: **{correlation:.3f}**")

                students_with_category = students.copy()
                students_with_category["Category"] = pd.cut(
                    students_with_category["Exam"],
                    bins=[float("-inf"), 60, 80, float("inf")],
                    labels=["Low", "Medium", "High"],
                    right=False,
                )

                left, right = st.columns(2)
                with left:
                    st.subheader("Summary statistics")
                    st.dataframe(
                        students.describe().round(2).T,
                        use_container_width=True,
                    )
                with right:
                    st.subheader("Students by exam category")
                    category_counts = (
                        students_with_category["Category"]
                        .value_counts()
                        .reindex(["Low", "Medium", "High"], fill_value=0)
                    )
                    st.bar_chart(category_counts)

                st.subheader("Charts")
                chart_left, chart_right = st.columns(2)
                with chart_left:
                    fig, ax = plt.subplots()
                    ax.scatter(students["Study Hours"], students["Exam"])
                    ax.set_xlabel("Study hours")
                    ax.set_ylabel("Exam marks")
                    ax.set_title("Study hours vs exam marks")
                    st.pyplot(fig)
                    plt.close(fig)

                    fig, ax = plt.subplots()
                    ax.boxplot(exam_rows["Exam"])
                    ax.set_ylabel("Exam marks")
                    ax.set_title("Exam marks box plot")
                    st.pyplot(fig)
                    plt.close(fig)

                with chart_right:
                    fig, ax = plt.subplots()
                    ax.hist(exam_rows["Exam"], bins=5, edgecolor="black")
                    ax.set_xlabel("Exam marks")
                    ax.set_ylabel("Number of students")
                    ax.set_title("Exam marks")
                    st.pyplot(fig)
                    plt.close(fig)

                    assignment_rows = students.dropna(subset=["Study Hours", "Assignment"])
                    fig, ax = plt.subplots()
                    heat = ax.hist2d(
                        assignment_rows["Study Hours"],
                        assignment_rows["Assignment"],
                        bins=5,
                    )
                    fig.colorbar(heat[3], ax=ax, label="Number of students")
                    ax.set_xlabel("Study hours")
                    ax.set_ylabel("Assignment marks")
                    ax.set_title("Study hours vs assignment marks")
                    st.pyplot(fig)
                    plt.close(fig)

                st.subheader("Student records with exam categories")
                st.dataframe(students_with_category, use_container_width=True)
