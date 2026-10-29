
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

#Folder containing this python file
BASE_DIR = Path(__file__).resolve().parent

# Load the CSV file
df = pd.read_csv(BASE_DIR / "student_performance.csv")

# Show DataFrame information
print("Dimensions:", df.shape)
print("\nData types:")
print(df.dtypes)
print("\nFirst rows:")
print(df.head())
print("\nSummary statistics:")
print(df.describe())

# Exam statistics and highest mark
print("\nExam mean:", df["Exam"].mean())
print("Exam median:", df["Exam"].median())
print("Exam standard deviation:", df["Exam"].std())

top_student = df.loc[df["Exam"].idxmax()]
print("Highest exam mark:", top_student["Exam"])
print("Student:", top_student["Student"])

# Correlation between study hours and exam marks
print("Study Hours and Exam correlation:",
      df["Study Hours"].corr(df["Exam"]))

# Add exam categories
df["Category"] = pd.cut(
    df["Exam"],
    bins=[float("-inf"), 60, 80, float("inf")],
    labels=["Low", "Medium", "High"],
    right=False
)

# Save the data and summary statistics
df.to_csv(BASE_DIR / "cleaned_student_performance.csv",
          index=False)
df.describe().to_csv(BASE_DIR / "summary_statistics.csv")

# Scatter plot
plt.scatter(df["Study Hours"], df["Exam"])
plt.xlabel("Study Hours")
plt.ylabel("Exam marks")
plt.title("Study Hours vs Exam Marks")
plt.show()

# Histogram
plt.hist(df["Exam"], bins=5, edgecolor="black")
plt.xlabel("Exam marks")
plt.ylabel("Number of students")
plt.title("Exam Marks")
plt.show()

# Box plot
plt.boxplot(df["Exam"])
plt.ylabel("Exam marks")
plt.title("Exam Marks Box Plot")
plt.show()

# Heatmap of Study Hours and Assignment marks
plt.hist2d(df["Study Hours"], df["Assignment"], bins=5)
plt.colorbar(label="Number of students")
plt.xlabel("Study Hours")
plt.ylabel("Assignment marks")
plt.title("Study Hours vs Assignment Marks")
plt.show()