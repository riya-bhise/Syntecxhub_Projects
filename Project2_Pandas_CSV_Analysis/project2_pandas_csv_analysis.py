"""
=============================================================
PROJECT 2: Pandas CSV Reader & Basic Analysis
Syntecxhub Data Science Internship
=============================================================
Topics Covered:
  - Read CSV/Excel into DataFrame; inspect head/tail/types
  - Compute summary stats (mean, median, min, max, count)
  - Filter rows, select columns, and slice subsets
  - Save filtered results to CSV/Excel
=============================================================
"""

import pandas as pd
import numpy as np
import os

print("=" * 60)
print("  PROJECT 2: Pandas CSV Reader & Basic Analysis")
print("=" * 60)

# ─────────────────────────────────────────────
# 0. CREATE SAMPLE DATA (if not present)
# ─────────────────────────────────────────────
CSV_FILE   = "sample_data.csv"
EXCEL_FILE = "sample_data.xlsx"

if not os.path.exists(CSV_FILE):
    data = {
        "Name":        ["Alice","Bob","Charlie","Diana","Eve","Frank","Grace",
                         "Henry","Iris","Jack","Karen","Leo","Mia","Noah","Olivia"],
        "Age":         [28,35,42,29,31,27,38,45,26,33,40,30,36,28,43],
        "Department":  ["Engineering","Marketing","Engineering","HR","Engineering",
                        "Marketing","HR","Engineering","Marketing","Engineering",
                        "HR","Marketing","Engineering","HR","Marketing"],
        "Salary":      [75000,55000,95000,48000,82000,52000,61000,110000,47000,
                        88000,67000,58000,91000,50000,62000],
        "JoiningDate": ["2020-03-15","2018-07-22","2015-01-10","2021-05-30",
                        "2019-11-05","2022-01-18","2017-08-14","2013-04-20",
                        "2023-02-28","2018-09-11","2016-06-03","2020-12-07",
                        "2017-03-25","2022-08-19","2014-10-31"],
        "City":        ["Mumbai","Delhi","Bangalore","Mumbai","Hyderabad","Chennai",
                        "Delhi","Bangalore","Mumbai","Pune","Delhi","Chennai",
                        "Hyderabad","Mumbai","Bangalore"],
    }
    pd.DataFrame(data).to_csv(CSV_FILE, index=False)
    print(f"  ↳ Created sample file: {CSV_FILE}")

# Also save as Excel for demonstration
df_temp = pd.read_csv(CSV_FILE)
df_temp.to_excel(EXCEL_FILE, index=False)

os.makedirs("pandas_output", exist_ok=True)

# ─────────────────────────────────────────────
# 1. READ CSV & EXCEL
# ─────────────────────────────────────────────
print("\n[1] Reading CSV & Excel Files")
print("-" * 40)

df_csv   = pd.read_csv(CSV_FILE, parse_dates=["JoiningDate"])
df_excel = pd.read_excel(EXCEL_FILE)

print(f"CSV   -> shape  : {df_csv.shape}  (rows x cols)")
print(f"Excel -> shape  : {df_excel.shape}")

# ─────────────────────────────────────────────
# 2. INSPECT — head / tail / dtypes / info
# ─────────────────────────────────────────────
print("\n[2] Inspection")
print("-" * 40)
df = df_csv.copy()

print("\n▶ head(5):")
print(df.head(5).to_string(index=False))

print("\n▶ tail(3):")
print(df.tail(3).to_string(index=False))

print("\n▶ Column dtypes:")
print(df.dtypes.to_string())

print("\n▶ DataFrame.info():")
df.info()

print("\n▶ Shape :", df.shape)
print("▶ Columns:", list(df.columns))
print("▶ Null counts:\n", df.isnull().sum().to_string())

# ─────────────────────────────────────────────
# 3. SUMMARY STATISTICS
# ─────────────────────────────────────────────
print("\n[3] Summary Statistics")
print("-" * 40)

print("\n▶ describe() — numeric columns:")
print(df.describe().round(2).to_string())

numeric_cols = ["Age", "Salary"]
for col in numeric_cols:
    print(f"\n  ── {col} ──")
    print(f"    Mean      : {df[col].mean():.2f}")
    print(f"    Median    : {df[col].median():.2f}")
    print(f"    Min       : {df[col].min()}")
    print(f"    Max       : {df[col].max()}")
    print(f"    Count     : {df[col].count()}")
    print(f"    Std Dev   : {df[col].std():.2f}")
    print(f"    Variance  : {df[col].var():.2f}")

print("\n▶ Value counts — Department:")
print(df["Department"].value_counts().to_string())

print("\n▶ Value counts — City:")
print(df["City"].value_counts().to_string())

print("\n▶ Mean Salary by Department:")
print(df.groupby("Department")["Salary"].mean().round(2).to_string())

print("\n▶ Mean Salary by City:")
print(df.groupby("City")["Salary"].mean().round(2).to_string())

# ─────────────────────────────────────────────
# 4. FILTER ROWS
# ─────────────────────────────────────────────
print("\n[4] Filtering Rows")
print("-" * 40)

# Single condition
eng_df = df[df["Department"] == "Engineering"]
print(f"\n▶ Engineering employees ({len(eng_df)} rows):")
print(eng_df[["Name", "Age", "Salary", "City"]].to_string(index=False))

# Salary > 70000
high_sal = df[df["Salary"] > 70000]
print(f"\n▶ Salary > 70,000 ({len(high_sal)} rows):")
print(high_sal[["Name", "Department", "Salary"]].to_string(index=False))

# Multiple conditions (AND)
senior_eng = df[(df["Department"] == "Engineering") & (df["Age"] > 35)]
print(f"\n▶ Engineering AND Age > 35 ({len(senior_eng)} rows):")
print(senior_eng[["Name", "Age", "Salary"]].to_string(index=False))

# OR condition
mumbai_delhi = df[(df["City"] == "Mumbai") | (df["City"] == "Delhi")]
print(f"\n▶ Mumbai OR Delhi ({len(mumbai_delhi)} rows):")
print(mumbai_delhi[["Name", "City", "Department"]].to_string(index=False))

# isin filter
selected_depts = df[df["Department"].isin(["HR", "Marketing"])]
print(f"\n▶ HR or Marketing via isin ({len(selected_depts)} rows):")
print(selected_depts[["Name", "Department", "Salary"]].to_string(index=False))

# Date filter
recent_joiners = df[df["JoiningDate"] >= "2020-01-01"]
print(f"\n▶ Joined after 2020-01-01 ({len(recent_joiners)} rows):")
print(recent_joiners[["Name", "JoiningDate", "Department"]].to_string(index=False))

# ─────────────────────────────────────────────
# 5. SELECT COLUMNS & SLICE SUBSETS
# ─────────────────────────────────────────────
print("\n[5] Selecting Columns & Slicing")
print("-" * 40)

# Select specific columns
subset = df[["Name", "Salary", "Department"]]
print("\n▶ Select ['Name','Salary','Department']:")
print(subset.head(5).to_string(index=False))

# loc — label-based
print("\n▶ loc rows 0-4, cols Name→Department:")
print(df.loc[0:4, "Name":"Department"].to_string(index=False))

# iloc — integer-based
print("\n▶ iloc first 3 rows, last 2 cols:")
print(df.iloc[:3, -2:].to_string(index=False))

# Sort
sorted_df = df.sort_values("Salary", ascending=False)
print("\n▶ Sorted by Salary (desc):")
print(sorted_df[["Name", "Salary", "Department"]].head(8).to_string(index=False))

# Top N per group
top_salary = (df.sort_values("Salary", ascending=False)
                .groupby("Department").head(2))
print("\n▶ Top-2 salaries per Department:")
print(top_salary[["Department","Name","Salary"]].sort_values(
      ["Department","Salary"], ascending=[True,False]).to_string(index=False))

# ─────────────────────────────────────────────
# 6. SAVE FILTERED RESULTS
# ─────────────────────────────────────────────
print("\n[6] Saving Filtered Results")
print("-" * 40)

# Save all to CSV
df.to_csv("pandas_output/full_dataset.csv", index=False)
print("  ✅ pandas_output/full_dataset.csv")

# Engineering employees → CSV
eng_df.to_csv("pandas_output/engineering_employees.csv", index=False)
print("  ✅ pandas_output/engineering_employees.csv")

# High salary → CSV
high_sal.to_csv("pandas_output/high_salary_above_70k.csv", index=False)
print("  ✅ pandas_output/high_salary_above_70k.csv")

# Recent joiners → CSV
recent_joiners.to_csv("pandas_output/recent_joiners_2020_onwards.csv", index=False)
print("  ✅ pandas_output/recent_joiners_2020_onwards.csv")

# Summary stats → Excel (multi-sheet)
with pd.ExcelWriter("pandas_output/analysis_summary.xlsx", engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="Full_Data", index=False)
    eng_df.to_excel(writer, sheet_name="Engineering", index=False)
    high_sal.to_excel(writer, sheet_name="HighSalary_70k+", index=False)
    df.groupby("Department")[["Age","Salary"]].mean().round(2).to_excel(
        writer, sheet_name="Dept_Stats")
    df.describe().round(2).to_excel(writer, sheet_name="Describe")
print("  ✅ pandas_output/analysis_summary.xlsx  (5 sheets)")

print("\n" + "=" * 60)
print("  Project 2 Complete — Pandas CSV Analysis ✅")
print("=" * 60)
