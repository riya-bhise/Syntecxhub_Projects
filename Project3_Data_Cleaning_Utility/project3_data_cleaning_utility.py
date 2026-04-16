"""
=============================================================
PROJECT 3: Data Cleaning Utility
Syntecxhub Data Science Internship
=============================================================
Topics Covered:
  - Detect and handle missing values (drop / fill / impute)
  - Fix incorrect dtypes (dates, numbers) and parse dates
  - Remove duplicates and standardize column names
  - Output a cleaned dataset and brief cleaning log
=============================================================
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

print("=" * 60)
print("       PROJECT 3: Data Cleaning Utility")
print("=" * 60)

# ─────────────────────────────────────────────
# 0. CREATE DIRTY SAMPLE DATA
# ─────────────────────────────────────────────
dirty_data = {
    "  Employee Name  ": [
        "Alice Johnson", "Bob Smith", "Charlie Brown", "diana prince",
        "EVE WILSON", "Frank Castle", "Grace Hopper", "Bob Smith",   # duplicate
        "Henry Ford", "   Iris West ", "Jack Ryan", None,
        "Karen Page", "Leo Fitz", "Mia Toretto"
    ],
    "AGE": [
        28, 35, "forty-two", 29, 31, 27, 38, 35,
        45, 26, "33", None, 40, 30, "36"
    ],
    "Department": [
        "engineering", "Marketing", "ENGINEERING", "hr", "Engineering",
        "marketing ", " HR", "Marketing",
        "engineering", "Marketing", "engineering", "hr",
        "HR", "marketing", "ENGINEERING"
    ],
    "Salary (INR)": [
        75000, "55,000", 95000, 48000, "82000", None, 61000, "55,000",
        "1,10,000", 47000, 88000, 67000,
        None, 58000, "91000"
    ],
    "Joining Date": [
        "15-03-2020", "22/07/2018", "2015-01-10", "30.05.2021", "05-11-2019",
        "18-01-2022", "14/08/2017", "22/07/2018",
        "20-04-2013", "28-02-2023", "11/09/2018", "03-06-2016",
        "9999-12-31", "07-12-2020", "25-03-2017"
    ],
    "Performance Score": [
        8.5, 7.0, None, 9.2, 8.8, 6.5, 9.5, 7.0,
        9.8, 7.3, 8.1, None,
        8.9, 7.6, 9.0
    ],
    "Email": [
        "alice@company.com", "bob@company.com", "charlie@company.com",
        "diana@company.com", "eve@company.com", "frank@company.com",
        "grace@company.com", "bob@company.com",
        "henry@company.com", "iris@company.com", "jack@company.com",
        "karen@company.com", "karen@company.com", "leo@company.com",
        "mia@company.com"
    ],
    "Phone": [
        "9876543210", "  8765432109", "7654321098", "6543210987", "9988776655",
        "abc-xyz", "9123456789", "8765432109",
        "9012345678", "8901234567", "", "7890123456",
        "6789012345", "9999", "5678901234"
    ],
}

os.makedirs("cleaning_output", exist_ok=True)

df_dirty = pd.DataFrame(dirty_data)
df_dirty.to_csv("cleaning_output/dirty_dataset.csv", index=False)
print(f"\n  Dirty dataset created: {df_dirty.shape[0]} rows × {df_dirty.shape[1]} cols")

# ─────────────────────────────────────────────
# CLEANING LOG
# ─────────────────────────────────────────────
log = []

def log_step(step, description, before=None, after=None):
    entry = {
        "step": step,
        "action": description,
        "before": before,
        "after": after,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    log.append(entry)
    before_str = f"  before={before}" if before is not None else ""
    after_str  = f"  after={after}"  if after  is not None else ""
    print(f"  [{step:02d}] {description}{before_str}{after_str}")

df = df_dirty.copy()

print("\n" + "─" * 60)
print("  CLEANING PIPELINE")
print("─" * 60)

# ─────────────────────────────────────────────
# STEP 1: Standardize column names
# ─────────────────────────────────────────────
print("\n[A] Column Name Standardization")
old_cols = list(df.columns)
df.columns = (
    df.columns
      .str.strip()
      .str.lower()
      .str.replace(r"[\s\(\)/]", "_", regex=True)
      .str.replace(r"_+", "_", regex=True)
      .str.strip("_")
)
new_cols = list(df.columns)
log_step(1, "Standardized column names", old_cols, new_cols)
print(f"    Old: {old_cols}")
print(f"    New: {new_cols}")

# ─────────────────────────────────────────────
# STEP 2: Remove exact duplicate rows
# ─────────────────────────────────────────────
print("\n[B] Duplicate Removal")
dup_count = df.duplicated().sum()
df = df.drop_duplicates()
log_step(2, "Removed exact duplicate rows", before=dup_count, after=0)
print(f"    Removed {dup_count} duplicate row(s). Remaining: {len(df)}")

# ─────────────────────────────────────────────
# STEP 3: Clean string columns
# ─────────────────────────────────────────────
print("\n[C] String Column Cleaning")

# Employee name
df["employee_name"] = df["employee_name"].str.strip().str.title()
df["employee_name"] = df["employee_name"].replace("None", np.nan)
null_names = df["employee_name"].isnull().sum()
log_step(3, "Stripped & title-cased employee_name; nulls found", after=null_names)

# Department
df["department"] = df["department"].str.strip().str.title()
log_step(4, "Stripped & title-cased department column")
print(f"    Unique departments: {df['department'].unique()}")

# ─────────────────────────────────────────────
# STEP 4: Fix Salary dtype (remove commas/symbols)
# ─────────────────────────────────────────────
print("\n[D] Salary Column — Fix dtype")

def clean_salary(val):
    if pd.isna(val):
        return np.nan
    cleaned = str(val).replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return np.nan

df["salary_inr"] = df["salary_inr"].apply(clean_salary)
null_sal = df["salary_inr"].isnull().sum()
log_step(5, "Cleaned salary (removed commas, cast to float)",
         before="mixed str/int", after=f"float64, {null_sal} nulls")
print(f"    Salary range: ₹{df['salary_inr'].min():,.0f} – ₹{df['salary_inr'].max():,.0f}")

# ─────────────────────────────────────────────
# STEP 5: Fix Age dtype
# ─────────────────────────────────────────────
print("\n[E] Age Column — Fix dtype")

def clean_age(val):
    if pd.isna(val):
        return np.nan
    try:
        return float(val)
    except (ValueError, TypeError):
        text_map = {"forty-two": 42, "thirty-three": 33}
        return text_map.get(str(val).strip().lower(), np.nan)

df["age"] = df["age"].apply(clean_age)
null_age = df["age"].isnull().sum()
log_step(6, "Fixed age dtype (text->numeric, cast to float)",
         before="mixed types", after=f"float64, {null_age} nulls")
print(f"    Age range: {df['age'].min():.0f} – {df['age'].max():.0f}")

# ─────────────────────────────────────────────
# STEP 6: Parse Joining Date (multiple formats)
# ─────────────────────────────────────────────
print("\n[F] Joining Date — Parse & Fix")

def parse_date(val):
    if pd.isna(val):
        return pd.NaT
    for fmt in ["%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%d.%m.%Y"]:
        try:
            d = datetime.strptime(str(val).strip(), fmt)
            # Flag absurd years as NaT
            if d.year > 2025 or d.year < 1990:
                return pd.NaT
            return d
        except ValueError:
            continue
    return pd.NaT

df["joining_date"] = df["joining_date"].apply(parse_date)
null_dates = df["joining_date"].isnull().sum()
log_step(7, "Parsed joining_date (multi-format, flagged invalid years)",
         before="string mixed formats", after=f"datetime64, {null_dates} NaT")
print(f"    Date range: {df['joining_date'].min().date()} – {df['joining_date'].max().date()}")

# ─────────────────────────────────────────────
# STEP 7: Phone validation & cleaning
# ─────────────────────────────────────────────
print("\n[G] Phone Column — Validate")

def clean_phone(val):
    if pd.isna(val):
        return np.nan
    digits = "".join(filter(str.isdigit, str(val)))
    if len(digits) == 10:
        return digits
    return np.nan

df["phone"] = df["phone"].apply(clean_phone)
null_phones = df["phone"].isnull().sum()
log_step(8, "Validated phone (10-digit, cleaned non-numeric)",
         after=f"{null_phones} invalid→NaN")

# ─────────────────────────────────────────────
# STEP 8: Handle missing values
# ─────────────────────────────────────────────
print("\n[H] Missing Value Treatment")

print("\n  ▶ Missing value counts BEFORE imputation:")
print(df.isnull().sum().to_string())

# Numeric: fill with median (robust to outliers)
before_salary_nulls = df["salary_inr"].isnull().sum()
df["salary_inr"] = df["salary_inr"].fillna(df["salary_inr"].median())
log_step(9, "Imputed salary_inr with median",
         before=before_salary_nulls, after=0)

before_age_nulls = df["age"].isnull().sum()
df["age"] = df["age"].fillna(df["age"].median())
log_step(10, "Imputed age with median",
         before=before_age_nulls, after=0)

before_perf_nulls = df["performance_score"].isnull().sum()
df["performance_score"] = df["performance_score"].fillna(
    df["performance_score"].median())
log_step(11, "Imputed performance_score with median",
         before=before_perf_nulls, after=0)

# Date: fill with median joining date
before_date_nulls = df["joining_date"].isnull().sum()
df["joining_date"] = df["joining_date"].fillna(
    pd.to_datetime(df["joining_date"].dropna().astype("int64").median()))
log_step(12, "Imputed joining_date with median date",
         before=before_date_nulls, after=0)

# Name: drop rows where name is null (unusable)
before_name_nulls = df["employee_name"].isnull().sum()
df = df.dropna(subset=["employee_name"])
log_step(13, "Dropped rows with null employee_name",
         before=before_name_nulls, after=0)

print("\n  ▶ Missing value counts AFTER imputation:")
print(df.isnull().sum().to_string())

# ─────────────────────────────────────────────
# STEP 9: Correct final dtypes
# ─────────────────────────────────────────────
print("\n[I] Final Dtype Correction")
df["age"]     = df["age"].astype(int)
df["salary_inr"] = df["salary_inr"].astype(int)
log_step(14, "Cast age and salary_inr to int")

# Add derived columns
df["years_experience"] = (
    (pd.Timestamp("today") - df["joining_date"]).dt.days // 365
)
log_step(15, "Added derived column: years_experience")

print("\n  ▶ Final dtypes:")
print(df.dtypes.to_string())

# ─────────────────────────────────────────────
# STEP 10: Final dataset overview
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("  CLEANED DATASET PREVIEW")
print("─" * 60)
print(df.to_string(index=False))

print(f"\n  Shape: {df.shape[0]} rows × {df.shape[1]} cols")

# ─────────────────────────────────────────────
# SAVE OUTPUTS
# ─────────────────────────────────────────────
print("\n[J] Saving Outputs")

# Cleaned CSV
df.to_csv("cleaning_output/cleaned_dataset.csv", index=False)
print("  ✅ cleaning_output/cleaned_dataset.csv")

# Excel with both sheets
with pd.ExcelWriter("cleaning_output/cleaning_report.xlsx", engine="openpyxl") as writer:
    df_dirty.to_excel(writer, sheet_name="Dirty_Data",   index=False)
    df.to_excel(      writer, sheet_name="Cleaned_Data", index=False)

    # Cleaning log sheet
    log_df = pd.DataFrame(log)
    log_df.to_excel(writer, sheet_name="Cleaning_Log", index=False)
print("  ✅ cleaning_output/cleaning_report.xlsx  (3 sheets)")

# Cleaning log as standalone CSV
log_df.to_csv("cleaning_output/cleaning_log.csv", index=False)
print("  ✅ cleaning_output/cleaning_log.csv")

# ─────────────────────────────────────────────
# PRINT CLEANING LOG
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("  CLEANING LOG SUMMARY")
print("─" * 60)
print(f"  {'Step':<5} {'Action':<60} {'Before':>10} {'After':>10}")
print("  " + "-" * 87)
for entry in log:
    b = str(entry["before"])[:10] if entry["before"] is not None else "-"
    a = str(entry["after"])[:10]  if entry["after"]  is not None else "-"
    print(f"  [{entry['step']:02d}]  {entry['action']:<58} {b:>10} {a:>10}")

print("\n" + "=" * 60)
print("  Project 3 Complete — Data Cleaning Utility ✅")
print("=" * 60)
