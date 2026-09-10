
import os
import csv
import pandas as pd


def load_dataset(file_path):
    """Load the student dataset from a local file or GitHub URL."""

    if file_path.startswith(("http://", "https://")):
        data_frame = pd.read_csv(file_path)
    else:
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Error: The file '{file_path}' was not found.")

        data_frame = pd.read_csv(file_path)

    print(f"[INFO] Successfully loaded dataset from '{file_path}'")
    print(
        f"[INFO] Total Rows: {len(data_frame)}, Total Columns: {len(data_frame.columns)}")

    return data_frame


def audit_dataset(df):

    audit_results = {}

    # 1. Dimensions
    total_rows = len(df)
    total_columns = len(df.columns)
    audit_results['total_rows'] = total_rows
    audit_results['total_columns'] = total_columns

    # 2. Missing values check
    missing_counts = {}
    total_missing = 0
    for col in df.columns:
        col_missing = int(df[col].isnull().sum())
        missing_counts[col] = col_missing
        total_missing = total_missing + col_missing
    audit_results['missing_counts'] = missing_counts
    audit_results['total_missing'] = total_missing

    # 3. Duplicate rows check
    duplicate_count = int(df.duplicated().sum())
    audit_results['duplicate_count'] = duplicate_count

    # 4. Column data types
    data_types = {}
    for col in df.columns:
        data_types[col] = str(df[col].dtype)
    audit_results['data_types'] = data_types

    # 5. Range validation for numeric attributes
    range_violations = []

    # Age should be between 15 and 22
    invalid_age = df[(df['age'] < 15) | (df['age'] > 22)]
    if len(invalid_age) > 0:
        range_violations.append(
            f"Age has {len(invalid_age)} values outside [15, 22].")

    # Grades G1, G2, G3 should be between 0 and 20
    for grade_col in ['G1', 'G2', 'G3']:
        invalid_grades = df[(df[grade_col] < 0) | (df[grade_col] > 20)]
        if len(invalid_grades) > 0:
            range_violations.append(
                f"{grade_col} has {len(invalid_grades)} values outside [0, 20].")

    # Absences should be non-negative
    invalid_absences = df[df['absences'] < 0]
    if len(invalid_absences) > 0:
        range_violations.append(
            f"Absences has {len(invalid_absences)} negative values.")

    audit_results['range_violations'] = range_violations

    # 6. Target Variable Analysis (G3)
    g3_values = df['G3'].tolist()
    g3_mean = sum(g3_values) / len(g3_values)
    g3_min = min(g3_values)
    g3_max = max(g3_values)

    # Calculate pass (G3 >= 10) vs at-risk (G3 < 10)
    pass_count = 0
    fail_count = 0
    for grade in g3_values:
        if grade >= 10:
            pass_count = pass_count + 1
        else:
            fail_count = fail_count + 1

    audit_results['g3_stats'] = {
        'mean': round(g3_mean, 2),
        'min': g3_min,
        'max': g3_max,
        'pass_count': pass_count,
        'fail_count': fail_count,
        'pass_percentage': round((pass_count / total_rows) * 100, 2),
        'fail_percentage': round((fail_count / total_rows) * 100, 2)
    }

    return audit_results


def print_audit_report(audit_results):

    print("=" * 60)
    print("           DATASET AUDIT REPORT (DW-3)")
    print("=" * 60)
    print(f"Total Records (Rows)    : {audit_results['total_rows']}")
    print(f"Total Attributes (Cols) : {audit_results['total_columns']}")
    print(f"Total Missing Values    : {audit_results['total_missing']}")
    print(f"Total Duplicate Rows    : {audit_results['duplicate_count']}")

    print("\n--- Range & Validity Checks ---")
    if len(audit_results['range_violations']) == 0:
        print(" [PASSED] All numerical features fall within valid expected boundaries.")
    else:
        for violation in audit_results['range_violations']:
            print(f" [WARNING] {violation}")

    print("\n--- Target Variable (G3) Summary ---")
    stats = audit_results['g3_stats']
    print(
        f" Final Grade Range      : {stats['min']} to {stats['max']} (Mean: {stats['mean']:.2f})")
    print(
        f" Passing Students (>=10): {stats['pass_count']} ({stats['pass_percentage']}%)")
    print(
        f" At-Risk Students (<10) : {stats['fail_count']} ({stats['fail_percentage']}%)")
    print("=" * 60)
