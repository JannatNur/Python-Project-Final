

import matplotlib.pyplot as plt
import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')


def compute_summary_statistics(df, numeric_columns=None):

    if numeric_columns is None:
        numeric_columns = df.select_dtypes(
            include=[np.number]).columns.tolist()

    stats_dict = {}
    for col in numeric_columns:
        values = df[col].dropna().tolist()
        n = len(values)
        if n == 0:
            continue

        col_mean = sum(values) / n

        # Standard deviation calculation
        variance = sum((x - col_mean) ** 2 for x in values) / \
            (n - 1) if n > 1 else 0
        col_std = math.sqrt(variance)

        # Median and Quartiles
        sorted_vals = sorted(values)
        col_min = sorted_vals[0]
        col_max = sorted_vals[-1]

        if n % 2 == 1:
            col_median = sorted_vals[n // 2]
        else:
            col_median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0

        stats_dict[col] = {
            'count': n,
            'mean': round(col_mean, 2),
            'std': round(col_std, 2),
            'min': col_min,
            'median': round(col_median, 2),
            'max': col_max
        }

    return stats_dict


def plot_target_distribution(df, output_path="figures/target_distribution.png"):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Final Grade (G3) Histogram
    grades = df['G3']
    axes[0].hist(grades, bins=21, range=(0, 20),
                 color='#2b5c8f', edgecolor='black', alpha=0.85)
    axes[0].axvline(x=10, color='crimson', linestyle='--',
                    linewidth=2, label='Passing Threshold (10/20)')
    axes[0].set_title("Distribution of Final Grades (G3)",
                      fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Final Grade (Scale: 0 - 20)", fontsize=11)
    axes[0].set_ylabel("Number of Students", fontsize=11)
    axes[0].set_xticks(range(0, 21, 2))
    axes[0].grid(axis='y', linestyle=':', alpha=0.7)
    axes[0].legend(loc='upper right')

    # 2. Pass vs At-Risk Bar Chart
    pass_count = int((df['G3'] >= 10).sum())
    fail_count = int((df['G3'] < 10).sum())
    categories = ['Passing (>=10)', 'At-Risk (<10)']
    counts = [pass_count, fail_count]
    colors = ['#2ca02c', '#d62728']

    bars = axes[1].bar(categories, counts, color=colors,
                       edgecolor='black', width=0.55)
    axes[1].set_title(
        "Student Classification Outcome (Target Balance)", fontsize=13, fontweight='bold')
    axes[1].set_ylabel("Student Count", fontsize=11)
    axes[1].set_ylim(0, max(counts) + 50)
    axes[1].grid(axis='y', linestyle=':', alpha=0.7)

    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        percentage = (height / len(df)) * 100
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 6,
                     f"{height} ({percentage:.1f}%)",
                     ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved target distribution plot to '{output_path}'.")


def plot_feature_relationships(df, output_path="figures/eda_relationships.png"):
    """
    Generate multi-panel exploratory plots showing key predictors vs Final Grade (G3).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Study Time vs Final Grade
    study_groups = df.groupby('studytime')['G3'].mean()
    study_labels = ['<2 hrs', '2-5 hrs', '5-10 hrs', '>10 hrs']
    axes[0, 0].bar(study_labels, study_groups.values,
                   color='#3470a3', edgecolor='black', width=0.5)
    axes[0, 0].set_title(
        "Average Final Grade by Weekly Study Time", fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel("Weekly Study Time Category", fontsize=10)
    axes[0, 0].set_ylabel("Average G3 Grade (0-20)", fontsize=10)
    axes[0, 0].set_ylim(0, 20)
    axes[0, 0].grid(axis='y', linestyle=':', alpha=0.6)
    for i, v in enumerate(study_groups.values):
        axes[0, 0].text(i, v + 0.4, f"{v:.2f}", ha='center', fontweight='bold')

    # 2. Number of Past Failures vs Final Grade
    fail_groups = df.groupby('failures')['G3'].mean()
    fail_labels = [f"{k} Failure(s)" for k in fail_groups.index]
    axes[0, 1].bar(fail_labels, fail_groups.values,
                   color='#d95f02', edgecolor='black', width=0.5)
    axes[0, 1].set_title("Average Final Grade by Past Failures",
                         fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel("Past Academic Failures Count", fontsize=10)
    axes[0, 1].set_ylabel("Average G3 Grade (0-20)", fontsize=10)
    axes[0, 1].set_ylim(0, 20)
    axes[0, 1].grid(axis='y', linestyle=':', alpha=0.6)
    for i, v in enumerate(fail_groups.values):
        axes[0, 1].text(i, v + 0.4, f"{v:.2f}", ha='center', fontweight='bold')

    # 3. School Absences vs Final Grade Scatter Plot
    axes[1, 0].scatter(df['absences'], df['G3'], alpha=0.6,
                       color='#7570b3', edgecolors='none', s=45)
    # Add simple trend line using numpy polyfit
    z = np.polyfit(df['absences'], df['G3'], 1)
    p = np.poly1d(z)
    x_vals = np.linspace(df['absences'].min(), df['absences'].max(), 100)
    axes[1, 0].plot(x_vals, p(x_vals), "r--", linewidth=2,
                    label=f"Trend line (slope={z[0]:.2f})")
    axes[1, 0].set_title("School Absences vs. Final Grade (G3)",
                         fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel("Number of Absences", fontsize=10)
    axes[1, 0].set_ylabel("Final Grade G3 (0-20)", fontsize=10)
    axes[1, 0].grid(True, linestyle=':', alpha=0.6)
    axes[1, 0].legend()

    # 4. Higher Education Goal vs Final Grade
    higher_groups = df.groupby('higher')['G3'].mean()
    higher_labels = ['No Higher Ed Plan', 'Wants Higher Ed']
    axes[1, 1].bar(higher_labels, higher_groups.values, color=[
                   '#e7298a', '#1b9e77'], edgecolor='black', width=0.45)
    axes[1, 1].set_title(
        "Impact of Higher Education Aspiration on G3", fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel("Average G3 Grade (0-20)", fontsize=10)
    axes[1, 1].set_ylim(0, 20)
    axes[1, 1].grid(axis='y', linestyle=':', alpha=0.6)
    for i, v in enumerate(higher_groups.values):
        axes[1, 1].text(i, v + 0.4, f"{v:.2f}", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved feature relationships plot to '{output_path}'.")


def plot_correlation_matrix(df, output_path="figures/correlation_matrix.png"):
    """
    Generate correlation matrix heatmap of numerical features.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Select key numerical features
    num_cols = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures',
                'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3']

    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax, shrink=0.8)

    ax.set_xticks(range(len(num_cols)))
    ax.set_yticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=45, ha="left", fontsize=9)
    ax.set_yticklabels(num_cols, fontsize=9)

    # Annotate correlation values inside cells
    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            val = corr.iloc[i, j]
            text_color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center",
                    va="center", color=text_color, fontsize=7.5)

    plt.title("Correlation Heatmap of Academic & Behavioral Features",
              fontsize=13, fontweight='bold', pad=25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved correlation matrix plot to '{output_path}'.")
