"""
This script is takes the json output of opik within the
opik_evaluation_results/ directory, and generates plots
from them.

Only the hist_geval_distrobution plot is used in the paper.
"""

import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import os

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12
})

DATA_DIR = "opik_evaluation_results"
OUTPUT_DIR = "plot_outputs"

files = glob.glob(os.path.join(DATA_DIR, "*.json"))
rows = []

if not files:
    raise RuntimeError("No JSON files found in current directory.")

print(f"Found {len(files)} JSON files")

for path in files:
    with open(path, "r") as f:
        data = json.load(f)

    for item in data:
        rows.append({
            "model": item.get("dataset.model_name"),
            "source_file": item.get("dataset.source_file"),
            "question": item.get("dataset.input"),
            "expected": item.get("dataset.expected_answer"),
            "geval": item.get("feedback_scores.g_eval_metric"),
            "hallucination": item.get("feedback_scores.hallucination_metric"),
        })

df = pd.DataFrame(rows)

def label_corpus(s):
    if not s:
        return "unknown"
    if "NOEMAILS" in s:
        return "without_emails"
    if "EMAILS" in s:
        return "with_emails"
    return "unknown"

df["corpus"] = df["source_file"].apply(label_corpus)

print("Loaded rows:", len(df))
print(df["corpus"].value_counts())

plt.figure(figsize=(8, 6))

for model in df["model"].dropna().unique():
    subset = df[df["model"] == model]
    plt.scatter(
        subset["hallucination"],
        subset["geval"],
        label=model,
        alpha=0.7
    )

plt.xlabel("Hallucination (lower = better)")
plt.ylabel("GEval (higher = better)")
plt.title("Accuracy vs Hallucination Across Models")

plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/scatter_accuracy_vs_hallucination.png", dpi=220)

plt.figure(figsize=(9, 6))

for model in df["model"].dropna().unique():
    subset = df[df["model"] == model]["geval"]
    plt.hist(subset, bins=10, alpha=0.5, label=model)

plt.xlabel("GEval score")
plt.ylabel("Count")
plt.title("Distribution of GEval Scores Per Model")

plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/hist_geval_distribution.png", dpi=220)

pivot = df.pivot_table(
    index="question",
    columns="model",
    values="geval",
    aggfunc="mean"
)

plt.figure(figsize=(11, 10))
im = plt.imshow(pivot, cmap="viridis", aspect="auto")

cbar = plt.colorbar(im)
cbar.set_label("GEval score", fontsize=14)

plt.yticks(
    range(len(pivot.index)),
    labels=[f"Q{i}" for i in range(1, len(pivot.index) + 1)]
)

plt.xticks(
    range(len(pivot.columns)),
    pivot.columns,
    rotation=20
)

plt.title("Per-Question Model Performance Heatmap")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/heatmap_question_difficulty.png", dpi=240)