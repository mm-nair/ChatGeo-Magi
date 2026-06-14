"""
This script generates the main GEval grouped bar chart (with vs without emails)
for the paper, computing all values directly from opik_evaluation_results/.
"""
import json
import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 16, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
})

DATA_DIR = "opik_evaluation_results"
OUTPUT_DIR = "plot_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

files = glob.glob(os.path.join(DATA_DIR, "*.json"))
if not files:
    raise RuntimeError(f"No JSON files found in {DATA_DIR}/")
print(f"Found {len(files)} JSON files")

rows = []
for path in files:
    with open(path, "r") as f:
        data = json.load(f)
    for item in data:
        rows.append({
            "model": item.get("dataset.model_name"),
            "source_file": item.get("dataset.source_file"),
            "geval": item.get("feedback_scores.g_eval_metric"),
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

means = df.pivot_table(index="model", columns="corpus", values="geval", aggfunc="mean")
model_order = ["llama3.3:70b", "mistral-nemo:12b", "gpt-oss:20b"]
means = means.reindex(model_order)

print("\nComputed mean GEval scores:")
print(means)

geval_without = means["without_emails"].to_numpy()
geval_with = means["with_emails"].to_numpy()

x = np.arange(len(model_order))
bar_width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - bar_width/2, geval_without, width=bar_width,
       label="Without Emails", color="#6aa9ff")
ax.bar(x + bar_width/2, geval_with, width=bar_width,
       label="With Emails", color="#ff9d6a")
ax.set_xticks(x)
ax.set_xticklabels(model_order, rotation=10)
ax.set_ylabel("GEval Score")
ax.set_title("GEval Scores by Model and Corpus")
ax.set_ylim(0.75, 1.0)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/geval_scores.png", dpi=200)
print("Saved geval_scores.png")