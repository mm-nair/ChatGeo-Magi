"""
This script was run with the eval_inference.py generated CSV files.
These results are then sent straight to Opik, which have been downloaded to opik_evaluation_results/
for plotting purposes.
"""

import csv
import time
import gc
from datetime import datetime
from dotenv import load_dotenv

import opik
from opik.evaluation import evaluate
from opik.evaluation.metrics import Hallucination, GEval

load_dotenv()

client = opik.Opik()

def load_goldens(path="goldens.csv"):
    goldens = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            goldens[row["question"]] = row["answer"]
    return goldens


def load_outputs(path, goldens):
    """
    Loads CSV and returns dict:

      model_name -> list(sample_rows)
    """
    rows_by_model = {}

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames

        model_columns = header[2:]   # skip id + question

        for row in reader:
            question = row["question"]

            for model_name in model_columns:
                rows_by_model.setdefault(model_name, []).append({
                    "input": question,
                    "expected_answer": goldens[question],
                    "model_output": row[model_name],
                    "model_name": model_name,
                    "source_file": path
                })

    return rows_by_model

hallucination = Hallucination(model="gpt-4o-mini")

geval = GEval(
    model="gpt-4o-mini",
    task_introduction="You are evaluating factual answers.",
    evaluation_criteria="""
Return ONLY a single number from 1 to 10.

Judge the answer by comparing it to EXPECTED_ANSWER.

A high score means:
- the answer clearly expresses the same core idea as EXPECTED_ANSWER
- it is accurate and aligned

A middle score means:
- the main idea is there
- but something important is missing or there are noticeable inaccuracies

A low score means:
- the answer does not really match EXPECTED_ANSWER
- it is wrong, irrelevant, or misleading

Do not grade based on writing style or length.
Reduce the score when the answer adds unsupported or invented details.

Be HARSH but FAIR. The top score possible means the answer could not be any better.

"""
)


# -----------------------------
# TASK (no model call!)
# -----------------------------

def evaluation_task(item):
    return {"output": item["model_output"]}


# -----------------------------
# MAIN
# -----------------------------

if __name__ == "__main__":
    goldens = load_goldens()

    csv_paths = [
        "EMAILS_outputs_final_2025-11-27_00-45-57.csv",
        "NOEMAILS_outputs_final_2025-12-08_02-24-41.csv",
    ]

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    for csv_path in csv_paths:
        print(f"\n### Processing CSV: {csv_path}")

        rows_by_model = load_outputs(csv_path, goldens)

        for model_name, rows in rows_by_model.items():
            pretty_csv = csv_path.replace(".csv", "").split("/")[-1]

            dataset_name = f"cgm-fromcsv-{pretty_csv}-{model_name}"
            experiment_name = f"CGM {model_name} • {pretty_csv} • {timestamp}"

            print(f"\n=== Evaluating {model_name} from {csv_path} ===")

            dataset = client.get_or_create_dataset(name=dataset_name)
            dataset.insert(rows)

            evaluate(
                experiment_name=experiment_name,
                project_name="ChatGeo-Magi",
                dataset=dataset,
                task=evaluation_task,
                scoring_metrics=[hallucination, geval],
                task_threads=2,
            )

            print(f"=== Finished {model_name} / {pretty_csv} ===\n")

            gc.collect()
            time.sleep(2)

    print("All done.")