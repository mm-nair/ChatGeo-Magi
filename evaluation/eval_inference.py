"""
This script was run on the system with/without the emails loaded in.
The final dataset of each was then taken, renamed to append their version, and
sent to Opik for evaluation.

Incremental is to prevent having to re-inference everything, in case a crash happens
in between models/configurations.
"""

import csv
import os
from datetime import datetime
from config import SystemConfig
from cgm import ChatGeoMagi

DATA_DIR = "inference_results"
BASE_MODELS = [
    "gpt-oss:20b",
    "mistral-nemo:12b",
    "llama3.3:70b"
]

def run_cmd(cmd):
    os.system(cmd)

def load_goldens(path="goldens.csv"):
    goldens = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            goldens.append({
                "id": len(goldens) + 1,
                "question": row["question"],
                "answer": row["answer"]
            })
    return goldens


def run_model(model_name, goldens):
    print(f"\n=== Pulling model: {model_name} ===")
    run_cmd(f"ollama pull {model_name}")

    config = SystemConfig()
    config.model_name = model_name

    llm, embeddings, retriever = ChatGeoMagi.get_shared_resources(config)
    model = ChatGeoMagi(config)
    model.initialize(llm, embeddings, retriever)

    outputs = []

    for g in goldens:
        print("Golden", g)
        result = model.invoke(g["question"])
        messages = result["messages"]
        assistant = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
        outputs.append((g["id"], assistant))

    print(f"\n=== Removing model: {model_name} ===")
    run_cmd(f"ollama rm {model_name}")

    return outputs


def write_output_incremental(goldens, results_per_model, ts):
    path = f"{DATA_DIR}/outputs_incremental_{ts}.csv"

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["id", "question"] + list(results_per_model.keys())
        writer.writerow(header)

        for g in goldens:
            row = [g["id"], g["question"]]
            for m in results_per_model.keys():
                row.append(results_per_model[m][g["id"] - 1][1])
            writer.writerow(row)

    print(f"Saved incremental output to {path}")


def write_output_final(goldens, results_per_model, ts):
    path = f"{DATA_DIR}/outputs_final_{ts}.csv"

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["id", "question"] + list(results_per_model.keys())
        writer.writerow(header)

        for g in goldens:
            row = [g["id"], g["question"]]
            for m in results_per_model.keys():
                row.append(results_per_model[m][g["id"] - 1][1])
            writer.writerow(row)

    print(f"Saved final output to {path}")


if __name__ == "__main__":
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    goldens = load_goldens()
    results_per_model = {}

    for m in BASE_MODELS:
        results_per_model[m] = run_model(m, goldens)
        write_output_incremental(goldens, results_per_model, ts)

    write_output_final(goldens, results_per_model, ts)