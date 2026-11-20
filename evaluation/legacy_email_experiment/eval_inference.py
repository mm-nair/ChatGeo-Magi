"""
This script was run on the system with/without the emails loaded in.
The final dataset of each was then taken, renamed to append their version, and
sent to Opik for evaluation.

Incremental is to prevent having to re-inference everything, in case a crash happens
in between models/configurations.
"""

import csv
import os
import argparse
from pathlib import Path
from uuid import uuid4
from golden_dataset import load_goldens as read_goldens
from datetime import datetime
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agentic"))
from config import SystemConfig
from cgm import ChatGeoMagi

DATA_DIR = str(Path(__file__).resolve().parents[1] / "revision_workspace/rag_ablation/legacy_inference")
BASE_MODELS = [
    "gpt-oss:20b",
    "mistral-nemo:12b",
    "llama3.3:70b"
]

def run_cmd(cmd):
    os.system(cmd)

def load_goldens(path=Path(__file__).with_name("goldens.csv")):
    return [{"id": index, "question": question, "answer": answer}
            for index, (question, answer) in enumerate(read_goldens(path).items(), 1)]


def run_model(model_name, goldens, isolated_sessions=True):
    run_id = uuid4().hex

    config = SystemConfig()
    config.model_name = model_name

    llm, embeddings, retriever = ChatGeoMagi.get_shared_resources(config)
    model = ChatGeoMagi(config)
    model.initialize(llm, embeddings, retriever)

    outputs = []

    for g in goldens:
        print("Golden", g)
        result = model.invoke(g["question"], thread_id=f"{run_id}-{g['id']}" if isolated_sessions else "1")
        messages = result["messages"]
        assistant = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
        outputs.append((g["id"], assistant))



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
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-session", action="store_true")
    args = parser.parse_args()
    os.makedirs(DATA_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    goldens = load_goldens()
    results_per_model = {}

    for m in BASE_MODELS:
        results_per_model[m] = run_model(m, goldens, isolated_sessions=not args.shared_session)
        write_output_incremental(goldens, results_per_model, ts)

    write_output_final(goldens, results_per_model, ts)