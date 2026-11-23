import csv
from pathlib import Path
import re

import numpy as np


def bootstrap(differences, seed=20261115, resamples=100000):
    differences = np.asarray(differences, dtype=float)
    if not len(differences) or not np.isfinite(differences).all():
        raise ValueError("Bootstrap needs finite paired observations")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(differences), (resamples, len(differences)))
    means = differences[indices].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def write_csv(path, rows):
    path = Path(path)
    if not rows:
        raise ValueError("Cannot write an empty row list to CSV")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_paired_scores(records, seed=20261115, resamples=100000):
    summaries, paired = [], []
    conditions = (
        ("no_rag", "rag")
        if any(row.get("condition") == "rag" for row in records)
        else ("noemails", "emails")
    )
    models = sorted({row.get("model", "default") for row in records})
    for model in models:
        groups = {}
        for row in records:
            if row.get("model", "default") != model:
                continue
            key = (row["id"], row["condition"])
            score = row.get("score")
            if (
                key in groups
                or isinstance(score, bool)
                or not isinstance(score, (float, int))
                or not 0 <= score <= 1
            ):
                raise ValueError("Duplicate or invalid score")
            groups[key] = row
        identifiers = sorted({key[0] for key in groups})
        if len(identifiers) != 25 or len(groups) != 50:
            raise ValueError("Expected 25 complete pairs per model")
        model_pairs = []
        for identifier in identifiers:
            first, second = [groups[identifier, condition] for condition in conditions]
            row = {
                "model": model,
                "id": identifier,
                conditions[0]: first["score"],
                conditions[1]: second["score"],
                "difference": second["score"] - first["score"],
                "empty_pair": bool(
                    first.get("empty_response") or second.get("empty_response")
                ),
            }
            model_pairs.append(row)
        differences = [row["difference"] for row in model_pairs]
        low, high = bootstrap(differences, seed=seed, resamples=resamples)
        nonempty = [row["difference"] for row in model_pairs if not row["empty_pair"]]
        summaries.append(
            {
                "model": model,
                "n": len(model_pairs),
                "first_condition": conditions[0],
                "second_condition": conditions[1],
                "first_mean": float(np.mean([row[conditions[0]] for row in model_pairs])),
                "second_mean": float(
                    np.mean([row[conditions[1]] for row in model_pairs])
                ),
                "difference": float(np.mean(differences)),
                "ci_low": float(low),
                "ci_high": float(high),
                "nonempty_pairs": len(nonempty),
                "nonempty_difference": float(np.mean(nonempty)) if nonempty else 0.0,
            }
        )
        paired.extend(model_pairs)
    return summaries, paired


def retrieval_metrics(expected, retrieved):
    if expected.get("verification_status") not in ("human_verified", "source_verified"):
        return {
            "hit_at_4": None,
            "recall_at_4": None,
            "precision_at_4": None,
            "support_coverage": None,
            "status": "pending_human_verification",
        }
    relevant = set(expected["relevant_chunk_ids"])
    if not relevant:
        raise ValueError("Verified retrieval item lacks relevant_chunk_ids")
    top = [item["chunk_id"] for item in retrieved[:4]]
    if len(set(top)) != len(top):
        raise ValueError("Duplicate chunk IDs in retrieved top-4")
    hits = len(relevant.intersection(top))
    facts = expected.get("support_facts", [])
    covered = sum(bool(set(fact["chunk_ids"]).intersection(top)) for fact in facts)
    return {
        "hit_at_4": int(hits > 0),
        "recall_at_4": hits / len(relevant),
        "precision_at_4": hits / 4,
        "support_coverage": covered / len(facts) if facts else None,
        "status": expected["verification_status"],
    }


def parse_coordinates(value):
    match = re.fullmatch(
        r"\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*", str(value)
    )
    return (float(match.group(1)), float(match.group(2))) if match else None


def argument_matches(name, expected, actual):
    if actual is None:
        return False
    if name in ("start_time", "end_time") and re.fullmatch(r"\d{4}", str(expected)):
        return str(actual) in (str(expected), f"{expected}-01-01")
    if name == "location":
        left, right = parse_coordinates(expected), parse_coordinates(actual)
        if left and right:
            return all(abs(a - b) < 1e-6 for a, b in zip(left, right))
    return str(actual).strip().lower() == str(expected).strip().lower()


def tool_metrics(expected, trace):
    calls = trace.get("calls")
    if not isinstance(calls, list):
        return {
            "tool_selection": False,
            "parameters_and_order": False,
            "execution": False,
            "numerical_agreement": None,
            "end_to_end": None,
        }
    required = expected["calls"]
    names = [call.get("name") for call in calls]
    required_names = [call["name"] for call in required]
    cursor = 0
    matched = []
    for item in required:
        found = None
        for index in range(cursor, len(calls)):
            call = calls[index]
            if call.get("name") == item["name"] and all(
                argument_matches(k, v, call.get("args", {}).get(k))
                for k, v in item.get("args", {}).items()
            ):
                found = call
                cursor = index + 1
                break
        matched.append(found)
    parameters_and_order = all(matched)
    cursor = 0
    executed = []
    for item in required:
        found = None
        for index in range(cursor, len(calls)):
            call = calls[index]
            if call.get("name") == item["name"] and call.get("success") is True:
                found = call
                cursor = index + 1
                break
        executed.append(found)
    execution = bool(required) and all(executed)
    return {
        "tool_selection": all(name in names for name in required_names),
        "parameters_and_order": parameters_and_order,
        "execution": execution,
        "numerical_agreement": None,
        "end_to_end": None,
    }
