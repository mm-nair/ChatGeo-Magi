"""End-to-end production smoke tests for the repaired ChatGeoMagi agent.

Tests the actual production initialization pathway (`ChatGeoMagi.setup_vector_store()`
followed by `ChatGeoMagi.initialize(llm, embeddings, retriever)`) across the 5 required
smoke-test scenarios:
1. Conceptual geomagnetism question requiring document retrieval.
2. NOAA geomagnetic field calculation.
3. Data retrieval followed by plotting.
4. Model provenance and units attribution.
5. Invalid-coordinate request (`95,0`) verifying no silent coordinate swap (`0,95`).
"""

import argparse
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agentic"))
sys.path.insert(0, str(ROOT / "evaluation"))

from agentic.cgm import ChatGeoMagi
from agentic.config import SystemConfig
from run_revision_inference import extract_trace
from vm_support import load_existing_credentials


SMOKE_TESTS = [
    {
        "id": "S01_document_retrieval",
        "category": "conceptual_document_retrieval",
        "prompt": (
            "According to the World Magnetic Model 2020 Technical Report, how are "
            "the blackout zones and caution zones around the magnetic poles defined "
            "in terms of horizontal intensity H, and why is compass navigation "
            "unreliable in those zones?"
        ),
    },
    {
        "id": "S02_noaa_calculation",
        "category": "noaa_calculation",
        "prompt": "Give total intensity at 51.5,-0.12 on 2026-01-01.",
    },
    {
        "id": "S03_retrieval_and_plotting",
        "category": "data_retrieval_and_plotting",
        "prompt": "Plot total intensity in London annually from 1999 to 2026.",
    },
    {
        "id": "S04_model_provenance",
        "category": "model_provenance",
        "prompt": "Give total intensity in London annually from 1999 to 2026 and identify the model.",
    },
    {
        "id": "S05_invalid_coordinates",
        "category": "invalid_coordinate_handling",
        "prompt": "Calculate inclination at 95,0 in 2026.",
    },
]


def evaluate_smoke_test(item, trace, image_dir):
    calls = trace.get("calls", [])
    tool_names = [call["name"] for call in calls]
    final = trace.get("final", "")
    category = item["category"]

    if category == "conceptual_document_retrieval":
        used_retriever = "retrieve_geomag_docs" in tool_names
        normalized_digits = final.replace(",", "").replace("\u202f", "").replace(" ", "")
        mentions_thresholds = "2000" in normalized_digits and "6000" in normalized_digits
        passed = used_retriever and mentions_thresholds and bool(final.strip())
        notes = {
            "used_retrieve_geomag_docs": used_retriever,
            "mentions_2000_and_6000_nT": mentions_thresholds,
        }
    elif category == "noaa_calculation":
        used_calc = "noaa_mag_api" in tool_names
        has_nt = "nT" in final or "nanotesla" in final.lower()
        passed = used_calc and has_nt and bool(final.strip())
        notes = {
            "used_noaa_mag_api": used_calc,
            "includes_nT_units": has_nt,
        }
    elif category == "data_retrieval_and_plotting":
        used_calc = "noaa_mag_api" in tool_names
        used_plot = "plot" in tool_names or "plot_many" in tool_names
        plot_files = (
            sorted(str(p.name) for p in image_dir.glob("*.png"))
            if image_dir.is_dir()
            else []
        )
        passed = used_calc and used_plot and len(plot_files) > 0 and bool(final.strip())
        notes = {
            "used_noaa_mag_api": used_calc,
            "used_plot_tool": used_plot,
            "generated_plots": plot_files,
        }
    elif category == "model_provenance":
        used_calc = "noaa_mag_api" in tool_names
        mentions_igrf = "IGRF" in final.upper()
        passed = used_calc and mentions_igrf and bool(final.strip())
        notes = {
            "used_noaa_mag_api": used_calc,
            "identifies_IGRF_model": mentions_igrf,
        }
    elif category == "invalid_coordinate_handling":
        swapped_calls = [
            call
            for call in calls
            if call["name"] == "noaa_mag_api"
            and str(call["args"].get("location", "")).replace(" ", "") in ("0,95", "0.0,95.0")
            and call.get("success", False)
        ]
        any_successful_calc = [
            call
            for call in calls
            if call["name"] == "noaa_mag_api" and call.get("success", False)
        ]
        rejected_in_final = any(
            token in final.lower()
            for token in [
                "invalid",
                "not valid",
                "outside the valid range",
                "valid range",
                "out of range",
                "[-90, 90]",
                "-90 to 90",
                "between -90 and 90",
                "between –90° and +90°",
                "between -90° and +90°",
                "cannot",
                "exceeds",
                "not a valid",
            ]
        )
        passed = (
            len(swapped_calls) == 0
            and len(any_successful_calc) == 0
            and rejected_in_final
            and bool(final.strip())
        )
        notes = {
            "silent_coordinate_swap_succeeded": len(swapped_calls) > 0,
            "any_calculator_call_succeeded": len(any_successful_calc) > 0,
            "rejected_invalid_latitude_in_response": rejected_in_final,
        }
    else:
        passed = bool(final.strip())
        notes = {}

    return {"passed": passed, "tool_names": tool_names, "checks": notes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-oss:20b")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--embedding-path", type=Path, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "revision_workspace/retrieval_evaluation/public_documents",
    )
    parser.add_argument("--preset", default="stella_s2p_cosine")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    image_dir = args.output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CGM_IMAGE_DIR"] = str(image_dir)
    os.environ["OPIK_TRACK_DISABLE"] = "true"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    load_existing_credentials()

    from langchain_ollama import ChatOllama

    config = SystemConfig.from_preset(
        args.preset,
        model_name=args.model,
        embedding_model=str(args.embedding_path),
        chroma_db_path=str(args.index),
        data_dir=str(args.data_dir),
        retrieval_k=4,
    )
    embeddings = ChatGeoMagi.create_embeddings(config, local_files_only=True)
    retriever = ChatGeoMagi.setup_vector_store(config, embeddings, strict=True)
    llm = ChatOllama(
        model=args.model,
        temperature=0,
        seed=20261115,
        num_ctx=8192,
        num_predict=2048,
    )

    results = []
    for item in SMOKE_TESTS:
        agent = ChatGeoMagi(config)
        agent.initialize(llm, embeddings, retriever=retriever)
        raw = agent.invoke(item["prompt"], thread_id=uuid4().hex)
        trace = extract_trace(raw)
        evaluation = evaluate_smoke_test(item, trace, image_dir)
        record = {
            "id": item["id"],
            "category": item["category"],
            "prompt": item["prompt"],
            "passed": evaluation["passed"],
            "tool_names": evaluation["tool_names"],
            "checks": evaluation["checks"],
            "calls": trace["calls"],
            "final": trace["final"],
        }
        results.append(record)
        (args.output / f"{item['id']}.json").write_text(json.dumps(record, indent=2))
        print(
            f"[SmokeTest] {item['id']}: passed={record['passed']} tools={record['tool_names']}",
            flush=True,
        )

    summary = {
        "total": len(results),
        "passed": sum(1 for r in results if r["passed"]),
        "all_passed": all(r["passed"] for r in results),
        "index": str(args.index),
        "preset": args.preset,
        "results": [
            {
                "id": r["id"],
                "category": r["category"],
                "passed": r["passed"],
                "tool_names": r["tool_names"],
                "checks": r["checks"],
                "final_preview": r["final"][:400],
            }
            for r in results
        ],
    }
    (args.output / "smoke_test_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
