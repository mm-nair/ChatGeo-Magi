import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import urllib.request
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentic.config import RETRIEVAL_PRESETS, SystemConfig
from golden_dataset import load_goldens
from public_retrieval import open_retriever


def extract_trace(result):
    calls = []
    responses = {}
    final = ""
    messages = []
    for message in result["messages"]:
        messages.append(message.model_dump(mode="json"))
        if getattr(message, "type", "") == "tool":
            responses[message.tool_call_id] = message
        for call in getattr(message, "tool_calls", []):
            calls.append({"id": call["id"], "name": call["name"], "args": call["args"]})
        if getattr(message, "type", "") == "ai" and not getattr(message, "tool_calls", []):
            final = message.content
    for call in calls:
        response = responses.get(call["id"])
        content = str(response.content) if response is not None else ""
        call["output"] = content
        call["success"] = (
            response is not None
            and getattr(response, "status", "success") != "error"
            and not content.startswith(
                (
                    "Could not find",
                    "Invalid data name",
                    "Calculator response lacks",
                    "Error:",
                    "Failed",
                )
            )
        )
    return {"calls": calls, "final": final, "messages": messages}


def verify_reusable_no_rag(reuse_dir: Path, args, current_source_hashes: dict):
    meta_file = reuse_dir / "metadata.json"
    if not meta_file.is_file():
        raise ValueError(f"Cannot reuse no_rag from {reuse_dir}: missing metadata.json")
    prior_meta = json.loads(meta_file.read_text())
    prior_args = prior_meta.get("arguments", {})
    for key in ("model", "model_digest", "temperature", "seed", "num_ctx", "num_predict"):
        if prior_args.get(key) != getattr(args, key):
            raise ValueError(
                f"Cannot reuse no_rag: parameter '{key}' differs ({prior_args.get(key)!r} vs {getattr(args, key)!r})"
            )
    if not prior_meta.get("calculator_tools_available_in_both_conditions"):
        raise ValueError("Cannot reuse no_rag: calculator tool availability differs")
    prior_hashes = prior_meta.get("source_hashes", {})
    for critical_file in ("agentic/prompts.py", "agentic/cgm.py", "agentic/config.py", "agentic/apis/noaa_apis.py"):
        if prior_hashes.get(critical_file) != current_source_hashes.get(critical_file):
            raise ValueError(
                f"Cannot reuse no_rag: source hash for {critical_file} changed"
            )
    return prior_meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["ablation", "tools"], required=True)
    parser.add_argument("--execute-inference", action="store_true")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-digest", required=True)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--num-predict", type=int, default=2048)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--embedding-path", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--preset",
        choices=sorted(RETRIEVAL_PRESETS.keys()),
        default="stella_s2p_cosine",
    )
    parser.add_argument("--retrieval-k", type=int, default=4)
    parser.add_argument(
        "--experiment-version",
        type=str,
        default="final_retrieval_revision_v1",
    )
    parser.add_argument(
        "--reuse-no-rag-from",
        type=Path,
        default=None,
        help="Optional path to a previous ablation run with identical model/prompt/settings from which to reuse no_rag outputs.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.execute_inference:
        parser.error(
            "Inference is disabled unless --execute-inference is explicitly supplied after VM authorization"
        )
    if args.experiment == "ablation" and not all(
        [args.index, args.embedding_path, args.manifest]
    ):
        parser.error(
            "Ablation requires the validated public index and cached Stella embedding model"
        )
    output = (
        args.output
        or ROOT
        / "revision_workspace"
        / ("rag_ablation" if args.experiment == "ablation" else "tool_evaluation")
        / "results"
        / uuid4().hex
    )
    if not output.resolve().is_relative_to(ROOT / "revision_workspace"):
        parser.error("Output must remain in revision_workspace")
    if output.exists():
        parser.error("Use a new output directory; previous runs are never overwritten")
    with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=10) as response:
        installed = json.load(response)
    if not any(
        row["name"] == args.model and row["digest"] == args.model_digest
        for row in installed["models"]
    ):
        parser.error(
            "Requested exact model digest is not installed; no model will be downloaded"
        )
    source_hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for folder in ["agentic", "evaluation"]
        for path in (ROOT / folder).rglob("*.py")
    }
    reused_no_rag_meta = None
    if args.reuse_no_rag_from is not None:
        reused_no_rag_meta = verify_reusable_no_rag(
            args.reuse_no_rag_from, args, source_hashes
        )
    output.mkdir(parents=True)
    os.environ["CGM_IMAGE_DIR"] = str(output / "images")
    os.environ["OPIK_TRACK_DISABLE"] = "true"
    from vm_support import load_existing_credentials

    load_existing_credentials()
    sys.path.insert(0, str(ROOT / "agentic"))
    from cgm import ChatGeoMagi
    from langchain_core.tools import create_retriever_tool
    from langchain_ollama import ChatOllama

    llm = ChatOllama(
        model=args.model,
        temperature=args.temperature,
        seed=args.seed,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
    )
    retriever = None
    index_metadata = None
    if args.experiment == "ablation":
        vector_retriever = open_retriever(
            args.index,
            args.embedding_path,
            args.manifest,
            preset=args.preset,
            k=args.retrieval_k,
        )
        retriever = create_retriever_tool(
            vector_retriever,
            "retrieve_geomag_docs",
            "Search and return information about geomagnetism, from authoritative public documents.",
        )
        meta_path = ChatGeoMagi.index_metadata_path(args.index)
        if meta_path.is_file():
            index_metadata = json.loads(meta_path.read_text())
        questions = [
            {"id": f"Q{index:02}", "prompt": question, "reference": answer}
            for index, (question, answer) in enumerate(
                load_goldens(ROOT / "evaluation/goldens.csv").items(), 1
            )
        ]
        conditions = ["rag", "no_rag"]
    else:
        questions = json.loads(
            (ROOT / "evaluation/benchmarks/tool_prompts.json").read_text()
        )
        conditions = ["tools_only"]
    metadata = {
        "experiment_version": args.experiment_version,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "index_metadata": index_metadata,
        "reused_no_rag_from": str(args.reuse_no_rag_from) if args.reuse_no_rag_from else None,
        "reused_no_rag_commit": reused_no_rag_meta.get("commit") if reused_no_rag_meta else None,
        "python": platform.python_version(),
        "packages": {
            distribution.metadata["Name"]: distribution.version
            for distribution in importlib.metadata.distributions()
        },
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)
        ),
        "calculator_tools_available_in_both_conditions": True,
        "model_digest": args.model_digest,
        "ollama_models": [
            {"name": row["name"], "digest": row["digest"]}
            for row in installed["models"]
        ],
        "source_hashes": source_hashes,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    config = SystemConfig.from_preset(
        args.preset,
        model_name=args.model,
        retrieval_k=args.retrieval_k,
        chroma_db_path=str(args.index) if args.index else "./chroma_db_stella_s2p_cosine_v1",
    )
    judge_inputs = []
    for question in questions:
        for condition in conditions:
            if condition == "no_rag" and args.reuse_no_rag_from is not None:
                prior_trace_path = args.reuse_no_rag_from / f"{question['id']}_no_rag.json"
                if not prior_trace_path.is_file():
                    raise FileNotFoundError(f"Missing {prior_trace_path}")
                trace = json.loads(prior_trace_path.read_text())
                if trace.get("question") != question["prompt"]:
                    raise ValueError(f"Question mismatch on {question['id']} in reused no_rag")
                trace["reused_from"] = str(prior_trace_path)
            else:
                model = ChatGeoMagi(config)
                model.initialize(llm, None, retriever if condition == "rag" else None)
                try:
                    result = model.invoke(question["prompt"], thread_id=uuid4().hex)
                    trace = extract_trace(result)
                except Exception as error:
                    trace = {"calls": [], "final": "", "error_type": type(error).__name__}
                trace.update(
                    {
                        "id": question["id"],
                        "condition": condition,
                        "question": question["prompt"],
                        "experiment_version": args.experiment_version,
                        "preset": args.preset if condition == "rag" else None,
                    }
                )
            print(
                question["id"],
                condition,
                "calls",
                len(trace["calls"]),
                "error",
                trace.get("error_type"),
                "reused" if trace.get("reused_from") else "generated",
                flush=True,
            )
            (output / f"{question['id']}_{condition}.json").write_text(
                json.dumps(trace, indent=2)
            )
            if "reference" in question:
                judge_inputs.append(
                    {
                        "id": question["id"],
                        "condition": condition,
                        "input": question["prompt"],
                        "expected_answer": question["reference"],
                        "model_output": trace["final"],
                    }
                )
    if judge_inputs:
        (output / "judge_inputs.json").write_text(json.dumps(judge_inputs, indent=2))


if __name__ == "__main__":
    main()
