import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

from benchmark_metrics import retrieval_metrics

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentic.cgm import ChatGeoMagi
from agentic.config import RETRIEVAL_PRESETS, SystemConfig


def chunk_identifier(text, metadata):
    return hashlib.sha256(
        (json.dumps(metadata, sort_keys=True, ensure_ascii=False) + text).encode()
    ).hexdigest()


def validate_manifest(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("public_only") is not True or not manifest.get("files"):
        raise ValueError("An explicit public-only corpus manifest is required")
    for record in manifest["files"]:
        path = Path(record["path"])
        if ChatGeoMagi.is_forbidden_source(str(path)):
            raise ValueError(f"Public manifest cannot contain email source: {path}")
        if (
            not record.get("source_url", "").startswith("https://")
            or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]
        ):
            raise ValueError("Unverified public corpus file")
    return manifest


def open_retriever(
    index,
    embedding_path,
    manifest_path,
    preset="stella_s2p_cosine",
    k=4,
    embedding_cls=None,
    chroma_cls=None,
):
    import os

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    manifest = validate_manifest(manifest_path)
    if not Path(index).is_dir() or not Path(embedding_path).is_dir():
        raise ValueError(
            "Existing public index and local Stella model required; downloads disabled"
        )
    if Path(manifest["index_path"]).resolve() != Path(index).resolve():
        raise ValueError("Manifest index does not match requested index")
    config = SystemConfig.from_preset(
        preset,
        chroma_db_path=str(index),
        embedding_model=str(embedding_path),
        retrieval_k=k,
        public_only=True,
    )
    expected_sig = config.index_signature()
    for key in (
        "preset",
        "embedding_model",
        "chunk_size",
        "chunk_overlap",
        "length_function",
        "trust_remote_code",
        "normalize_embeddings",
        "distance_metric",
        "query_instruction",
    ):
        if manifest.get(key) != expected_sig[key]:
            raise ValueError(
                f"Manifest configuration mismatch on '{key}': manifest has {manifest.get(key)!r}, requested preset '{preset}' requires {expected_sig[key]!r}"
            )
    expected_hashes = ChatGeoMagi.compute_document_hashes(
        [Path(row["path"]).resolve() for row in manifest["files"]],
        public_only=True,
    )
    embeddings = ChatGeoMagi.create_embeddings(
        config,
        local_files_only=True,
        embedding_cls=embedding_cls,
    )
    store, _ = ChatGeoMagi.validate_existing_index(
        config,
        embeddings=embeddings,
        expected_hashes=expected_hashes,
        chroma_cls=chroma_cls,
    )
    return store.as_retriever(search_kwargs={"k": config.retrieval_k})


def compute_hubness_diagnostics(questions, retrieved, k=4):
    all_relevant = set()
    scored_ids = set()
    for q in questions:
        if q.get("verification_status") in ("human_verified", "source_verified"):
            scored_ids.add(q["id"])
            all_relevant.update(q.get("relevant_chunk_ids", []))

    counts_all = Counter()
    counts_scored = Counter()
    chunk_meta = {}
    for qid, docs in retrieved.items():
        for doc in docs[:k]:
            cid = doc["chunk_id"]
            counts_all[cid] += 1
            if qid in scored_ids:
                counts_scored[cid] += 1
            if cid not in chunk_meta:
                meta = doc.get("metadata") or {}
                chunk_meta[cid] = {
                    "source": Path(str(meta.get("source", ""))).name,
                    "page": meta.get("page"),
                    "start_index": meta.get("start_index"),
                }

    unrelated_repeated = [
        {
            "chunk_id": cid,
            "count_scored": count,
            "count_all": counts_all[cid],
            "is_any_relevant": cid in all_relevant,
            **chunk_meta.get(cid, {}),
        }
        for cid, count in counts_scored.most_common()
        if count > 1 and cid not in all_relevant
    ]
    top_chunks = [
        {
            "chunk_id": cid,
            "count_scored": count,
            "count_all": counts_all[cid],
            "is_any_relevant": cid in all_relevant,
            **chunk_meta.get(cid, {}),
        }
        for cid, count in counts_scored.most_common(10)
    ]
    return {
        "scored_questions": len(scored_ids),
        "total_questions": len(questions),
        "total_slots_scored": len(scored_ids) * k,
        "unique_chunks_scored": len(counts_scored),
        "unique_chunks_all": len(counts_all),
        "max_chunk_frequency_scored": max(counts_scored.values(), default=0),
        "repeated_unrelated_chunk_count": len(unrelated_repeated),
        "repeated_unrelated_slot_Recoveries": sum(
            item["count_scored"] for item in unrelated_repeated
        ),
        "repeated_unrelated_chunks": unrelated_repeated[:10],
        "top_chunks_scored": top_chunks,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "evaluation/benchmarks/retrieval.json",
    )
    parser.add_argument("--retrieved", type=Path)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--embedding-path", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--preset",
        choices=sorted(RETRIEVAL_PRESETS.keys()),
        default="stella_s2p_cosine",
    )
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "revision_workspace/retrieval_evaluation/results_corrected",
    )
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT / "revision_workspace"):
        parser.error("Outputs must remain in revision_workspace")
    questions = json.loads(args.benchmark.read_text())
    if args.retrieved:
        retrieved = json.loads(args.retrieved.read_text())
    else:
        if not all([args.index, args.embedding_path, args.manifest]):
            parser.error(
                "Provide cached retrieval results or a public index, local embedding path and manifest"
            )
        retriever = open_retriever(
            args.index,
            args.embedding_path,
            args.manifest,
            preset=args.preset,
            k=args.k,
        )
        retrieved = {}
        for question in questions:
            documents = retriever.invoke(question["question"])
            retrieved[question["id"]] = [
                {
                    "chunk_id": chunk_identifier(
                        document.page_content, document.metadata
                    ),
                    "text": document.page_content,
                    "metadata": document.metadata,
                }
                for document in documents
            ]
    if set(retrieved) != {question["id"] for question in questions}:
        raise ValueError("Retrieval results must cover every benchmark question exactly")
    rows = [
        {"id": question["id"], **retrieval_metrics(question, retrieved[question["id"]], k=args.k)}
        for question in questions
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "retrieved.json").write_text(json.dumps(retrieved, indent=2))
    (args.output / "metrics.json").write_text(json.dumps(rows, indent=2))
    summary = {}
    for metric in ["hit_at_4", "recall_at_4", "precision_at_4", "support_coverage"]:
        values = [row[metric] for row in rows if row[metric] is not None]
        summary[metric] = {
            "n": len(values),
            "mean": sum(values) / len(values) if values else None,
            "hits": sum(1 for v in values if v > 0) if values else 0,
        }
    summary["hubness"] = compute_hubness_diagnostics(questions, retrieved, k=args.k)
    summary["configuration"] = {
        "preset": args.preset,
        "k": args.k,
        "benchmark": str(args.benchmark),
        "index": str(args.index) if args.index else None,
        "manifest": str(args.manifest) if args.manifest else None,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
