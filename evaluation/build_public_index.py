import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentic.cgm import ChatGeoMagi
from agentic.config import RETRIEVAL_PRESETS
from benchmark_metrics import retrieval_metrics, write_csv
from public_retrieval import (
    build_config_for_preset,
    load_public_chunks,
    open_retriever,
    validate_manifest,
)


def build_index(
    manifest_path,
    embedding_path,
    index_path,
    preset="stella_s2p_cosine",
    chunk_size=1000,
    chunk_overlap=200,
    overwrite=False,
    embedding_cls=None,
    chroma_cls=None,
    data_dir=None,
):
    manifest, files = validate_manifest(manifest_path, data_dir=data_dir)
    embedding_path = Path(embedding_path).resolve()
    if not embedding_path.is_dir():
        raise ValueError(
            "Cached embedding model directory is required; automatic downloads are disabled"
        )
    index_path = Path(index_path).resolve()
    if index_path.exists():
        if not overwrite:
            raise ValueError(
                f"Index directory {index_path} already exists; use a new versioned directory or --overwrite"
            )
        shutil.rmtree(index_path)

    config = build_config_for_preset(
        preset=preset,
        index_path=index_path,
        embedding_path=embedding_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = load_public_chunks(
        manifest_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        data_dir=data_dir,
    )
    expected_hashes = {
        str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(files)
    }
    embeddings = ChatGeoMagi.create_embeddings(
        config,
        local_files_only=True,
        embedding_cls=embedding_cls,
    )
    ChatGeoMagi.build_index(
        config,
        embeddings,
        chroma_cls=chroma_cls,
        preloaded_splits=(chunks, expected_hashes),
    )

    index_meta = json.loads(ChatGeoMagi.index_metadata_path(index_path).read_text())
    record = dict(manifest)
    record.update(
        {
            "index_path": str(index_path),
            "embedding_path": str(embedding_path),
            **index_meta,
        }
    )
    (index_path / "indexed_manifest.json").write_text(json.dumps(record, indent=2))
    return record


def evaluate_retrieval(
    index_path,
    embedding_path,
    manifest_path,
    benchmark_path,
    output_path,
    preset="stella_s2p_cosine",
    k=4,
    data_dir=None,
):
    retriever = open_retriever(
        index_path,
        embedding_path,
        manifest_path,
        preset=preset,
        k=k,
        data_dir=data_dir,
    )
    benchmark = json.loads(Path(benchmark_path).read_text())
    rows = []
    traces = []
    for item in benchmark:
        docs = retriever.invoke(item["question"])
        retrieved = [
            {
                "chunk_id": doc.metadata["chunk_id"],
                "source": doc.metadata.get("source_file")
                or Path(doc.metadata.get("source", "")).name,
                "page": doc.metadata.get("page"),
                "start_index": doc.metadata.get("start_index"),
                "excerpt": doc.page_content[:240],
            }
            for doc in docs
        ]
        metrics = retrieval_metrics(item, retrieved)
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "preset": preset,
                **metrics,
            }
        )
        traces.append(
            {
                "id": item["id"],
                "question": item["question"],
                "preset": preset,
                "retrieved": retrieved,
                "metrics": metrics,
            }
        )
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    write_csv(output_path / "metrics.csv", rows)
    (output_path / "metrics.json").write_text(json.dumps(rows, indent=2))
    (output_path / "traces.json").write_text(json.dumps(traces, indent=2))

    scored = [r for r in rows if r["hit_at_4"] is not None]
    all_slots = [c["chunk_id"] for t in traces for c in t["retrieved"][:k]]
    scored_ids = {r["id"] for r in scored}
    scored_slots = [
        c["chunk_id"]
        for t in traces
        if t["id"] in scored_ids
        for c in t["retrieved"][:k]
    ]
    from collections import Counter

    slot_counts = Counter(scored_slots)
    summary = {
        "preset": preset,
        "k": k,
        "total_questions": len(rows),
        "scored_questions": len(scored),
        "hits_at_4": sum(r["hit_at_4"] for r in scored),
        "hit_at_4": (
            sum(r["hit_at_4"] for r in scored) / len(scored) if scored else None
        ),
        "recall_at_4": (
            sum(r["recall_at_4"] for r in scored) / len(scored) if scored else None
        ),
        "precision_at_4": (
            sum(r["precision_at_4"] for r in scored) / len(scored) if scored else None
        ),
        "support_coverage_at_4": (
            sum(
                r["support_coverage"]
                for r in scored
                if r["support_coverage"] is not None
            )
            / len([r for r in scored if r["support_coverage"] is not None])
            if any(r["support_coverage"] is not None for r in scored)
            else None
        ),
        "total_slots_scored": len(scored_slots),
        "unique_chunks_scored": len(set(scored_slots)),
        "unique_chunks_all": len(set(all_slots)),
        "max_chunk_frequency_scored": max(slot_counts.values()) if slot_counts else 0,
        "top_repeated_chunks_scored": slot_counts.most_common(10),
    }
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2))
    return rows, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a public-corpus Chroma index and evaluate top-k retrieval."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "evaluation/benchmarks/public_corpus_manifest.json",
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--embedding-path", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument(
        "--preset",
        choices=sorted(RETRIEVAL_PRESETS.keys()),
        default="stella_s2p_cosine",
    )
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "evaluation/benchmarks/retrieval.json",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--overwrite-index", action="store_true")
    args = parser.parse_args()
    if not args.skip_build:
        build_index(
            args.manifest,
            args.embedding_path,
            args.index,
            preset=args.preset,
            overwrite=args.overwrite_index,
            data_dir=args.data_dir,
        )
    _, summary = evaluate_retrieval(
        args.index / "indexed_manifest.json"
        if (args.index / "indexed_manifest.json").is_file()
        else args.index,
        args.embedding_path,
        (args.index / "indexed_manifest.json")
        if (args.index / "indexed_manifest.json").is_file()
        else args.manifest,
        args.benchmark,
        args.output,
        preset=args.preset,
        k=args.k,
        data_dir=args.data_dir,
    )
    print(json.dumps(summary, indent=2))
