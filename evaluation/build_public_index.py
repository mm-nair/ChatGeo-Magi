import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentic.cgm import ChatGeoMagi
from agentic.config import RETRIEVAL_PRESETS, SystemConfig
from public_retrieval import chunk_identifier, validate_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--embedding-path", required=True, type=Path)
    parser.add_argument(
        "--preset",
        choices=sorted(RETRIEVAL_PRESETS.keys()),
        default="stella_s2p_cosine",
    )
    parser.add_argument("--retrieval-k", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "revision_workspace/retrieval_evaluation/public_index_stella_s2p_cosine_v1",
    )
    parser.add_argument("--manifest-output", type=Path)
    args = parser.parse_args()
    if (
        not args.output.resolve().is_relative_to(ROOT / "revision_workspace")
        or args.output.exists()
    ):
        parser.error("Use a fresh index directory inside revision_workspace")
    manifest = validate_manifest(args.manifest)
    if not args.embedding_path.is_dir():
        parser.error("Cached Stella model directory required; downloads disabled")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    config = SystemConfig.from_preset(
        args.preset,
        chroma_db_path=str(args.output),
        embedding_model=str(args.embedding_path),
        retrieval_k=args.retrieval_k,
        public_only=True,
    )
    file_paths = [Path(record["path"]).resolve() for record in manifest["files"]]
    embeddings = ChatGeoMagi.create_embeddings(config, local_files_only=True)
    _, chunks, index_metadata = ChatGeoMagi.build_index(
        config,
        embeddings,
        file_paths=file_paths,
    )

    catalog = [
        {
            "chunk_id": chunk_identifier(chunk.page_content, chunk.metadata),
            "text": chunk.page_content,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ]
    (args.output / "chunk_catalog.json").write_text(json.dumps(catalog, indent=2))
    parent_catalog = args.output.parent / "chunk_catalog.json"
    if not parent_catalog.exists():
        parent_catalog.write_text(json.dumps(catalog, indent=2))

    indexed_manifest = dict(manifest)
    indexed_manifest.update(
        {
            "index_path": str(args.output.resolve()),
            **index_metadata,
        }
    )
    (args.output / "indexed_manifest.json").write_text(
        json.dumps(indexed_manifest, indent=2)
    )
    if args.manifest_output is not None:
        args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_output.write_text(json.dumps(indexed_manifest, indent=2))
    else:
        parent_manifest = args.output.parent / "indexed_manifest.json"
        if not parent_manifest.exists():
            parent_manifest.write_text(json.dumps(indexed_manifest, indent=2))


if __name__ == "__main__":
    main()
