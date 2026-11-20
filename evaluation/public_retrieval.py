import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentic.cgm import ChatGeoMagi
from agentic.config import SystemConfig

FORBIDDEN_NAME_TOKENS = ("email", "support_email", "noaa_email", "customer", "inbox")


def chunk_identifier(text, metadata):
    payload = json.dumps(
        {"text": text, "metadata": metadata}, sort_keys=True, ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resolve_entry_path(entry, manifest_dir: Path, data_dir=None) -> Path:
    raw_path = entry.get("path") or entry.get("filename")
    if not raw_path:
        raise ValueError("Manifest entry lacks 'path' or 'filename'")
    candidate = Path(raw_path)
    if data_dir is not None:
        base_candidate = Path(data_dir) / candidate.name
        if base_candidate.exists():
            return base_candidate.resolve()
    if candidate.is_absolute():
        return candidate.resolve()
    return (manifest_dir / candidate).resolve()


def validate_manifest(manifest_path, data_dir=None):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("public_only") is not True or not manifest.get("files"):
        raise ValueError("Manifest must explicitly mark non-empty public_only files")
    files = []
    for entry in manifest["files"]:
        raw_name = str(entry.get("path") or entry.get("filename") or "")
        if any(token in raw_name.lower() for token in FORBIDDEN_NAME_TOKENS):
            raise ValueError(
                f"Forbidden support-email token in manifest path: {raw_name}"
            )
        path = _resolve_entry_path(entry, manifest_path.parent, data_dir=data_dir)
        if any(token in path.name.lower() for token in FORBIDDEN_NAME_TOKENS):
            raise ValueError(
                f"Forbidden support-email token in manifest path: {path}"
            )
        if (
            not entry.get("source_url")
            or not path.is_file()
            or path.suffix.lower() not in (".pdf", ".txt")
        ):
            raise ValueError(f"Invalid public document entry: {entry}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry.get("sha256"):
            raise ValueError(f"Manifest hash mismatch for {path}")
        files.append(path)
    return manifest, files


def load_public_chunks(manifest_path, chunk_size=1000, chunk_overlap=200, data_dir=None):
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    _, files = validate_manifest(manifest_path, data_dir=data_dir)
    documents = []
    for path in sorted(files):
        loader = (
            PyPDFLoader(str(path))
            if path.suffix.lower() == ".pdf"
            else TextLoader(str(path), encoding="utf-8", autodetect_encoding=True)
        )
        for doc in loader.load():
            doc.metadata["source"] = str(path)
            doc.metadata["source_file"] = path.name
            documents.append(doc)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    seen = set()
    for chunk in chunks:
        stable_metadata = {
            "source_file": Path(chunk.metadata.get("source", "")).name,
            "page": chunk.metadata.get("page"),
            "start_index": chunk.metadata.get("start_index"),
        }
        identifier = chunk_identifier(chunk.page_content, stable_metadata)
        if identifier in seen:
            raise ValueError("Duplicate chunk identifier")
        seen.add(identifier)
        chunk.metadata["chunk_id"] = identifier
    return chunks


def build_config_for_preset(
    preset="stella_s2p_cosine",
    index_path=None,
    embedding_path=None,
    k=4,
    chunk_size=1000,
    chunk_overlap=200,
):
    overrides = {
        "retrieval_k": k,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "public_only": True,
    }
    if index_path is not None:
        overrides["chroma_db_path"] = str(index_path)
    if embedding_path is not None:
        overrides["embedding_model"] = str(embedding_path)
    return SystemConfig.from_preset(preset, **overrides)


def open_retriever(
    index_path,
    embedding_path,
    manifest_path,
    preset="stella_s2p_cosine",
    k=4,
    embedding_cls=None,
    chroma_cls=None,
    data_dir=None,
):
    manifest, files = validate_manifest(manifest_path, data_dir=data_dir)
    index_path = Path(index_path).resolve()
    embedding_path = Path(embedding_path).resolve()
    if not index_path.is_dir() or not embedding_path.is_dir():
        raise ValueError("Existing index and cached embedding directory are required")
    if (
        manifest.get("index_path")
        and Path(manifest["index_path"]).resolve() != index_path
    ):
        raise ValueError("Index path does not match manifest")

    config = build_config_for_preset(
        preset=preset,
        index_path=index_path,
        embedding_path=embedding_path,
        k=k,
        chunk_size=manifest.get("chunk_size", 1000),
        chunk_overlap=manifest.get("chunk_overlap", 200),
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
    vector_store = ChatGeoMagi.validate_existing_index(
        config,
        embeddings=embeddings,
        expected_hashes=expected_hashes,
        chroma_cls=chroma_cls,
    )
    return vector_store.as_retriever(search_kwargs={"k": int(config.retrieval_k)})
