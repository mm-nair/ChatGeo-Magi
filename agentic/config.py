from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

STELLA_MODEL_ID = "NovaSearch/stella_en_1.5B_v5"
STELLA_DEFAULT_REVISION = "7817065102fd9e1b031fe874e910c01f40b2f001"
STELLA_S2P_QUERY_PROMPT = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: "
)


def load_env_files() -> None:
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent / ".env",
        Path(__file__).resolve().parents[1] / ".env",
    ]
    seen = set()
    try:
        from dotenv import find_dotenv, load_dotenv

        cwd_env = find_dotenv(usecwd=True)
        if cwd_env:
            resolved = Path(cwd_env).resolve()
            seen.add(resolved)
            load_dotenv(resolved, override=False)
        for path in candidates:
            if path.is_file():
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    load_dotenv(resolved, override=False)
        return
    except ImportError:
        pass

    for path in candidates:
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        for raw_line in resolved.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            if key:
                os.environ.setdefault(key, value)


RETRIEVAL_PRESETS = {
    "legacy_l2": {
        "trust_remote_code": False,
        "normalize_embeddings": False,
        "distance_metric": "l2",
        "query_instruction": "",
    },
    "causal_norm_cosine": {
        "trust_remote_code": False,
        "normalize_embeddings": True,
        "distance_metric": "cosine",
        "query_instruction": "",
    },
    "causal_s2p_cosine": {
        "trust_remote_code": False,
        "normalize_embeddings": True,
        "distance_metric": "cosine",
        "query_instruction": STELLA_S2P_QUERY_PROMPT,
    },
    "norm_cosine": {
        "trust_remote_code": True,
        "normalize_embeddings": True,
        "distance_metric": "cosine",
        "query_instruction": "",
    },
    "stella_s2p_cosine": {
        "trust_remote_code": True,
        "normalize_embeddings": True,
        "distance_metric": "cosine",
        "query_instruction": STELLA_S2P_QUERY_PROMPT,
    },
}


def canonical_embedding_model(model_name_or_path: str) -> str:
    value = str(model_name_or_path)
    if "stella" in value.lower():
        return STELLA_MODEL_ID
    return value


def extract_embedding_revision(model_name_or_path: str, default: str = STELLA_DEFAULT_REVISION) -> str:
    parts = Path(str(model_name_or_path)).parts
    if "snapshots" in parts:
        idx = parts.index("snapshots")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return default


@dataclass
class SystemConfig:
    chroma_db_path: str = "./chroma_db_stella_s2p_cosine_v1"
    data_dir: str = "./data"
    model_name: str = "gpt-oss:20b"
    embedding_model: str = STELLA_MODEL_ID
    embedding_revision: str = STELLA_DEFAULT_REVISION
    embedding_dimension: int = 1024
    request_timeout: float = 300.0
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 4
    trust_remote_code: bool = True
    normalize_embeddings: bool = True
    distance_metric: str = "cosine"
    query_instruction: str = STELLA_S2P_QUERY_PROMPT
    preset: str = "stella_s2p_cosine"
    public_only: bool = True

    def __post_init__(self):
        if self.retrieval_k <= 0:
            raise ValueError("retrieval_k must be a positive integer")
        if self.distance_metric not in ("l2", "cosine", "ip"):
            raise ValueError(f"Unsupported distance_metric: {self.distance_metric}")
        self.embedding_revision = extract_embedding_revision(
            self.embedding_model, self.embedding_revision
        )

    @classmethod
    def from_preset(cls, preset: str, **overrides) -> "SystemConfig":
        if preset not in RETRIEVAL_PRESETS:
            raise ValueError(f"Unknown retrieval preset: {preset}")
        params = dict(RETRIEVAL_PRESETS[preset])
        params["preset"] = preset
        params.update(overrides)
        return cls(**params)

    def retrieval_signature(self) -> dict:
        return {
            "preset": self.preset,
            "embedding_model": canonical_embedding_model(self.embedding_model),
            "embedding_revision": self.embedding_revision,
            "embedding_dimension": int(self.embedding_dimension),
            "chunk_size": int(self.chunk_size),
            "chunk_overlap": int(self.chunk_overlap),
            "length_function": "len",
            "trust_remote_code": bool(self.trust_remote_code),
            "normalize_embeddings": bool(self.normalize_embeddings),
            "distance_metric": self.distance_metric,
            "query_instruction": self.query_instruction,
            "retrieval_k": int(self.retrieval_k),
            "public_only": bool(self.public_only),
        }

    def index_signature(self) -> dict:
        sig = self.retrieval_signature()
        sig.pop("retrieval_k", None)
        return sig

    def config_hash(self, document_hashes: dict | None = None) -> str:
        payload = {
            "signature": self.index_signature(),
            "document_hashes": dict(sorted((document_hashes or {}).items())),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
