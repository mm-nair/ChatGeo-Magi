import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from config import SystemConfig
except ImportError:
    from agentic.config import SystemConfig

try:
    from prompts import system_prompt
except ImportError:
    from agentic.prompts import system_prompt


FORBIDDEN_EMAIL_PATTERNS = (
    "email",
    "emails",
    "inbox",
    "mbox",
    "support_mail",
    "qa_email",
    "customer_support",
)
FORBIDDEN_EMAIL_SUFFIXES = (".eml", ".mbox", ".msg", ".pst")


class ChatGeoMagi:
    def __init__(self, config: SystemConfig):
        self.config = config

        self.llm = None
        self.embeddings = None
        self.vector_store = None

        self.memory = None
        self.retriever = None
        self.agent = None

    @staticmethod
    def is_forbidden_source(path_str: str) -> bool:
        path = Path(str(path_str))
        lower_name = path.name.lower()
        lower_parts = [part.lower() for part in path.parts]
        if path.suffix.lower() in FORBIDDEN_EMAIL_SUFFIXES:
            return True
        for pattern in FORBIDDEN_EMAIL_PATTERNS:
            if pattern in lower_name or any(pattern == part for part in lower_parts):
                return True
        return False

    @staticmethod
    def discover_corpus_files(
        config: SystemConfig, file_paths: Optional[Iterable[Path]] = None
    ) -> List[Path]:
        if file_paths is not None:
            resolved = [Path(p).resolve() for p in file_paths]
        else:
            root = Path(config.data_dir)
            if not root.exists():
                return []
            resolved = sorted(
                p.resolve()
                for p in root.rglob("*")
                if p.is_file() and p.suffix.lower() in (".pdf", ".txt")
            )
        for path in resolved:
            if path.suffix.lower() not in (".pdf", ".txt"):
                raise ValueError(f"Unsupported corpus file extension: {path}")
            if config.public_only and ChatGeoMagi.is_forbidden_source(str(path)):
                raise ValueError(
                    f"Public-only RAG database cannot index support email or forbidden source: {path}"
                )
        return sorted(resolved, key=lambda p: str(p))

    @staticmethod
    def compute_document_hashes(
        file_paths: Iterable[Path], public_only: bool = True
    ) -> Dict[str, str]:
        hashes: Dict[str, str] = {}
        for raw_path in sorted((Path(p).resolve() for p in file_paths), key=lambda p: str(p)):
            if public_only and ChatGeoMagi.is_forbidden_source(str(raw_path)):
                raise ValueError(
                    f"Public-only RAG database cannot include support email source: {raw_path}"
                )
            hashes[str(raw_path)] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        return hashes

    @staticmethod
    def collection_metadata(config: SystemConfig) -> Dict[str, str]:
        if config.distance_metric not in ("l2", "cosine", "ip"):
            raise ValueError(f"Unsupported distance_metric: {config.distance_metric}")
        return {"hnsw:space": config.distance_metric}

    @staticmethod
    def index_metadata_path(chroma_db_path: str | Path) -> Path:
        return Path(chroma_db_path) / "index_metadata.json"

    @staticmethod
    def create_embeddings(
        config: SystemConfig,
        local_files_only: bool = False,
        embedding_cls=None,
    ):
        if embedding_cls is None:
            from langchain_huggingface.embeddings import HuggingFaceEmbeddings

            embedding_cls = HuggingFaceEmbeddings
        encode_kwargs = {"normalize_embeddings": bool(config.normalize_embeddings)}
        query_encode_kwargs = {"normalize_embeddings": bool(config.normalize_embeddings)}
        if config.query_instruction:
            query_encode_kwargs["prompt"] = config.query_instruction
        model_kwargs = {"trust_remote_code": bool(config.trust_remote_code)}
        if local_files_only:
            model_kwargs["local_files_only"] = True
        return embedding_cls(
            model_name=str(config.embedding_model),
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            query_encode_kwargs=query_encode_kwargs,
        )

    @staticmethod
    def load_and_split_documents(
        config: SystemConfig, file_paths: Optional[Iterable[Path]] = None
    ) -> Tuple[List, Dict[str, str]]:
        from langchain_community.document_loaders import PyPDFLoader, TextLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        files = ChatGeoMagi.discover_corpus_files(config, file_paths=file_paths)
        if not files:
            raise ValueError("No public corpus files found to index")
        document_hashes = ChatGeoMagi.compute_document_hashes(
            files, public_only=config.public_only
        )
        documents = []
        for path in files:
            loader = (
                PyPDFLoader(str(path))
                if path.suffix.lower() == ".pdf"
                else TextLoader(str(path))
            )
            documents.extend(loader.load())
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            add_start_index=True,
        )
        splits = splitter.split_documents(documents)
        if not splits:
            raise ValueError("No document chunks produced during splitting")
        return splits, document_hashes

    @staticmethod
    def _probe_query_dimension(embeddings, expected_dim: int) -> int:
        if embeddings is None or not hasattr(embeddings, "embed_query"):
            return expected_dim
        vec = embeddings.embed_query("dimension verification query")
        actual_dim = len(vec)
        if expected_dim and actual_dim != expected_dim:
            raise ValueError(
                f"Query embedding dimension {actual_dim} does not match configured dimension {expected_dim}"
            )
        return actual_dim

    @staticmethod
    def build_index(
        config: SystemConfig,
        embeddings,
        file_paths: Optional[Iterable[Path]] = None,
        batch_size: int = 128,
        chroma_cls=None,
        preloaded_splits: Optional[Tuple[List, Dict[str, str]]] = None,
    ):
        if chroma_cls is None:
            from langchain_chroma import Chroma

            chroma_cls = Chroma
        splits, document_hashes = (
            preloaded_splits
            if preloaded_splits is not None
            else ChatGeoMagi.load_and_split_documents(config, file_paths=file_paths)
        )
        query_dim = ChatGeoMagi._probe_query_dimension(
            embeddings, config.embedding_dimension
        )
        db_path = Path(config.chroma_db_path)
        db_path.mkdir(parents=True, exist_ok=True)
        vector_store = chroma_cls(
            collection_name="cgmdocs",
            embedding_function=embeddings,
            persist_directory=str(db_path),
            collection_metadata=ChatGeoMagi.collection_metadata(config),
        )
        for start in range(0, len(splits), batch_size):
            batch = splits[start : start + batch_size]
            vector_store.add_documents(documents=batch)
        sample = vector_store._collection.get(limit=1, include=["embeddings"])
        stored_embeddings = sample.get("embeddings")
        if stored_embeddings is not None and len(stored_embeddings) > 0:
            doc_dim = len(stored_embeddings[0])
            if doc_dim != query_dim:
                raise ValueError(
                    f"Document embedding dimension {doc_dim} does not match query embedding dimension {query_dim}"
                )
        else:
            doc_dim = query_dim
        metadata = {
            **config.retrieval_signature(),
            "embedding_dimension": int(doc_dim),
            "chunk_count": len(splits),
            "document_hashes": document_hashes,
            "config_hash": config.config_hash(document_hashes),
        }
        ChatGeoMagi.index_metadata_path(db_path).write_text(
            json.dumps(metadata, indent=2)
        )
        return vector_store, splits, metadata

    @staticmethod
    def validate_existing_index(
        config: SystemConfig,
        embeddings=None,
        expected_hashes: Optional[Dict[str, str]] = None,
        chroma_cls=None,
    ):
        if chroma_cls is None:
            from langchain_chroma import Chroma

            chroma_cls = Chroma
        db_path = Path(config.chroma_db_path)
        meta_path = ChatGeoMagi.index_metadata_path(db_path)
        if not db_path.is_dir() or not meta_path.is_file():
            raise ValueError(
                f"Existing Chroma index at {db_path} lacks index_metadata.json and cannot be silently reused"
            )
        recorded = json.loads(meta_path.read_text())
        expected_sig = config.index_signature()
        for key, expected_val in expected_sig.items():
            if recorded.get(key) != expected_val:
                raise ValueError(
                    f"Incompatible Chroma index configuration for '{key}': index has {recorded.get(key)!r}, config requires {expected_val!r}"
                )
        if expected_hashes is not None:
            if recorded.get("document_hashes") != expected_hashes:
                raise ValueError(
                    "Existing Chroma index document hashes do not match the requested corpus"
                )
        query_dim = ChatGeoMagi._probe_query_dimension(
            embeddings, config.embedding_dimension
        )
        if recorded.get("embedding_dimension") != query_dim:
            raise ValueError(
                f"Recorded index embedding dimension {recorded.get('embedding_dimension')} does not match query embedding dimension {query_dim}"
            )
        vector_store = chroma_cls(
            collection_name="cgmdocs",
            embedding_function=embeddings,
            persist_directory=str(db_path),
            collection_metadata=ChatGeoMagi.collection_metadata(config),
        )
        if not vector_store._collection.count():
            raise ValueError("Public index is empty")
        coll_meta = getattr(vector_store._collection, "metadata", None) or {}
        recorded_space = coll_meta.get("hnsw:space", "l2")
        if recorded_space != config.distance_metric:
            raise ValueError(
                f"Chroma collection distance metric {recorded_space!r} does not match configured {config.distance_metric!r}"
            )
        sample = vector_store._collection.get(limit=1, include=["embeddings"])
        stored_embeddings = sample.get("embeddings")
        if stored_embeddings is not None and len(stored_embeddings) > 0:
            doc_dim = len(stored_embeddings[0])
            if doc_dim != query_dim:
                raise ValueError(
                    f"Stored vector dimension {doc_dim} does not match query dimension {query_dim}"
                )
        indexed = vector_store.get(include=["metadatas"])
        permitted = (
            set(expected_hashes.keys())
            if expected_hashes is not None
            else set((recorded.get("document_hashes") or {}).keys())
        )
        for metadata in indexed.get("metadatas") or []:
            source = str(Path(metadata.get("source", "")).resolve())
            if config.public_only and ChatGeoMagi.is_forbidden_source(source):
                raise ValueError(
                    f"Index contains forbidden support email source: {source}"
                )
            if permitted and source not in permitted:
                raise ValueError(f"Index contains source outside manifest: {source}")
        return vector_store, recorded

    @staticmethod
    def setup_vector_store(
        config: SystemConfig,
        embeddings,
        strict: bool = False,
        batch_size: int = 128,
        chroma_cls=None,
    ):
        from langchain_core.tools import create_retriever_tool

        print("Loading vector store")
        expected_hashes = None
        if Path(config.data_dir).exists():
            files = ChatGeoMagi.discover_corpus_files(config)
            if files:
                expected_hashes = ChatGeoMagi.compute_document_hashes(
                    files, public_only=config.public_only
                )
        if os.path.exists(config.chroma_db_path):
            try:
                vector_store, _ = ChatGeoMagi.validate_existing_index(
                    config,
                    embeddings=embeddings,
                    expected_hashes=expected_hashes,
                    chroma_cls=chroma_cls,
                )
            except ValueError as err:
                if strict:
                    raise
                version_suffix = config.config_hash(expected_hashes)[:8]
                versioned_path = f"{config.chroma_db_path}_{config.preset}_{version_suffix}"
                print(
                    f"Existing index at {config.chroma_db_path} is incompatible ({err}); using versioned index {versioned_path}"
                )
                config.chroma_db_path = versioned_path
                if os.path.exists(config.chroma_db_path):
                    vector_store, _ = ChatGeoMagi.validate_existing_index(
                        config,
                        embeddings=embeddings,
                        expected_hashes=expected_hashes,
                        chroma_cls=chroma_cls,
                    )
                else:
                    vector_store, splits, _ = ChatGeoMagi.build_index(
                        config,
                        embeddings,
                        batch_size=batch_size,
                        chroma_cls=chroma_cls,
                    )
                    print(f"Loaded {len(splits)} document chunks")
        else:
            vector_store, splits, _ = ChatGeoMagi.build_index(
                config,
                embeddings,
                batch_size=batch_size,
                chroma_cls=chroma_cls,
            )
            print(f"Loaded {len(splits)} document chunks")

        vector_retriever = vector_store.as_retriever(
            search_kwargs={"k": config.retrieval_k}
        )
        return create_retriever_tool(
            vector_retriever,
            "retrieve_geomag_docs",
            "Search and return information about geomagnetism, from authoritative public documents like the World Magnetic Model technical report.",
        )

    @staticmethod
    def get_shared_resources(config: SystemConfig, local_files_only: bool = False):
        from langchain_ollama import ChatOllama

        embeddings = ChatGeoMagi.create_embeddings(
            config, local_files_only=local_files_only
        )
        llm = ChatOllama(model=config.model_name)
        retriever = ChatGeoMagi.setup_vector_store(config, embeddings)

        return llm, embeddings, retriever

    def initialize(self, llm, embeddings, retriever):
        from langchain_core.messages import SystemMessage
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.prebuilt import create_react_agent

        try:
            from apis.noaa_apis import contour_map, noaa_mag_api, plot, plot_many
        except ImportError:
            from agentic.apis.noaa_apis import contour_map, noaa_mag_api, plot, plot_many

        self.llm = llm
        self.embeddings = embeddings
        self.retriever = retriever
        self.memory = MemorySaver()

        tools = [
            noaa_mag_api,
            plot,
            plot_many,
            contour_map,
        ]

        if self.retriever is not None:
            tools.insert(0, self.retriever)

        self.agent = create_react_agent(
            self.llm,
            tools,
            checkpointer=self.memory,
            prompt=SystemMessage(content=system_prompt),
        )

    @staticmethod
    def _activate_request_context(question: str) -> None:
        try:
            from apis.noaa_apis import set_active_request_context
        except ImportError:
            from agentic.apis.noaa_apis import set_active_request_context
        set_active_request_context(question)

    def invoke(self, question: str, thread_id: str = "1"):
        self._activate_request_context(question)
        message = {"role": "user", "content": question}

        return self.agent.invoke(
            {"messages": [message]},
            config={"configurable": {"thread_id": thread_id}},
        )

    def stream_invoke(self, question: str, thread_id: str = "1"):
        self._activate_request_context(question)
        message = {"role": "user", "content": question}

        return self.agent.stream(
            {"messages": [message]},
            config={"configurable": {"thread_id": thread_id}},
            stream_mode="values",
        )

    def traced_invoke(self, question: str, thread_id: str = "1"):
        from opik.integrations.langchain import OpikTracer

        self._activate_request_context(question)
        message = {"role": "user", "content": question}
        tracer = OpikTracer(graph=self.agent.get_graph(xray=True))

        return self.agent.invoke(
            {"messages": [message]},
            config={"configurable": {"thread_id": thread_id}, "callbacks": [tracer]},
        )

    def visualize_agent(self, filepath="agent_graph.png"):
        self.agent.get_graph(xray=True).draw_mermaid_png(
            output_file_path=filepath,
        )

        print(self.agent.get_graph(xray=True).draw_ascii())
        print(f"Agent graph saved to {filepath}")
