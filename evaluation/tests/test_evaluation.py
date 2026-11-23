import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

from agentic.cgm import ChatGeoMagi
from agentic.config import (
    STELLA_DEFAULT_REVISION,
    STELLA_MODEL_ID,
    STELLA_S2P_QUERY_PROMPT,
    SystemConfig,
)
from benchmark_metrics import (
    argument_matches,
    bootstrap,
    retrieval_metrics,
    summarize_paired_scores,
    tool_metrics,
)
from golden_dataset import load_goldens
from openai_reference_judge import (
    JUDGE_INSTRUCTIONS,
    assert_public_only_input,
    judge_payload,
)
from plots import generate_plots
from public_retrieval import (
    chunk_identifier,
    load_public_chunks,
    open_retriever,
    validate_manifest,
)
from run_revision_inference import verify_reusable_no_rag
from score_live_tools import numbers, score_question


class FakeEmbeddingFunction:
    instances = []

    def __init__(
        self,
        model_name,
        model_kwargs=None,
        encode_kwargs=None,
        query_encode_kwargs=None,
        dim=1024,
    ):
        self.model_name = model_name
        self.model_kwargs = model_kwargs or {}
        self.encode_kwargs = encode_kwargs or {}
        self.query_encode_kwargs = query_encode_kwargs or {}
        self.dim = dim
        self.doc_calls = []
        self.query_calls = []
        FakeEmbeddingFunction.instances.append(self)

    def embed_documents(self, texts):
        self.doc_calls.append((list(texts), dict(self.encode_kwargs)))
        return [[1.0] * self.dim for _ in texts]

    def embed_query(self, text):
        self.query_calls.append((text, dict(self.query_encode_kwargs)))
        return [1.0] * self.dim


class FakeCollection:
    def __init__(self, metadata=None, dim=1024):
        self.metadata = dict(metadata or {})
        self.docs = []
        self.dim = dim

    def count(self):
        return len(self.docs)

    def get(self, limit=None, include=None):
        items = self.docs[:limit] if limit else self.docs
        return {
            "embeddings": [[1.0] * self.dim for _ in items],
            "metadatas": [d.metadata for d in items],
        }


class FakeChroma:
    stores = {}

    def __init__(
        self,
        collection_name,
        embedding_function,
        persist_directory,
        collection_metadata=None,
    ):
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.persist_directory = str(persist_directory)
        if self.persist_directory not in FakeChroma.stores:
            FakeChroma.stores[self.persist_directory] = FakeCollection(
                metadata=collection_metadata,
                dim=getattr(embedding_function, "dim", 1024),
            )
        self._collection = FakeChroma.stores[self.persist_directory]

    def add_documents(self, documents):
        self.embedding_function.embed_documents([d.page_content for d in documents])
        self._collection.docs.extend(documents)

    def get(self, include=None):
        return {"metadatas": [d.metadata for d in self._collection.docs]}

    def as_retriever(self, search_kwargs=None):
        return {"search_kwargs": dict(search_kwargs or {}), "store": self}


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        FakeEmbeddingFunction.instances.clear()
        FakeChroma.stores.clear()

    def test_original_headerless_goldens(self):
        self.assertEqual(len(load_goldens(ROOT / "evaluation/goldens.csv")), 25)

    def test_duplicate_goldens_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "goldens.csv"
            path.write_text("question,answer\nq,a\nq,b\n")
            with self.assertRaises(ValueError):
                load_goldens(path)

    def test_bootstrap_pairing(self):
        self.assertEqual(bootstrap([0.25] * 5, resamples=1000), [0.25, 0.25])
        self.assertEqual(
            bootstrap([-0.2, 0, 0.1], resamples=1000),
            bootstrap([-0.2, 0, 0.1], resamples=1000),
        )
        with self.assertRaises(ValueError):
            bootstrap([])

    def test_retrieval_annotations(self):
        expected = {"verification_status": "pending"}
        self.assertIsNone(retrieval_metrics(expected, [])["hit_at_4"])
        expected = {
            "verification_status": "human_verified",
            "relevant_chunk_ids": ["a", "b"],
            "support_facts": [{"chunk_ids": ["a"]}, {"chunk_ids": ["b"]}],
        }
        metrics = retrieval_metrics(expected, [{"chunk_id": "a"}, {"chunk_id": "c"}])
        self.assertEqual(metrics["hit_at_4"], 1)
        self.assertEqual(metrics["recall_at_4"], 0.5)
        self.assertEqual(metrics["precision_at_4"], 0.25)
        self.assertEqual(metrics["support_coverage"], 0.5)
        with self.assertRaises(ValueError):
            retrieval_metrics(expected, [{"chunk_id": "a"}, {"chunk_id": "a"}])

    def test_tools_require_traces(self):
        expected = {"calls": [{"name": "noaa_mag_api", "args": {"start_time": "2026"}}]}
        result = tool_metrics(expected, {"final": "The declination is 1 degree"})
        self.assertFalse(result["tool_selection"])
        self.assertFalse(result["execution"])
        self.assertIsNone(result["end_to_end"])
        result = tool_metrics(
            expected,
            {
                "calls": [
                    {
                        "name": "noaa_mag_api",
                        "args": {"start_time": "2026-01-01"},
                        "success": True,
                    }
                ]
            },
        )
        self.assertTrue(result["execution"])
        self.assertIsNone(result["numerical_agreement"])

    def test_equivalent_coordinate_arguments(self):
        self.assertTrue(argument_matches("location", "51.500, -0.120", "51.5,-0.12"))
        self.assertFalse(argument_matches("location", "51.5,0.12", "51.5,-0.12"))

    def test_unicode_number_extraction(self):
        self.assertEqual(
            numbers("49\u202f100.5 nT and \u201116.98374\u00b0"), [49100.5, -16.98374]
        )

    def test_corrected_summary_requires_complete_pairs(self):
        with self.assertRaises(ValueError):
            summarize_paired_scores(
                [{"model": "test", "id": "Q01", "condition": "rag", "score": 1}]
            )

    def test_tool_sequence(self):
        expected = {
            "calls": [
                {"name": "noaa_mag_api", "args": {}},
                {"name": "plot", "args": {}},
            ]
        }
        trace = {
            "calls": [
                {"name": "plot", "args": {}, "success": True},
                {"name": "noaa_mag_api", "args": {}, "success": True},
            ]
        }
        self.assertFalse(tool_metrics(expected, trace)["parameters_and_order"])
        self.assertFalse(tool_metrics(expected, trace)["execution"])

    def test_tool_execution_independent_of_strict_parameters_and_numbers(self):
        expected = json.loads(
            (ROOT / "evaluation/benchmarks/tool_expected.json").read_text()
        )[6]
        reference = expected["authoritative_calls"][0]
        trace = {"calls": [{
            "name": "noaa_mag_api",
            "args": {**expected["calls"][0]["args"], "end_time": "2026-12-31"},
            "success": True,
            "output": json.dumps({
                "component": reference["component"],
                "values": reference["values"],
                "provenance": {"returned_model": reference["returned_model"]},
            }),
        }]}
        result = score_question(expected, trace)
        self.assertTrue(result["execution"])
        self.assertFalse(result["parameters_and_order"])
        self.assertTrue(result["tool_numerical_agreement"])
        output = json.loads(trace["calls"][0]["output"])
        output["values"][0]["value"] += 100
        trace["calls"][0]["output"] = json.dumps(output)
        result = score_question(expected, trace)
        self.assertTrue(result["execution"])
        self.assertFalse(result["tool_numerical_agreement"])
        for invalid_output in ["Error: unavailable", '{"values": []}',
                               json.dumps({**output, "values": [{"date": 2026, "value": float("nan")}]})]:
            trace["calls"][0]["output"] = invalid_output
            self.assertFalse(score_question(expected, trace)["execution"])
        trace["calls"][0]["success"] = False
        self.assertFalse(score_question(expected, trace)["execution"])

    def test_documented_evaluation_arguments(self):
        import argparse
        import ast
        import re
        import shlex
        from agentic.config import RETRIEVAL_PRESETS
        from plots import FIGURE_BUILDERS, OUTPUT_DIR, RESULTS_DIR

        commands = re.findall(r"^python (evaluation/[^\n]+)",
                              (ROOT / "README.md").read_text().replace("\\\n", ""), re.MULTILINE)
        self.assertTrue(commands)
        for command in commands:
            arguments = shlex.split(command)
            script = ROOT / arguments[0]
            self.assertTrue(script.is_file(), arguments[0])
            tree = ast.parse(script.read_text())
            entry = next(node for node in tree.body if
                         isinstance(node, ast.FunctionDef) and node.name == "main") if any(
                             isinstance(node, ast.FunctionDef) and node.name == "main" for node in tree.body
                         ) else tree.body[-1]
            statements = []
            for node in entry.body:
                if isinstance(node, ast.Assign) and any(isinstance(child, ast.Call) and
                    isinstance(child.func, ast.Attribute) and child.func.attr == "parse_args"
                    for child in ast.walk(node)):
                    break
                statements.append(node)
            namespace = {"argparse": argparse, "Path": Path, "ROOT": ROOT,
                         "RETRIEVAL_PRESETS": RETRIEVAL_PRESETS,
                         "FIGURE_BUILDERS": FIGURE_BUILDERS, "OUTPUT_DIR": OUTPUT_DIR,
                         "RESULTS_DIR": RESULTS_DIR}
            exec(compile(ast.Module(body=statements, type_ignores=[]), str(script), "exec"), namespace)
            namespace["parser"].parse_args(arguments[1:])

    def test_index_cli_passes_directory_to_retrieval(self):
        import argparse
        import ast
        from unittest.mock import Mock
        from agentic.config import RETRIEVAL_PRESETS

        source = ROOT / "evaluation/build_public_index.py"
        entry = ast.parse(source.read_text()).body[-1]
        with tempfile.TemporaryDirectory() as directory:
            index_path = Path(directory) / "index"
            index_path.mkdir()
            manifest_path = index_path / "indexed_manifest.json"
            manifest_path.write_text('{}')
            for skip_build in (False, True):
                build_index = Mock()
                evaluate_retrieval = Mock(return_value=([], {}))
                namespace = {"argparse": argparse, "Path": Path, "ROOT": ROOT,
                             "RETRIEVAL_PRESETS": RETRIEVAL_PRESETS, "json": json,
                             "build_index": build_index, "evaluate_retrieval": evaluate_retrieval}
                arguments = [str(source), "--index", str(index_path), "--embedding-path", "cached",
                             "--output", str(Path(directory) / "results")]
                if skip_build:
                    arguments.append("--skip-build")
                with patch.object(sys, "argv", arguments):
                    exec(compile(ast.Module(body=entry.body, type_ignores=[]), str(source), "exec"), namespace)
                self.assertEqual(build_index.call_count, int(not skip_build))
                self.assertEqual(evaluate_retrieval.call_args.args[0], index_path)
                self.assertEqual(evaluate_retrieval.call_args.args[2], manifest_path)

    def test_judge_cli_keeps_condition_labels_separate(self):
        import openai_reference_judge

        fake_openai = types.ModuleType("openai")
        fake_openai.OpenAI = lambda: object()
        with tempfile.TemporaryDirectory() as directory:
            inputs = Path(directory) / "judge_inputs.json"
            inputs.write_text(json.dumps([
                {"id": "Q01", "condition": condition, "input": "question",
                 "expected_answer": "reference", "model_output": "answer"}
                for condition in ("rag", "no_rag")
            ]))
            output = Path(directory) / "judging"
            arguments = ["openai_reference_judge.py", "--inputs",
                         f"no_rag={inputs}:no_rag", f"rag_15doc_corrected={inputs}:rag",
                         "--output", str(output)]
            with patch.dict(sys.modules, {"openai": fake_openai}), patch.object(sys, "argv", arguments), \
                 patch.object(openai_reference_judge, "run_calibration", return_value=[{"passed": True}]), \
                 patch.object(openai_reference_judge, "judge_single", return_value=(1.0, {"score": 1.0, "raw_score_0_10": 10})) as judge_single:
                openai_reference_judge.main()
            self.assertEqual(judge_single.call_count, 2)
            self.assertEqual({row["condition"] for row in json.loads((output / "all_scores.json").read_text())},
                             {"no_rag", "rag_15doc_corrected"})

    def test_benchmark_integrity(self):
        prompts = json.loads(
            (ROOT / "evaluation/benchmarks/tool_prompts.json").read_text()
        )
        expected = json.loads(
            (ROOT / "evaluation/benchmarks/tool_expected.json").read_text()
        )
        self.assertEqual(len(prompts), 20)
        self.assertEqual(
            [row["id"] for row in prompts], [row["id"] for row in expected]
        )
        self.assertTrue(all(set(row) == {"id", "prompt"} for row in prompts))

    def test_judge_reference_payload(self):
        payload = json.loads(
            judge_payload(
                {
                    "input": "question",
                    "expected_answer": "reference",
                    "model_output": "response",
                }
            )
        )
        self.assertEqual(
            payload,
            {
                "QUESTION": "question",
                "EXPECTED_ANSWER": "reference",
                "MODEL_OUTPUT": "response",
            },
        )
        self.assertIn("EXPECTED_ANSWER", JUDGE_INSTRUCTIONS)

    def test_chunk_identifier_stability(self):
        self.assertEqual(
            chunk_identifier("text", {"page": 1, "source": "doc"}),
            chunk_identifier("text", {"source": "doc", "page": 1}),
        )
        self.assertNotEqual(chunk_identifier("text", {}), chunk_identifier("other", {}))

    def test_retrieval_benchmark_chunk_ids_match_portable_chunker(self):
        # 1. Confirm load_public_chunks produces identical portable chunk IDs across
        # different absolute directory paths.
        class FakeDoc:
            def __init__(self, page_content, metadata):
                self.page_content = page_content
                self.metadata = dict(metadata)

        class FakeTextLoader:
            def __init__(self, path, encoding="utf-8", autodetect_encoding=True):
                self.path = Path(path)

            def load(self):
                return [FakeDoc(self.path.read_text(encoding="utf-8"), {"source": str(self.path)})]

        class FakeSplitter:
            def __init__(self, chunk_size, chunk_overlap, length_function=len, add_start_index=True):
                self.chunk_size = chunk_size
                self.add_start_index = add_start_index

            def split_documents(self, documents):
                out = []
                for doc in documents:
                    meta = dict(doc.metadata)
                    meta["start_index"] = 0
                    out.append(FakeDoc(doc.page_content, meta))
                return out

        fake_loaders = types.ModuleType("langchain_community.document_loaders")
        fake_loaders.PyPDFLoader = FakeTextLoader
        fake_loaders.TextLoader = FakeTextLoader
        fake_splitters = types.ModuleType("langchain_text_splitters")
        fake_splitters.RecursiveCharacterTextSplitter = FakeSplitter

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            content = "The World Magnetic Model is updated every five years."
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            manifest_path = tmp_path / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "public_only": True,
                        "files": [
                            {
                                "filename": "wmm_public.txt",
                                "source_url": "https://www.ncei.noaa.gov/products/world-magnetic-model",
                                "sha256": digest,
                            }
                        ],
                    }
                )
            )
            dir_a = tmp_path / "machine_a"
            dir_b = tmp_path / "machine_b" / "nested"
            dir_a.mkdir(parents=True)
            dir_b.mkdir(parents=True)
            (dir_a / "wmm_public.txt").write_text(content, encoding="utf-8")
            (dir_b / "wmm_public.txt").write_text(content, encoding="utf-8")
            with patch.dict(
                sys.modules,
                {
                    "langchain_community": types.ModuleType("langchain_community"),
                    "langchain_community.document_loaders": fake_loaders,
                    "langchain_text_splitters": fake_splitters,
                },
            ):
                chunks_a = load_public_chunks(manifest_path, data_dir=dir_a)
                chunks_b = load_public_chunks(manifest_path, data_dir=dir_b)
            expected_id = chunk_identifier(
                content, {"source_file": "wmm_public.txt", "page": None, "start_index": 0}
            )
            self.assertEqual(chunks_a[0].metadata["chunk_id"], expected_id)
            self.assertEqual(chunks_b[0].metadata["chunk_id"], expected_id)

        # 2. Structural and cross-benchmark portable ID consistency across all four files.
        bench_dir = ROOT / "evaluation/benchmarks"
        benchmarks = {
            name: json.loads((bench_dir / name).read_text(encoding="utf-8"))
            for name in (
                "retrieval.json",
                "retrieval_2doc.json",
                "retrieval_heldout.json",
                "retrieval_heldout_2doc.json",
            )
        }
        for name, items in benchmarks.items():
            for item in items:
                rel_ids = item.get("relevant_chunk_ids", [])
                for cid in rel_ids:
                    self.assertEqual(len(cid), 64, f"{name}:{item['id']} invalid chunk_id")
                    self.assertTrue(all(ch in "0123456789abcdef" for ch in cid))
                for sf in item.get("support_facts", []):
                    for cid in sf.get("chunk_ids", []):
                        self.assertIn(cid, rel_ids, f"{name}:{item['id']} support_fact id not in relevant_chunk_ids")

        b15 = {x["id"]: set(x["relevant_chunk_ids"]) for x in benchmarks["retrieval.json"]}
        b2 = {x["id"]: set(x["relevant_chunk_ids"]) for x in benchmarks["retrieval_2doc.json"]}
        for qid, ids2 in b2.items():
            if ids2:
                self.assertTrue(ids2.issubset(b15[qid]), f"Portable 2-doc IDs for {qid} missing from 15-doc benchmark")

        h15 = {x["id"]: set(x["relevant_chunk_ids"]) for x in benchmarks["retrieval_heldout.json"]}
        h2 = {x["id"]: set(x["relevant_chunk_ids"]) for x in benchmarks["retrieval_heldout_2doc.json"]}
        self.assertEqual(h15, h2)

        # 3. When local reconstructed corpus indexes exist, confirm 100% ID resolution
        # and exact reproducibility of recorded 15-doc retrieval hits (22/25 and 13/14).
        ws = ROOT / "revision_workspace/final_retrieval_revision"
        index_specs = [
            ("expanded_stella_s2p_cosine", ("retrieval.json", "retrieval_heldout.json")),
            ("two_doc_stella_s2p_cosine", ("retrieval_2doc.json", "retrieval_heldout_2doc.json")),
        ]
        for db_name, bench_names in index_specs:
            db_path = ws / "indexes" / db_name / "chroma.sqlite3"
            if not db_path.is_file():
                continue
            con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            try:
                cur = con.cursor()
                cur.execute(
                    "SELECT id, key, string_value, int_value, float_value, bool_value FROM embedding_metadata"
                )
                by_id = {}
                for eid, key, sv, iv, fv, bv in cur.fetchall():
                    d = by_id.setdefault(eid, {})
                    d[key] = (
                        sv if sv is not None else (
                            iv if iv is not None else (
                                fv if fv is not None else (bool(bv) if bv is not None else None)
                            )
                        )
                    )
            finally:
                con.close()
            portable_ids = set()
            for meta in by_id.values():
                text = meta.pop("chroma:document")
                stable_meta = {
                    "source_file": Path(meta.get("source", "")).name,
                    "page": meta.get("page"),
                    "start_index": meta.get("start_index"),
                }
                cid = chunk_identifier(text, stable_meta)
                self.assertNotIn(cid, portable_ids)
                portable_ids.add(cid)
            for bname in bench_names:
                for item in benchmarks[bname]:
                    for cid in item.get("relevant_chunk_ids", []):
                        self.assertIn(cid, portable_ids, f"{bname}:{item['id']} chunk_id not in {db_name}")
                    for sf in item.get("support_facts", []):
                        for cid in sf.get("chunk_ids", []):
                            self.assertIn(cid, portable_ids, f"{bname}:{item['id']} support_fact id not in {db_name}")

        for run_id, bname, expected_hits in (
            ("expanded_stella_s2p_cosine_full25", "retrieval.json", 22),
            ("expanded_stella_s2p_cosine_heldout14", "retrieval_heldout.json", 13),
        ):
            ret_path = ws / run_id / "retrieved.json"
            if not ret_path.is_file():
                continue
            raw_retrieved = json.loads(ret_path.read_text(encoding="utf-8"))
            hits = 0
            for item in benchmarks[bname]:
                docs = [
                    {
                        "chunk_id": chunk_identifier(
                            doc["text"],
                            {
                                "source_file": Path(doc["metadata"].get("source", "")).name,
                                "page": doc["metadata"].get("page"),
                                "start_index": doc["metadata"].get("start_index"),
                            },
                        )
                    }
                    for doc in raw_retrieved[item["id"]]
                ]
                hits += retrieval_metrics(item, docs)["hit_at_4"]
            self.assertEqual(hits, expected_hits)

    def test_inference_isolation_wiring(self):
        import ast

        source = (
            ROOT / "evaluation/legacy_email_experiment/eval_inference.py"
        ).read_text()
        tree = ast.parse(source)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_model"
        )
        module = ast.Module(body=[function], type_ignores=[])

        class FakeConfig:
            pass

        class FakeAgent:
            threads = []

            @staticmethod
            def get_shared_resources(config):
                return None, None, None

            def __init__(self, config):
                pass

            def initialize(self, *args):
                pass

            def invoke(self, question, thread_id):
                self.threads.append(thread_id)
                return {"messages": [type("Message", (), {"content": "ok"})()]}

        namespace = {
            "SystemConfig": FakeConfig,
            "ChatGeoMagi": FakeAgent,
            "uuid4": lambda: type("ID", (), {"hex": "run"})(),
        }
        exec(compile(module, "<isolated_function>", "exec"), namespace)
        goldens = [{"id": 1, "question": "first"}, {"id": 2, "question": "second"}]
        namespace["run_model"]("fake", goldens)
        self.assertEqual(FakeAgent.threads, ["run-1", "run-2"])
        FakeAgent.threads.clear()
        namespace["run_model"]("fake", goldens, isolated_sessions=False)
        self.assertEqual(FakeAgent.threads, ["1", "1"])

    def test_query_instructions_only_on_queries_and_presets(self):
        cfg_c = SystemConfig.from_preset("stella_s2p_cosine", retrieval_k=6)
        emb_c = ChatGeoMagi.create_embeddings(
            cfg_c, local_files_only=True, embedding_cls=FakeEmbeddingFunction
        )
        self.assertEqual(emb_c.encode_kwargs, {"normalize_embeddings": True})
        self.assertNotIn("prompt", emb_c.encode_kwargs)
        self.assertEqual(
            emb_c.query_encode_kwargs,
            {"normalize_embeddings": True, "prompt": STELLA_S2P_QUERY_PROMPT},
        )
        self.assertEqual(
            ChatGeoMagi.collection_metadata(cfg_c), {"hnsw:space": "cosine"}
        )
        self.assertEqual(cfg_c.retrieval_k, 6)

        cfg_b = SystemConfig.from_preset("norm_cosine")
        emb_b = ChatGeoMagi.create_embeddings(
            cfg_b, local_files_only=True, embedding_cls=FakeEmbeddingFunction
        )
        self.assertEqual(emb_b.encode_kwargs, {"normalize_embeddings": True})
        self.assertEqual(emb_b.query_encode_kwargs, {"normalize_embeddings": True})

        cfg_a = SystemConfig.from_preset("legacy_l2")
        emb_a = ChatGeoMagi.create_embeddings(
            cfg_a, local_files_only=True, embedding_cls=FakeEmbeddingFunction
        )
        self.assertEqual(emb_a.encode_kwargs, {"normalize_embeddings": False})
        self.assertEqual(emb_a.query_encode_kwargs, {"normalize_embeddings": False})
        self.assertEqual(ChatGeoMagi.collection_metadata(cfg_a), {"hnsw:space": "l2"})

    def test_public_only_rejects_support_emails(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            data_dir.mkdir()
            email_file = data_dir / "support_emails_2024.txt"
            email_file.write_text("Private support email content")
            cfg = SystemConfig(data_dir=str(data_dir), public_only=True)
            with self.assertRaises(ValueError):
                ChatGeoMagi.discover_corpus_files(cfg)

            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "public_only": True,
                        "files": [
                            {
                                "path": str(email_file),
                                "source_url": "https://example.gov/doc.txt",
                                "sha256": "0" * 64,
                            }
                        ],
                    }
                )
            )
            with self.assertRaises(ValueError):
                validate_manifest(manifest_path)

    def test_index_validation_and_rebuild_on_config_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "data"
            data_dir.mkdir()
            doc_file = data_dir / "wmm_public.txt"
            doc_file.write_text("The World Magnetic Model is produced by NOAA and BGS.")
            fake_doc = type(
                "Doc",
                (),
                {
                    "page_content": doc_file.read_text(),
                    "metadata": {
                        "source": str(doc_file.resolve()),
                        "start_index": 0,
                    },
                },
            )()
            hashes = ChatGeoMagi.compute_document_hashes([doc_file], public_only=True)

            index_dir = tmp_path / "index_v1"
            cfg_a = SystemConfig.from_preset(
                "legacy_l2",
                data_dir=str(data_dir),
                chroma_db_path=str(index_dir),
                embedding_model=f"/cache/models--NovaSearch--stella_en_1.5B_v5/snapshots/{STELLA_DEFAULT_REVISION}",
            )
            emb_a = ChatGeoMagi.create_embeddings(
                cfg_a, local_files_only=True, embedding_cls=FakeEmbeddingFunction
            )
            ChatGeoMagi.build_index(
                cfg_a,
                emb_a,
                chroma_cls=FakeChroma,
                preloaded_splits=([fake_doc], hashes),
            )
            meta_a = json.loads(ChatGeoMagi.index_metadata_path(index_dir).read_text())
            self.assertEqual(meta_a["embedding_model"], STELLA_MODEL_ID)
            self.assertEqual(meta_a["embedding_revision"], STELLA_DEFAULT_REVISION)
            self.assertFalse(meta_a["normalize_embeddings"])
            self.assertEqual(meta_a["distance_metric"], "l2")
            self.assertEqual(meta_a["document_hashes"], hashes)

            # Attempting to validate or strictly open legacy_l2 index with stella_s2p_cosine must fail
            cfg_c = SystemConfig.from_preset(
                "stella_s2p_cosine",
                data_dir=str(data_dir),
                chroma_db_path=str(index_dir),
                retrieval_k=5,
            )
            emb_c = ChatGeoMagi.create_embeddings(
                cfg_c, local_files_only=True, embedding_cls=FakeEmbeddingFunction
            )
            with self.assertRaises(ValueError):
                ChatGeoMagi.validate_existing_index(
                    cfg_c,
                    embeddings=emb_c,
                    expected_hashes=hashes,
                    chroma_cls=FakeChroma,
                )

            # Dimension mismatch must also fail
            emb_bad_dim = FakeEmbeddingFunction("test", dim=512)
            with self.assertRaises(ValueError):
                ChatGeoMagi.validate_existing_index(
                    cfg_a,
                    embeddings=emb_bad_dim,
                    expected_hashes=hashes,
                    chroma_cls=FakeChroma,
                )

            # setup_vector_store with changed embedding config must NOT reuse index_dir or destroy it;
            # it must create a versioned index directory and configure retrieval_k=5
            fake_tools_mod = types.ModuleType("langchain_core.tools")
            fake_tools_mod.create_retriever_tool = lambda retriever, name, desc: {
                "retriever": retriever,
                "name": name,
                "description": desc,
            }
            with patch.dict(
                sys.modules, {"langchain_core.tools": fake_tools_mod}
            ), patch.object(
                ChatGeoMagi,
                "load_and_split_documents",
                return_value=([fake_doc], hashes),
            ):
                tool_obj = ChatGeoMagi.setup_vector_store(
                    cfg_c, emb_c, strict=False, chroma_cls=FakeChroma
                )
            self.assertNotEqual(Path(cfg_c.chroma_db_path), index_dir)
            self.assertTrue(index_dir.exists())
            self.assertTrue(Path(cfg_c.chroma_db_path).exists())
            self.assertEqual(tool_obj["retriever"]["search_kwargs"], {"k": 5})

            # open_retriever must reject preset mismatch against manifest
            emb_dir = tmp_path / "stella_cache"
            emb_dir.mkdir()
            manifest_c = {
                "public_only": True,
                "files": [
                    {
                        "path": str(doc_file.resolve()),
                        "source_url": "https://www.ncei.noaa.gov/products/world-magnetic-model",
                        "sha256": hashes[str(doc_file.resolve())],
                    }
                ],
                "index_path": str(Path(cfg_c.chroma_db_path).resolve()),
                **json.loads(
                    ChatGeoMagi.index_metadata_path(cfg_c.chroma_db_path).read_text()
                ),
            }
            manifest_path = Path(cfg_c.chroma_db_path) / "indexed_manifest.json"
            manifest_path.write_text(json.dumps(manifest_c, indent=2))
            with self.assertRaises(ValueError):
                open_retriever(
                    cfg_c.chroma_db_path,
                    emb_dir,
                    manifest_path,
                    preset="legacy_l2",
                    embedding_cls=FakeEmbeddingFunction,
                    chroma_cls=FakeChroma,
                )
            retriever_ok = open_retriever(
                cfg_c.chroma_db_path,
                emb_dir,
                manifest_path,
                preset="stella_s2p_cosine",
                k=4,
                embedding_cls=FakeEmbeddingFunction,
                chroma_cls=FakeChroma,
            )
            self.assertEqual(retriever_ok["search_kwargs"], {"k": 4})

    def test_openai_judge_confidentiality_and_no_rag_reuse_validation(self):
        with self.assertRaises(PermissionError):
            assert_public_only_input(
                ROOT
                / "evaluation/legacy_email_experiment/opik_evaluation_results/gpt_emails.json",
                [],
            )
        with self.assertRaises(PermissionError):
            assert_public_only_input(
                ROOT / "evaluation/results/test_inputs.json",
                [{"id": "Q01", "condition": "emails"}],
            )
        assert_public_only_input(
            ROOT / "evaluation/results/test_inputs.json",
            [{"id": "Q01", "condition": "rag"}],
        )

        with tempfile.TemporaryDirectory() as tmp:
            reuse_dir = Path(tmp)
            hashes = {
                "agentic/prompts.py": "h1",
                "agentic/cgm.py": "h2",
                "agentic/config.py": "h3",
                "agentic/apis/noaa_apis.py": "h4",
            }
            (reuse_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "arguments": {
                            "model": "gpt-oss:20b",
                            "model_digest": "d1",
                            "temperature": 0.0,
                            "seed": 20261115,
                            "num_ctx": 8192,
                            "num_predict": 2048,
                        },
                        "calculator_tools_available_in_both_conditions": True,
                        "source_hashes": hashes,
                    }
                )
            )
            args_ok = type(
                "Args",
                (),
                {
                    "model": "gpt-oss:20b",
                    "model_digest": "d1",
                    "temperature": 0.0,
                    "seed": 20261115,
                    "num_ctx": 8192,
                    "num_predict": 2048,
                },
            )()
            verify_reusable_no_rag(reuse_dir, args_ok, hashes)
            args_bad = type(
                "Args",
                (),
                {
                    "model": "gpt-oss:20b",
                    "model_digest": "d2",
                    "temperature": 0.0,
                    "seed": 20261115,
                    "num_ctx": 8192,
                    "num_predict": 2048,
                },
            )()
            with self.assertRaises(ValueError):
                verify_reusable_no_rag(reuse_dir, args_bad, hashes)

    def test_published_results_match_manuscript_statistics(self):
        results_dir = ROOT / "evaluation/results"

        # 1. Retrieval summary
        with (results_dir / "retrieval_summary.csv").open() as handle:
            ret_rows = {r["run_id"]: r for r in csv.DictReader(handle)}
        full_ret = ret_rows["expanded_stella_s2p_cosine_full25"]
        self.assertEqual(int(full_ret["hits_at_4"]), 22)
        self.assertEqual(int(full_ret["scored_questions"]), 25)

        heldout_ret = ret_rows["expanded_stella_s2p_cosine_heldout14"]
        self.assertEqual(int(heldout_ret["hits_at_4"]), 13)
        self.assertEqual(int(heldout_ret["scored_questions"]), 14)

        # 2. RAG vs. no-RAG statistical summary and per-question scores
        rag_stats = json.loads(
            (results_dir / "rag_statistical_summary.json").read_text()
        )["primary_openai_judge"]["all_25_questions"]
        self.assertAlmostEqual(rag_stats["no_rag"]["mean"], 0.668, places=6)
        self.assertAlmostEqual(
            rag_stats["rag_15doc_corrected"]["mean"], 0.784, places=6
        )
        comp = rag_stats["paired_15doc_minus_no_rag"]
        self.assertAlmostEqual(comp["mean_diff"], 0.116, places=6)
        self.assertEqual([comp["ci95_low"], comp["ci95_high"]], [0.024, 0.224])

        # Re-verify paired bootstrap and cross-document Hit@4 directly from rag_question_scores.csv
        with (results_dir / "rag_question_scores.csv").open() as handle:
            q_rows = list(csv.DictReader(handle))
        self.assertEqual(len(q_rows), 25)
        crossdoc_hits = sum(
            int(r["expanded_15doc_stella_s2p_crossdoc_hit4"]) for r in q_rows
        )
        self.assertEqual(crossdoc_hits, 24)
        diffs = [
            float(r["rag_15doc_score_gpt4omini"]) - float(r["no_rag_score_gpt4omini"])
            for r in q_rows
        ]
        self.assertAlmostEqual(sum(diffs) / len(diffs), 0.116, places=6)
        ci = [round(v, 3) for v in bootstrap(diffs, seed=20261115, resamples=100000)]
        self.assertEqual(ci, [0.024, 0.224])

        # 3. Tool benchmark summary
        tool_stats = json.loads((results_dir / "tool_summary.json").read_text())
        self.assertEqual(tool_stats["tool_selection"]["passed"], 16)
        self.assertEqual(tool_stats["tool_selection"]["n"], 16)
        self.assertEqual(tool_stats["tool_numerical_agreement"]["passed"], 16)
        self.assertEqual(tool_stats["parameters_and_order"]["passed"], 12)
        self.assertEqual(tool_stats["execution"]["passed"], 16)
        with (results_dir / "tool_per_question.csv").open() as handle:
            tool_rows = list(csv.DictReader(handle))
        self.assertEqual(sum(row["execution"] == "True" for row in tool_rows), 16)
        self.assertEqual(sum(row["parameters_and_order"] == "True" for row in tool_rows), 12)
        self.assertEqual(tool_stats["invalid_request_handling"]["passed"], 3)
        self.assertEqual(tool_stats["invalid_request_handling"]["n"], 4)

    def test_plots_generate_offline_without_api_keys(self):
        import os

        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True), \
             patch("socket.socket.connect", side_effect=AssertionError("Plotting must stay offline")):
            out_dir = Path(tmp) / "plots"
            written = generate_plots(
                figure="all",
                results_dir=ROOT / "evaluation/results",
                output_dir=out_dir,
            )
            self.assertEqual(len(written), 10)
            for path in written:
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
