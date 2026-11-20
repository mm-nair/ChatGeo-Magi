import csv
import json
from pathlib import Path
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
from public_retrieval import chunk_identifier, open_retriever, validate_manifest
from run_revision_inference import verify_reusable_no_rag
from score_live_tools import numbers


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
        self.assertEqual(tool_stats["invalid_request_handling"]["passed"], 3)
        self.assertEqual(tool_stats["invalid_request_handling"]["n"], 4)

    def test_plots_generate_offline_without_api_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
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
