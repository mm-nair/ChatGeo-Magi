"""Generate publication tables, reproducibility metadata, and vector PDF/PNG figures
for `revision_workspace/final_retrieval_revision/`.
"""

import csv
import json
from pathlib import Path
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

from golden_dataset import load_goldens
from revision_analysis import bootstrap, save_plot, write_csv


RETRIEVAL_RUN_SPECS = [
    (
        "two_doc_legacy_l2_orig22",
        "two_doc_283_chunks",
        "orig_22_verified",
        "A_legacy_l2",
        "Config A: Original unnormalized L2 (causal fallback)",
    ),
    (
        "two_doc_causal_norm_cosine_orig22",
        "two_doc_283_chunks",
        "orig_22_verified",
        "A2_causal_norm_cosine",
        "Diagnostic: Causal fallback + normalized cosine",
    ),
    (
        "two_doc_causal_s2p_cosine_orig22",
        "two_doc_283_chunks",
        "orig_22_verified",
        "A3_causal_s2p_cosine",
        "Diagnostic: Causal fallback + normalized cosine + s2p_query",
    ),
    (
        "two_doc_norm_cosine_orig22",
        "two_doc_283_chunks",
        "orig_22_verified",
        "B_norm_cosine",
        "Config B: Bidirectional Stella + normalized cosine",
    ),
    (
        "two_doc_stella_s2p_cosine_orig22",
        "two_doc_283_chunks",
        "orig_22_verified",
        "C_stella_s2p_cosine",
        "Config C: Bidirectional Stella + normalized cosine + s2p_query",
    ),
    (
        "two_doc_legacy_l2_multi22",
        "two_doc_283_chunks",
        "multichunk_22_verified",
        "A_legacy_l2",
        "Config A: Original unnormalized L2 (multi-chunk GT)",
    ),
    (
        "two_doc_norm_cosine_multi22",
        "two_doc_283_chunks",
        "multichunk_22_verified",
        "B_norm_cosine",
        "Config B: Bidirectional Stella + normalized cosine (multi-chunk GT)",
    ),
    (
        "two_doc_stella_s2p_cosine_multi22",
        "two_doc_283_chunks",
        "multichunk_22_verified",
        "C_stella_s2p_cosine",
        "Config C: Bidirectional Stella + normalized cosine + s2p_query (multi-chunk GT)",
    ),
    (
        "two_doc_legacy_l2_heldout14",
        "two_doc_283_chunks",
        "heldout_14_verified",
        "A_legacy_l2",
        "Config A: Original unnormalized L2 (Held-out 14)",
    ),
    (
        "two_doc_causal_norm_cosine_heldout14",
        "two_doc_283_chunks",
        "heldout_14_verified",
        "A2_causal_norm_cosine",
        "Diagnostic: Causal fallback + normalized cosine (Held-out 14)",
    ),
    (
        "two_doc_causal_s2p_cosine_heldout14",
        "two_doc_283_chunks",
        "heldout_14_verified",
        "A3_causal_s2p_cosine",
        "Diagnostic: Causal fallback + s2p_query (Held-out 14)",
    ),
    (
        "two_doc_norm_cosine_heldout14",
        "two_doc_283_chunks",
        "heldout_14_verified",
        "B_norm_cosine",
        "Config B: Bidirectional Stella + normalized cosine (Held-out 14)",
    ),
    (
        "two_doc_stella_s2p_cosine_heldout14",
        "two_doc_283_chunks",
        "heldout_14_verified",
        "C_stella_s2p_cosine",
        "Config C: Bidirectional Stella + normalized cosine + s2p_query (Held-out 14)",
    ),
    (
        "expanded_legacy_l2_full25",
        "expanded_15_doc_8863_chunks",
        "expanded_25_verified",
        "A_legacy_l2",
        "Config A: Original unnormalized L2 (Expanded 15-doc, 25 Qs)",
    ),
    (
        "expanded_norm_cosine_full25",
        "expanded_15_doc_8863_chunks",
        "expanded_25_verified",
        "B_norm_cosine",
        "Config B: Bidirectional Stella + normalized cosine (Expanded 15-doc, 25 Qs)",
    ),
    (
        "expanded_stella_s2p_cosine_full25",
        "expanded_15_doc_8863_chunks",
        "expanded_25_verified",
        "C_stella_s2p_cosine",
        "Config C: Bidirectional Stella + normalized cosine + s2p_query (Expanded 15-doc, 25 Qs)",
    ),
    (
        "expanded_legacy_l2_heldout14",
        "expanded_15_doc_8863_chunks",
        "expanded_heldout_14",
        "A_legacy_l2",
        "Config A: Original unnormalized L2 (Expanded 15-doc, Held-out 14)",
    ),
    (
        "expanded_norm_cosine_heldout14",
        "expanded_15_doc_8863_chunks",
        "expanded_heldout_14",
        "B_norm_cosine",
        "Config B: Bidirectional Stella + normalized cosine (Expanded 15-doc, Held-out 14)",
    ),
    (
        "expanded_stella_s2p_cosine_heldout14",
        "expanded_15_doc_8863_chunks",
        "expanded_heldout_14",
        "C_stella_s2p_cosine",
        "Config C: Bidirectional Stella + normalized cosine + s2p_query (Expanded 15-doc, Held-out 14)",
    ),
]


def load_metrics_by_id(folder: Path):
    metrics = json.loads((folder / "metrics.json").read_text())
    retrieved = json.loads((folder / "retrieved.json").read_text())
    by_id = {}
    for row in metrics:
        qid = row["id"]
        chunks = retrieved.get(qid, [])
        by_id[qid] = {
            **row,
            "retrieved_chunk_ids": [c["chunk_id"] for c in chunks],
            "retrieved_sources": [
                f"{Path(c['metadata'].get('source', '')).name}:p{int(c['metadata'].get('page', 0)) + 1}"
                for c in chunks
            ],
        }
    return by_id


def build_retrieval_tables(rev_dir: Path):
    summary_rows = []
    for folder_name, corpus, benchmark_set, config_code, description in RETRIEVAL_RUN_SPECS:
        summary_path = rev_dir / folder_name / "summary.json"
        if not summary_path.is_file():
            continue
        data = json.loads(summary_path.read_text())
        hub = data["hubness"]
        summary_rows.append(
            {
                "run_id": folder_name,
                "corpus": corpus,
                "benchmark": benchmark_set,
                "config_code": config_code,
                "description": description,
                "scored_questions": data["hit_at_4"]["n"],
                "hits_at_4": data["hit_at_4"]["hits"],
                "hit_at_4": round(data["hit_at_4"]["mean"], 4),
                "recall_at_4": round(data["recall_at_4"]["mean"], 4),
                "precision_at_4": round(data["precision_at_4"]["mean"], 4),
                "support_coverage_at_4": round(data["support_coverage"]["mean"], 4),
                "total_slots_scored": hub["total_slots_scored"],
                "unique_chunks_scored": hub["unique_chunks_scored"],
                "unique_chunks_all": hub["unique_chunks_all"],
                "max_chunk_frequency_scored": hub["max_chunk_frequency_scored"],
                "repeated_unrelated_chunk_count": hub["repeated_unrelated_chunk_count"],
                "repeated_unrelated_slot_recoveries": hub[
                    "repeated_unrelated_slot_Recoveries"
                ],
            }
        )
    write_csv(rev_dir / "retrieval_summary.csv", summary_rows)

    # Per-question before/after table across the 25 primary questions + 14 held-out questions
    orig_bench = {
        q["id"]: q
        for q in json.loads(
            (ROOT / "revision_workspace/retrieval_evaluation/annotated_benchmark.json").read_text()
        )
    }
    heldout_bench = {
        q["id"]: q
        for q in json.loads((rev_dir / "annotated_heldout_benchmark.json").read_text())
    }
    exp_bench = {
        q["id"]: q
        for q in json.loads((rev_dir / "annotated_expanded_benchmark.json").read_text())
    }

    m_a_orig = load_metrics_by_id(rev_dir / "two_doc_legacy_l2_orig22")
    m_b_orig = load_metrics_by_id(rev_dir / "two_doc_norm_cosine_orig22")
    m_c_orig = load_metrics_by_id(rev_dir / "two_doc_stella_s2p_cosine_orig22")
    m_c_multi = load_metrics_by_id(rev_dir / "two_doc_stella_s2p_cosine_multi22")
    m_a_exp = load_metrics_by_id(rev_dir / "expanded_legacy_l2_full25")
    m_b_exp = load_metrics_by_id(rev_dir / "expanded_norm_cosine_full25")
    m_c_exp = load_metrics_by_id(rev_dir / "expanded_stella_s2p_cosine_full25")

    m_a_ho = load_metrics_by_id(rev_dir / "two_doc_legacy_l2_heldout14")
    m_b_ho = load_metrics_by_id(rev_dir / "two_doc_norm_cosine_heldout14")
    m_c_ho = load_metrics_by_id(rev_dir / "two_doc_stella_s2p_cosine_heldout14")
    m_a_ho_exp = load_metrics_by_id(rev_dir / "expanded_legacy_l2_heldout14")
    m_b_ho_exp = load_metrics_by_id(rev_dir / "expanded_norm_cosine_heldout14")
    m_c_ho_exp = load_metrics_by_id(rev_dir / "expanded_stella_s2p_cosine_heldout14")

    before_after_rows = []
    for qid in sorted(orig_bench.keys()):
        q = orig_bench[qid]
        q_exp = exp_bench[qid]
        ra = m_a_orig[qid]
        rb = m_b_orig[qid]
        rc = m_c_orig[qid]
        rcm = m_c_multi[qid]
        rea = m_a_exp[qid]
        reb = m_b_exp[qid]
        rec = m_c_exp[qid]
        before_after_rows.append(
            {
                "id": qid,
                "split": "primary_25",
                "in_two_doc_corpus": q["verification_status"] == "source_verified",
                "source_document": q_exp.get("source_document") or q.get("source_document") or "",
                "question": q["question"],
                "two_doc_A_legacy_l2_hit4": ra.get("hit_at_4", ""),
                "two_doc_B_norm_cosine_hit4": rb.get("hit_at_4", ""),
                "two_doc_C_stella_s2p_hit4": rc.get("hit_at_4", ""),
                "two_doc_C_stella_s2p_recall4": rc.get("recall_at_4", ""),
                "two_doc_C_stella_s2p_precision4": rc.get("precision_at_4", ""),
                "two_doc_C_multichunk_hit4": rcm.get("hit_at_4", ""),
                "expanded_A_legacy_l2_hit4": rea.get("hit_at_4", ""),
                "expanded_B_norm_cosine_hit4": reb.get("hit_at_4", ""),
                "expanded_C_stella_s2p_hit4": rec.get("hit_at_4", ""),
                "expanded_C_stella_s2p_recall4": rec.get("recall_at_4", ""),
                "two_doc_A_top_sources": "; ".join(ra["retrieved_sources"]),
                "two_doc_C_top_sources": "; ".join(rc["retrieved_sources"]),
                "expanded_C_top_sources": "; ".join(rec["retrieved_sources"]),
            }
        )

    for qid in sorted(heldout_bench.keys()):
        q = heldout_bench[qid]
        ra = m_a_ho[qid]
        rb = m_b_ho[qid]
        rc = m_c_ho[qid]
        rea = m_a_ho_exp[qid]
        reb = m_b_ho_exp[qid]
        rec = m_c_ho_exp[qid]
        before_after_rows.append(
            {
                "id": qid,
                "split": "heldout_14",
                "in_two_doc_corpus": True,
                "source_document": q["source_document"],
                "question": q["question"],
                "two_doc_A_legacy_l2_hit4": ra.get("hit_at_4", ""),
                "two_doc_B_norm_cosine_hit4": rb.get("hit_at_4", ""),
                "two_doc_C_stella_s2p_hit4": rc.get("hit_at_4", ""),
                "two_doc_C_stella_s2p_recall4": rc.get("recall_at_4", ""),
                "two_doc_C_stella_s2p_precision4": rc.get("precision_at_4", ""),
                "two_doc_C_multichunk_hit4": rc.get("hit_at_4", ""),
                "expanded_A_legacy_l2_hit4": rea.get("hit_at_4", ""),
                "expanded_B_norm_cosine_hit4": reb.get("hit_at_4", ""),
                "expanded_C_stella_s2p_hit4": rec.get("hit_at_4", ""),
                "expanded_C_stella_s2p_recall4": rec.get("recall_at_4", ""),
                "two_doc_A_top_sources": "; ".join(ra["retrieved_sources"]),
                "two_doc_C_top_sources": "; ".join(rc["retrieved_sources"]),
                "expanded_C_top_sources": "; ".join(rec["retrieved_sources"]),
            }
        )

    write_csv(rev_dir / "retrieval_before_after.csv", before_after_rows)
    return summary_rows, before_after_rows


def mean_ci(values, seed=20261115):
    arr = np.asarray(values, dtype=float)
    low, high = bootstrap(arr, seed=seed)
    return float(arr.mean()), float(low), float(high)


def build_inference_tables(rev_dir: Path):
    old_ablation_dir = ROOT / "revision_workspace/rag_ablation"
    old_traces_dir = old_ablation_dir / "results/vm_run_01"
    old_judged_dir = old_ablation_dir / "local_judged"

    new_traces_dir = rev_dir / "ablation_results/vm_run_corrected"
    new_judged_dir = rev_dir / "ablation_results/local_judged_corrected"

    if not new_judged_dir.is_dir():
        return None, None

    goldens = load_goldens(ROOT / "evaluation/goldens.csv")
    questions = [
        {"id": f"Q{i:02d}", "question": q, "expected": a}
        for i, (q, a) in enumerate(goldens.items(), 1)
    ]
    m_c_orig = load_metrics_by_id(rev_dir / "two_doc_stella_s2p_cosine_orig22")
    m_a_orig = load_metrics_by_id(rev_dir / "two_doc_legacy_l2_orig22")

    # Optional expanded RAG supplemental run for Q22-Q24
    exp_q22_q24_judged_dir = rev_dir / "ablation_results/local_judged_expanded_q22_q24"

    rows = []
    for item in questions:
        qid = item["id"]
        old_rag_j = json.loads((old_judged_dir / f"ablation_rag_{qid}.json").read_text())
        old_norag_j = json.loads((old_judged_dir / f"ablation_no_rag_{qid}.json").read_text())
        new_rag_j = json.loads((new_judged_dir / f"ablation_rag_{qid}.json").read_text())
        new_norag_j = json.loads((new_judged_dir / f"ablation_no_rag_{qid}.json").read_text())

        old_rag_t = json.loads((old_traces_dir / f"{qid}_rag.json").read_text())
        old_norag_t = json.loads((old_traces_dir / f"{qid}_no_rag.json").read_text())
        new_rag_t = json.loads((new_traces_dir / f"{qid}_rag.json").read_text())
        new_norag_t = json.loads((new_traces_dir / f"{qid}_no_rag.json").read_text())

        old_rag_calls = [c for c in old_rag_t.get("calls", []) if c["name"] == "retrieve_geomag_docs"]
        new_rag_calls = [c for c in new_rag_t.get("calls", []) if c["name"] == "retrieve_geomag_docs"]

        in_two_doc = qid not in ("Q22", "Q23", "Q24")
        exp_rag_score = ""
        if exp_q22_q24_judged_dir.is_dir():
            p_exp = exp_q22_q24_judged_dir / f"ablation_rag_{qid}.json"
            if p_exp.is_file():
                exp_rag_score = json.loads(p_exp.read_text())["score"]

        rows.append(
            {
                "id": qid,
                "in_two_doc_corpus": in_two_doc,
                "corpus_status": "in_two_doc_corpus" if in_two_doc else "missing_from_two_doc_in_expanded_15_doc",
                "question": item["question"],
                "broken_rag_hit_at_4": m_a_orig[qid].get("hit_at_4", "") if in_two_doc else "",
                "corrected_rag_hit_at_4": m_c_orig[qid].get("hit_at_4", "") if in_two_doc else "",
                "broken_rag_score": old_rag_j["score"],
                "old_no_rag_score": old_norag_j["score"],
                "corrected_rag_score": new_rag_j["score"],
                "new_no_rag_score": new_norag_j["score"],
                "diff_corrected_rag_minus_no_rag": round(new_rag_j["score"] - new_norag_j["score"], 4),
                "diff_corrected_rag_minus_broken_rag": round(new_rag_j["score"] - old_rag_j["score"], 4),
                "broken_rag_retrieval_calls": len(old_rag_calls),
                "corrected_rag_retrieval_calls": len(new_rag_calls),
                "new_no_rag_tool_calls": len(new_norag_t.get("calls", [])),
                "broken_rag_empty": old_rag_j["empty_response"],
                "corrected_rag_empty": new_rag_j["empty_response"],
                "new_no_rag_empty": new_norag_j["empty_response"],
                "expanded_corpus_rag_score": exp_rag_score,
            }
        )

    write_csv(rev_dir / "inference_comparison.csv", rows)

    def summarize_subset(subset_rows, label):
        broken = [r["broken_rag_score"] for r in subset_rows]
        old_nr = [r["old_no_rag_score"] for r in subset_rows]
        corr = [r["corrected_rag_score"] for r in subset_rows]
        new_nr = [r["new_no_rag_score"] for r in subset_rows]

        diff_corr_nr = [c - n for c, n in zip(corr, new_nr)]
        diff_corr_brk = [c - b for c, b in zip(corr, broken)]
        diff_brk_nr = [b - n for b, n in zip(broken, old_nr)]

        m_brk, l_brk, h_brk = mean_ci(broken)
        m_onr, l_onr, h_onr = mean_ci(old_nr)
        m_cor, l_cor, h_cor = mean_ci(corr)
        m_nnr, l_nnr, h_nnr = mean_ci(new_nr)

        d_cn, dl_cn, dh_cn = mean_ci(diff_corr_nr)
        d_cb, dl_cb, dh_cb = mean_ci(diff_corr_brk)
        d_bn, dl_bn, dh_bn = mean_ci(diff_brk_nr)

        nonempty_cn = [
            r["corrected_rag_score"] - r["new_no_rag_score"]
            for r in subset_rows
            if not r["corrected_rag_empty"] and not r["new_no_rag_empty"]
        ]

        return {
            "subset": label,
            "n": len(subset_rows),
            "broken_rag": {"mean": round(m_brk, 4), "ci95": [round(l_brk, 4), round(h_brk, 4)]},
            "old_no_rag": {"mean": round(m_onr, 4), "ci95": [round(l_onr, 4), round(h_onr, 4)]},
            "corrected_rag": {"mean": round(m_cor, 4), "ci95": [round(l_cor, 4), round(h_cor, 4)]},
            "new_no_rag": {"mean": round(m_nnr, 4), "ci95": [round(l_nnr, 4), round(h_nnr, 4)]},
            "paired_corrected_minus_no_rag": {
                "mean": round(d_cn, 4),
                "ci95": [round(dl_cn, 4), round(dh_cn, 4)],
                "improved_count": sum(1 for x in diff_corr_nr if x > 1e-9),
                "regressed_count": sum(1 for x in diff_corr_nr if x < -1e-9),
                "tied_count": sum(1 for x in diff_corr_nr if abs(x) <= 1e-9),
                "nonempty_n": len(nonempty_cn),
                "nonempty_mean": round(float(np.mean(nonempty_cn)), 4) if nonempty_cn else None,
            },
            "paired_corrected_minus_broken_rag": {
                "mean": round(d_cb, 4),
                "ci95": [round(dl_cb, 4), round(dh_cb, 4)],
                "improved_count": sum(1 for x in diff_corr_brk if x > 1e-9),
                "regressed_count": sum(1 for x in diff_corr_brk if x < -1e-9),
                "tied_count": sum(1 for x in diff_corr_brk if abs(x) <= 1e-9),
            },
            "paired_broken_minus_old_no_rag": {
                "mean": round(d_bn, 4),
                "ci95": [round(dl_bn, 4), round(dh_bn, 4)],
            },
            "empty_responses": {
                "broken_rag": sum(1 for r in subset_rows if r["broken_rag_empty"]),
                "corrected_rag": sum(1 for r in subset_rows if r["corrected_rag_empty"]),
                "new_no_rag": sum(1 for r in subset_rows if r["new_no_rag_empty"]),
            },
            "retrieval_tool_usage": {
                "broken_rag_questions_with_calls": sum(
                    1 for r in subset_rows if r["broken_rag_retrieval_calls"] > 0
                ),
                "broken_rag_total_calls": sum(
                    r["broken_rag_retrieval_calls"] for r in subset_rows
                ),
                "corrected_rag_questions_with_calls": sum(
                    1 for r in subset_rows if r["corrected_rag_retrieval_calls"] > 0
                ),
                "corrected_rag_total_calls": sum(
                    r["corrected_rag_retrieval_calls"] for r in subset_rows
                ),
            },
        }

    all_25 = summarize_subset(rows, "all_25_questions")
    in_corpus_22 = summarize_subset(
        [r for r in rows if r["in_two_doc_corpus"]], "in_two_doc_corpus_22_questions"
    )
    out_of_corpus_3 = [r for r in rows if not r["in_two_doc_corpus"]]

    summary = {
        "all_25_questions": all_25,
        "in_two_doc_corpus_22_questions": in_corpus_22,
        "out_of_two_doc_corpus_3_questions": {
            "n": len(out_of_corpus_3),
            "question_ids": [r["id"] for r in out_of_corpus_3],
            "broken_rag_mean": round(
                float(np.mean([r["broken_rag_score"] for r in out_of_corpus_3])), 4
            ),
            "corrected_two_doc_rag_mean": round(
                float(np.mean([r["corrected_rag_score"] for r in out_of_corpus_3])), 4
            ),
            "new_no_rag_mean": round(
                float(np.mean([r["new_no_rag_score"] for r in out_of_corpus_3])), 4
            ),
            "expanded_15_doc_rag_mean": (
                round(
                    float(
                        np.mean(
                            [
                                r["expanded_corpus_rag_score"]
                                for r in out_of_corpus_3
                                if isinstance(r["expanded_corpus_rag_score"], (int, float))
                            ]
                        )
                    ),
                    4,
                )
                if any(
                    isinstance(r["expanded_corpus_rag_score"], (int, float))
                    for r in out_of_corpus_3
                )
                else None
            ),
        },
    }
    (rev_dir / "inference_summary.json").write_text(json.dumps(summary, indent=2))
    return rows, summary


def generate_figures(rev_dir: Path, retrieval_summary_rows, inference_rows, inference_summary):
    fig_dir = rev_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    by_run = {r["run_id"]: r for r in retrieval_summary_rows}

    # --- Figure 1: Retrieval Performance Before vs. After ---
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), layout="constrained")

    # Panel A: 2-document benchmark across Configs A, B, C (Dev 22 vs Held-out 14) + Expanded 15-doc (25 Qs & Held-out 14)
    categories = [
        "2-Doc Corpus\nPrimary (n=22)",
        "2-Doc Corpus\nHeld-Out (n=14)",
        "Expanded 15-Doc\nAll Qs (n=25)",
        "Expanded 15-Doc\nHeld-Out (n=14)",
    ]
    a_vals = [
        by_run["two_doc_legacy_l2_orig22"]["hit_at_4"] * 100,
        by_run["two_doc_legacy_l2_heldout14"]["hit_at_4"] * 100,
        by_run["expanded_legacy_l2_full25"]["hit_at_4"] * 100,
        by_run["expanded_legacy_l2_heldout14"]["hit_at_4"] * 100,
    ]
    b_vals = [
        by_run["two_doc_norm_cosine_orig22"]["hit_at_4"] * 100,
        by_run["two_doc_norm_cosine_heldout14"]["hit_at_4"] * 100,
        by_run["expanded_norm_cosine_full25"]["hit_at_4"] * 100,
        by_run["expanded_norm_cosine_heldout14"]["hit_at_4"] * 100,
    ]
    c_vals = [
        by_run["two_doc_stella_s2p_cosine_orig22"]["hit_at_4"] * 100,
        by_run["two_doc_stella_s2p_cosine_heldout14"]["hit_at_4"] * 100,
        by_run["expanded_stella_s2p_cosine_full25"]["hit_at_4"] * 100,
        by_run["expanded_stella_s2p_cosine_heldout14"]["hit_at_4"] * 100,
    ]

    x = np.arange(len(categories))
    w = 0.25
    b1 = axes[0].bar(x - w, a_vals, width=w, label="Config A: Legacy Unnorm. L2", color="#CC79A7")
    b2 = axes[0].bar(x, b_vals, width=w, label="Config B: Bidirectional + Norm. Cosine", color="#56B4E9")
    b3 = axes[0].bar(x + w, c_vals, width=w, label="Config C: Bidirectional + Norm. Cosine + s2p_query", color="#009E73")

    for bars in (b1, b2, b3):
        for bar in bars:
            val = bar.get_height()
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                val + 1.5,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    axes[0].set_ylim(0, 108)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories)
    axes[0].set_ylabel("Top-4 Retrieval Hit Rate (Hit@4, %)")
    axes[0].set_title("(a) Top-4 Retrieval Accuracy Across Corpora & Benchmarks")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    axes[0].legend(loc="upper left", fontsize=8.2, framealpha=0.92)

    # Panel B: Root-Cause Decomposition on 2-Doc Primary Benchmark (n=22)
    ablation_labels = [
        "Config A:\nLegacy L2\n(Causal)",
        "Causal +\nNorm. Cosine\n(No trust_remote)",
        "Causal +\nNorm. Cosine\n+ s2p_query",
        "Config B:\nBidirectional +\nNorm. Cosine",
        "Config C:\nBidirectional +\nCosine + s2p",
    ]
    ablation_runs = [
        "two_doc_legacy_l2_orig22",
        "two_doc_causal_norm_cosine_orig22",
        "two_doc_causal_s2p_cosine_orig22",
        "two_doc_norm_cosine_orig22",
        "two_doc_stella_s2p_cosine_orig22",
    ]
    hit_rates = [by_run[k]["hit_at_4"] * 100 for k in ablation_runs]
    uniq_counts = [by_run[k]["unique_chunks_scored"] for k in ablation_runs]

    x2 = np.arange(len(ablation_labels))
    colors2 = ["#CC79A7", "#E69F00", "#D55E00", "#56B4E9", "#009E73"]
    bars2 = axes[1].bar(x2, hit_rates, width=0.52, color=colors2)
    for bar, uniq in zip(bars2, uniq_counts):
        val = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            val + 1.5,
            f"{val:.1f}%\n({uniq} uniq)",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[1].set_ylim(0, 112)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(ablation_labels, fontsize=8.5)
    axes[1].set_ylabel("Hit@4 (%) [with unique top-4 chunks across 22 Qs]")
    axes[1].set_title("(b) Root-Cause Ablation on 22 Verified Questions")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    save_plot(fig, fig_dir / "fig1_retrieval_before_after")
    save_plot(fig, rev_dir / "fig1_retrieval_before_after")

    if inference_rows is None or inference_summary is None:
        return

    # --- Figure 2: Inference Correctness (Corrected RAG vs. Broken RAG vs. No-RAG) ---
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), layout="constrained")

    s25 = inference_summary["all_25_questions"]
    s22 = inference_summary["in_two_doc_corpus_22_questions"]

    groups = ["All 25 Benchmark Questions\n(n = 25)", "In-Corpus Questions\n(n = 22, in 2-Doc Index)"]
    xg = np.arange(len(groups))
    wg = 0.24

    cond_specs = [
        ("new_no_rag", "No RAG (Parametric Only)", "#0072B2", -wg),
        ("broken_rag", "Previous Broken RAG (Legacy L2)", "#CC79A7", 0.0),
        ("corrected_rag", "Corrected RAG (Stella s2p Cosine)", "#009E73", wg),
    ]
    for key, label, color, offset in cond_specs:
        means = [s25[key]["mean"], s22[key]["mean"]]
        lows = [s25[key]["mean"] - s25[key]["ci95"][0], s22[key]["mean"] - s22[key]["ci95"][0]]
        highs = [s25[key]["ci95"][1] - s25[key]["mean"], s22[key]["ci95"][1] - s22[key]["mean"]]
        bars = axes[0].bar(
            xg + offset,
            means,
            width=wg,
            yerr=[lows, highs],
            capsize=4,
            label=label,
            color=color,
            error_kw={"elinewidth": 1.2},
        )
        for bar, m in zip(bars, means):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                m + 0.015,
                f"{m:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )

    axes[0].set_ylim(0, 1.05)
    axes[0].set_xticks(xg)
    axes[0].set_xticklabels(groups)
    axes[0].set_ylabel("Mean Reference-Judge Correctness Score (0–1)")
    axes[0].set_title("(a) Mean Answer Correctness (with 95% Bootstrap CI)")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    axes[0].legend(loc="lower right", fontsize=8.5, framealpha=0.92)

    # Panel B: Paired differences with 95% bootstrap CI
    diff_items = [
        (
            "Broken RAG − No RAG\n(All 25 Qs, Previous)",
            s25["paired_broken_minus_old_no_rag"],
            "#CC79A7",
        ),
        (
            "Corrected RAG − Broken RAG\n(All 25 Qs)",
            s25["paired_corrected_minus_broken_rag"],
            "#56B4E9",
        ),
        (
            "Corrected RAG − No RAG\n(All 25 Qs)",
            s25["paired_corrected_minus_no_rag"],
            "#009E73",
        ),
        (
            "Corrected RAG − No RAG\n(In-Corpus 22 Qs)",
            s22["paired_corrected_minus_no_rag"],
            "#0072B2",
        ),
    ]
    yd = np.arange(len(diff_items))
    axes[1].axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    for idx, (label, dspec, color) in enumerate(diff_items):
        m = dspec["mean"]
        lo, hi = dspec["ci95"]
        axes[1].errorbar(
            m,
            idx,
            xerr=[[m - lo], [hi - m]],
            fmt="o",
            color=color,
            markersize=7,
            capsize=5,
            linewidth=2.0,
        )
        axes[1].text(
            m,
            idx + 0.18,
            f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
    axes[1].set_yticks(yd)
    axes[1].set_yticklabels([x[0] for x in diff_items])
    axes[1].set_xlim(-0.25, 0.45)
    axes[1].set_ylim(-0.5, len(diff_items) - 0.3)
    axes[1].set_xlabel("Paired Score Difference (95% Paired Bootstrap CI)")
    axes[1].set_title("(b) Paired Treatment Effects of Retrieval Repair")
    axes[1].grid(axis="x", linestyle="--", alpha=0.4)

    save_plot(fig, fig_dir / "fig2_inference_rag_vs_norag")
    save_plot(fig, rev_dir / "fig2_inference_rag_vs_norag")

    # --- Figure 3: Per-Question Diagnostic Comparison (Q01-Q25) ---
    fig, axes = plt.subplots(2, 1, figsize=(12.0, 6.2), sharex=True, layout="constrained")

    qids = [r["id"] for r in inference_rows]
    xq = np.arange(len(qids))
    no_rag_scores = [r["new_no_rag_score"] for r in inference_rows]
    broken_scores = [r["broken_rag_score"] for r in inference_rows]
    corr_scores = [r["corrected_rag_score"] for r in inference_rows]
    diffs = [r["diff_corrected_rag_minus_no_rag"] for r in inference_rows]

    axes[0].plot(
        xq,
        no_rag_scores,
        marker="s",
        linestyle=":",
        color="#0072B2",
        label="No RAG",
        alpha=0.85,
    )
    axes[0].plot(
        xq,
        broken_scores,
        marker="^",
        linestyle="--",
        color="#CC79A7",
        label="Broken RAG (Legacy L2)",
        alpha=0.85,
    )
    axes[0].plot(
        xq,
        corr_scores,
        marker="o",
        linestyle="-",
        color="#009E73",
        label="Corrected RAG (Stella s2p Cosine)",
        linewidth=1.8,
    )
    axes[0].axvspan(20.5, 23.5, color="#F0E442", alpha=0.22, label="Q22–Q24 (Missing from 2-Doc; in Expanded 15-Doc)")
    axes[0].set_ylim(-0.05, 1.08)
    axes[0].set_ylabel("Judge Score (0–1)")
    axes[0].set_title("(a) Per-Question Answer Correctness Across Conditions (Q01–Q25)")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(loc="lower left", ncol=2, fontsize=8.5, framealpha=0.92)

    bar_colors = [
        "#009E73" if d > 0 else ("#D55E00" if d < 0 else "#999999") for d in diffs
    ]
    axes[1].axhline(0.0, color="black", linewidth=0.9)
    axes[1].bar(xq, diffs, color=bar_colors, width=0.62)
    axes[1].axvspan(20.5, 23.5, color="#F0E442", alpha=0.22)
    for i, r in enumerate(inference_rows):
        hit_c = r["corrected_rag_hit_at_4"]
        marker_txt = "Hit" if hit_c == 1 else ("Miss" if hit_c == 0 else "OOC")
        y_pos = diffs[i] + (0.03 if diffs[i] >= 0 else -0.07)
        axes[1].text(
            i,
            y_pos,
            marker_txt,
            ha="center",
            va="bottom" if diffs[i] >= 0 else "top",
            fontsize=7.2,
            color="#222222",
        )

    axes[1].set_ylim(-0.65, 0.85)
    axes[1].set_xticks(xq)
    axes[1].set_xticklabels(qids, rotation=0, fontsize=8.5)
    axes[1].set_ylabel("Corrected RAG − No RAG")
    axes[1].set_xlabel("Benchmark Question ID (Hit = Supporting Chunk in Top-4; OOC = Out of 2-Doc Corpus)")
    axes[1].set_title("(b) Per-Question Paired Gain (Corrected RAG minus No RAG) & Retrieval Status")
    axes[1].grid(True, linestyle="--", alpha=0.35)

    save_plot(fig, fig_dir / "fig3_per_question_diagnostics")
    save_plot(fig, rev_dir / "fig3_per_question_diagnostics")


def write_experimental_config(rev_dir: Path, retrieval_summary_rows, inference_summary):
    from agentic.config import RETRIEVAL_PRESETS, STELLA_DEFAULT_REVISION, STELLA_MODEL_ID, STELLA_S2P_QUERY_PROMPT

    two_doc_manifest = json.loads(
        (rev_dir / "indexes/two_doc_stella_s2p_cosine/indexed_manifest.json").read_text()
    )
    expanded_manifest = json.loads(
        (rev_dir / "indexes/expanded_stella_s2p_cosine/indexed_manifest.json").read_text()
    )
    smoke_path = rev_dir / "smoke_tests/smoke_test_summary.json"
    smoke_summary = json.loads(smoke_path.read_text()) if smoke_path.is_file() else None

    config_payload = {
        "experiment_phase": "FINAL Retrieval Repair and Scientific Re-Evaluation",
        "git_branch": "revision/offline-scientific-audit",
        "git_commit_at_run": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "llm_model": {
            "name": "gpt-oss:20b",
            "digest": "17052f91a42e97930aa6e28a6c6c06a983e6a58dbb00434885a0cf5313e376f7",
            "temperature": 0,
            "seed": 20261115,
            "num_ctx": 8192,
            "num_predict": 2048,
        },
        "embedding_model": {
            "model_id": STELLA_MODEL_ID,
            "revision": STELLA_DEFAULT_REVISION,
            "dimension": 1024,
            "s2p_query_prompt": STELLA_S2P_QUERY_PROMPT,
            "presets": RETRIEVAL_PRESETS,
        },
        "corpora": {
            "two_document_benchmark": {
                "chunk_count": two_doc_manifest["chunk_count"],
                "config_hash_stella_s2p_cosine": two_doc_manifest["config_hash"],
                "document_hashes": two_doc_manifest["document_hashes"],
                "files": two_doc_manifest["files"],
            },
            "expanded_15_document_benchmark": {
                "chunk_count": expanded_manifest["chunk_count"],
                "config_hash_stella_s2p_cosine": expanded_manifest["config_hash"],
                "document_hashes": expanded_manifest["document_hashes"],
                "files": expanded_manifest["files"],
            },
        },
        "reproduction_commands": {
            "unit_tests": "python -m unittest discover -s evaluation/tests -v",
            "prepare_benchmarks": "python evaluation/prepare_retrieval_benchmarks.py",
            "build_two_doc_index": (
                "python evaluation/build_public_index.py "
                "--manifest revision_workspace/retrieval_evaluation/public_manifest.json "
                "--embedding-path $STELLA_SNAPSHOT --preset stella_s2p_cosine "
                "--output revision_workspace/final_retrieval_revision/indexes/two_doc_stella_s2p_cosine"
            ),
            "evaluate_two_doc_retrieval": (
                "python evaluation/public_retrieval.py "
                "--index revision_workspace/final_retrieval_revision/indexes/two_doc_stella_s2p_cosine "
                "--embedding-path $STELLA_SNAPSHOT "
                "--manifest revision_workspace/final_retrieval_revision/indexes/two_doc_stella_s2p_cosine/indexed_manifest.json "
                "--preset stella_s2p_cosine "
                "--benchmark revision_workspace/retrieval_evaluation/annotated_benchmark.json "
                "--output revision_workspace/final_retrieval_revision/two_doc_stella_s2p_cosine_orig22"
            ),
            "evaluate_heldout_retrieval": (
                "python evaluation/public_retrieval.py "
                "--index revision_workspace/final_retrieval_revision/indexes/two_doc_stella_s2p_cosine "
                "--embedding-path $STELLA_SNAPSHOT "
                "--manifest revision_workspace/final_retrieval_revision/indexes/two_doc_stella_s2p_cosine/indexed_manifest.json "
                "--preset stella_s2p_cosine "
                "--benchmark revision_workspace/final_retrieval_revision/annotated_heldout_benchmark.json "
                "--output revision_workspace/final_retrieval_revision/two_doc_stella_s2p_cosine_heldout14"
            ),
            "run_50_generation_ablation": (
                "python evaluation/run_revision_inference.py --experiment ablation --execute-inference "
                "--model gpt-oss:20b --model-digest 17052f91a42e97930aa6e28a6c6c06a983e6a58dbb00434885a0cf5313e376f7 "
                "--temperature 0 --seed 20261115 --num-ctx 8192 --num-predict 2048 "
                "--index revision_workspace/final_retrieval_revision/indexes/two_doc_stella_s2p_cosine "
                "--embedding-path $STELLA_SNAPSHOT "
                "--manifest revision_workspace/final_retrieval_revision/indexes/two_doc_stella_s2p_cosine/indexed_manifest.json "
                "--preset stella_s2p_cosine --retrieval-k 4 "
                "--output revision_workspace/final_retrieval_revision/ablation_results/vm_run_corrected"
            ),
            "judge_ablation": (
                "python evaluation/local_reference_judge.py --model gpt-oss:20b "
                "--model-digest 17052f91a42e97930aa6e28a6c6c06a983e6a58dbb00434885a0cf5313e376f7 "
                "--input revision_workspace/final_retrieval_revision/ablation_results/vm_run_corrected/judge_inputs.json "
                "--output revision_workspace/final_retrieval_revision/ablation_results/local_judged_corrected"
            ),
            "run_smoke_tests": (
                "python evaluation/run_smoke_tests.py --model gpt-oss:20b "
                "--index revision_workspace/final_retrieval_revision/indexes/two_doc_stella_s2p_cosine "
                "--embedding-path $STELLA_SNAPSHOT --preset stella_s2p_cosine "
                "--output revision_workspace/final_retrieval_revision/smoke_tests"
            ),
        },
        "retrieval_summary": retrieval_summary_rows,
        "inference_summary": inference_summary,
        "smoke_test_summary": smoke_summary,
    }
    (rev_dir / "experimental_configuration.json").write_text(
        json.dumps(config_payload, indent=2)
    )


def main():
    rev_dir = ROOT / "revision_workspace/final_retrieval_revision"
    retrieval_summary_rows, _ = build_retrieval_tables(rev_dir)
    inference_rows, inference_summary = build_inference_tables(rev_dir)
    generate_figures(rev_dir, retrieval_summary_rows, inference_rows, inference_summary)
    write_experimental_config(rev_dir, retrieval_summary_rows, inference_summary)
    print("Generated final revision CSVs, JSONs, and PDF/PNG figures.")


if __name__ == "__main__":
    main()
