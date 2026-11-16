"""Generate final publication-ready tables, statistical summaries, sensitivity analyses,
reproducibility metadata, and unclipped vector PDF/PNG figures in
`revision_workspace/final_publication_handoff/`.
"""

import json
from pathlib import Path
import platform
import shutil
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


def mean_ci(values, seed=20261115, resamples=100000):
    arr = np.asarray(values, dtype=float)
    low, high = bootstrap(arr, seed=seed, resamples=resamples)
    return float(arr.mean()), float(low), float(high)


def load_judge_scores(folder: Path, prefix: str):
    out = {}
    for q_idx in range(1, 26):
        qid = f"Q{q_idx:02d}"
        p = folder / f"{prefix}_{qid}.json"
        if p.is_file():
            out[qid] = json.loads(p.read_text())
    return out


def load_traces(folder: Path, cond: str):
    out = {}
    for q_idx in range(1, 26):
        qid = f"Q{q_idx:02d}"
        p = folder / f"{qid}_{cond}.json"
        if p.is_file():
            out[qid] = json.loads(p.read_text())
    return out


def summarize_condition(scores_list, empty_flags, call_counts):
    m, lo, hi = mean_ci(scores_list)
    return {
        "n": len(scores_list),
        "mean": round(m, 4),
        "sd": round(float(np.std(scores_list, ddof=1)), 4) if len(scores_list) > 1 else 0.0,
        "median": round(float(np.median(scores_list)), 4),
        "iqr": round(
            float(np.percentile(scores_list, 75) - np.percentile(scores_list, 25)), 4
        ),
        "ci95_low": round(lo, 4),
        "ci95_high": round(hi, 4),
        "exact_1_0_count": int(sum(1 for s in scores_list if s >= 0.999)),
        "empty_count": int(sum(1 for e in empty_flags if e)),
        "questions_calling_retriever": int(sum(1 for c in call_counts if c > 0)),
        "total_retriever_calls": int(sum(call_counts)),
    }


def summarize_paired_diff(a_scores, b_scores, a_empty, b_empty):
    diffs = [a - b for a, b in zip(a_scores, b_scores)]
    m, lo, hi = mean_ci(diffs)
    nonempty_diffs = [
        a - b
        for a, b, ae, be in zip(a_scores, b_scores, a_empty, b_empty)
        if not ae and not be
    ]
    ne_m, ne_lo, ne_hi = mean_ci(nonempty_diffs) if nonempty_diffs else (0.0, 0.0, 0.0)
    return {
        "n": len(diffs),
        "mean_diff": round(m, 4),
        "ci95_low": round(lo, 4),
        "ci95_high": round(hi, 4),
        "improved_count": int(sum(1 for d in diffs if d > 1e-9)),
        "unchanged_count": int(sum(1 for d in diffs if abs(d) <= 1e-9)),
        "worsened_count": int(sum(1 for d in diffs if d < -1e-9)),
        "nonempty_n": len(nonempty_diffs),
        "nonempty_mean_diff": round(ne_m, 4),
        "nonempty_ci95_low": round(ne_lo, 4),
        "nonempty_ci95_high": round(ne_hi, 4),
    }


def main():
    rev_dir = ROOT / "revision_workspace/final_retrieval_revision"
    out_dir = ROOT / "revision_workspace/final_publication_handoff"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Copy retrieval CSVs into final_publication_handoff for self-contained access
    for csv_name in ("retrieval_summary.csv", "retrieval_before_after.csv"):
        src = rev_dir / csv_name
        if src.is_file():
            shutil.copy2(src, out_dir / csv_name)

    # Load goldens
    goldens = load_goldens(ROOT / "evaluation/goldens.csv")
    questions = [
        {"id": f"Q{i:02d}", "question": q, "expected": a}
        for i, (q, a) in enumerate(goldens.items(), 1)
    ]

    # Load retrieval metrics per question
    m_a_2doc = {
        r["id"]: r
        for r in json.loads(
            (rev_dir / "two_doc_legacy_l2_orig22/metrics.json").read_text()
        )
    }
    m_c_2doc = {
        r["id"]: r
        for r in json.loads(
            (rev_dir / "two_doc_stella_s2p_cosine_orig22/metrics.json").read_text()
        )
    }
    m_a_15doc = {
        r["id"]: r
        for r in json.loads(
            (rev_dir / "expanded_legacy_l2_full25/metrics.json").read_text()
        )
    }
    m_c_15doc = {
        r["id"]: r
        for r in json.loads(
            (rev_dir / "expanded_stella_s2p_cosine_full25/metrics.json").read_text()
        )
    }

    # Load traces
    traces_2doc_dir = rev_dir / "ablation_results/vm_run_corrected"
    traces_15doc_dir = out_dir / "runs/expanded_15doc_ablation"
    traces_broken_dir = ROOT / "revision_workspace/rag_ablation/results/vm_run_01"

    t_norag = load_traces(traces_15doc_dir, "no_rag")
    t_rag2 = load_traces(traces_2doc_dir, "rag")
    t_rag15 = load_traces(traces_15doc_dir, "rag")
    t_broken = load_traces(traces_broken_dir, "rag")

    # Load primary OpenAI GPT-4o-mini judge scores
    openai_judge_dir = out_dir / "openai_judged"
    j_norag = load_judge_scores(openai_judge_dir, "no_rag")
    j_rag2 = load_judge_scores(openai_judge_dir, "rag_2doc")
    j_rag15 = load_judge_scores(openai_judge_dir, "rag_15doc")
    openai_meta = json.loads((openai_judge_dir / "metadata.json").read_text())

    # Load secondary local GPT-OSS judge scores (for historical continuity, kept separate)
    local_2doc_dir = rev_dir / "ablation_results/local_judged_corrected"
    local_broken_dir = ROOT / "revision_workspace/rag_ablation/local_judged"
    local_15doc_dir = out_dir / "local_judged_15doc"

    l_norag = load_judge_scores(local_2doc_dir, "ablation_no_rag")
    l_rag2 = load_judge_scores(local_2doc_dir, "ablation_rag")
    l_broken = load_judge_scores(local_broken_dir, "ablation_rag")
    l_rag15 = (
        load_judge_scores(local_15doc_dir, "ablation_rag")
        if local_15doc_dir.is_dir()
        else {}
    )

    exp_bench = {
        q["id"]: q
        for q in json.loads((rev_dir / "annotated_expanded_benchmark.json").read_text())
    }

    question_rows = []
    for item in questions:
        qid = item["id"]
        in_2doc = qid not in ("Q22", "Q23", "Q24")
        nr_calls = [
            c
            for c in t_norag[qid].get("calls", [])
            if c["name"] == "retrieve_geomag_docs"
        ]
        r2_calls = [
            c
            for c in t_rag2[qid].get("calls", [])
            if c["name"] == "retrieve_geomag_docs"
        ]
        r15_calls = [
            c
            for c in t_rag15[qid].get("calls", [])
            if c["name"] == "retrieve_geomag_docs"
        ]
        brk_calls = [
            c
            for c in t_broken[qid].get("calls", [])
            if c["name"] == "retrieve_geomag_docs"
        ]

        s_nr = j_norag[qid]["score"]
        s_r2 = j_rag2[qid]["score"]
        s_r15 = j_rag15[qid]["score"]

        # In the 15-doc corpus, Q05 and Q10 retrieve valid cross-document passages
        # (Reeve 2010 tutorial on Magsat; Encyclopedia of Geomagnetism p. 958 on ring current).
        strict_15_hit = m_c_15doc[qid].get("hit_at_4", 0)
        crossdoc_15_hit = 1 if (strict_15_hit == 1 or qid in ("Q05", "Q10")) else 0

        question_rows.append(
            {
                "id": qid,
                "in_two_doc_corpus": in_2doc,
                "reference_flag": (
                    "reference_artifact_q06_q12"
                    if qid in ("Q06", "Q12")
                    else "standard"
                ),
                "source_document": exp_bench[qid].get("source_document", ""),
                "question": item["question"],
                "two_doc_legacy_l2_hit4": m_a_2doc[qid].get("hit_at_4", "")
                if in_2doc
                else "",
                "two_doc_stella_s2p_hit4": m_c_2doc[qid].get("hit_at_4", "")
                if in_2doc
                else "",
                "expanded_15doc_legacy_l2_hit4": m_a_15doc[qid].get("hit_at_4", 0),
                "expanded_15doc_stella_s2p_strict_hit4": strict_15_hit,
                "expanded_15doc_stella_s2p_crossdoc_hit4": crossdoc_15_hit,
                "no_rag_score_gpt4omini": s_nr,
                "rag_2doc_score_gpt4omini": s_r2,
                "rag_15doc_score_gpt4omini": s_r15,
                "diff_15doc_minus_norag_gpt4omini": round(s_r15 - s_nr, 4),
                "diff_2doc_minus_norag_gpt4omini": round(s_r2 - s_nr, 4),
                "diff_15doc_minus_2doc_gpt4omini": round(s_r15 - s_r2, 4),
                "no_rag_empty": j_norag[qid]["empty_response"],
                "rag_2doc_empty": j_rag2[qid]["empty_response"],
                "rag_15doc_empty": j_rag15[qid]["empty_response"],
                "no_rag_retriever_calls": len(nr_calls),
                "rag_2doc_retriever_calls": len(r2_calls),
                "rag_15doc_retriever_calls": len(r15_calls),
                "broken_rag_retriever_calls": len(brk_calls),
                "no_rag_reason_gpt4omini": j_norag[qid]["reason"],
                "rag_2doc_reason_gpt4omini": j_rag2[qid]["reason"],
                "rag_15doc_reason_gpt4omini": j_rag15[qid]["reason"],
                "local_gptoss_no_rag_score": l_norag[qid]["score"],
                "local_gptoss_rag_2doc_score": l_rag2[qid]["score"],
                "local_gptoss_rag_15doc_score": (
                    l_rag15[qid]["score"] if qid in l_rag15 else ""
                ),
                "local_gptoss_broken_rag_score": l_broken[qid]["score"],
            }
        )

    write_csv(out_dir / "question_scores.csv", question_rows)

    def analyze_subset(subset_rows, subset_name):
        nr = [r["no_rag_score_gpt4omini"] for r in subset_rows]
        r2 = [r["rag_2doc_score_gpt4omini"] for r in subset_rows]
        r15 = [r["rag_15doc_score_gpt4omini"] for r in subset_rows]

        nr_e = [r["no_rag_empty"] for r in subset_rows]
        r2_e = [r["rag_2doc_empty"] for r in subset_rows]
        r15_e = [r["rag_15doc_empty"] for r in subset_rows]

        nr_c = [r["no_rag_retriever_calls"] for r in subset_rows]
        r2_c = [r["rag_2doc_retriever_calls"] for r in subset_rows]
        r15_c = [r["rag_15doc_retriever_calls"] for r in subset_rows]

        return {
            "subset": subset_name,
            "n": len(subset_rows),
            "judge_model": openai_meta["returned_models"][0]
            if openai_meta.get("returned_models")
            else "gpt-4o-mini",
            "no_rag": summarize_condition(nr, nr_e, nr_c),
            "rag_2doc_corrected": summarize_condition(r2, r2_e, r2_c),
            "rag_15doc_corrected": summarize_condition(r15, r15_e, r15_c),
            "paired_15doc_minus_no_rag": summarize_paired_diff(r15, nr, r15_e, nr_e),
            "paired_2doc_minus_no_rag": summarize_paired_diff(r2, nr, r2_e, nr_e),
            "paired_15doc_minus_2doc": summarize_paired_diff(r15, r2, r15_e, r2_e),
        }

    s_all25 = analyze_subset(question_rows, "all_25_questions")
    s_excl_q06_q12 = analyze_subset(
        [r for r in question_rows if r["id"] not in ("Q06", "Q12")],
        "sensitivity_excluding_q06_q12_n23",
    )
    s_in2doc_22 = analyze_subset(
        [r for r in question_rows if r["in_two_doc_corpus"]],
        "in_two_doc_corpus_n22",
    )
    s_jacobs_3 = analyze_subset(
        [r for r in question_rows if not r["in_two_doc_corpus"]],
        "jacobs_biography_q22_q24_n3",
    )

    # Also summarize local GPT-OSS judge separately if available
    local_summary = {}
    if len(l_rag15) == 25:
        lnr = [r["local_gptoss_no_rag_score"] for r in question_rows]
        lr2 = [r["local_gptoss_rag_2doc_score"] for r in question_rows]
        lr15 = [r["local_gptoss_rag_15doc_score"] for r in question_rows]
        lbrk = [r["local_gptoss_broken_rag_score"] for r in question_rows]
        nr_e = [r["no_rag_empty"] for r in question_rows]
        r2_e = [r["rag_2doc_empty"] for r in question_rows]
        r15_e = [r["rag_15doc_empty"] for r in question_rows]
        brk_e = [l_broken[r["id"]]["empty_response"] for r in question_rows]
        local_summary = {
            "judge_model": "gpt-oss:20b (local self-judge, secondary)",
            "no_rag": summarize_condition(
                lnr, nr_e, [r["no_rag_retriever_calls"] for r in question_rows]
            ),
            "rag_broken_l2": summarize_condition(
                lbrk, brk_e, [r["broken_rag_retriever_calls"] for r in question_rows]
            ),
            "rag_2doc_corrected": summarize_condition(
                lr2, r2_e, [r["rag_2doc_retriever_calls"] for r in question_rows]
            ),
            "rag_15doc_corrected": summarize_condition(
                lr15, r15_e, [r["rag_15doc_retriever_calls"] for r in question_rows]
            ),
            "paired_15doc_minus_no_rag": summarize_paired_diff(lr15, lnr, r15_e, nr_e),
            "paired_2doc_minus_no_rag": summarize_paired_diff(lr2, lnr, r2_e, nr_e),
            "paired_15doc_minus_2doc": summarize_paired_diff(lr15, lr2, r15_e, r2_e),
        }

    # Build model_scores.csv
    model_score_rows = []
    for sub in (s_all25, s_excl_q06_q12, s_in2doc_22, s_jacobs_3):
        j_name = sub["judge_model"]
        for cond_key, cond_label in (
            ("no_rag", "No RAG (Parametric Baseline)"),
            ("rag_2doc_corrected", "Corrected 2-Doc RAG (stella_s2p_cosine)"),
            ("rag_15doc_corrected", "Corrected 15-Doc Full-Corpus RAG (stella_s2p_cosine)"),
        ):
            c = sub[cond_key]
            diff_vs_nr = (
                sub["paired_15doc_minus_no_rag"]
                if cond_key == "rag_15doc_corrected"
                else (
                    sub["paired_2doc_minus_no_rag"]
                    if cond_key == "rag_2doc_corrected"
                    else None
                )
            )
            model_score_rows.append(
                {
                    "judge": j_name,
                    "subset": sub["subset"],
                    "condition_code": cond_key,
                    "condition_label": cond_label,
                    "n": c["n"],
                    "mean": c["mean"],
                    "sd": c["sd"],
                    "median": c["median"],
                    "iqr": c["iqr"],
                    "ci95_low": c["ci95_low"],
                    "ci95_high": c["ci95_high"],
                    "exact_1_0_count": c["exact_1_0_count"],
                    "empty_count": c["empty_count"],
                    "questions_calling_retriever": c["questions_calling_retriever"],
                    "total_retriever_calls": c["total_retriever_calls"],
                    "paired_diff_vs_no_rag": diff_vs_nr["mean_diff"] if diff_vs_nr else 0.0,
                    "paired_diff_ci95_low": diff_vs_nr["ci95_low"] if diff_vs_nr else "",
                    "paired_diff_ci95_high": diff_vs_nr["ci95_high"] if diff_vs_nr else "",
                    "improved_vs_no_rag": diff_vs_nr["improved_count"] if diff_vs_nr else "",
                    "unchanged_vs_no_rag": diff_vs_nr["unchanged_count"] if diff_vs_nr else "",
                    "worsened_vs_no_rag": diff_vs_nr["worsened_count"] if diff_vs_nr else "",
                }
            )

    if local_summary:
        for cond_key, cond_label in (
            ("no_rag", "No RAG (Parametric Baseline)"),
            ("rag_broken_l2", "Original Broken 2-Doc RAG (legacy_l2)"),
            ("rag_2doc_corrected", "Corrected 2-Doc RAG (stella_s2p_cosine)"),
            ("rag_15doc_corrected", "Corrected 15-Doc Full-Corpus RAG (stella_s2p_cosine)"),
        ):
            c = local_summary[cond_key]
            diff_vs_nr = (
                local_summary["paired_15doc_minus_no_rag"]
                if cond_key == "rag_15doc_corrected"
                else (
                    local_summary["paired_2doc_minus_no_rag"]
                    if cond_key == "rag_2doc_corrected"
                    else None
                )
            )
            model_score_rows.append(
                {
                    "judge": local_summary["judge_model"],
                    "subset": "all_25_questions",
                    "condition_code": cond_key,
                    "condition_label": cond_label,
                    "n": c["n"],
                    "mean": c["mean"],
                    "sd": c["sd"],
                    "median": c["median"],
                    "iqr": c["iqr"],
                    "ci95_low": c["ci95_low"],
                    "ci95_high": c["ci95_high"],
                    "exact_1_0_count": c["exact_1_0_count"],
                    "empty_count": c["empty_count"],
                    "questions_calling_retriever": c["questions_calling_retriever"],
                    "total_retriever_calls": c["total_retriever_calls"],
                    "paired_diff_vs_no_rag": diff_vs_nr["mean_diff"] if diff_vs_nr else "",
                    "paired_diff_ci95_low": diff_vs_nr["ci95_low"] if diff_vs_nr else "",
                    "paired_diff_ci95_high": diff_vs_nr["ci95_high"] if diff_vs_nr else "",
                    "improved_vs_no_rag": diff_vs_nr["improved_count"] if diff_vs_nr else "",
                    "unchanged_vs_no_rag": diff_vs_nr["unchanged_count"] if diff_vs_nr else "",
                    "worsened_vs_no_rag": diff_vs_nr["worsened_count"] if diff_vs_nr else "",
                }
            )

    write_csv(out_dir / "model_scores.csv", model_score_rows)

    full_stats = {
        "primary_openai_judge": {
            "all_25_questions": s_all25,
            "sensitivity_excluding_q06_q12_n23": s_excl_q06_q12,
            "in_two_doc_corpus_n22": s_in2doc_22,
            "jacobs_biography_q22_q24_n3": s_jacobs_3,
            "judge_metadata": openai_meta,
        },
        "secondary_local_gptoss_judge": local_summary,
    }
    (out_dir / "statistical_summary.json").write_text(json.dumps(full_stats, indent=2))

    # Build experimental_configuration.json
    run15_meta = json.loads((traces_15doc_dir / "metadata.json").read_text())
    run2_meta = json.loads((traces_2doc_dir / "metadata.json").read_text())
    exp_config = {
        "branch": "revision/offline-scientific-audit",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "python_version": platform.python_version(),
        "answering_model": {
            "name": "gpt-oss:20b",
            "digest": run15_meta["model_digest"],
            "temperature": 0.0,
            "seed": 20261115,
            "num_ctx": 8192,
            "num_predict": 2048,
            "session_isolation": "fresh_ChatGeoMagi_instance_and_uuid4_thread_per_question",
        },
        "primary_judge_model": {
            "provider": "OpenAI API",
            "requested_model": openai_meta["requested_model"],
            "returned_models": openai_meta["returned_models"],
            "temperature": openai_meta["temperature"],
            "seed": openai_meta["seed"],
            "score_scale": openai_meta["score_scale"],
            "calibration_cases": openai_meta["calibration_cases"],
            "calibration_passed": openai_meta["calibration_passed"],
            "evaluated_responses": openai_meta["evaluated_responses"],
            "token_usage_and_cost": openai_meta["usage"],
        },
        "retrieval_configuration": run15_meta["index_metadata"],
        "two_document_index_configuration": run2_meta["index_metadata"],
        "bootstrap_configuration": {
            "seed": 20261115,
            "resamples": 100000,
            "method": "paired_percentile_bootstrap",
        },
    }
    (out_dir / "experimental_configuration.json").write_text(
        json.dumps(exp_config, indent=2)
    )

    # --- Generate Publication Figures ---
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    # Figure 1: Retrieval Before and After
    ret_summary_rows = json.loads(
        json.dumps(
            [
                row
                for row in (
                    # Load from rev_dir / retrieval_summary.csv via simple dict reader
                    __import__("csv").DictReader(
                        (rev_dir / "retrieval_summary.csv").open()
                    )
                )
            ]
        )
    )
    by_run = {r["run_id"]: r for r in ret_summary_rows}

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.7), layout="constrained")
    categories = [
        "2-Doc Corpus\nPrimary (n=22)",
        "2-Doc Corpus\nHeld-Out (n=14)",
        "Expanded 15-Doc\nAll Qs (n=25)",
        "Expanded 15-Doc\nHeld-Out (n=14)",
    ]
    a_vals = [
        float(by_run["two_doc_legacy_l2_orig22"]["hit_at_4"]) * 100,
        float(by_run["two_doc_legacy_l2_heldout14"]["hit_at_4"]) * 100,
        float(by_run["expanded_legacy_l2_full25"]["hit_at_4"]) * 100,
        float(by_run["expanded_legacy_l2_heldout14"]["hit_at_4"]) * 100,
    ]
    b_vals = [
        float(by_run["two_doc_norm_cosine_orig22"]["hit_at_4"]) * 100,
        float(by_run["two_doc_norm_cosine_heldout14"]["hit_at_4"]) * 100,
        float(by_run["expanded_norm_cosine_full25"]["hit_at_4"]) * 100,
        float(by_run["expanded_norm_cosine_heldout14"]["hit_at_4"]) * 100,
    ]
    c_vals = [
        float(by_run["two_doc_stella_s2p_cosine_orig22"]["hit_at_4"]) * 100,
        float(by_run["two_doc_stella_s2p_cosine_heldout14"]["hit_at_4"]) * 100,
        float(by_run["expanded_stella_s2p_cosine_full25"]["hit_at_4"]) * 100,
        float(by_run["expanded_stella_s2p_cosine_heldout14"]["hit_at_4"]) * 100,
    ]
    x = np.arange(len(categories))
    w = 0.25
    b1 = axes[0].bar(
        x - w, a_vals, width=w, label="Config A: Legacy Unnorm. L2", color="#CC79A7"
    )
    b2 = axes[0].bar(
        x,
        b_vals,
        width=w,
        label="Config B: Bidirectional + Norm. Cosine",
        color="#56B4E9",
    )
    b3 = axes[0].bar(
        x + w,
        c_vals,
        width=w,
        label="Config C: Bidirectional + Norm. Cosine + s2p_query",
        color="#009E73",
    )
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
    axes[0].set_ylim(0, 112)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories)
    axes[0].set_ylabel("Top-4 Retrieval Hit Rate (Hit@4, %)")
    axes[0].set_title("(a) Top-4 Retrieval Accuracy Across Corpora & Benchmarks")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    axes[0].legend(loc="upper left", fontsize=8.2, framealpha=0.92)

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
    hit_rates = [float(by_run[k]["hit_at_4"]) * 100 for k in ablation_runs]
    uniq_counts = [int(by_run[k]["unique_chunks_scored"]) for k in ablation_runs]
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
    axes[1].set_ylim(0, 114)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(ablation_labels, fontsize=8.5)
    axes[1].set_ylabel("Hit@4 (%) [with unique top-4 chunks across 22 Qs]")
    axes[1].set_title("(b) Root-Cause Ablation on 22 Verified Questions")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    save_plot(fig, fig_dir / "retrieval_before_after")

    # Figure 2: RAG vs No-RAG Answer Correctness (Judged by OpenAI GPT-4o-mini) — UNCLIPPED CIs
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), layout="constrained")

    subsets = [
        ("All 25 Questions\n(Primary Benchmark, n=25)", s_all25),
        ("Excluding Q06 & Q12\n(Ref. Artifact Sensitivity, n=23)", s_excl_q06_q12),
    ]
    xg = np.arange(len(subsets))
    wg = 0.25

    cond_specs = [
        ("no_rag", "No RAG (Parametric Baseline)", "#0072B2", -wg),
        ("rag_2doc_corrected", "Corrected 2-Doc RAG (283 chunks)", "#56B4E9", 0.0),
        ("rag_15doc_corrected", "Corrected 15-Doc Full-Corpus RAG (8,863 chunks)", "#009E73", wg),
    ]
    axes[0].axhline(1.0, color="#666666", linestyle=":", linewidth=0.9)
    for key, label, color, offset in cond_specs:
        means = [sub[key]["mean"] for _, sub in subsets]
        lows = [sub[key]["mean"] - sub[key]["ci95_low"] for _, sub in subsets]
        highs = [sub[key]["ci95_high"] - sub[key]["mean"] for _, sub in subsets]
        ci_tops = [sub[key]["ci95_high"] for _, sub in subsets]
        bars = axes[0].bar(
            xg + offset,
            means,
            width=wg,
            yerr=[lows, highs],
            capsize=4,
            label=label,
            color=color,
            error_kw={"elinewidth": 1.3},
        )
        for bar, m, top in zip(bars, means, ci_tops):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                top + 0.02,
                f"{m:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )

    axes[0].set_ylim(0, 1.16)
    axes[0].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes[0].set_xticks(xg)
    axes[0].set_xticklabels([label for label, _ in subsets])
    axes[0].set_ylabel("Mean GPT-4o-mini Correctness Score (0–1)")
    axes[0].set_title("(a) Answer Correctness by Condition (95% Bootstrap CI)")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    axes[0].legend(loc="lower right", fontsize=8.2, framealpha=0.94)

    # Panel B: Paired Treatment Effects (with wide unclipped x-axis limits)
    diff_items = [
        (
            "15-Doc RAG − 2-Doc RAG\n(All 25 Qs)",
            s_all25["paired_15doc_minus_2doc"],
            "#E69F00",
        ),
        (
            "2-Doc RAG − No RAG\n(All 25 Qs)",
            s_all25["paired_2doc_minus_no_rag"],
            "#56B4E9",
        ),
        (
            "15-Doc RAG − No RAG\n(All 25 Qs, Primary)",
            s_all25["paired_15doc_minus_no_rag"],
            "#009E73",
        ),
        (
            "15-Doc RAG − No RAG\n(Excl. Q06 & Q12, n=23)",
            s_excl_q06_q12["paired_15doc_minus_no_rag"],
            "#0072B2",
        ),
    ]
    yd = np.arange(len(diff_items))
    axes[1].axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    min_lo = min(item[1]["ci95_low"] for item in diff_items)
    max_hi = max(item[1]["ci95_high"] for item in diff_items)
    for idx, (label, dspec, color) in enumerate(diff_items):
        m = dspec["mean_diff"]
        lo = dspec["ci95_low"]
        hi = dspec["ci95_high"]
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
    axes[1].set_xlim(min(-0.20, min_lo - 0.08), max(0.55, max_hi + 0.08))
    axes[1].set_ylim(-0.5, len(diff_items) - 0.25)
    axes[1].set_xlabel("Paired Score Difference (GPT-4o-mini, 95% Bootstrap CI)")
    axes[1].set_title("(b) Paired Treatment Effects (Unclipped 95% CIs)")
    axes[1].grid(axis="x", linestyle="--", alpha=0.4)

    save_plot(fig, fig_dir / "rag_vs_no_rag")

    # Figure 3: Per-Question Differences (Full-Corpus 15-Doc RAG vs No RAG, GPT-4o-mini)
    fig, axes = plt.subplots(2, 1, figsize=(12.2, 6.4), sharex=True, layout="constrained")
    qids = [r["id"] for r in question_rows]
    xq = np.arange(len(qids))
    nr_s = [r["no_rag_score_gpt4omini"] for r in question_rows]
    r2_s = [r["rag_2doc_score_gpt4omini"] for r in question_rows]
    r15_s = [r["rag_15doc_score_gpt4omini"] for r in question_rows]
    diffs_15_nr = [r["diff_15doc_minus_norag_gpt4omini"] for r in question_rows]

    axes[0].plot(
        xq, nr_s, marker="s", linestyle=":", color="#0072B2", label="No RAG", alpha=0.85
    )
    axes[0].plot(
        xq,
        r2_s,
        marker="^",
        linestyle="--",
        color="#56B4E9",
        label="Corrected 2-Doc RAG",
        alpha=0.85,
    )
    axes[0].plot(
        xq,
        r15_s,
        marker="o",
        linestyle="-",
        color="#009E73",
        label="Corrected 15-Doc Full-Corpus RAG",
        linewidth=1.8,
    )
    axes[0].axvspan(
        20.5,
        23.5,
        color="#F0E442",
        alpha=0.22,
        label="Q22–Q24 (Jacobs Biography: Missing in 2-Doc, Present in 15-Doc)",
    )
    for ref_q_idx in (5, 11):  # Q06 (idx 5) and Q12 (idx 11)
        axes[0].axvline(ref_q_idx, color="#D55E00", linestyle=":", alpha=0.55)
    axes[0].set_ylim(-0.05, 1.08)
    axes[0].set_ylabel("GPT-4o-mini Score (0–1)")
    axes[0].set_title("(a) Per-Question Answer Correctness Across Conditions (Q01–Q25)")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(loc="lower left", ncol=2, fontsize=8.3, framealpha=0.92)

    bar_colors = [
        "#009E73" if d > 1e-9 else ("#D55E00" if d < -1e-9 else "#999999")
        for d in diffs_15_nr
    ]
    bars = axes[1].bar(xq, diffs_15_nr, width=0.62, color=bar_colors)
    axes[1].axhline(0.0, color="black", linewidth=0.9)
    axes[1].axvspan(20.5, 23.5, color="#F0E442", alpha=0.22)
    for bar, d in zip(bars, diffs_15_nr):
        if abs(d) > 1e-9:
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                d + (0.03 if d > 0 else -0.08),
                f"{d:+.1f}",
                ha="center",
                va="bottom" if d > 0 else "top",
                fontsize=7.8,
                fontweight="bold",
            )
    axes[1].set_ylim(-1.15, 1.18)
    axes[1].set_xticks(xq)
    axes[1].set_xticklabels(qids)
    axes[1].set_ylabel("Score Change\n(15-Doc RAG − No RAG)")
    axes[1].set_xlabel("Benchmark Question ID")
    axes[1].set_title(
        "(b) Per-Question Paired Gain from Full-Corpus Corrected RAG (Judged by GPT-4o-mini)"
    )
    axes[1].grid(True, linestyle="--", alpha=0.35)

    save_plot(fig, fig_dir / "question_differences")
    print("Generated all tables, statistical summaries, and figures in", out_dir)


if __name__ == "__main__":
    main()
