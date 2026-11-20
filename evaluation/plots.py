"""Generate publication figures for ChatGeo-Magi directly from checked-in result tables.

Usage:
    python evaluation/plots.py
    python evaluation/plots.py --figure rag
    python evaluation/plots.py --figure retrieval
    python evaluation/plots.py --figure questions
    python evaluation/plots.py --figure tools
    python evaluation/plots.py --figure london
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths and Matplotlib style configuration (edit here to customize figures)
# ---------------------------------------------------------------------------
EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
OUTPUT_DIR = EVAL_DIR / "plot_outputs"

DPI = 400
FONT_SIZE = 10
TITLE_SIZE = 11
LABEL_SIZE = 10
LINE_WIDTH = 1.8

COLORS = {
    "navy": "#0072B2",
    "sky": "#56B4E9",
    "teal": "#009E73",
    "amber": "#E69F00",
    "vermilion": "#D55E00",
    "rose": "#CC79A7",
    "yellow": "#F0E442",
    "gray": "#7F7F7F",
}

FIGSIZE_RETRIEVAL = (10.5, 4.6)
FIGSIZE_RAG = (11.6, 4.8)
FIGSIZE_QUESTIONS = (12.0, 6.2)
FIGSIZE_TOOLS = (9.2, 4.5)
FIGSIZE_LONDON = (8.8, 4.6)


def apply_style():
    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": LABEL_SIZE,
            "xtick.labelsize": FONT_SIZE - 1,
            "ytick.labelsize": FONT_SIZE - 1,
            "legend.fontsize": FONT_SIZE - 1.5,
            "lines.linewidth": LINE_WIDTH,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_csv_rows(path: Path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def save_figure(fig, output_stem: Path, export_png: bool = True):
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(pdf_path)
    saved = [pdf_path]
    if export_png:
        png_path = output_stem.with_suffix(".png")
        fig.savefig(png_path, dpi=DPI)
        saved.append(png_path)
    plt.close(fig)
    return saved


# ---------------------------------------------------------------------------
# Figure 1: Top-4 Retrieval Performance (Hit@4, Recall@4, Coverage@4)
# ---------------------------------------------------------------------------
def plot_retrieval(results_dir: Path = RESULTS_DIR, output_dir: Path = OUTPUT_DIR, export_png: bool = True):
    """Plot final top-4 Stella retrieval performance across corpora and benchmarks."""
    apply_style()
    summary_rows = {
        row["run_id"]: row
        for row in read_csv_rows(Path(results_dir) / "retrieval_summary.csv")
    }
    question_rows = read_csv_rows(Path(results_dir) / "rag_question_scores.csv")

    full15 = summary_rows["expanded_stella_s2p_cosine_full25"]
    held15 = summary_rows["expanded_stella_s2p_cosine_heldout14"]
    sub2 = summary_rows["two_doc_stella_s2p_cosine_orig22"]
    held2 = summary_rows["two_doc_stella_s2p_cosine_heldout14"]

    strict_15_hits = int(full15["hits_at_4"])
    strict_15_n = int(full15["scored_questions"])
    crossdoc_15_hits = sum(
        int(row["expanded_15doc_stella_s2p_crossdoc_hit4"]) for row in question_rows
    )

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_RETRIEVAL, layout="constrained")

    # Panel (a): Hit@4 across the primary and held-out benchmarks
    bench_labels = [
        f"15-Doc Full Corpus\nPrimary Strict\n(n={strict_15_n})",
        f"15-Doc Full Corpus\nPrimary Cross-Doc\n(n={strict_15_n})",
        f"15-Doc Full Corpus\nHeld-Out\n(n={held15['scored_questions']})",
        f"2-Doc Subset\nPrimary In-Corpus\n(n={sub2['scored_questions']})",
        f"2-Doc Subset\nHeld-Out\n(n={held2['scored_questions']})",
    ]
    hit_rates = [
        float(full15["hit_at_4"]) * 100.0,
        (crossdoc_15_hits / strict_15_n) * 100.0,
        float(held15["hit_at_4"]) * 100.0,
        float(sub2["hit_at_4"]) * 100.0,
        float(held2["hit_at_4"]) * 100.0,
    ]
    hit_counts = [
        f"{strict_15_hits}/{strict_15_n}",
        f"{crossdoc_15_hits}/{strict_15_n}",
        f"{held15['hits_at_4']}/{held15['scored_questions']}",
        f"{sub2['hits_at_4']}/{sub2['scored_questions']}",
        f"{held2['hits_at_4']}/{held2['scored_questions']}",
    ]
    bar_colors = [
        COLORS["navy"],
        COLORS["teal"],
        COLORS["teal"],
        COLORS["sky"],
        COLORS["sky"],
    ]

    x = np.arange(len(bench_labels))
    bars = axes[0].bar(x, hit_rates, width=0.56, color=bar_colors)
    for bar, rate, count_str in zip(bars, hit_rates, hit_counts):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            rate + 1.8,
            f"{rate:.1f}%\n({count_str})",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE - 1.5,
            fontweight="bold",
        )
    axes[0].set_ylim(0, 114)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(bench_labels, fontsize=FONT_SIZE - 1.5)
    axes[0].set_ylabel("Top-4 Retrieval Hit Rate (Hit@4, %)")
    axes[0].set_title("(a) Top-4 Retrieval Hit@4 by Corpus and Benchmark")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    # Panel (b): Multi-metric profile on the 15-document full corpus
    metric_groups = [
        f"15-Doc Primary Benchmark\n(n={strict_15_n}, 8,863 chunks)",
        f"15-Doc Held-Out Benchmark\n(n={held15['scored_questions']}, 8,863 chunks)",
    ]
    xm = np.arange(len(metric_groups))
    width = 0.24
    metric_specs = [
        (
            "Strict Hit@4",
            [float(full15["hit_at_4"]) * 100.0, float(held15["hit_at_4"]) * 100.0],
            COLORS["navy"],
            -width,
        ),
        (
            "Support Coverage@4",
            [
                float(full15["support_coverage_at_4"]) * 100.0,
                float(held15["support_coverage_at_4"]) * 100.0,
            ],
            COLORS["teal"],
            0.0,
        ),
        (
            "Chunk Recall@4",
            [float(full15["recall_at_4"]) * 100.0, float(held15["recall_at_4"]) * 100.0],
            COLORS["sky"],
            width,
        ),
    ]
    for label, vals, color, offset in metric_specs:
        mbars = axes[1].bar(xm + offset, vals, width=width, label=label, color=color)
        for bar, val in zip(mbars, vals):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2.0,
                val + 1.8,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=FONT_SIZE - 1.5,
            )

    axes[1].set_ylim(0, 114)
    axes[1].set_xticks(xm)
    axes[1].set_xticklabels(metric_groups)
    axes[1].set_ylabel("Retrieval Score (%)")
    axes[1].set_title("(b) Full-Corpus Retrieval Metrics (stella_s2p_cosine, k=4)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    axes[1].legend(loc="lower right", framealpha=0.94)

    return save_figure(fig, Path(output_dir) / "retrieval_performance", export_png=export_png)


# ---------------------------------------------------------------------------
# Figure 2: RAG vs. No-RAG Answer Correctness and Paired Bootstrap CIs
# ---------------------------------------------------------------------------
def plot_rag_comparison(results_dir: Path = RESULTS_DIR, output_dir: Path = OUTPUT_DIR, export_png: bool = True):
    """Plot RAG vs. No-RAG GPT-4o-mini correctness scores and paired 95% bootstrap CIs."""
    apply_style()
    stats = json.loads(
        (Path(results_dir) / "rag_statistical_summary.json").read_text(encoding="utf-8")
    )["primary_openai_judge"]
    s_all25 = stats["all_25_questions"]
    s_excl23 = stats["sensitivity_excluding_q06_q12_n23"]

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_RAG, layout="constrained")

    subsets = [
        (f"All 25 Questions\n(Primary Benchmark, n={s_all25['n']})", s_all25),
        (f"Excluding Q06 & Q12\n(Ref. Artifact Sensitivity, n={s_excl23['n']})", s_excl23),
    ]
    xg = np.arange(len(subsets))
    wg = 0.25

    cond_specs = [
        ("no_rag", "No RAG (Parametric Baseline)", COLORS["navy"], -wg),
        ("rag_2doc_corrected", "2-Doc Subset RAG (283 chunks)", COLORS["sky"], 0.0),
        (
            "rag_15doc_corrected",
            "Full-Corpus 15-Doc RAG (8,863 chunks)",
            COLORS["teal"],
            wg,
        ),
    ]

    axes[0].axhline(1.0, color=COLORS["gray"], linestyle=":", linewidth=0.9)
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
        for bar, mean_val, top in zip(bars, means, ci_tops):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2.0,
                top + 0.02,
                f"{mean_val:.3f}",
                ha="center",
                va="bottom",
                fontsize=FONT_SIZE - 1.5,
                fontweight="bold",
            )

    axes[0].set_ylim(0, 1.16)
    axes[0].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes[0].set_xticks(xg)
    axes[0].set_xticklabels([label for label, _ in subsets])
    axes[0].set_ylabel("Mean GPT-4o-mini Correctness Score (0–1)")
    axes[0].set_title("(a) Answer Correctness by Condition (95% Bootstrap CI)")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    axes[0].legend(loc="lower right", framealpha=0.94)

    diff_items = [
        (
            "15-Doc RAG − 2-Doc RAG\n(All 25 Qs, n=25)",
            s_all25["paired_15doc_minus_2doc"],
            COLORS["amber"],
        ),
        (
            "2-Doc RAG − No RAG\n(All 25 Qs, n=25)",
            s_all25["paired_2doc_minus_no_rag"],
            COLORS["sky"],
        ),
        (
            "15-Doc RAG − No RAG\n(All 25 Qs, Primary, n=25)",
            s_all25["paired_15doc_minus_no_rag"],
            COLORS["teal"],
        ),
        (
            "15-Doc RAG − No RAG\n(Excl. Q06 & Q12, n=23)",
            s_excl23["paired_15doc_minus_no_rag"],
            COLORS["navy"],
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
            fontsize=FONT_SIZE - 1.5,
            fontweight="bold",
        )

    axes[1].set_yticks(yd)
    axes[1].set_yticklabels([item[0] for item in diff_items])
    axes[1].set_xlim(min(-0.15, min_lo - 0.08), max(0.35, max_hi + 0.08))
    axes[1].set_ylim(-0.5, len(diff_items) - 0.25)
    axes[1].set_xlabel("Paired Score Difference (GPT-4o-mini, 95% Bootstrap CI)")
    axes[1].set_title("(b) Paired Treatment Effects (95% Bootstrap CIs)")
    axes[1].grid(axis="x", linestyle="--", alpha=0.4)

    return save_figure(fig, Path(output_dir) / "rag_comparison", export_png=export_png)


# ---------------------------------------------------------------------------
# Figure 3: Per-Question Answer Correctness and Paired Differences (Q01-Q25)
# ---------------------------------------------------------------------------
def plot_question_differences(results_dir: Path = RESULTS_DIR, output_dir: Path = OUTPUT_DIR, export_png: bool = True):
    """Plot per-question correctness scores and paired RAG gains across Q01-Q25."""
    apply_style()
    question_rows = read_csv_rows(Path(results_dir) / "rag_question_scores.csv")

    qids = [row["id"] for row in question_rows]
    xq = np.arange(len(qids))
    nr_scores = [float(row["no_rag_score_gpt4omini"]) for row in question_rows]
    r2_scores = [float(row["rag_2doc_score_gpt4omini"]) for row in question_rows]
    r15_scores = [float(row["rag_15doc_score_gpt4omini"]) for row in question_rows]
    diffs_15_nr = [float(row["diff_15doc_minus_norag_gpt4omini"]) for row in question_rows]

    fig, axes = plt.subplots(2, 1, figsize=FIGSIZE_QUESTIONS, sharex=True, layout="constrained")

    axes[0].plot(
        xq,
        nr_scores,
        marker="s",
        linestyle=":",
        color=COLORS["navy"],
        label="No RAG (Parametric Baseline)",
        alpha=0.85,
    )
    axes[0].plot(
        xq,
        r2_scores,
        marker="^",
        linestyle="--",
        color=COLORS["sky"],
        label="2-Doc Subset RAG",
        alpha=0.85,
    )
    axes[0].plot(
        xq,
        r15_scores,
        marker="o",
        linestyle="-",
        color=COLORS["teal"],
        label="Full-Corpus 15-Doc RAG",
        linewidth=LINE_WIDTH,
    )
    axes[0].axvspan(
        20.5,
        23.5,
        color=COLORS["yellow"],
        alpha=0.22,
        label="Q22–Q24 (Encyclopedia of Geomagnetism: Present in 15-Doc Corpus)",
    )
    axes[0].set_ylim(-0.05, 1.08)
    axes[0].set_ylabel("GPT-4o-mini Score (0–1)")
    axes[0].set_title("(a) Per-Question Answer Correctness Across Conditions (Q01–Q25)")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(loc="lower left", ncol=2, framealpha=0.92)

    bar_colors = [
        COLORS["teal"] if d > 1e-9 else (COLORS["vermilion"] if d < -1e-9 else COLORS["gray"])
        for d in diffs_15_nr
    ]
    bars = axes[1].bar(xq, diffs_15_nr, width=0.62, color=bar_colors)
    axes[1].axhline(0.0, color="black", linewidth=0.9)
    axes[1].axvspan(20.5, 23.5, color=COLORS["yellow"], alpha=0.22)
    for bar, d in zip(bars, diffs_15_nr):
        if abs(d) > 1e-9:
            axes[1].text(
                bar.get_x() + bar.get_width() / 2.0,
                d + (0.03 if d > 0 else -0.08),
                f"{d:+.1f}",
                ha="center",
                va="bottom" if d > 0 else "top",
                fontsize=FONT_SIZE - 2.0,
                fontweight="bold",
            )
    axes[1].set_ylim(-1.15, 1.18)
    axes[1].set_xticks(xq)
    axes[1].set_xticklabels(qids)
    axes[1].set_ylabel("Score Change\n(15-Doc RAG − No RAG)")
    axes[1].set_xlabel("Benchmark Question ID")
    axes[1].set_title("(b) Per-Question Paired Gain from Full-Corpus 15-Doc RAG")
    axes[1].grid(True, linestyle="--", alpha=0.35)

    return save_figure(fig, Path(output_dir) / "question_differences", export_png=export_png)


# ---------------------------------------------------------------------------
# Figure 4: Scientific Tool-Calling Benchmark Performance
# ---------------------------------------------------------------------------
def plot_tool_performance(results_dir: Path = RESULTS_DIR, output_dir: Path = OUTPUT_DIR, export_png: bool = True):
    """Plot tool-calling benchmark accuracy across supported and invalid requests."""
    apply_style()
    summary = json.loads(
        (Path(results_dir) / "tool_summary.json").read_text(encoding="utf-8")
    )

    items = [
        ("Tool Selection\n(Supported Tasks)", summary["tool_selection"], COLORS["teal"]),
        (
            "NOAA Numerical Agreement\n(Authoritative Reference)",
            summary["tool_numerical_agreement"],
            COLORS["teal"],
        ),
        (
            "Returned-Model Attribution\n(WMM / IGRF / DGRF)",
            summary["model_attribution_presence"],
            COLORS["teal"],
        ),
        (
            "Final-Answer Numbers\n(Within Tolerance)",
            summary["final_number_presence"],
            COLORS["sky"],
        ),
        (
            "Strict Endpoint Parameters\n(YYYY-01-01 Convention)",
            summary["parameters_and_order"],
            COLORS["navy"],
        ),
        (
            "Invalid-Request Rejection\n(Out-of-Range Dates / Coords)",
            summary["invalid_request_handling"],
            COLORS["amber"],
        ),
    ]

    fig, ax = plt.subplots(figsize=FIGSIZE_TOOLS, layout="constrained")
    y = np.arange(len(items))[::-1]
    rates = [item[1]["rate"] * 100.0 for item in items]
    colors = [item[2] for item in items]

    bars = ax.barh(y, rates, height=0.58, color=colors)
    for bar, (_, spec, _) in zip(bars, items):
        rate = spec["rate"] * 100.0
        passed = spec["passed"]
        total = spec["n"]
        ax.text(
            rate + 1.5,
            bar.get_y() + bar.get_height() / 2.0,
            f"{passed}/{total} ({rate:.1f}%)",
            va="center",
            ha="left",
            fontsize=FONT_SIZE - 1,
            fontweight="bold",
        )

    ax.set_yticks(y)
    ax.set_yticklabels([item[0] for item in items])
    ax.set_xlim(0, 118)
    ax.set_xlabel("Pass Rate (%)")
    ax.set_title("Scientific Tool-Calling Benchmark Performance (GPT-OSS 20B, 20 Prompts)")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    return save_figure(fig, Path(output_dir) / "tool_performance", export_png=export_png)


# ---------------------------------------------------------------------------
# Figure 5: London Total Magnetic Intensity Demonstration (1999-2026)
# ---------------------------------------------------------------------------
def plot_london_intensity(results_dir: Path = RESULTS_DIR, output_dir: Path = OUTPUT_DIR, export_png: bool = True):
    """Plot the London total magnetic field intensity series (1999-2026, IGRF2025)."""
    apply_style()
    rows = read_csv_rows(Path(results_dir) / "london_intensity_series.csv")
    summary = json.loads(
        (Path(results_dir) / "london_intensity_summary.json").read_text(encoding="utf-8")
    )

    years = np.array([float(row["decimal_year"]) for row in rows])
    intensities = np.array([float(row["total_intensity_nT"]) for row in rows])

    fig, ax = plt.subplots(figsize=FIGSIZE_LONDON, layout="constrained")
    ax.plot(
        years,
        intensities,
        marker="o",
        markersize=4.5,
        color=COLORS["navy"],
        label=f"NOAA {summary['returned_model']} (Annual Jan 1 Estimates)",
    )

    # Endpoint linear trend line for visual comparison
    endpoint_line = intensities[0] + summary["endpoint_nT_per_year"] * (years - years[0])
    ax.plot(
        years,
        endpoint_line,
        linestyle="--",
        color=COLORS["vermilion"],
        linewidth=1.4,
        label=(
            f"Endpoint Trend: +{summary['endpoint_nT_per_year']:.2f} nT/yr "
            f"(+{summary['net_change_nT']:.1f} nT total)"
        ),
    )

    ax.annotate(
        f"{int(years[0])}: {intensities[0]:,.1f} nT",
        xy=(years[0], intensities[0]),
        xytext=(years[0] + 1.2, intensities[0] - 95),
        arrowprops={"arrowstyle": "->", "color": COLORS["navy"], "lw": 1.0},
        fontsize=FONT_SIZE - 1,
        fontweight="bold",
    )
    ax.annotate(
        f"{int(years[-1])}: {intensities[-1]:,.1f} nT",
        xy=(years[-1], intensities[-1]),
        xytext=(years[-1] - 7.5, intensities[-1] + 55),
        arrowprops={"arrowstyle": "->", "color": COLORS["navy"], "lw": 1.0},
        fontsize=FONT_SIZE - 1,
        fontweight="bold",
    )

    ax.set_xlim(1998, 2027)
    ax.set_ylim(48140, 49220)
    ax.set_xlabel("Year (Annual January 1 Epoch)")
    ax.set_ylabel("Total Magnetic Field Intensity F (nT)")
    ax.set_title(
        f"London ({summary['latitude']:.2f}°N, {abs(summary['longitude']):.2f}°W) "
        f"Total Intensity ({int(years[0])}–{int(years[-1])}, {summary['returned_model']})"
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", framealpha=0.94)

    return save_figure(fig, Path(output_dir) / "london_intensity", export_png=export_png)


FIGURE_BUILDERS = {
    "retrieval": plot_retrieval,
    "rag": plot_rag_comparison,
    "questions": plot_question_differences,
    "tools": plot_tool_performance,
    "london": plot_london_intensity,
}


def generate_plots(
    figure: str = "all",
    results_dir: Path = RESULTS_DIR,
    output_dir: Path = OUTPUT_DIR,
    export_png: bool = True,
):
    selected = (
        list(FIGURE_BUILDERS.keys()) if figure == "all" else [figure]
    )
    generated = []
    for name in selected:
        builder = FIGURE_BUILDERS[name]
        paths = builder(results_dir=results_dir, output_dir=output_dir, export_png=export_png)
        generated.extend(paths)
    return generated


def main():
    parser = argparse.ArgumentParser(
        description="Generate ChatGeo-Magi publication figures from evaluation/results/."
    )
    parser.add_argument(
        "--figure",
        choices=["all", *FIGURE_BUILDERS.keys()],
        default="all",
        help="Generate all figures (default) or a single named figure.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Path to the checked-in results directory (default: evaluation/results).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where generated plots will be saved (default: evaluation/plot_outputs).",
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Export only vector PDFs without high-resolution PNG copies.",
    )
    args = parser.parse_args()
    saved = generate_plots(
        figure=args.figure,
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        export_png=not args.no_png,
    )
    for path in saved:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
