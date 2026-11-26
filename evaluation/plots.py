import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
OUTPUT_DIR = EVAL_DIR / "plot_outputs"

DPI = 400
FONT_SIZE = 10
LABEL_SIZE = 10

COLORS = {
    "navy": "#0072B2",
    "teal": "#009E73",
    "amber": "#E69F00",
    "vermilion": "#D55E00",
    "gray": "#7F7F7F",
}

FIGSIZE_RETRIEVAL = (4.6, 3.6)
FIGSIZE_RAG = (5.0, 3.9)
FIGSIZE_QUESTIONS = (10.0, 3.6)


def apply_style():
    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.labelsize": LABEL_SIZE,
            "xtick.labelsize": FONT_SIZE - 1,
            "ytick.labelsize": FONT_SIZE - 1,
            "legend.fontsize": FONT_SIZE - 1.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
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


def plot_retrieval(results_dir: Path = RESULTS_DIR, output_dir: Path = OUTPUT_DIR, export_png: bool = True):
    apply_style()
    summary_rows = {
        row["run_id"]: row
        for row in read_csv_rows(Path(results_dir) / "retrieval_summary.csv")
    }

    full15 = summary_rows["expanded_stella_s2p_cosine_full25"]
    held15 = summary_rows["expanded_stella_s2p_cosine_heldout14"]

    entries = [
        (
            f"Primary (n={int(full15['scored_questions'])})",
            int(full15["hits_at_4"]),
            int(full15["scored_questions"]),
            float(full15["hit_at_4"]) * 100.0,
            COLORS["navy"],
        ),
        (
            f"Held-out (n={int(held15['scored_questions'])})",
            int(held15["hits_at_4"]),
            int(held15["scored_questions"]),
            float(held15["hit_at_4"]) * 100.0,
            COLORS["teal"],
        ),
    ]

    fig, ax = plt.subplots(figsize=FIGSIZE_RETRIEVAL, layout="constrained")
    x = np.arange(len(entries))
    rates = [item[3] for item in entries]
    colors = [item[4] for item in entries]

    ax.set_axisbelow(True)
    bars = ax.bar(x, rates, width=0.5, color=colors)
    for bar, (_, hits, total, rate, _) in zip(bars, entries):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            rate + 1.8,
            f"{hits}/{total} ({rate:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE - 0.5,
            fontweight="bold",
            color="#222222",
        )

    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xticks(x)
    ax.set_xticklabels([item[0] for item in entries])
    ax.set_ylabel("Top-4 retrieval hit rate (%)")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    return save_figure(fig, Path(output_dir) / "retrieval_performance", export_png=export_png)


def plot_rag_comparison(results_dir: Path = RESULTS_DIR, output_dir: Path = OUTPUT_DIR, export_png: bool = True):
    apply_style()
    s_all25 = json.loads(
        (Path(results_dir) / "rag_statistical_summary.json").read_text(encoding="utf-8")
    )["primary_openai_judge"]["all_25_questions"]

    conditions = [
        ("No RAG", s_all25["no_rag"], COLORS["navy"]),
        ("15-doc RAG", s_all25["rag_15doc_corrected"], COLORS["teal"]),
    ]
    paired = s_all25["paired_15doc_minus_no_rag"]

    fig, ax = plt.subplots(figsize=FIGSIZE_RAG, layout="constrained")
    x = np.arange(len(conditions))
    means = [spec["mean"] for _, spec, _ in conditions]
    lows = [spec["mean"] - spec["ci95_low"] for _, spec, _ in conditions]
    highs = [spec["ci95_high"] - spec["mean"] for _, spec, _ in conditions]
    colors = [color for _, _, color in conditions]

    ax.set_axisbelow(True)
    bars = ax.bar(
        x,
        means,
        width=0.48,
        yerr=[lows, highs],
        capsize=5,
        color=colors,
        error_kw={"ecolor": "#222222", "elinewidth": 1.3, "capthick": 1.2},
    )

    for bar, (_, spec, _) in zip(bars, conditions):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            spec["ci95_low"] - 0.035,
            f"{spec['mean']:.3f}",
            ha="center",
            va="top",
            fontsize=FONT_SIZE,
            fontweight="bold",
            color="white",
        )

    top_ci = max(spec["ci95_high"] for _, spec, _ in conditions)
    bracket_y = top_ci + 0.042
    bracket_h = 0.018
    ax.plot(
        [0, 0, 1, 1],
        [bracket_y - bracket_h, bracket_y, bracket_y, bracket_y - bracket_h],
        color="#222222",
        linewidth=1.1,
    )
    ax.text(
        0.5,
        bracket_y + 0.016,
        (
            f"Mean Diff = {paired['mean_diff']:+.3f} "
            f"(95% CI [{paired['ci95_low']:+.3f}, {paired['ci95_high']:+.3f}])"
        ),
        ha="center",
        va="bottom",
        fontsize=FONT_SIZE - 0.8,
        fontweight="bold",
        color="#222222",
    )

    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _, _ in conditions])
    ax.set_ylabel("Mean correctness score")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    return save_figure(fig, Path(output_dir) / "rag_comparison", export_png=export_png)


def plot_question_differences(results_dir: Path = RESULTS_DIR, output_dir: Path = OUTPUT_DIR, export_png: bool = True):
    apply_style()
    question_rows = read_csv_rows(Path(results_dir) / "rag_question_scores.csv")

    qids = [row["id"] for row in question_rows]
    xq = np.arange(len(qids))
    diffs_15_nr = [float(row["diff_15doc_minus_norag_gpt4omini"]) for row in question_rows]

    fig, ax = plt.subplots(figsize=FIGSIZE_QUESTIONS, layout="constrained")
    ax.set_axisbelow(True)

    bar_colors = [
        COLORS["teal"] if d > 1e-9 else (COLORS["vermilion"] if d < -1e-9 else COLORS["gray"])
        for d in diffs_15_nr
    ]
    bars = ax.bar(xq, diffs_15_nr, width=0.62, color=bar_colors, zorder=2)
    ax.axhline(0.0, color="#222222", linewidth=1.0, zorder=3)

    for bar, d in zip(bars, diffs_15_nr):
        if abs(d) > 1e-9:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                d + (0.025 if d > 0 else -0.025),
                f"{d:+.1f}",
                ha="center",
                va="bottom" if d > 0 else "top",
                fontsize=FONT_SIZE - 2.0,
                fontweight="bold",
                color="#222222",
            )

    legend_handles = [
        mpatches.Patch(facecolor=COLORS["teal"], label="Improved"),
        mpatches.Patch(facecolor=COLORS["vermilion"], label="Worsened"),
    ]

    ax.set_xlim(-0.7, len(qids) - 0.3)
    ax.set_ylim(-0.46, 1.16)
    ax.set_yticks([-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticks(xq)
    ax.set_xticklabels(qids, fontsize=FONT_SIZE - 1.5)
    ax.set_xlabel("Question")
    ax.set_ylabel("Score diff (15-doc RAG & No RAG)")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        ncol=len(legend_handles),
        frameon=False,
    )

    return save_figure(fig, Path(output_dir) / "question_differences", export_png=export_png)


FIGURE_BUILDERS = {
    "retrieval": plot_retrieval,
    "rag": plot_rag_comparison,
    "questions": plot_question_differences,
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
