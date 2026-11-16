import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from revision_analysis import ROOT, bootstrap, write_csv, save_plot


def summarize(records):
    summaries, paired = [], []
    conditions = ('no_rag', 'rag') if any(row['condition'] == 'rag' for row in records) else ('noemails', 'emails')
    for model in sorted({row['model'] for row in records}):
        groups = {}
        for row in records:
            if row['model'] != model:
                continue
            key = row['id'], row['condition']
            if key in groups or not isinstance(row['score'], (float, int)) or not 0 <= row['score'] <= 1:
                raise ValueError('Duplicate or invalid score')
            groups[key] = row
        identifiers = sorted({key[0] for key in groups})
        if len(identifiers) != 25 or len(groups) != 50:
            raise ValueError('Expected 25 complete pairs per model')
        model_pairs = []
        for identifier in identifiers:
            first, second = [groups[identifier, condition] for condition in conditions]
            row = {'model': model, 'id': identifier, conditions[0]: first['score'], conditions[1]: second['score'], 'difference': second['score'] - first['score'], 'empty_pair': first['empty_response'] or second['empty_response'], 'original_first': first.get('original_score'), 'original_second': second.get('original_score')}
            model_pairs.append(row)
        differences = [row['difference'] for row in model_pairs]
        low, high = bootstrap(differences)
        nonempty = [row['difference'] for row in model_pairs if not row['empty_pair']]
        summary = {'model': model, 'n': len(model_pairs), 'first_condition': conditions[0], 'second_condition': conditions[1], 'first_mean': np.mean([row[conditions[0]] for row in model_pairs]), 'second_mean': np.mean([row[conditions[1]] for row in model_pairs]), 'difference': np.mean(differences), 'ci_low': low, 'ci_high': high, 'nonempty_pairs': len(nonempty), 'nonempty_difference': np.mean(nonempty)}
        if all(row['original_first'] is not None for row in model_pairs):
            summary.update({'original_first_mean': np.mean([row['original_first'] for row in model_pairs]), 'original_second_mean': np.mean([row['original_second'] for row in model_pairs]), 'original_difference': np.mean([row['original_second'] - row['original_first'] for row in model_pairs])})
        summaries.append(summary)
        paired.extend(model_pairs)
    return summaries, paired


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'revision_workspace/corrected_judging')
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT / 'revision_workspace'):
        parser.error('Output must remain in revision_workspace')
    records = []
    for path in args.input.glob('*.json'):
        row = json.loads(path.read_text())
        if isinstance(row, dict) and 'score' in row and 'condition' in row:
            records.append(row)
    summaries, paired = summarize(records)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / 'model_summary.csv', summaries)
    write_csv(args.output / 'question_analysis.csv', paired)
    plots = args.output / 'plots'
    plots.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42})
    figure, axes = plt.subplots(figsize=(8, 4.5), layout='constrained')
    positions = np.arange(len(summaries))
    for offset, key, label, color in [(-.18, 'first_mean', summaries[0]['first_condition'], '#0072B2'), (.18, 'second_mean', summaries[0]['second_condition'], '#D55E00')]:
        axes.bar(positions + offset, [row[key] for row in summaries], width=.36, label=label, color=color)
    axes.set(ylim=(0, 1), xticks=positions, xticklabels=[row['model'] for row in summaries], ylabel='Local reference-judge score')
    axes.legend(loc='lower right')
    save_plot(figure, plots / 'reference_scores')
    figure, axes = plt.subplots(figsize=(8, 4.5), layout='constrained')
    for index, row in enumerate(summaries):
        axes.errorbar(row['difference'], index, xerr=[[row['difference'] - row['ci_low']], [row['ci_high'] - row['difference']]], fmt='o', capsize=4)
    axes.axvline(0, color='black', linewidth=.8)
    axes.set(xlim=(-1, 1), yticks=positions, yticklabels=[row['model'] for row in summaries], xlabel='Second minus first condition (95% paired bootstrap CI)')
    save_plot(figure, plots / 'paired_differences')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
