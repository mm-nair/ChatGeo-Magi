import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


from golden_dataset import load_goldens


def bootstrap(differences, seed=20261115, resamples=100000):
    differences = np.asarray(differences, dtype=float)
    if not len(differences) or not np.isfinite(differences).all():
        raise ValueError('Bootstrap needs finite paired observations')
    generator = np.random.default_rng(seed)
    means = differences[generator.integers(0, len(differences), (resamples, len(differences)))].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(output):
    output.mkdir(parents=True, exist_ok=True)
    goldens = load_goldens(ROOT / 'evaluation/goldens.csv')
    questions = list(goldens)
    groups = {}
    validation = []
    manifest = {}
    for path in sorted((ROOT / 'evaluation/opik_evaluation_results').glob('*.json')):
        manifest[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        records = json.loads(path.read_text())
        model = records[0]['dataset.model_name']
        condition = 'noemails' if 'noemails' in path.name else 'emails'
        source = ROOT / 'evaluation/inference_results' / records[0]['dataset.source_file']
        manifest[str(source.relative_to(ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
        with source.open(newline='') as stream:
            inference = list(csv.DictReader(stream))
        if len({row['question'] for row in inference}) != len(inference):
            raise ValueError('Duplicate inference question')
        if any(int(row['id']) != questions.index(row['question']) + 1 for row in inference):
            raise ValueError('Inference identifier does not match golden question order')
        inference = {row['question']: row for row in inference}
        mapped = {}
        for record in records:
            if record['dataset.model_name'] != model or record['dataset.source_file'] != source.name:
                raise ValueError('Mixed model or source within export')
            question = record['dataset.input']
            score = record['feedback_scores.g_eval_metric']
            if question in mapped or question not in goldens:
                raise ValueError('Duplicate or unknown exported question')
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('Invalid score; inspect source rather than silently excluding')
            if record['dataset.expected_answer'] != goldens[question]:
                raise ValueError('Reference mismatch')
            response = record['dataset.model_output']
            if not isinstance(response, str) or response != inference[question][model]:
                raise ValueError('Inference response mismatch')
            validation.append({'model': model, 'condition': condition, 'question_id': questions.index(question) + 1,
                               'output_matches_dataset': record['output.output'] == response,
                               'output_ends_ellipsis': str(record['output.output']).endswith('...'),
                               'empty': not response.strip(), 'tool_marker': '[TOOL_CALLS]' in response})
            mapped[question] = record
        if set(mapped) != set(goldens) or set(inference) != set(goldens):
            raise ValueError('Missing questions')
        if (model, condition) in groups:
            raise ValueError('Duplicate condition')
        groups[model, condition] = mapped
    if len(groups) != 6:
        raise ValueError('Expected six conditions')
    summaries, question_rows, descriptions = [], [], []
    for model in sorted({key[0] for key in groups}):
        values = {}
        for condition in ['noemails', 'emails']:
            values[condition] = np.array([groups[model, condition][q]['feedback_scores.g_eval_metric'] for q in questions])
            scores = values[condition]
            descriptions.append({'model': model, 'condition': condition, 'n': len(scores), 'mean': scores.mean(),
                                 'sd': scores.std(ddof=1), 'median': np.median(scores), 'min': scores.min(), 'max': scores.max(),
                                 'ceiling_count': int((scores == 1).sum()), 'distribution': json.dumps({str(v): int((scores == v).sum()) for v in np.unique(scores)})})
        differences = values['emails'] - values['noemails']
        low, high = bootstrap(differences)
        valid = np.array([all(groups[model, c][q]['dataset.model_output'].strip() for c in ['emails', 'noemails']) for q in questions])
        sensitivity = differences[valid]
        sensitivity_low, sensitivity_high = bootstrap(sensitivity)
        summaries.append({'model': model, 'n': len(questions), 'noemails_mean': values['noemails'].mean(),
                          'emails_mean': values['emails'].mean(), 'mean_difference': differences.mean(), 'ci_low': low, 'ci_high': high,
                          'nonempty_pairs': int(valid.sum()), 'nonempty_noemails_mean': values['noemails'][valid].mean(),
                          'nonempty_emails_mean': values['emails'][valid].mean(), 'nonempty_difference': sensitivity.mean(),
                          'nonempty_ci_low': sensitivity_low, 'nonempty_ci_high': sensitivity_high})
        for i, question in enumerate(questions):
            question_rows.append({'model': model, 'question_id': i + 1, 'question': question,
                                  'noemails_score': values['noemails'][i], 'emails_score': values['emails'][i],
                                  'difference': differences[i],
                                  'noemails_empty': not groups[model, 'noemails'][question]['dataset.model_output'].strip(),
                                  'emails_empty': not groups[model, 'emails'][question]['dataset.model_output'].strip()})
    write_csv(output / 'model_summary.csv', summaries)
    write_csv(output / 'question_analysis.csv', question_rows)
    write_csv(output / 'score_distribution.csv', descriptions)
    write_csv(output / 'validation.csv', validation)
    manifest['evaluation/goldens.csv'] = hashlib.sha256((ROOT / 'evaluation/goldens.csv').read_bytes()).hexdigest()
    (output / 'input_manifest.json').write_text(json.dumps(manifest, indent=2))
    (output / 'analysis_config.json').write_text(json.dumps({'seed': 20261115, 'resamples': 100000, 'interval': 'paired percentile bootstrap', 'numpy': np.__version__, 'matplotlib': matplotlib.__version__}, indent=2))
    plots = output / 'plots'
    plots.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42})
    labels = [r['model'] for r in summaries]
    positions = np.arange(len(labels))
    figure, axes = plt.subplots(figsize=(8, 4.5), layout='constrained')
    for offset, condition, color in [(-.18, 'noemails', '#0072B2'), (.18, 'emails', '#D55E00')]:
        axes.bar(positions + offset, [r[condition + '_mean'] for r in summaries], width=.36, label=condition, color=color)
    axes.set(ylim=(0, 1), ylabel='Mean G-Eval score', xticks=positions, xticklabels=labels)
    axes.legend(loc='lower right')
    save_plot(figure, plots / 'mean_scores')
    figure, axes = plt.subplots(figsize=(8, 4.5), layout='constrained')
    for i, row in enumerate(summaries):
        axes.errorbar(row['mean_difference'], i, xerr=[[row['mean_difference'] - row['ci_low']], [row['ci_high'] - row['mean_difference']]], fmt='o', color='#0072B2', capsize=4)
    axes.axvline(0, color='black', linewidth=.8)
    axes.set(xlim=(-1, 1), yticks=positions, yticklabels=labels, xlabel='Email minus no-email score (95% paired bootstrap CI)')
    save_plot(figure, plots / 'paired_differences')
    figure, axes = plt.subplots(figsize=(11, 3.4), layout='constrained')
    matrix = [[r['difference'] for r in question_rows if r['model'] == model] for model in labels]
    mesh = axes.imshow(matrix, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
    axes.set(yticks=positions, yticklabels=labels, xticks=np.arange(len(questions)), xticklabels=np.arange(1, len(questions) + 1), xlabel='Question number in original headerless goldens.csv')
    figure.colorbar(mesh, ax=axes, label='Email minus no-email score')
    save_plot(figure, plots / 'question_differences')
    lines = ['# Historical paired analysis', '', 'Difference is email minus no-email. All 25 question pairs per model are retained, including empty responses. Exact question/reference/inference matching passed. No missing or invalid scores were silently dropped.', '', '100,000 paired percentile bootstrap resamples, seed 20261115. These intervals describe question resampling conditional on one saved run and judge per condition. If the historical runs used the checked-in shared-session behavior, question independence fails; intervals are descriptive and may understate uncertainty. They do not establish equivalence, causal email effects, or population generalization.', '']
    for row in summaries:
        lines.append(f"- {row['model']}: no-email {row['noemails_mean']:.6f}, email {row['emails_mean']:.6f}, difference {row['mean_difference']:+.6f}, 95% CI [{row['ci_low']:.6f}, {row['ci_high']:.6f}]. Nonempty-pair sensitivity n={row['nonempty_pairs']}: difference {row['nonempty_difference']:+.6f}, CI [{row['nonempty_ci_low']:.6f}, {row['nonempty_ci_high']:.6f}].")
    lines += ['', '## Largest changes and lowest scores', '']
    for model in labels:
        selected = [r for r in question_rows if r['model'] == model]
        lines.append(f'### {model}')
        for label, candidates in [('Largest improvement', [r for r in selected if r['difference'] > 0]), ('Largest regression', [r for r in selected if r['difference'] < 0])]:
            if candidates:
                row = max(candidates, key=lambda r: abs(r['difference']))
                lines.append(f"- {label}: Q{row['question_id']}, {row['difference']:+.6f}.")
            else:
                lines.append(f'- {label}: none.')
        for row in sorted(selected, key=lambda r: -abs(r['difference']))[:5]:
            lines.append(f"- Q{row['question_id']}: {row['question']} No-email {row['noemails_score']:.6f}; email {row['emails_score']:.6f}; difference {row['difference']:+.6f}.")
        for row in sorted(selected, key=lambda r: min(r['emails_score'], r['noemails_score']))[:3]:
            lines.append(f"- Low score Q{row['question_id']}: no-email {row['noemails_score']:.6f}, email {row['emails_score']:.6f}.")
    lines += ['', '## Cross-model disagreement', '']
    for question in questions:
        selected = [r for r in question_rows if r['question'] == question]
        if min(r['difference'] for r in selected) < 0 < max(r['difference'] for r in selected):
            lines.append(f"- Q{selected[0]['question_id']}: " + '; '.join(f"{r['model']} {r['difference']:+.6f}" for r in selected))
    lines += ['', 'See [outlier inspection](OUTLIERS.md), [distributions](score_distribution.csv), [validation](validation.csv), and [plots](plots/). Full response text is deliberately not copied into handoff artifacts.']
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(summaries, indent=2))
    print('Output/dataset mismatches:', sum(not r['output_matches_dataset'] for r in validation))


def save_plot(figure, path):
    figure.savefig(path.with_suffix('.pdf'))
    figure.savefig(path.with_suffix('.png'), dpi=400)
    plt.close(figure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'revision_workspace/statistical_analysis')
    arguments = parser.parse_args()
    if not arguments.output.resolve().is_relative_to((ROOT / 'revision_workspace').resolve()):
        parser.error('Analysis artifacts must stay in revision_workspace')
    analyze(arguments.output)
