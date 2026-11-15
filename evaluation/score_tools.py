import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark_metrics import tool_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--traces', required=True, type=Path)
    parser.add_argument('--expected', type=Path, default=ROOT / 'evaluation/benchmarks/tool_expected.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'revision_workspace/tool_evaluation/scored')
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT / 'revision_workspace'):
        parser.error('Output must remain in revision_workspace')
    expected = json.loads(args.expected.read_text())
    rows = []
    for question in expected:
        path = args.traces / f"{question['id']}_tools_only.json"
        if not path.exists():
            rows.append({'id': question['id'], 'status': 'missing_trace'})
        else:
            trace = json.loads(path.read_text())
            if trace['id'] != question['id']:
                raise ValueError('Trace identifier mismatch')
            rows.append({'id': question['id'], 'status': 'scored', **tool_metrics(question, trace)})
    summary = {'total_questions': len(rows), 'missing_traces': sum(row['status'] == 'missing_trace' for row in rows)}
    for metric in ['tool_selection', 'parameters_and_order', 'execution', 'numerical_agreement', 'units_and_attribution', 'end_to_end']:
        values = [row[metric] for row in rows if row.get(metric) is not None]
        summary[metric] = {'n': len(values), 'mean': sum(values) / len(values) if values else None}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'per_question.json').write_text(json.dumps(rows, indent=2))
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
