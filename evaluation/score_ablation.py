import argparse
import ast
import importlib.metadata
import json
from pathlib import Path

from revision_analysis import ROOT, bootstrap, write_csv


def judge_payload(row):
    return json.dumps({'QUESTION': row['input'], 'EXPECTED_ANSWER': row['expected_answer'], 'MODEL_OUTPUT': row['model_output']}, ensure_ascii=False)


def historical_rubric():
    tree = ast.parse((ROOT / 'evaluation/eval_opik.py').read_text())
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'geval' for target in node.targets))
    return ast.literal_eval(next(keyword.value for keyword in assignment.value.keywords if keyword.arg == 'evaluation_criteria'))


def compare(rows, output):
    groups = {}
    for row in rows:
        if row['condition'] not in ('rag', 'no_rag'):
            raise ValueError('Unexpected ablation condition')
        key = row['id'], row['condition']
        if key in groups or isinstance(row['score'], bool) or not isinstance(row['score'], (int, float)) or not 0 <= row['score'] <= 1:
            raise ValueError('Duplicate or invalid ablation score')
        groups[key] = row['score']
    identifiers = sorted({key[0] for key in groups})
    if not identifiers or any((identifier, condition) not in groups for identifier in identifiers for condition in ('rag', 'no_rag')):
        raise ValueError('Missing paired score')
    paired = [{'id': identifier, 'rag': groups[identifier, 'rag'], 'no_rag': groups[identifier, 'no_rag'], 'difference': groups[identifier, 'rag'] - groups[identifier, 'no_rag']} for identifier in identifiers]
    low, high = bootstrap([row['difference'] for row in paired])
    write_csv(output / 'paired_scores.csv', paired)
    (output / 'summary.json').write_text(json.dumps({'n': len(paired), 'difference': sum(row['difference'] for row in paired) / len(paired), 'ci_low': low, 'ci_high': high, 'seed': 20261115, 'resamples': 100000}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--judge-inputs', type=Path)
    parser.add_argument('--scored', type=Path)
    parser.add_argument('--execute-judge', action='store_true')
    parser.add_argument('--judge-model', default='gpt-4o-mini')
    parser.add_argument('--output', type=Path, default=ROOT / 'revision_workspace/rag_ablation/scoring')
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT / 'revision_workspace') or args.output.exists():
        parser.error('Use a fresh output directory inside revision_workspace')
    if bool(args.judge_inputs) == bool(args.scored):
        parser.error('Supply exactly one of --judge-inputs or --scored')
    args.output.mkdir(parents=True)
    if args.scored:
        compare(json.loads(args.scored.read_text()), args.output)
        return
    inputs = json.loads(args.judge_inputs.read_text())
    if len(inputs) != 50 or len({(row['id'], row['condition']) for row in inputs}) != 50:
        raise ValueError('Expected 25 questions in each of two conditions')
    payloads = [{'id': row['id'], 'condition': row['condition'], 'payload': judge_payload(row)} for row in inputs]
    rubric = historical_rubric()
    (args.output / 'judge_payloads.json').write_text(json.dumps(payloads, indent=2))
    (args.output / 'rubric.txt').write_text(rubric)
    if not args.execute_judge:
        return
    from opik.evaluation.metrics import GEval
    metric = GEval(model=args.judge_model, task_introduction='You are evaluating factual answers.', evaluation_criteria=rubric, track=False)
    scores = []
    for row in payloads:
        result = metric.score(output=row['payload'])
        scores.append({'id': row['id'], 'condition': row['condition'], 'score': result.value, 'reason': result.reason})
        (args.output / 'scores.json').write_text(json.dumps(scores, indent=2))
    (args.output / 'judge_metadata.json').write_text(json.dumps({'judge_model': args.judge_model, 'opik_version': importlib.metadata.version('opik'), 'reference_payload_explicit': True, 'historical_comparability': False}, indent=2))
    compare(scores, args.output)


if __name__ == '__main__':
    main()
