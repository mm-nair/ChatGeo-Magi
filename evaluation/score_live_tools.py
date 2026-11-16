import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark_metrics import tool_metrics
from revision_analysis import write_csv

UNICODE_DASHES = str.maketrans({
    '\u2212': '-',
    '\u2010': '-',
    '\u2011': '-',
    '\u2012': '-',
    '\u2013': '-',
    '\u2014': '-',
    '\u2015': '-',
})


def normalize_text(text):
    return str(text).translate(UNICODE_DASHES)


def numbers(text):
    normalized = re.sub(r'(?<=\d)[,\u202f\u00a0](?=\d{3}(?:\D|$))', '', normalize_text(text))
    return [float(match) for match in re.findall(r'(?<![A-Za-z])[-+]?\d+(?:\.\d+)?', normalized)]


def score_question(expected, trace):
    structural = tool_metrics(expected, trace)
    result = {'id': expected['id'], 'tool_selection': structural['tool_selection'], 'parameters_and_order': structural['parameters_and_order'], 'execution': structural['execution'], 'tool_numerical_agreement': None, 'final_number_presence': None, 'model_attribution_presence': None, 'plot_data_agreement': None, 'human_review': 'pending', 'error_type': trace.get('error_type')}
    references = expected.get('authoritative_calls', [])
    if not references:
        result['rejection_review'] = 'required'
        return result
    successful = []
    for reference in references:
        matched = False
        for call in trace.get('calls', []):
            if call['name'] != 'noaa_mag_api' or call.get('args', {}).get('data_name') != reference['component']:
                continue
            try:
                output = json.loads(call['output'])
                actual = output['values']
                matched = len(actual) == len(reference['values']) and all(abs(float(left['date']) - right['date']) < 1e-6 and abs(left['value'] - right['value']) <= reference['tolerance'] for left, right in zip(actual, reference['values']))
                matched = matched and output['provenance']['returned_model'] == reference['returned_model']
            except (ValueError, KeyError, TypeError):
                matched = False
            if matched:
                break
        successful.append(matched)
    result['tool_numerical_agreement'] = all(successful)
    final = str(trace.get('final', ''))
    final_numbers = numbers(final)
    result['final_number_presence'] = all(any(abs(number - row['value']) <= reference['tolerance'] for number in final_numbers) for reference in references for row in reference['values'])
    normalized_final = re.sub(r'[\s\-_]+', '', normalize_text(final)).lower()
    result['model_attribution_presence'] = all(re.sub(r'[\s\-_]+', '', normalize_text(reference['returned_model'])).lower() in normalized_final for reference in references)
    if any(call['name'] == 'plot' for call in expected['calls']):
        reference = references[0]
        result['plot_data_agreement'] = False
        for call in trace.get('calls', []):
            if call['name'] == 'plot':
                values = call.get('args', {}).get('y', [])
                dates = call.get('args', {}).get('x', [])
                try:
                    result['plot_data_agreement'] = len(values) == len(reference['values']) == len(dates) and all(abs(value - row['value']) <= reference['tolerance'] and abs(float(year) - row['date']) < 1e-6 for year, value, row in zip(dates, values, reference['values']))
                except (ValueError, TypeError):
                    pass
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--traces', required=True, type=Path)
    parser.add_argument('--expected', type=Path, default=ROOT / 'revision_workspace/tool_evaluation/authoritative/expected.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'revision_workspace/tool_evaluation/live_scored')
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT / 'revision_workspace'):
        parser.error('Output must remain in revision_workspace')
    rows = []
    for expected in json.loads(args.expected.read_text()):
        path = args.traces / f"{expected['id']}_tools_only.json"
        if not path.exists():
            raise ValueError(f'Missing trace {expected["id"]}')
        rows.append(score_question(expected, json.loads(path.read_text())))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'per_question.json').write_text(json.dumps(rows, indent=2))
    fields = sorted({key for row in rows for key in row})
    write_csv(args.output / 'per_question.csv', [{key: row.get(key) for key in fields} for row in rows])
    summary = {}
    for metric in ['tool_selection', 'parameters_and_order', 'execution', 'tool_numerical_agreement', 'final_number_presence', 'model_attribution_presence', 'plot_data_agreement']:
        values = [row[metric] for row in rows if row[metric] is not None]
        summary[metric] = {'passed': sum(values), 'n': len(values), 'rate': sum(values) / len(values) if values else None}
    summary['note'] = 'Automated evidence checks, not human-adjudicated scientific end-to-end accuracy. Invalid prompts require separate review.'
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
