import math


def retrieval_metrics(expected, retrieved, k=4):
    if expected.get('verification_status') not in ('human_verified', 'source_verified'):
        return {'status': 'unannotated', 'hit_at_4': None, 'recall_at_4': None, 'precision_at_4': None, 'support_coverage': None}
    relevant = set(expected.get('relevant_chunk_ids', []))
    if not relevant:
        raise ValueError('Verified annotation requires relevant chunk IDs')
    ranked = [row['chunk_id'] for row in retrieved[:k]]
    if len(ranked) != len(set(ranked)):
        raise ValueError('Duplicate retrieved chunk IDs')
    hits = relevant.intersection(ranked)
    facts = expected.get('support_facts', [])
    coverage = sum(bool(set(fact['chunk_ids']).intersection(ranked)) for fact in facts) / len(facts) if facts else None
    return {'status': 'scored', 'hit_at_4': int(bool(hits)), 'recall_at_4': len(hits) / len(relevant), 'precision_at_4': len(hits) / k, 'support_coverage': coverage}


def argument_matches(name, actual, expected):
    if actual is None:
        return expected is None
    if name in ('start_time', 'end_time'):
        from agentic.apis.scientific import parse_date
        try:
            return parse_date(actual) == parse_date(expected)
        except (ValueError, TypeError):
            return False
    if isinstance(expected, (float, int)):
        return isinstance(actual, (float, int)) and not isinstance(actual, bool) and math.isclose(actual, expected, abs_tol=1e-6)
    return actual == expected


def tool_metrics(expected, trace):
    calls = trace.get('calls', [])
    required = expected['calls']
    selected = []
    previous_index = -1
    for required_call in required:
        candidates = [(index, call) for index, call in enumerate(calls) if index > previous_index and call['name'] == required_call['name']]
        match = next(((index, call) for index, call in candidates if all(argument_matches(name, call.get('args', {}).get(name), value) for name, value in required_call.get('args', {}).items())), None)
        if match is None:
            selected.append(None)
        else:
            previous_index, call = match
            selected.append(call)
    if expected.get('reject'):
        return {'tool_selection': None, 'parameters_and_order': None, 'execution': None, 'numerical_agreement': None, 'units_and_attribution': None, 'end_to_end': trace.get('review', {}).get('appropriate_rejection') if trace.get('review', {}).get('human_verified') else None}
    selection = all(any(call['name'] == required_call['name'] for call in calls) for required_call in required)
    parameters = all(call is not None for call in selected)
    execution = parameters and all(call.get('success') is True for call in selected)
    references = expected.get('reference_values', [])
    numerical = None
    if references and expected.get('reference_verification') == 'authoritative_verified':
        numerical = True
        for reference in references:
            observed = trace.get('numerical_values', {}).get(reference['name'])
            numerical = numerical and isinstance(observed, (int, float)) and not isinstance(observed, bool) and math.isfinite(observed) and abs(observed - reference['value']) <= reference['tolerance']
    review = trace.get('review', {})
    attribution = review.get('units_and_attribution') if review.get('human_verified') else None
    end_to_end = None if numerical is None or attribution is None or not review.get('human_verified') else bool(selection and parameters and execution and numerical and attribution and review.get('answer_correct') is True)
    return {'tool_selection': selection, 'parameters_and_order': parameters, 'execution': execution, 'numerical_agreement': numerical, 'units_and_attribution': attribution, 'end_to_end': end_to_end}
