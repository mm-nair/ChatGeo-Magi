import argparse
from datetime import date
import json
from pathlib import Path

import requests

from vm_support import load_existing_credentials

ROOT = Path(__file__).resolve().parents[1]
CALCULATOR = 'https://www.ngdc.noaa.gov/geomag-web/calculators/calculateIgrfwmm'


def main():
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'revision_workspace/tool_evaluation/authoritative')
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT / 'revision_workspace'):
        parser.error('Output must remain in revision_workspace')
    load_existing_credentials()
    args.output.mkdir(parents=True, exist_ok=True)
    expected = json.loads((ROOT / 'evaluation/benchmarks/tool_expected.json').read_text())
    coordinate_cache = {}
    response_cache = {}
    for question in expected:
        reference_calls = []
        for call in question['calls']:
            if call['name'] != 'noaa_mag_api':
                continue
            parameters = call['args']
            location = parameters['location']
            if location not in coordinate_cache:
                if location == 'London':
                    response = requests.get('https://geocode.maps.co/search', params={'q': location, 'api_key': os.environ['GEOCODE_API_KEY']}, timeout=30)
                    response.raise_for_status()
                    geocoded = response.json()[0]
                    coordinate_cache[location] = [float(geocoded['lat']), float(geocoded['lon'])]
                else:
                    coordinate_cache[location] = [float(value) for value in location.split(',')]
            latitude, longitude = coordinate_cache[location]
            start = parameters['start_time']
            end = parameters.get('end_time', start)
            start_date = date.fromisoformat(start if len(start) > 4 else start + '-01-01')
            end_date = date.fromisoformat(end if len(end) > 4 else end + '-01-01')
            model = 'WMM' if start_date.year >= 2025 else 'IGRF'
            query = {'lat1': latitude, 'lon1': longitude, 'model': model, 'startYear': start_date.year, 'startMonth': start_date.month, 'startDay': start_date.day, 'endYear': end_date.year, 'endMonth': end_date.month, 'endDay': end_date.day, 'elevation': 0, 'elevationUnits': 'K', 'coordinateSystem': 'M', 'dateStepSize': 1, 'resultFormat': 'json'}
            cache_key = json.dumps(query, sort_keys=True)
            if cache_key not in response_cache:
                response = requests.get(CALCULATOR, params={**query, 'key': os.environ['CALC_API_KEY']}, timeout=60)
                response.raise_for_status()
                payload = response.json()
                if not payload.get('result'):
                    raise ValueError('NOAA returned no authoritative results')
                response_cache[cache_key] = payload
            payload = response_cache[cache_key]
            component = parameters['data_name']
            reference_calls.append({'args': parameters, 'query_without_key': query, 'returned_model': payload.get('model'), 'calculator_version': payload.get('version'), 'units': payload['units'][component], 'component': component, 'tolerance': .011 if component in ('declination', 'inclination', 'declination_sv', 'inclination_sv') else .11, 'values': [{'date': row['date'], 'value': row[component]} for row in payload['result']]})
        question['authoritative_calls'] = reference_calls
        question['reference_verification'] = 'authoritative_verified' if reference_calls else 'not_applicable'
        question['review_status'] = 'machine_collected_not_human_verified'
    (args.output / 'expected.json').write_text(json.dumps(expected, indent=2))
    (args.output / 'noaa_responses.json').write_text(json.dumps([{'query_without_key': json.loads(key), 'payload': payload} for key, payload in response_cache.items()], indent=2))
    (args.output / 'geocoded_coordinates.json').write_text(json.dumps(coordinate_cache, indent=2))
    print('Independent NOAA reference requests:', len(response_cache))


if __name__ == '__main__':
    main()
