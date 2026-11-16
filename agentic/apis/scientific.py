import json
import math
import re
from datetime import date


MODEL_SOURCE = 'https://www.ngdc.noaa.gov/geomag/calculators/help/igrfwmmHelp.html'


def parse_date(time_str=None):
    if time_str is None or str(time_str).lower() in ('now', 'today'):
        return date.today()
    value = str(time_str)
    if re.fullmatch(r'\d{4}', value):
        return date(int(value), 1, 1)
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
        raise ValueError('Use YYYY or YYYY-MM-DD; ambiguous dates are not accepted')
    return date.fromisoformat(value)


def select_model(start_time=None, end_time=None):
    start_date = parse_date(start_time)
    end_date = parse_date(end_time) if end_time is not None else start_date
    if end_date < start_date:
        raise ValueError('End date precedes start date')
    if start_date < date(1900, 1, 1) or end_date >= date(2030, 1, 1):
        raise ValueError('Supported dates: 1900-01-01 through 2029-12-31; pre-1900 gufm1 and later extrapolation are unsupported')
    model_type = 'WMM' if start_date >= date(2025, 1, 1) else 'IGRF'
    return model_type, start_date, end_date


def coordinates(location):
    if ',' not in location:
        return None
    if not re.match(r'^\s*[+\-]?(?:\d|\.|nan\b|inf\b)', location, re.IGNORECASE):
        return None
    values = location.split(',')
    try:
        latitude, longitude = [float(value.strip()) for value in values]
    except ValueError as error:
        raise ValueError('Coordinates must be latitude,longitude in decimal degrees') from error
    if not math.isfinite(latitude) or not math.isfinite(longitude) or not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError('Coordinates outside latitude/longitude bounds')
    return latitude, longitude


def provenance(payload, model_type, start_date, end_date, latitude, longitude):
    if not isinstance(payload, dict) or payload.get('error') or payload.get('errors'):
        raise ValueError('Calculator returned an error or invalid JSON structure')
    if not isinstance(payload.get('result'), list) or not payload['result']:
        raise ValueError('Calculator returned no data')
    if any(not isinstance(row, dict) for row in payload['result']):
        raise ValueError('Calculator returned invalid result rows')
    return {
        'source': MODEL_SOURCE,
        'requested_model': model_type,
        'returned_model': payload.get('model', 'not supplied'),
        'returned_model_metadata': {key: value for key, value in payload.items() if any(word in key.lower() for word in ['model', 'epoch', 'valid', 'uncert'])},
        'policy_validity': ['2025-01-01' if model_type == 'WMM' else '1900-01-01', '2029-12-31'],
        'calculator_version': payload.get('version', 'not supplied'),
        'epoch': payload.get('epoch', 'not supplied'),
        'uncertainty': payload.get('uncertainty', 'see row-level uncertainty metadata' if any(any('uncert' in key.lower() for key in row) for row in payload['result']) else 'not supplied; no uncertainty inferred'),
        'requested_start': start_date.isoformat(),
        'requested_end': end_date.isoformat(),
        'latitude': latitude,
        'longitude': longitude,
        'elevation_km': 0,
        'elevation_reference': 'MSL',
        'interpretation': 'Model estimates, not observations. IGRF 2025 onward and WMM are predictive. Requested family does not establish the server model version.',
    }


def summarize_component(payload, data_name):
    units_metadata = payload['units'] if isinstance(payload.get('units'), dict) else {}
    units = units_metadata.get(data_name, 'not supplied')
    rows = []
    for entry in payload['result']:
        value = entry.get(data_name)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError('Calculator returned missing or nonfinite component')
        rows.append({'date': entry.get('date', entry.get('year', 'not supplied')), 'value': value,
                     'model': entry.get('model', payload.get('model', 'not supplied')),
                     'uncertainty_metadata': {key: value for key, value in entry.items() if any(word in key.lower() for word in ['uncert', 'error', 'sigma'])}})
    reported_units = units
    if data_name.endswith('_sv') and units != 'not supplied' and 'year' not in units.lower():
        units = f'{units} per year'
    result = {'component': data_name, 'units': units, 'reported_units': reported_units, 'values': rows, 'provenance': payload['provenance'], 'uncertainty_units': {key: value for key, value in units_metadata.items() if 'uncert' in key.lower()}}
    if len(rows) > 1 and data_name == 'totalintensity':
        times = [decimal_year(row['date']) for row in rows]
        if all(value is not None for value in times) and all(b > a for a, b in zip(times, times[1:])):
            result['endpoint_average_change'] = {'per_year': (rows[-1]['value'] - rows[0]['value']) / (times[-1] - times[0]), 'units': f'{units} per year', 'definition': 'Last minus first divided by elapsed decimal years; not a fitted slope or instantaneous secular variation'}
        else:
            result['endpoint_average_change'] = 'unavailable: missing or non-increasing dates'
    return json.dumps(result, allow_nan=False)


def decimal_year(value):
    try:
        if isinstance(value, (float, int)) or re.fullmatch(r'\d{4}(\.\d+)?', str(value)):
            result = float(value)
            return result if math.isfinite(result) else None
        parsed = date.fromisoformat(str(value))
        return parsed.year + (parsed - date(parsed.year, 1, 1)).days / (date(parsed.year + 1, 1, 1) - date(parsed.year, 1, 1)).days
    except (ValueError, TypeError):
        return None
