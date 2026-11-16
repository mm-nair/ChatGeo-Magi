import json
import sys
import types
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch, Mock

from agentic.apis import scientific


class ScientificTests(unittest.TestCase):
    def test_date_policy(self):
        cases = [('1900', None, 'IGRF'), ('1999', '2026', 'IGRF'), ('2024', '2025', 'IGRF'), ('2025', '2029-12-31', 'WMM'), ('2026-05-01', None, 'WMM')]
        for start, end, expected in cases:
            with self.subTest(start=start, end=end):
                self.assertEqual(scientific.select_model(start, end)[0], expected)

    def test_invalid_dates(self):
        for start, end in [('1899', None), ('2030', None), ('2025', '2030'), ('2026', '2025'), ('2025-02-29', None), ('03/04/2025', None), ('no', None)]:
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                scientific.select_model(start, end)
        self.assertEqual(scientific.parse_date('2024-02-29'), date(2024, 2, 29))
        self.assertEqual(scientific.parse_date('1999'), date(1999, 1, 1))

    def test_coordinates(self):
        self.assertEqual(scientific.coordinates('51.5,-0.12'), (51.5, -.12))
        self.assertIsNone(scientific.coordinates('London'))
        self.assertIsNone(scientific.coordinates('London, UK'))
        for location in ['91,0', '0,181', 'nan,0', '0,inf', '1,2,3']:
            with self.assertRaises(ValueError):
                scientific.coordinates(location)

    def test_provenance_and_trend(self):
        payload = {'result': [{'year': 1999, 'totalintensity': 48000}, {'year': 2026, 'totalintensity': 48810}], 'units': {'totalintensity': 'nT'}, 'model': 'IGRF-14'}
        payload['provenance'] = scientific.provenance(payload, 'IGRF', date(1999, 1, 1), date(2026, 1, 1), 51.5, -.12)
        result = json.loads(scientific.summarize_component(payload, 'totalintensity'))
        self.assertEqual(result['endpoint_average_change']['per_year'], 30)
        self.assertEqual(result['provenance']['returned_model'], 'IGRF-14')
        self.assertEqual(result['units'], 'nT')
        self.assertIn('not supplied', result['provenance']['uncertainty'])

    def test_missing_metadata(self):
        payload = {'result': [{'totalintensity': 48000}, {'totalintensity': 48001}]}
        payload['provenance'] = scientific.provenance(payload, 'WMM', date(2025, 1, 1), date(2026, 1, 1), 0, 0)
        result = json.loads(scientific.summarize_component(payload, 'totalintensity'))
        self.assertEqual(result['units'], 'not supplied')
        self.assertEqual(result['provenance']['returned_model'], 'not supplied')
        self.assertIsInstance(result['endpoint_average_change'], str)
        payload['result'][0]['totalintensity'] = float('nan')
        with self.assertRaises(ValueError):
            scientific.summarize_component(payload, 'totalintensity')

    def test_captured_noaa_schema(self):
        payload = json.loads((Path(__file__).parent / 'fixtures/noaa_wmm2025.json').read_text())
        payload['provenance'] = scientific.provenance(payload, 'WMM', date(2026, 1, 1), date(2026, 1, 1), 51.5, -.12)
        result = json.loads(scientific.summarize_component(payload, 'totalintensity_sv'))
        self.assertEqual(result['values'][0]['value'], 36.5)
        self.assertEqual(result['units'], 'nanoTesla (nT) per year')
        self.assertEqual(result['reported_units'], 'nanoTesla (nT)')
        self.assertEqual(result['provenance']['calculator_version'], '1.2.1')
        self.assertEqual(result['provenance']['returned_model'], 'WMM-2025')
        self.assertEqual(result['values'][0]['uncertainty_metadata']['totalintensity_uncertainty'], 138)
        self.assertEqual(result['uncertainty_units']['totalintensity_uncertainty'], 'nanoTesla (nT)')
        self.assertEqual(result['provenance']['uncertainty'], 'see row-level uncertainty metadata')

    def test_invalid_payload(self):
        for payload in [{}, {'result': []}, {'error': 'failure'}, [], {'result': [1]}]:
            with self.assertRaises(ValueError):
                scientific.provenance(payload, 'WMM', date(2025, 1, 1), date(2025, 1, 1), 0, 0)


class WrapperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        module = types.ModuleType('langchain_core.tools')
        module.tool = lambda function: function
        with patch.dict(sys.modules, {'langchain_core': types.ModuleType('langchain_core'), 'langchain_core.tools': module}):
            from agentic.apis import noaa_apis
            cls.api = noaa_apis

    def test_request_and_single_date(self):
        response = Mock()
        response.json.return_value = {'result': [{'date': '2026-01-01', 'declination': 1.2}], 'units': {'declination': 'degrees'}, 'model': 'WMM2025'}
        with patch.object(self.api, 'CALC_API_KEY', 'test-placeholder'), patch.object(self.api.requests, 'get', return_value=response) as request:
            result = json.loads(self.api.noaa_mag_api('declination', '51.5,-0.12', '2026'))
            params = request.call_args.kwargs['params']
            self.assertEqual(params['startYear'], params['endYear'])
            self.assertEqual(params['model'], 'WMM')
            self.assertEqual(request.call_args.kwargs['timeout'], 30)
            self.assertEqual(result['values'][0]['date'], '2026-01-01')
            self.assertNotIn('test-placeholder', json.dumps(result))

    def test_crossing_range(self):
        response = Mock()
        response.json.return_value = {'result': [{'year': 1999, 'totalintensity': 48000}]}
        with patch.object(self.api, 'CALC_API_KEY', 'test-placeholder'), patch.object(self.api.requests, 'get', return_value=response) as request:
            success, payload = self.api.fetch_field_data('0,0', '1999', '2026')
            self.assertTrue(success)
            self.assertEqual(request.call_args.kwargs['params']['model'], 'IGRF')
            self.assertEqual(payload['provenance']['returned_model'], 'not supplied')

    def test_failures(self):
        for failure in [self.api.requests.Timeout('secret-containing-url'), self.api.requests.HTTPError('secret-containing-url')]:
            with patch.object(self.api, 'CALC_API_KEY', 'test-placeholder'), patch.object(self.api.requests, 'get', side_effect=failure):
                success, message = self.api.fetch_field_data('0,0', '2026')
                self.assertFalse(success)
                self.assertNotIn('secret', message)
        with patch.object(self.api.requests, 'get') as request:
            self.assertFalse(self.api.fetch_field_data('0,0', '1899')[0])
            request.assert_not_called()

    def test_bad_json_and_missing_component(self):
        response = Mock()
        response.json.side_effect = ValueError('invalid JSON')
        with patch.object(self.api, 'CALC_API_KEY', 'test-placeholder'), patch.object(self.api.requests, 'get', return_value=response):
            self.assertFalse(self.api.fetch_field_data('0,0', '2026')[0])
        response.json.side_effect = None
        response.json.return_value = {'result': [{'year': 2026}]}
        with patch.object(self.api, 'CALC_API_KEY', 'test-placeholder'), patch.object(self.api.requests, 'get', return_value=response):
            self.assertEqual(self.api.noaa_mag_api('declination', '0,0', '2026'), 'Calculator response lacks valid component data')

    def test_grid_bounds_and_provenance(self):
        response = Mock()
        response.json.return_value = {'result': [{'totalintensity': 48000}]}
        with patch.object(self.api, 'GRID_CALC_API_KEY', 'test-placeholder'), patch.object(self.api.requests, 'get', return_value=response) as request:
            success, payload = self.api.fetch_grid_data('0,0', 'totalintensity', start_time='1999')
            self.assertTrue(success)
            self.assertEqual(payload['provenance']['requested_model'], 'IGRF')
            self.assertEqual(request.call_args.kwargs['timeout'], 30)
            request.reset_mock()
            self.assertFalse(self.api.fetch_grid_data('89,0', 'totalintensity', start_time='2026')[0])
            request.assert_not_called()

    def test_missing_key(self):
        with patch.object(self.api, 'CALC_API_KEY', ''), patch.object(self.api.requests, 'get') as request:
            self.assertFalse(self.api.fetch_field_data('0,0', '2026')[0])
            request.assert_not_called()


if __name__ == '__main__':
    unittest.main()
