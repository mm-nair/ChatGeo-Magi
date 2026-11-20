import importlib
import json
import os
import runpy
import sys
import tempfile
import types
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch, Mock

from PIL import Image

from agentic.apis import scientific
from agentic.config import load_env_files


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

    def setUp(self):
        self.api.set_active_request_context(None)

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

    def test_swapped_invalid_coordinates_rejected(self):
        # 1. Direct swap from user prompt containing invalid 95,0
        self.api.set_active_request_context('Compute declination at latitude,longitude 95,0 for 2025-01-01.')
        with patch.object(self.api, 'CALC_API_KEY', 'test-placeholder'), patch.object(self.api.requests, 'get') as request:
            success, msg = self.api.fetch_field_data('0,95', '2025-01-01')
            self.assertFalse(success)
            self.assertIn('Refusing to swap invalid user coordinates', msg)
            request.assert_not_called()

        # 2. Retry swap after 95,0 was rejected in the same turn even without prompt text
        self.api.set_active_request_context(None)
        with patch.object(self.api, 'CALC_API_KEY', 'test-placeholder'), patch.object(self.api.requests, 'get') as request:
            success1, _ = self.api.fetch_field_data('95,0', '2025-01-01')
            self.assertFalse(success1)
            success2, msg2 = self.api.fetch_field_data('0,95', '2025-01-01')
            self.assertFalse(success2)
            self.assertIn('Refusing to swap invalid user coordinates', msg2)
            request.assert_not_called()

        # 3. Legitimate request for 0,95 succeeds
        self.api.set_active_request_context('Compute declination at latitude,longitude 0,95 for 2025-01-01.')
        response = Mock()
        response.json.return_value = {'result': [{'date': '2025-01-01', 'declination': 0.5}], 'units': {'declination': 'degrees'}, 'model': 'WMM-2025'}
        with patch.object(self.api, 'CALC_API_KEY', 'test-placeholder'), patch.object(self.api.requests, 'get', return_value=response) as request:
            success3, _ = self.api.fetch_field_data('0,95', '2025-01-01')
            self.assertTrue(success3)
            request.assert_called_once()

    def test_contour_map_with_base_map(self):
        fake_image = Image.new('RGB', (8, 8), color=(120, 160, 200))
        with tempfile.TemporaryDirectory() as tmpdir, \
             patch.dict(os.environ, {'CGM_IMAGE_DIR': tmpdir}), \
             patch.object(self.api, 'get_map_image', return_value=fake_image) as fetch_map:
            output_path = self.api.contour_map(
                x=[0.0, 1.0],
                y=[0.0, 1.0],
                z=[[1.0, 2.0], [3.0, 4.0]],
                show_base_map=True,
                extent=(0.0, 1.0, 0.0, 1.0),
            )
            fetch_map.assert_called_once_with((0.0, 1.0, 0.0, 1.0))
            self.assertIsInstance(output_path, str)
            self.assertTrue(output_path.endswith('.png'))
            self.assertTrue(Path(output_path).is_file())
            self.assertGreater(Path(output_path).stat().st_size, 0)

    def test_dotenv_loading_for_cli_streamlit_and_precedence(self):
        root = Path(__file__).resolve().parents[2]
        agentic_dir = root / 'agentic'

        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / '.env'
            env_file.write_text(
                'CALC_API_KEY=dotenv_calc_placeholder\n'
                'GEOCODE_API_KEY="dotenv_geocode_placeholder"\n'
                'MAPBOX_API_KEY=dotenv_mapbox_placeholder\n',
                encoding='utf-8',
            )
            orig_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                # 1. Direct load_env_files() + exported variable precedence
                with patch.dict(os.environ, {'CALC_API_KEY': 'exported_calc_placeholder'}, clear=True):
                    load_env_files()
                    self.assertEqual(os.environ['CALC_API_KEY'], 'exported_calc_placeholder')
                    self.assertEqual(os.environ['GEOCODE_API_KEY'], 'dotenv_geocode_placeholder')
                    self.assertEqual(os.environ['MAPBOX_API_KEY'], 'dotenv_mapbox_placeholder')

                # 2. Fresh import of noaa_apis picks up .env credentials while respecting exported overrides
                fake_tools = types.ModuleType('langchain_core.tools')
                fake_tools.tool = lambda function: function
                apis_pkg = sys.modules['agentic.apis']
                orig_attr = getattr(apis_pkg, 'noaa_apis', None)
                try:
                    with patch.dict(os.environ, {'CALC_API_KEY': 'exported_calc_placeholder'}, clear=True), \
                         patch.dict(sys.modules, {'langchain_core': types.ModuleType('langchain_core'), 'langchain_core.tools': fake_tools}):
                        sys.modules.pop('agentic.apis.noaa_apis', None)
                        if hasattr(apis_pkg, 'noaa_apis'):
                            delattr(apis_pkg, 'noaa_apis')
                        fresh_api = importlib.import_module('agentic.apis.noaa_apis')
                        self.assertEqual(fresh_api.CALC_API_KEY, 'exported_calc_placeholder')
                        self.assertEqual(fresh_api.GRID_CALC_API_KEY, 'exported_calc_placeholder')
                        self.assertEqual(fresh_api.GEOCODE_API_KEY, 'dotenv_geocode_placeholder')
                        self.assertEqual(fresh_api.MAPBOX_API_KEY, 'dotenv_mapbox_placeholder')
                finally:
                    if orig_attr is not None:
                        setattr(apis_pkg, 'noaa_apis', orig_attr)

                # 3. CLI entry point (agentic/cgmterm.py) loads .env before agent initialization
                seen_cli = {}

                class FakeCGM:
                    @staticmethod
                    def get_shared_resources(cfg):
                        seen_cli['CALC_API_KEY'] = os.getenv('CALC_API_KEY')
                        seen_cli['GEOCODE_API_KEY'] = os.getenv('GEOCODE_API_KEY')
                        return None, None, None

                    def __init__(self, cfg):
                        pass

                    def initialize(self, *args):
                        pass

                    def visualize_agent(self):
                        pass

                fake_cgm_mod = types.ModuleType('cgm')
                fake_cgm_mod.ChatGeoMagi = FakeCGM
                with patch.dict(os.environ, {'CALC_API_KEY': 'exported_cli_placeholder'}, clear=True), \
                     patch.dict(sys.modules, {'cgm': fake_cgm_mod}), \
                     patch('builtins.input', return_value='q'):
                    runpy.run_path(str(agentic_dir / 'cgmterm.py'), run_name='__main__')
                self.assertEqual(seen_cli['CALC_API_KEY'], 'exported_cli_placeholder')
                self.assertEqual(seen_cli['GEOCODE_API_KEY'], 'dotenv_geocode_placeholder')

                # 4. Streamlit entry point (agentic/cgmwebsite.py) loads .env on startup
                class FakeSessionState(dict):
                    __setattr__ = dict.__setitem__

                fake_st = types.ModuleType('streamlit')
                fake_st.session_state = FakeSessionState()
                fake_st.cache_resource = lambda fn: fn
                fake_torch = types.ModuleType('torch')
                fake_torch.classes = types.SimpleNamespace(__path__=[])
                with patch.dict(os.environ, {'CALC_API_KEY': 'exported_web_placeholder'}, clear=True), \
                     patch.dict(sys.modules, {'streamlit': fake_st, 'torch': fake_torch}):
                    runpy.run_path(str(agentic_dir / 'cgmwebsite.py'), run_name='test_cgmwebsite')
                    self.assertEqual(os.environ['CALC_API_KEY'], 'exported_web_placeholder')
                    self.assertEqual(os.environ['GEOCODE_API_KEY'], 'dotenv_geocode_placeholder')
            finally:
                os.chdir(orig_cwd)


if __name__ == '__main__':
    unittest.main()
