import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'evaluation'))

from benchmark_metrics import retrieval_metrics, tool_metrics
from golden_dataset import load_goldens
from revision_analysis import bootstrap


class EvaluationTests(unittest.TestCase):
    def test_original_headerless_goldens(self):
        self.assertEqual(len(load_goldens(ROOT / 'evaluation/goldens.csv')), 25)

    def test_duplicate_goldens_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'revision_workspace') as directory:
            path = Path(directory) / 'goldens.csv'
            path.write_text('question,answer\nq,a\nq,b\n')
            with self.assertRaises(ValueError):
                load_goldens(path)

    def test_bootstrap_pairing(self):
        self.assertEqual(bootstrap([.25] * 5, resamples=1000), [.25, .25])
        self.assertEqual(bootstrap([-.2, 0, .1], resamples=1000), bootstrap([-.2, 0, .1], resamples=1000))
        with self.assertRaises(ValueError):
            bootstrap([])

    def test_retrieval_annotations(self):
        expected = {'verification_status': 'pending'}
        self.assertIsNone(retrieval_metrics(expected, [])['hit_at_4'])
        expected = {'verification_status': 'human_verified', 'relevant_chunk_ids': ['a', 'b'], 'support_facts': [{'chunk_ids': ['a']}, {'chunk_ids': ['b']}]}
        metrics = retrieval_metrics(expected, [{'chunk_id': 'a'}, {'chunk_id': 'c'}])
        self.assertEqual(metrics['hit_at_4'], 1)
        self.assertEqual(metrics['recall_at_4'], .5)
        self.assertEqual(metrics['precision_at_4'], .25)
        self.assertEqual(metrics['support_coverage'], .5)
        with self.assertRaises(ValueError):
            retrieval_metrics(expected, [{'chunk_id': 'a'}, {'chunk_id': 'a'}])

    def test_tools_require_traces(self):
        expected = {'calls': [{'name': 'noaa_mag_api', 'args': {'start_time': '2026'}}]}
        result = tool_metrics(expected, {'final': 'The declination is 1 degree'})
        self.assertFalse(result['tool_selection'])
        self.assertFalse(result['execution'])
        self.assertIsNone(result['end_to_end'])
        result = tool_metrics(expected, {'calls': [{'name': 'noaa_mag_api', 'args': {'start_time': '2026-01-01'}, 'success': True}]})
        self.assertTrue(result['execution'])
        self.assertIsNone(result['numerical_agreement'])

    def test_equivalent_coordinate_arguments(self):
        from benchmark_metrics import argument_matches
        self.assertTrue(argument_matches('location', '51.500, -0.120', '51.5,-0.12'))
        self.assertFalse(argument_matches('location', '51.5,0.12', '51.5,-0.12'))

    def test_unicode_number_extraction(self):
        from score_live_tools import numbers
        self.assertEqual(numbers('49\u202f100.5 nT and \u201116.98374\u00b0'), [49100.5, -16.98374])

    def test_corrected_summary_requires_complete_pairs(self):
        from analyze_reference_judging import summarize
        with self.assertRaises(ValueError):
            summarize([{'model': 'test', 'id': 'Q01', 'condition': 'rag', 'score': 1}])

    def test_tool_sequence(self):
        expected = {'calls': [{'name': 'noaa_mag_api', 'args': {}}, {'name': 'plot', 'args': {}}]}
        trace = {'calls': [{'name': 'plot', 'args': {}, 'success': True}, {'name': 'noaa_mag_api', 'args': {}, 'success': True}]}
        self.assertFalse(tool_metrics(expected, trace)['parameters_and_order'])

    def test_benchmark_integrity(self):
        prompts = json.loads((ROOT / 'evaluation/benchmarks/tool_prompts.json').read_text())
        expected = json.loads((ROOT / 'evaluation/benchmarks/tool_expected.json').read_text())
        self.assertEqual(len(prompts), 20)
        self.assertEqual([row['id'] for row in prompts], [row['id'] for row in expected])
        self.assertTrue(all(set(row) == {'id', 'prompt'} for row in prompts))

    def test_judge_reference_payload(self):
        from score_ablation import judge_payload, historical_rubric
        payload = json.loads(judge_payload({'input': 'question', 'expected_answer': 'reference', 'model_output': 'response'}))
        self.assertEqual(payload, {'QUESTION': 'question', 'EXPECTED_ANSWER': 'reference', 'MODEL_OUTPUT': 'response'})
        self.assertIn('EXPECTED_ANSWER', historical_rubric())

    def test_chunk_identifier_stability(self):
        from public_retrieval import chunk_identifier
        self.assertEqual(chunk_identifier('text', {'page': 1, 'source': 'doc'}), chunk_identifier('text', {'source': 'doc', 'page': 1}))
        self.assertNotEqual(chunk_identifier('text', {}), chunk_identifier('other', {}))

    def test_inference_isolation_wiring(self):
        import ast
        source = (ROOT / 'evaluation/eval_inference.py').read_text()
        tree = ast.parse(source)
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run_model')
        module = ast.Module(body=[function], type_ignores=[])
        class FakeConfig:
            pass
        class FakeAgent:
            threads = []
            @staticmethod
            def get_shared_resources(config):
                return None, None, None
            def __init__(self, config):
                pass
            def initialize(self, *args):
                pass
            def invoke(self, question, thread_id):
                self.threads.append(thread_id)
                return {'messages': [type('Message', (), {'content': 'ok'})()]}
        namespace = {'SystemConfig': FakeConfig, 'ChatGeoMagi': FakeAgent, 'uuid4': lambda: type('ID', (), {'hex': 'run'})()}
        exec(compile(module, '<isolated_function>', 'exec'), namespace)
        goldens = [{'id': 1, 'question': 'first'}, {'id': 2, 'question': 'second'}]
        namespace['run_model']('fake', goldens)
        self.assertEqual(FakeAgent.threads, ['run-1', 'run-2'])
        FakeAgent.threads.clear()
        namespace['run_model']('fake', goldens, isolated_sessions=False)
        self.assertEqual(FakeAgent.threads, ['1', '1'])


if __name__ == '__main__':
    unittest.main()
