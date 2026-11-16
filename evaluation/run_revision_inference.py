import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import urllib.request
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from golden_dataset import load_goldens
from public_retrieval import open_retriever


def extract_trace(result):
    calls = []
    responses = {}
    final = ''
    messages = []
    for message in result['messages']:
        messages.append(message.model_dump(mode='json'))
        if getattr(message, 'type', '') == 'tool':
            responses[message.tool_call_id] = message
        for call in getattr(message, 'tool_calls', []):
            calls.append({'id': call['id'], 'name': call['name'], 'args': call['args']})
        if getattr(message, 'type', '') == 'ai' and not getattr(message, 'tool_calls', []):
            final = message.content
    for call in calls:
        response = responses.get(call['id'])
        content = str(response.content) if response is not None else ''
        call['output'] = content
        call['success'] = response is not None and getattr(response, 'status', 'success') != 'error' and not content.startswith(('Could not find', 'Invalid data name', 'Calculator response lacks', 'Error:', 'Failed'))
    return {'calls': calls, 'final': final, 'messages': messages}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', choices=['ablation', 'tools'], required=True)
    parser.add_argument('--execute-inference', action='store_true')
    parser.add_argument('--model', required=True)
    parser.add_argument('--model-digest', required=True)
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--seed', type=int, default=20261115)
    parser.add_argument('--num-ctx', type=int, default=8192)
    parser.add_argument('--num-predict', type=int, default=2048)
    parser.add_argument('--index', type=Path)
    parser.add_argument('--embedding-path', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not args.execute_inference:
        parser.error('Inference is disabled unless --execute-inference is explicitly supplied after VM authorization')
    if args.experiment == 'ablation' and not all([args.index, args.embedding_path, args.manifest]):
        parser.error('Ablation requires the validated public index and cached Stella embedding model')
    output = args.output or ROOT / 'revision_workspace' / ('rag_ablation' if args.experiment == 'ablation' else 'tool_evaluation') / 'results' / uuid4().hex
    if not output.resolve().is_relative_to(ROOT / 'revision_workspace'):
        parser.error('Output must remain in revision_workspace')
    if output.exists():
        parser.error('Use a new output directory; previous runs are never overwritten')
    with urllib.request.urlopen('http://localhost:11434/api/tags', timeout=10) as response:
        installed = json.load(response)
    if not any(row['name'] == args.model and row['digest'] == args.model_digest for row in installed['models']):
        parser.error('Requested exact model digest is not installed; no model will be downloaded')
    output.mkdir(parents=True)
    os.environ['CGM_IMAGE_DIR'] = str(output / 'images')
    os.environ['OPIK_TRACK_DISABLE'] = 'true'
    from vm_support import load_existing_credentials
    load_existing_credentials()
    sys.path.insert(0, str(ROOT / 'agentic'))
    from cgm import ChatGeoMagi
    from config import SystemConfig
    from langchain_ollama import ChatOllama
    from langchain_core.tools import create_retriever_tool
    llm = ChatOllama(model=args.model, temperature=args.temperature, seed=args.seed, num_ctx=args.num_ctx, num_predict=args.num_predict)
    retriever = None
    if args.experiment == 'ablation':
        retriever = create_retriever_tool(open_retriever(args.index, args.embedding_path, args.manifest), 'retrieve_geomag_docs', 'Search and return information about geomagnetism, from authoritative public documents.')
        questions = [{'id': f'Q{index:02}', 'prompt': question, 'reference': answer} for index, (question, answer) in enumerate(load_goldens(ROOT / 'evaluation/goldens.csv').items(), 1)]
        conditions = ['rag', 'no_rag']
    else:
        questions = json.loads((ROOT / 'evaluation/benchmarks/tool_prompts.json').read_text())
        conditions = ['tools_only']
    metadata = {'arguments': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, 'python': platform.python_version(), 'packages': {distribution.metadata['Name']: distribution.version for distribution in importlib.metadata.distributions()}, 'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), 'dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT)), 'calculator_tools_available_in_both_conditions': True, 'model_digest': args.model_digest, 'ollama_models': [{'name': row['name'], 'digest': row['digest']} for row in installed['models']], 'source_hashes': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for folder in ['agentic', 'evaluation'] for path in (ROOT / folder).rglob('*.py')}}
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    config = SystemConfig(model_name=args.model)
    judge_inputs = []
    for question in questions:
        for condition in conditions:
            model = ChatGeoMagi(config)
            model.initialize(llm, None, retriever if condition == 'rag' else None)
            try:
                result = model.invoke(question['prompt'], thread_id=uuid4().hex)
                trace = extract_trace(result)
            except Exception as error:
                trace = {'calls': [], 'final': '', 'error_type': type(error).__name__}
            trace.update({'id': question['id'], 'condition': condition, 'question': question['prompt']})
            print(question['id'], condition, 'calls', len(trace['calls']), 'error', trace.get('error_type'), flush=True)
            (output / f"{question['id']}_{condition}.json").write_text(json.dumps(trace, indent=2))
            if 'reference' in question:
                judge_inputs.append({'id': question['id'], 'condition': condition, 'input': question['prompt'], 'expected_answer': question['reference'], 'model_output': trace['final']})
    if judge_inputs:
        (output / 'judge_inputs.json').write_text(json.dumps(judge_inputs, indent=2))


if __name__ == '__main__':
    main()
