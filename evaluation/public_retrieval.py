import argparse
import hashlib
import json
from pathlib import Path

from benchmark_metrics import retrieval_metrics

ROOT = Path(__file__).resolve().parents[1]


def chunk_identifier(text, metadata):
    return hashlib.sha256((json.dumps(metadata, sort_keys=True, ensure_ascii=False) + text).encode()).hexdigest()


def validate_manifest(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get('public_only') is not True or not manifest.get('files'):
        raise ValueError('An explicit public-only corpus manifest is required')
    for record in manifest['files']:
        path = Path(record['path'])
        if not record.get('source_url', '').startswith('https://') or hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('Unverified public corpus file')
    return manifest


def open_retriever(index, embedding_path, manifest_path):
    import os
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    manifest = validate_manifest(manifest_path)
    if not Path(index).is_dir() or not Path(embedding_path).is_dir():
        raise ValueError('Existing public index and local Stella model required; downloads disabled')
    if Path(manifest['index_path']).resolve() != Path(index).resolve():
        raise ValueError('Manifest index does not match requested index')
    if manifest.get('embedding_model') != 'NovaSearch/stella_en_1.5B_v5' or manifest.get('chunk_size') != 1000 or manifest.get('chunk_overlap') != 200 or manifest.get('length_function') != 'len':
        raise ValueError('Manifest differs from reconstructed historical retrieval settings')
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    embeddings = HuggingFaceEmbeddings(model_name=str(embedding_path), model_kwargs={'local_files_only': True})
    store = Chroma(collection_name='cgmdocs', embedding_function=embeddings, persist_directory=str(index))
    if not store._collection.count():
        raise ValueError('Public index is empty')
    indexed = store.get(include=['metadatas'])
    permitted = {str(Path(row['path']).resolve()) for row in manifest['files']}
    if any(str(Path(metadata.get('source', '')).resolve()) not in permitted for metadata in indexed['metadatas']):
        raise ValueError('Index contains sources outside public manifest')
    return store.as_retriever(search_kwargs={'k': 4})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=Path, default=ROOT / 'evaluation/benchmarks/retrieval.json')
    parser.add_argument('--retrieved', type=Path)
    parser.add_argument('--index', type=Path)
    parser.add_argument('--embedding-path', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'revision_workspace/retrieval_evaluation/results')
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT / 'revision_workspace'):
        parser.error('Outputs must remain in revision_workspace')
    questions = json.loads(args.benchmark.read_text())
    if args.retrieved:
        retrieved = json.loads(args.retrieved.read_text())
    else:
        if not all([args.index, args.embedding_path, args.manifest]):
            parser.error('Provide cached retrieval results or a public index, local embedding path and manifest')
        retriever = open_retriever(args.index, args.embedding_path, args.manifest)
        retrieved = {}
        for question in questions:
            documents = retriever.invoke(question['question'])
            retrieved[question['id']] = [{'chunk_id': chunk_identifier(document.page_content, document.metadata), 'text': document.page_content, 'metadata': document.metadata} for document in documents]
    if set(retrieved) != {question['id'] for question in questions}:
        raise ValueError('Retrieval results must cover every benchmark question exactly')
    rows = [{'id': question['id'], **retrieval_metrics(question, retrieved[question['id']])} for question in questions]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'retrieved.json').write_text(json.dumps(retrieved, indent=2))
    (args.output / 'metrics.json').write_text(json.dumps(rows, indent=2))
    summary = {}
    for metric in ['hit_at_4', 'recall_at_4', 'precision_at_4', 'support_coverage']:
        values = [row[metric] for row in rows if row[metric] is not None]
        summary[metric] = {'n': len(values), 'mean': sum(values) / len(values) if values else None}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
