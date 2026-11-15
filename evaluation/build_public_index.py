import argparse
import json
import os
from pathlib import Path

from public_retrieval import ROOT, validate_manifest, chunk_identifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--embedding-path', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'revision_workspace/retrieval_evaluation/public_index')
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT / 'revision_workspace') or args.output.exists():
        parser.error('Use a fresh index directory inside revision_workspace')
    manifest = validate_manifest(args.manifest)
    if not args.embedding_path.is_dir():
        parser.error('Cached Stella model directory required; downloads disabled')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    documents = []
    for record in manifest['files']:
        path = Path(record['path']).resolve()
        if path.suffix.lower() not in ('.pdf', '.txt'):
            raise ValueError('Only original PDF/TXT loader types are supported')
        loader = PyPDFLoader(str(path)) if path.suffix.lower() == '.pdf' else TextLoader(str(path))
        documents.extend(loader.load())
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True).split_documents(documents)
    if not chunks:
        raise ValueError('No public chunks loaded')
    embeddings = HuggingFaceEmbeddings(model_name=str(args.embedding_path), model_kwargs={'local_files_only': True})
    store = Chroma(collection_name='cgmdocs', embedding_function=embeddings, persist_directory=str(args.output))
    for chunk in chunks:
        store.add_documents([chunk])
    (args.output.parent / 'chunk_catalog.json').write_text(json.dumps([{'chunk_id': chunk_identifier(chunk.page_content, chunk.metadata), 'text': chunk.page_content, 'metadata': chunk.metadata} for chunk in chunks], indent=2))
    manifest.update({'index_path': str(args.output.resolve()), 'embedding_model': 'NovaSearch/stella_en_1.5B_v5', 'chunk_size': 1000, 'chunk_overlap': 200, 'length_function': 'len', 'chunk_count': len(chunks)})
    (args.output.parent / 'indexed_manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
