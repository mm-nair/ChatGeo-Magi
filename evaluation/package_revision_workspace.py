import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRECTORIES = {'private', 'public_documents', 'public_index', 'vm-venv', '__pycache__', 'vm_scripts'}
ALLOWED_SUFFIXES = {'.md', '.csv', '.json', '.pdf', '.png', '.txt', '.log'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'revision_workspace/revision_handoff.zip')
    args = parser.parse_args()
    workspace = ROOT / 'revision_workspace'
    if not args.output.resolve().is_relative_to(workspace):
        parser.error('Archive must remain in revision_workspace')
    secrets = []
    environment_path = Path.home() / 'chat-geo-magi/.env'
    if environment_path.exists():
        from dotenv import dotenv_values
        secrets = [value.encode() for name, value in dotenv_values(environment_path).items() if 'KEY' in name and value and len(value) >= 8]
        from vm_support import load_existing_credentials
        import os
        load_existing_credentials()
        secrets.extend(value.encode() for name, value in os.environ.items() if name.endswith('API_KEY') and len(value) >= 8)
    manifest = []
    candidates = []
    for path in sorted(workspace.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        relative = path.relative_to(workspace)
        if any(part in EXCLUDED_DIRECTORIES for part in relative.parts):
            continue
        if path.stat().st_size > 10_000_000:
            continue
        if path.name in ('potemra_pages.json', 'chunk_catalog.json', 'calculator_form.html', 'handoff_manifest.json'):
            continue
        content = path.read_bytes()
        if any(secret in content for secret in secrets):
            raise ValueError(f'Secret detected in candidate artifact {relative}; archive not created')
        candidates.append((path, relative))
        manifest.append({'path': str(relative), 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()})
    with zipfile.ZipFile(args.output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path, relative in candidates:
            archive.write(path, 'revision_workspace/' + str(relative))
        archive.writestr('revision_workspace/handoff_manifest.json', json.dumps(manifest, indent=2))
    print(json.dumps({'archive': str(args.output), 'files': len(manifest), 'bytes': args.output.stat().st_size, 'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest()}, indent=2))


if __name__ == '__main__':
    main()
