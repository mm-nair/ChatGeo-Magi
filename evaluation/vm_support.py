import ast
import os
from pathlib import Path


def load_existing_credentials(environment_path=None, legacy_api_path=None):
    from dotenv import dotenv_values
    environment_path = Path(environment_path or Path.home() / 'chat-geo-magi/.env')
    if environment_path.exists():
        for name, value in dotenv_values(environment_path).items():
            if value and name in ('OPENAI_API_KEY', 'OPIK_API_KEY', 'OPIK_WORKSPACE', 'OPIK_PROJECT_NAME'):
                os.environ.setdefault(name, value)
    legacy_api_path = Path(legacy_api_path or Path.home() / 'chat-geo-magi/apis/noaa_apis.py')
    if legacy_api_path.exists():
        for node in ast.parse(legacy_api_path.read_text()).body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in ('CALC_API_KEY', 'GEOCODE_API_KEY', 'GRID_CALC_API_KEY', 'MAPBOX_API_KEY'):
                        try:
                            value = ast.literal_eval(node.value)
                        except (ValueError, TypeError):
                            continue
                        if isinstance(value, str) and value and value != 'REDACTED':
                            os.environ.setdefault(target.id, value)
    return {name: bool(os.getenv(name)) for name in ('OPENAI_API_KEY', 'OPIK_API_KEY', 'CALC_API_KEY', 'GEOCODE_API_KEY')}
