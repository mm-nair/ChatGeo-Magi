"""
This script outlines the basic parameters of the agent.
"""

from dataclasses import dataclass

@dataclass
class SystemConfig:
    chroma_db_path: str = "./chroma_db"
    data_dir: str = "./data"
    model_name: str = "gpt-oss:20b"
    embedding_model: str = "NovaSearch/stella_en_1.5B_v5"
    request_timeout: float = 300.0