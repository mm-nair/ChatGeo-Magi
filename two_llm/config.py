"""
LEGACY: Many portions of this script may be broken now,
and is solely to demonstrate how the two-LLM architecture was
constrcuted. No plots nor data were made from this architecture.

This script has the configuration options for the two-LLM system.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, TypedDict

from apis.noaa_apis import (
    magnetic_declination,
    magnetic_inclination,
    total_intensity,
    plot,
    gridplot
)

class ToolConfig(TypedDict):
    name: str
    params: List[str]
    description: str
    fn: Callable

@dataclass
class SystemConfig:
    chroma_db_dir: str = "./chroma_db"
    data_dir: str = "./data"
    model_name: str = "gemma2:27b"
    embedding_model: str = "NovaSearch/stella_en_1.5B_v5"
    request_timeout: float = 300.0

TOOLS: Dict[str, ToolConfig] = {
    "magnetic_declination": {
        "name": "MAGDEC",
        "params": ["location_string", "date OR now"],
        "description": "Magnetic declination at a location, at a specific time. If unspecified, set the date to 'now'",
        "fn": magnetic_declination,
    },
    "magnetic_inclination": {
        "name": "MAGINC",
        "params": ["location_string", "date OR now"],
        "description": "Magnetic inclination at a location, at a specific time. If unspecified, set the date to 'now'",
        "fn": magnetic_inclination,
    },
    "total_intensity": {
        "name": "TOTINT",
        "params": ["location_string", "date OR now"],
        "description": "The total strength/intensity of the magnetic field at a location, at a specific time. If unspecified, set the date to 'now'",
        "fn": total_intensity,
    },
    "plot": {
        "name": "PLOT",
        "params": ["data to plot (eg DECLINATION or INTENSITY)", "location_string", "start date", "end date OR now"],
        "description": "Returns the plot image name of a certain location from the start date to the end date",
        "fn": plot,
    },
    "gridplot": {
        "name": "GRIDPLOT",
        "params": ["data to plot (eg DECLINATION or INTENSITY)", "location_string", "date OR now", "zoom (0.5-5.0) <OPTIONAL>"],
        "description": "Returns the contour map image of the magnetic field component given at the date given. If unspecified, set the date to 'now'",
        "fn": gridplot,
    }
}
