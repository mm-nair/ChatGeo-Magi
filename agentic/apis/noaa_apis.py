from typing import Dict, Tuple, Optional, Union, List
import contextvars
import math
import re
from matplotlib.figure import Figure

from langchain_core.tools import tool
from .scientific import parse_date as scientific_date, select_model, coordinates, provenance, summarize_component
from datetime import date

from PIL import Image
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import requests

import uuid
import os

try:
    from config import load_env_files
except ImportError:
    from ..config import load_env_files

load_env_files()

_ACTIVE_QUESTION: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "cgm_active_question", default=None
)
_REJECTED_COORDS: contextvars.ContextVar[Tuple[Tuple[float, float], ...]] = contextvars.ContextVar(
    "cgm_rejected_coords", default=()
)


def set_active_request_context(question: Optional[str]) -> None:
    _ACTIVE_QUESTION.set(question)
    _REJECTED_COORDS.set(())


def _extract_raw_pair(location: str) -> Optional[Tuple[float, float]]:
    if not isinstance(location, str) or "," not in location:
        return None
    parts = location.split(",")
    if len(parts) != 2:
        return None
    try:
        lat, lon = float(parts[0].strip()), float(parts[1].strip())
        if math.isfinite(lat) and math.isfinite(lon):
            return lat, lon
    except ValueError:
        return None
    return None


def _out_of_range_pairs_in_question(question: Optional[str]) -> List[Tuple[float, float]]:
    if not question:
        return []
    pairs = []
    for match in re.finditer(r"([+\-]?\d+(?:\.\d+)?)\s*,\s*([+\-]?\d+(?:\.\d+)?)", question):
        try:
            first = float(match.group(1))
            second = float(match.group(2))
        except ValueError:
            continue
        if not (-90.0 <= first <= 90.0 and -180.0 <= second <= 180.0):
            pairs.append((first, second))
    return pairs

BASE_URLS = {
    "calculator": "https://www.ngdc.noaa.gov/geomag-web/calculators/calculateIgrfwmm",
    "grid": "https://www.ngdc.noaa.gov/geomag-web/calculators/calculateIgrfgrid",
    "geocode": "https://geocode.maps.co/search",
    "mapbox": "https://api.mapbox.com/styles/v1/mapbox/streets-v12/static"
}

GEOCODE_API_KEY = os.getenv("GEOCODE_API_KEY", "")
MAPBOX_API_KEY = os.getenv("MAPBOX_API_KEY", "")
CALC_API_KEY = os.getenv("CALC_API_KEY", "")
GRID_CALC_API_KEY = os.getenv("CALC_API_KEY", "")

valid_components = ["declination", "inclination", "totalintensity", "horintensity", "xcomponent", "ycomponent", "zcomponent", "declination_sv", "inclination_sv", "totalintensity_sv", "horintensity_sv", "xcomponent_sv", "ycomponent_sv", "zcomponent_sv"]
components_string = ', '.join(valid_components)

def save_figure_to_temp(fig):
    output_dir = os.getenv("CGM_IMAGE_DIR", "tempimages")
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{uuid.uuid4().hex}.png"
    path = os.path.join(output_dir, filename)
    fig.savefig(path)
    plt.close(fig)

    print(f"Saved figure to {path}")
    
    return path

def parse_date(time_str: Optional[str] = None) -> Tuple[int, int, int]:
    parsed = scientific_date(time_str)
    return parsed.year, parsed.month, parsed.day


def resolve_coordinates(location):
    raw_pair = _extract_raw_pair(location)
    try:
        explicit = coordinates(location)
    except ValueError as error:
        if raw_pair is not None:
            current = list(_REJECTED_COORDS.get())
            if raw_pair not in current:
                current.append(raw_pair)
                _REJECTED_COORDS.set(tuple(current))
        raise ValueError(
            f"{error} (latitude must be in [-90, 90], longitude in [-180, 180]; do not swap or alter invalid user coordinates)"
        ) from error
    if explicit is not None:
        lat, lon = explicit
        candidates = list(_REJECTED_COORDS.get()) + _out_of_range_pairs_in_question(_ACTIVE_QUESTION.get())
        for bad_lat, bad_lon in candidates:
            if math.isclose(lat, bad_lon, abs_tol=1e-6) and math.isclose(lon, bad_lat, abs_tol=1e-6):
                raise ValueError(
                    f"Refusing to swap invalid user coordinates ({bad_lat:g},{bad_lon:g} -> {lat:g},{lon:g}); latitude must be in [-90, 90] and longitude in [-180, 180]"
                )
        return explicit
    if not GEOCODE_API_KEY:
        raise ValueError("Place-name lookup is unavailable; provide latitude,longitude")
    response = requests.get(BASE_URLS["geocode"], params={"q": location, "api_key": GEOCODE_API_KEY}, timeout=30)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list) or not data:
        raise ValueError("Location not found")
    return coordinates(f"{data[0]['lat']},{data[0]['lon']}")


@tool
def get_coordinates(location: str) -> Tuple[bool, Union[Tuple[float, float], str]]:
    """Resolve place names or explicit latitude,longitude decimal coordinates."""
    try:
        return True, resolve_coordinates(location)
    except (requests.RequestException, ValueError, KeyError, TypeError):
        return False, "Coordinate lookup failed; check location or supply coordinates"


def get_model_type(year: int) -> str:
    return select_model(str(year))[0]


def fetch_field_data(location: str, start_time: Optional[str] = None,
                     end_time: Optional[str] = None) -> Tuple[bool, Union[Dict, str]]:
    try:
        model_type, start_date, end_date = select_model(start_time, end_time)
        if not CALC_API_KEY:
            return False, "Calculator credentials are not configured"
        lat, lon = resolve_coordinates(location)
        params = {
            "lat1": lat, "lon1": lon, "model": model_type,
            "startYear": start_date.year, "startMonth": start_date.month, "startDay": start_date.day,
            "endYear": end_date.year, "endMonth": end_date.month, "endDay": end_date.day,
            "elevation": 0, "elevationUnits": "K", "coordinateSystem": "M",
            "dateStepSize": 1, "key": CALC_API_KEY, "resultFormat": "json"
        }
        response = requests.get(BASE_URLS["calculator"], params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        payload_provenance = provenance(payload, model_type, start_date, end_date, lat, lon)
        payload_provenance["calculator_url"] = BASE_URLS["calculator"]
        payload["provenance"] = payload_provenance
        return True, payload
    except requests.RequestException:
        return False, "NOAA request failed (HTTP or transport error)"
    except (ValueError, KeyError, TypeError) as error:
        return False, "Invalid date, location, or calculator response: " + (str(error) if isinstance(error, ValueError) and not isinstance(error, requests.RequestException) else type(error).__name__)


def fetch_grid_data(location: str, component: str, bounds: float = 5.0,
                    start_time: Optional[str] = None) -> Tuple[bool, Union[Dict, str]]:
    try:
        model_type, start_date, end_date = select_model(start_time)
        if component not in valid_components or not 0 < bounds <= 90:
            return False, "Invalid grid component or bounds"
        if not GRID_CALC_API_KEY:
            return False, "Grid calculator credentials are not configured"
        lat, lon = resolve_coordinates(location)
        coordinates(f"{lat - bounds},{lon - bounds}")
        coordinates(f"{lat + bounds},{lon + bounds}")
        params = {
            "lat1": lat - bounds, "lon1": lon - bounds,
            "lat2": lat + bounds, "lon2": lon + bounds,
            "magneticComponent": component, "model": model_type,
            "startYear": start_date.year, "startMonth": start_date.month, "startDay": start_date.day,
            "key": GRID_CALC_API_KEY, "resultFormat": "json"
        }
        response = requests.get(BASE_URLS["grid"], params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        payload["provenance"] = provenance(payload, model_type, start_date, end_date, lat, lon)
        payload["provenance"]["calculator_url"] = BASE_URLS["grid"]
        payload["provenance"]["elevation_reference"] = "server default; not specified"
        payload["provenance"]["elevation_km"] = "server default; not specified"
        return True, payload
    except (requests.RequestException, ValueError, KeyError, TypeError):
        return False, "Grid request failed or returned invalid data"

def get_map_image(bounding_box: Tuple[float, float, float, float], 
                    size: str = "640x480") -> Image.Image:
    bbox_str = f"[{bounding_box[1]},{bounding_box[0]},{bounding_box[3]},{bounding_box[2]}]"
    params = {
        "access_token": MAPBOX_API_KEY
    }

    url = f"{BASE_URLS['mapbox']}/{bbox_str}/{size}"
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch map: {response.status_code}")
    
    return Image.open(BytesIO(response.content))

@tool
def noaa_mag_api(data_name: str, location: str, start_time: Optional[str] = None, end_time: Optional[str] = None) -> str:
    """
    Fetch magnetic field data from NOAA Magnetic Calculator APIs, from start date (default to today) to end date (default to start date).
    Dates must be YYYY or YYYY-MM-DD. Location can be a place name or latitude,longitude in decimal degrees (latitude in [-90, 90], longitude in [-180, 180]).
    NEVER swap latitude and longitude (e.g., never change 95,0 to 0,95) or modify invalid user coordinates/dates. If coordinates or dates are invalid, report the error directly.
    Supported dates are 1900 through 2029. Historical or mixed ranges use IGRF; wholly 2025-2029 ranges use WMM.
    Report requested and returned model identity separately; never infer a returned version.
    The output includes units and provenance. Missing metadata is explicitly marked.
    If you want to call multiple times for different data, please do these in seperate tool steps, dont do them together.
    Please select data name from the following list:

    declination, inclination, totalintensity, horintensity
    xcomponent, ycomponent, zcomponent
    
    All of these can be appended with _sv to get secular variation data, e.g. declination_sv.
    """

    if data_name not in valid_components:
        return f"Invalid data name: {data_name}. Please choose from: {components_string}"

    success, result = fetch_field_data(location, start_time, end_time)
    
    if not success:
        return f"Could not find {data_name}: {result}"
    
    try:
        return summarize_component(result, data_name)
    except (ValueError, KeyError, TypeError):
        return "Calculator response lacks valid component data"

@tool
def plot(
    title: str,
    x: List[Union[float, int, str]],
    y: List[Union[float, int]],
    xlabel: str = "",
    ylabel: str = "",
    marker: str = "o",
    figsize: Tuple[int, int] = (10, 6),
) -> str:
    """
    Generalized plotting function.
    x and y should be sequences of equal length.
    Returns path to saved plot image. Once you call this tool, the plot will render to the user.
    """
    if len(x) != len(y):
        return "Error: x and y data must have the same length.", None

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y, marker=marker, linestyle='-', label=ylabel or "Data")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)    
    ax.grid(True)

    if ylabel:
        ax.legend()

    print(fig)

    return save_figure_to_temp(fig)

@tool
def plot_many(
    title: str,
    x: List[Union[float, int, str]],
    ys: List[List[Union[float, int]]],
    labels: Optional[List[str]] = None,
    xlabel: str = "",
    ylabel: str = "",
    markers: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> str:
    """
    Plot multiple y datasets against a common x dataset.
    
    Args:
        title: Plot title.
        x: X-axis data.
        ys: List of Y datasets to plot.
        labels: Optional list of labels for each Y dataset.
        xlabel, ylabel: Axis labels.
        markers: Optional list of markers for each Y dataset.
        figsize: Figure size in inches.
        
    Returns:
        Path to saved figure image file.
    """
    if not ys or not all(len(y) == len(x) for y in ys):
        return "Error: All y datasets must have the same length as x.", None

    fig, ax = plt.subplots(figsize=figsize)

    for i, y in enumerate(ys):
        marker = markers[i] if markers and i < len(markers) else 'o'
        label = labels[i] if labels and i < len(labels) else f"Dataset {i+1}"
        ax.plot(x, y, marker=marker, linestyle='-', label=label)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    
    if labels:
        ax.legend()

    return save_figure_to_temp(fig)

@tool
def contour_map(
    x: List[float],
    y: List[float],
    z: List[List[float]],
    title: str = "Contour Map",
    xlabel: str = "X",
    ylabel: str = "Y",
    cmap: str = "viridis",
    show_base_map: bool = False,
    extent: Optional[Tuple[float, float, float, float]] = None,
    alpha: float = 0.6,
    figsize: Tuple[int, int] = (10, 8),
) -> str:
    """
    Create a contour plot from grid data.
    
    Args:
        x, y: 1D arrays or lists of x and y coordinates (lengths must match z dimensions).
        z: 2D array-like (list of lists or numpy array) of z-values with shape (len(y), len(x)).
        title: Plot title.
        xlabel, ylabel: Axis labels.
        cmap: Matplotlib colormap string.
        show_base_map: Whether to show a map image beneath the contour.
        extent: The bounding box of the map (xmin, ymin, xmax, ymax) if show_base_map is True.
        alpha: Transparency of contour overlay.
        figsize: Figure size in inches.
        
    Returns:
        Path to saved figure image file.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    X, Y = np.meshgrid(x, y)
    Z = np.array(z)

    fig, ax = plt.subplots(figsize=figsize)

    if show_base_map:
        if extent is None:
            return "Error: extent must be provided if show_base_map is True.", None
        try:
            base_map = get_map_image(extent)
            ax.imshow(base_map, extent=extent, aspect='auto', zorder=1)
        except Exception as e:
            return f"Failed to fetch base map: {str(e)}", None

    contour = ax.contourf(X, Y, Z, cmap=cmap, alpha=alpha, zorder=2)
    fig.colorbar(contour, ax=ax, label="Value")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)

    return save_figure_to_temp(fig)
