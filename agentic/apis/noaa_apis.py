from typing import Dict, Tuple, Optional, Union, List
from matplotlib.figure import Figure

from langchain_core.tools import tool
from dateutil.parser import parse
from datetime import date

from PIL import Image
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import requests

import uuid
import os

BASE_URLS = {
    "calculator": "https://www.ngdc.noaa.gov/geomag-web/calculators/calculateIgrfwmm",
    "grid": "https://www.ngdc.noaa.gov/geomag-web/calculators/calculateIgrfgrid",
    "geocode": "https://geocode.maps.co/search",
    "mapbox": "https://api.mapbox.com/styles/v1/mapbox/streets-v12/static"
}

GEOCODE_API_KEY = "REDACTED"
MAPBOX_API_KEY = "REDACTED"
CALC_API_KEY = "REDACTED"
GRID_CALC_API_KEY = "REDACTED"

valid_components = ["declination", "inclination", "totalintensity", "horintensity", "xcomponent", "ycomponent", "zcomponent", "declination_sv", "inclination_sv", "totalintensity_sv", "horintensity_sv", "xcomponent_sv", "ycomponent_sv", "zcomponent_sv"]
components_string = ', '.join(valid_components)

def save_figure_to_temp(fig):
    output_dir = "tempimages"
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{uuid.uuid4().hex}.png"
    path = os.path.join(output_dir, filename)
    fig.savefig(path)
    plt.close(fig)

    print(f"Saved figure to {path}")
    
    return path

def parse_date(time_str: Optional[str] = None) -> Tuple[int, int, int]:
    """Convert date string to (year, month, day) tuple."""
    if not time_str or time_str.lower() in ["now", "no", ""]:
        today = date.today()
        return today.year, today.month, today.day
    
    parsed = parse(time_str, fuzzy=True)
    return parsed.year, parsed.month, parsed.day

@tool
def get_coordinates(location: str) -> Tuple[bool, Union[Tuple[float, float], str]]:
    """Get coordinates for a location using the geocoding API"""
    params = {"q": location, "api_key": GEOCODE_API_KEY}
    response = requests.get(BASE_URLS["geocode"], params=params)
    data = response.json()
    
    if not data:
        return False, "Location not found"
    
    return True, (float(data[0]["lat"]), float(data[0]["lon"]))

def get_model_type(year: int) -> str:
    """Determine appropriate model type based on year."""
    if year < 1590:
        raise ValueError("Date must be after 1590 for IGRF calculations")
    return "IGRF" if year < 2025 else "WMM"

def fetch_field_data(location: str, start_time: Optional[str] = None, 
                    end_time: Optional[str] = None) -> Tuple[bool, Union[Dict, str]]:
    """Fetch geomagnetic field data from NOAA API."""
    success, coords = get_coordinates(location)
    
    if not success:
        return False, coords

    lat, lon = coords
    start_year, start_month, start_day = parse_date(start_time)
    
    try:
        model_type = get_model_type(start_year)
    except ValueError as e:
        return False, str(e)

    params = {
        "lat1": lat,
        "lon1": lon,
        "model": model_type,
        "startYear": start_year,
        "startMonth": start_month,
        "startDay": start_day,
        "key": CALC_API_KEY,
        "resultFormat": "json"
    }

    if end_time:
        end_year, end_month, end_day = parse_date(end_time)
        params.update({
            "endYear": end_year,
            "endMonth": end_month,
            "endDay": end_day
        })

    response = requests.get(BASE_URLS["calculator"], params=params)
    return True, response.json()

def fetch_grid_data(location: str, component: str, bounds: float = 5.0, 
                    start_time: Optional[str] = None) -> Tuple[bool, Union[Dict, str]]:
    """Fetch grid data for contour plotting."""
    success, coords = get_coordinates(location)
    if not success:
        return False, coords

    lat, lon = coords
    start_year, start_month, start_day = parse_date(start_time)
    
    try:
        model_type = get_model_type(start_year)
    except ValueError as e:
        return False, str(e)

    params = {
        "lat1": lat - bounds,
        "lon1": lon - bounds,
        "lat2": lat + bounds,
        "lon2": lon + bounds,
        "magneticComponent": component,
        "model": model_type,
        "startYear": start_year,
        "startMonth": start_month,
        "startDay": start_day,
        "key": GRID_CALC_API_KEY,
        "resultFormat": "json"
    }

    response = requests.get(BASE_URLS["grid"], params=params)
    return True, response.json()

def get_map_image(bounding_box: Tuple[float, float, float, float], 
                    size: str = "640x480") -> Image.Image:
    """Fetch map image from Mapbox API."""
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
    Fetch magnetic field data from NOAA Magnetic Calculator APIs, from start date (default to now) to end date (default to now).
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
    
    results = result.get('result', [])
    if not results:
        return f"No {data_name} data found."

    if len(results) == 1:
        val = results[0][data_name]
        units = result.get('units', {}).get(data_name, '')

        return f"{val:.2f} {units}"
    else:
        summary = []
        for entry in results:
            year = entry.get('year') or entry.get('date') or 'unknown'
            val = entry[data_name]
            units = result.get('units', {}).get(data_name, '')
            summary.append(f"{year}: {val:.2f} {units}")

        return f"{data_name} over time:\n" + "\n".join(summary)

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
            base_map, = get_map_image(extent)
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