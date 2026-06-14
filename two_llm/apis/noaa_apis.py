"""
LEGACY: Many portions of this script may be broken now,
and is solely to demonstrate how the two-LLM architecture was
constrcuted. No plots nor data were made from this architecture.

This script has the API capabilities for the two-LLM system.
"""

import matplotlib.pyplot as plt
import numpy as np
import time
import requests
from datetime import date
from dateutil.parser import parse
from PIL import Image
from io import BytesIO
from typing import Dict, Tuple, Optional, Union, List

# Constants
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

FIELD_COMPONENTS = {
    "DECLINATION": ("declination", "Declination", "d"),
    "MAGDEC": ("declination", "Declination", "d"),
    "INTENSITY": ("totalintensity", "Total Intensity", "f"),
    "MAGINC": ("totalintensity", "Total Intensity", "f"),
    "TOTINT": ("totalintensity", "Total Intensity", "f"),
}

class GeomagneticAPI:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @staticmethod
    def parse_date(time_str: Optional[str] = None) -> Tuple[int, int, int]:
        """Convert date string to (year, month, day) tuple."""
        if not time_str or time_str.lower() in ["now", "no", ""]:
            today = date.today()
            return today.year, today.month, today.day
        
        parsed = parse(time_str, fuzzy=True)
        return parsed.year, parsed.month, parsed.day

    @staticmethod
    def get_coordinates(location: str) -> Tuple[bool, Union[Tuple[float, float], str]]:
        """Get coordinates for a location using geocoding API."""
        params = {"q": location, "api_key": GEOCODE_API_KEY}
        response = requests.get(BASE_URLS["geocode"], params=params)
        data = response.json()
        
        if not data:
            return False, "Location not found"
        
        return True, (float(data[0]["lat"]), float(data[0]["lon"])), response.url
    
    @staticmethod
    def remove_api_keys(urls: List[str]) -> List[str]:
        """Remove API keys from URLs for security when displaying to the client."""
        new_urls = []
        for url in urls:
            for key in [GEOCODE_API_KEY, MAPBOX_API_KEY, CALC_API_KEY, GRID_CALC_API_KEY]:
                if key in url:
                    new_urls.append(url.replace(key, "[API_KEY_REDACTED]"))
        
        return new_urls
        

    def get_model_type(self, year: int) -> str:
        """Determine appropriate model type based on year."""
        if year < 1590:
            raise ValueError("Date must be after 1590 for IGRF calculations")
        return "IGRF" if year < 2025 else "WMM"

    def fetch_field_data(self, location: str, start_time: Optional[str] = None, 
                        end_time: Optional[str] = None) -> Tuple[bool, Union[Dict, str]]:
        """Fetch geomagnetic field data from NOAA API."""
        success, coords, grid_url = self.get_coordinates(location)
        
        if not success:
            return False, coords

        lat, lon = coords
        start_year, start_month, start_day = self.parse_date(start_time)
        
        try:
            model_type = self.get_model_type(start_year)
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
            end_year, end_month, end_day = self.parse_date(end_time)
            params.update({
                "endYear": end_year,
                "endMonth": end_month,
                "endDay": end_day
            })

        response = requests.get(BASE_URLS["calculator"], params=params)
        return True, response.json(), [response.url, grid_url] 

    def fetch_grid_data(self, location: str, component: str, bounds: float = 5.0, 
                       start_time: Optional[str] = None) -> Tuple[bool, Union[Dict, str]]:
        """Fetch grid data for contour plotting."""
        success, coords, grid_url = self.get_coordinates(location)
        if not success:
            return False, coords

        lat, lon = coords
        start_year, start_month, start_day = self.parse_date(start_time)
        
        try:
            model_type = self.get_model_type(start_year)
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
        return True, response.json(), [response.url, grid_url]

    def get_map_image(self, bounding_box: Tuple[float, float, float, float], 
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
        
        return Image.open(BytesIO(response.content)), response.url

api = GeomagneticAPI()

def magnetic_declination(location: str, start_time: Optional[str] = None) -> str:
    """Get magnetic declination for a location."""
    success, result, urls = api.fetch_field_data(location, start_time)
    
    if not success:
        return "Could not find declination.", None
    
    declination = result['result'][0]['declination']
    direction = "east" if declination > 0 else "west"
    return f"{abs(declination)}deg {direction}", api.remove_api_keys(urls)

def magnetic_inclination(location: str, start_time: Optional[str] = None) -> str:
    """Get magnetic inclination for a location."""
    success, result, urls = api.fetch_field_data(location, start_time)
    
    if not success:
        return "Could not find inclination.", None
    
    return f"{result['result'][0]['inclination']}deg", api.remove_api_keys(urls)

def total_intensity(location: str, start_time: Optional[str] = None) -> str:
    """Get total magnetic field intensity for a location."""
    success, result, urls = api.fetch_field_data(location, start_time)
    
    if not success:
        return "Could not find the total intensity.", None
    
    return f"{result['result'][0]['totalintensity']}nT", api.remove_api_keys(urls)

def plot(data_name: str, location: str, start_time: Optional[str] = None, 
         end_time: Optional[str] = None) -> str:
    """Create a time series plot of magnetic field components."""
    if data_name not in FIELD_COMPONENTS:
        return "API currently does not have access to that data", None
    
    success, result, urls = api.fetch_field_data(location, start_time, end_time)
    
    if not success:
        return f"{result} Could not create plot.", None
    
    result_data, units = result["result"], result["units"]
    start_year = api.parse_date(start_time)[0]
    years = [start_year + i for i in range(len(result_data))]
    
    data_key, data_display, _ = FIELD_COMPONENTS[data_name]
    values = [entry[data_key] for entry in result_data]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(years, values, marker='o', label=f"{data_display} Data")
    ax.set_title(f"{data_display} in {location} from {start_time or 'now'} to {end_time or 'now'}")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"{data_display} {units[data_key]}")
    ax.grid(True)
    ax.legend()
    
    return fig, api.remove_api_keys(urls)

def gridplot(data_name: str, location: str, start_time: Optional[str] = None, zoom: Optional[str] = "1.0") -> str:
    """Create a contour plot of magnetic field components."""
    if data_name not in FIELD_COMPONENTS:
        return "API currently does not have access to that data", None
    
    final_zoom = zoom

    try:
        final_zoom = float(zoom)
    except ValueError:
        final_zoom = zoom if (type(zoom) is float) or (type(zoom) is int) else 1.0

    print(final_zoom)

    data_key, data_display, component = FIELD_COMPONENTS[data_name]
    success, result, urls = api.fetch_grid_data(location, component, bounds=(5 / final_zoom), start_time=start_time)
    
    if not success:
        return f"{result} Could not create plot.", None
    
    results, units = result["result"], result["units"]
    latitudes = sorted(set(result['latitude'] for result in results))
    longitudes = sorted(set(result['longitude'] for result in results))
    
    declination_grid = np.zeros((len(latitudes), len(longitudes)))
    for result in results:
        lat_idx = latitudes.index(result['latitude'])
        lon_idx = longitudes.index(result['longitude'])
        declination_grid[lat_idx, lon_idx] = result[data_key]
    
    bounds = (min(latitudes), min(longitudes), max(latitudes), max(longitudes))
    try:
        base_map, map_url = api.get_map_image(bounds)
        urls.append(map_url)
    except Exception as e:
        return f"Failed to fetch map: {str(e)}", None

    fig, ax = plt.subplots(figsize=(10, 8))
    X, Y = np.meshgrid(longitudes, latitudes)
    
    ax.imshow(base_map, extent=(bounds[1], bounds[3], bounds[0], bounds[2]), 
             zorder=1, aspect='auto')
    
    contour = ax.contourf(X, Y, declination_grid, cmap='viridis', alpha=0.6, zorder=2)
    fig.colorbar(contour, ax=ax, label=f'{data_display} {units[data_key]}')
    
    ax.set_title(f'{data_display} Contour Map around {location}')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    return fig, api.remove_api_keys(urls)