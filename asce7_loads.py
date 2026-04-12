"""
ASCE 7 style preliminary hazard and load-combination helpers.

These functions are intended for schematic-level sizing and visualization.
They do not replace a project-specific code study by a licensed engineer.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen


USGS_ASCE7_22_ENDPOINT = "https://earthquake.usgs.gov/ws/designmaps/asce7-22.json"
DEFAULT_FLOOR_HEIGHT_FT = 14.0
DEFAULT_EXPOSURE = "B"
DEFAULT_SITE_CLASS = "D"
DEFAULT_RISK_CATEGORY = "II"
DEFAULT_SYSTEM = "OMF"

OCCUPANCY_DEFAULTS = {
    "Office": {
        "dead_psf": 50.0,
        "live_psf": 50.0,
        "risk_category": "II",
        "importance_factor": 1.0,
    },
    "Retail": {
        "dead_psf": 65.0,
        "live_psf": 100.0,
        "risk_category": "II",
        "importance_factor": 1.0,
    },
    "Warehouse": {
        "dead_psf": 80.0,
        "live_psf": 125.0,
        "risk_category": "II",
        "importance_factor": 1.0,
    },
    "Residential": {
        "dead_psf": 45.0,
        "live_psf": 40.0,
        "risk_category": "II",
        "importance_factor": 1.0,
    },
    "Hospital": {
        "dead_psf": 85.0,
        "live_psf": 80.0,
        "risk_category": "IV",
        "importance_factor": 1.25,
    },
}

SYSTEM_RESPONSE_FACTORS = {
    "OMF": 3.5,
    "SMF": 6.0,
}

EXPOSURE_CONSTANTS = {
    "B": {"alpha": 7.0, "zg": 1200.0},
    "C": {"alpha": 9.5, "zg": 900.0},
    "D": {"alpha": 11.5, "zg": 700.0},
}

CITY_DATA = {
    "Albuquerque": {"latitude": 35.0844, "longitude": -106.6504, "wind_mph": 115, "ground_snow_pg": 10},
    "Atlanta": {"latitude": 33.7490, "longitude": -84.3880, "wind_mph": 105, "ground_snow_pg": 5},
    "Austin": {"latitude": 30.2672, "longitude": -97.7431, "wind_mph": 115, "ground_snow_pg": 0},
    "Baltimore": {"latitude": 39.2904, "longitude": -76.6122, "wind_mph": 115, "ground_snow_pg": 20},
    "Boston": {"latitude": 42.3601, "longitude": -71.0589, "wind_mph": 115, "ground_snow_pg": 30},
    "Charlotte": {"latitude": 35.2271, "longitude": -80.8431, "wind_mph": 105, "ground_snow_pg": 5},
    "Chicago": {"latitude": 41.8781, "longitude": -87.6298, "wind_mph": 115, "ground_snow_pg": 25},
    "Cleveland": {"latitude": 41.4993, "longitude": -81.6944, "wind_mph": 115, "ground_snow_pg": 30},
    "Colorado Springs": {"latitude": 38.8339, "longitude": -104.8214, "wind_mph": 115, "ground_snow_pg": 30},
    "Columbus": {"latitude": 39.9612, "longitude": -82.9988, "wind_mph": 115, "ground_snow_pg": 20},
    "Dallas": {"latitude": 32.7767, "longitude": -96.7970, "wind_mph": 115, "ground_snow_pg": 5},
    "Denver": {"latitude": 39.7392, "longitude": -104.9903, "wind_mph": 115, "ground_snow_pg": 30},
    "Detroit": {"latitude": 42.3314, "longitude": -83.0458, "wind_mph": 115, "ground_snow_pg": 25},
    "El Paso": {"latitude": 31.7619, "longitude": -106.4850, "wind_mph": 115, "ground_snow_pg": 5},
    "Fort Worth": {"latitude": 32.7555, "longitude": -97.3308, "wind_mph": 115, "ground_snow_pg": 5},
    "Fresno": {"latitude": 36.7378, "longitude": -119.7871, "wind_mph": 95, "ground_snow_pg": 0},
    "Houston": {"latitude": 29.7604, "longitude": -95.3698, "wind_mph": 130, "ground_snow_pg": 0},
    "Indianapolis": {"latitude": 39.7684, "longitude": -86.1581, "wind_mph": 115, "ground_snow_pg": 20},
    "Jacksonville": {"latitude": 30.3322, "longitude": -81.6557, "wind_mph": 130, "ground_snow_pg": 0},
    "Kansas City": {"latitude": 39.0997, "longitude": -94.5786, "wind_mph": 115, "ground_snow_pg": 20},
    "Las Vegas": {"latitude": 36.1699, "longitude": -115.1398, "wind_mph": 105, "ground_snow_pg": 0},
    "Long Beach": {"latitude": 33.7701, "longitude": -118.1937, "wind_mph": 100, "ground_snow_pg": 0},
    "Los Angeles": {"latitude": 34.0522, "longitude": -118.2437, "wind_mph": 95, "ground_snow_pg": 0},
    "Louisville": {"latitude": 38.2527, "longitude": -85.7585, "wind_mph": 115, "ground_snow_pg": 10},
    "Memphis": {"latitude": 35.1495, "longitude": -90.0490, "wind_mph": 115, "ground_snow_pg": 5},
    "Mesa": {"latitude": 33.4152, "longitude": -111.8315, "wind_mph": 105, "ground_snow_pg": 0},
    "Miami": {"latitude": 25.7617, "longitude": -80.1918, "wind_mph": 175, "ground_snow_pg": 0},
    "Milwaukee": {"latitude": 43.0389, "longitude": -87.9065, "wind_mph": 115, "ground_snow_pg": 35},
    "Minneapolis": {"latitude": 44.9778, "longitude": -93.2650, "wind_mph": 115, "ground_snow_pg": 50},
    "Nashville": {"latitude": 36.1627, "longitude": -86.7816, "wind_mph": 115, "ground_snow_pg": 5},
    "New Orleans": {"latitude": 29.9511, "longitude": -90.0715, "wind_mph": 140, "ground_snow_pg": 0},
    "New York": {"latitude": 40.7128, "longitude": -74.0060, "wind_mph": 115, "ground_snow_pg": 25},
    "Oakland": {"latitude": 37.8044, "longitude": -122.2711, "wind_mph": 100, "ground_snow_pg": 0},
    "Oklahoma City": {"latitude": 35.4676, "longitude": -97.5164, "wind_mph": 115, "ground_snow_pg": 10},
    "Omaha": {"latitude": 41.2565, "longitude": -95.9345, "wind_mph": 115, "ground_snow_pg": 25},
    "Philadelphia": {"latitude": 39.9526, "longitude": -75.1652, "wind_mph": 115, "ground_snow_pg": 20},
    "Phoenix": {"latitude": 33.4484, "longitude": -112.0740, "wind_mph": 105, "ground_snow_pg": 0},
    "Portland": {"latitude": 45.5152, "longitude": -122.6784, "wind_mph": 110, "ground_snow_pg": 10},
    "Raleigh": {"latitude": 35.7796, "longitude": -78.6382, "wind_mph": 110, "ground_snow_pg": 5},
    "Sacramento": {"latitude": 38.5816, "longitude": -121.4944, "wind_mph": 95, "ground_snow_pg": 0},
    "San Antonio": {"latitude": 29.4241, "longitude": -98.4936, "wind_mph": 115, "ground_snow_pg": 0},
    "San Diego": {"latitude": 32.7157, "longitude": -117.1611, "wind_mph": 95, "ground_snow_pg": 0},
    "San Francisco": {"latitude": 37.7749, "longitude": -122.4194, "wind_mph": 100, "ground_snow_pg": 0},
    "San Jose": {"latitude": 37.3382, "longitude": -121.8863, "wind_mph": 95, "ground_snow_pg": 0},
    "Seattle": {"latitude": 47.6062, "longitude": -122.3321, "wind_mph": 110, "ground_snow_pg": 20},
    "Tampa": {"latitude": 27.9506, "longitude": -82.4572, "wind_mph": 130, "ground_snow_pg": 0},
    "Tulsa": {"latitude": 36.1539, "longitude": -95.9928, "wind_mph": 115, "ground_snow_pg": 10},
    "Tucson": {"latitude": 32.2226, "longitude": -110.9747, "wind_mph": 105, "ground_snow_pg": 0},
    "Virginia Beach": {"latitude": 36.8529, "longitude": -75.9780, "wind_mph": 130, "ground_snow_pg": 10},
    "Washington, DC": {"latitude": 38.9072, "longitude": -77.0369, "wind_mph": 115, "ground_snow_pg": 15},
}


def list_supported_cities() -> list[str]:
    return sorted(CITY_DATA.keys())


def normalize_city_name(city: str) -> str:
    city_key = city.strip()
    for known in CITY_DATA:
        if known.lower() == city_key.lower():
            return known
    raise KeyError(f"Unsupported city '{city}'.")


def get_city_data(city: str) -> dict[str, Any]:
    return CITY_DATA[normalize_city_name(city)].copy()


def get_occupancy_defaults(occupancy: str) -> dict[str, Any]:
    for key, value in OCCUPANCY_DEFAULTS.items():
        if key.lower() == occupancy.strip().lower():
            return value.copy()
    raise KeyError(f"Unsupported occupancy '{occupancy}'.")


def _story_elevations(height_ft: float, floor_height_ft: float = DEFAULT_FLOOR_HEIGHT_FT) -> list[float]:
    num_levels = max(1, int(math.ceil(height_ft / floor_height_ft)))
    return [round(min((index + 1) * floor_height_ft, height_ft), 1) for index in range(num_levels)]


def _kz(exposure: str, z_ft: float) -> float:
    exposure_key = exposure.upper()
    if exposure_key not in EXPOSURE_CONSTANTS:
        raise KeyError(f"Unsupported exposure '{exposure}'.")
    constants = EXPOSURE_CONSTANTS[exposure_key]
    alpha = constants["alpha"]
    zg = constants["zg"]
    z_eval = max(15.0, min(z_ft, zg))
    return round(2.01 * (z_eval / zg) ** (2.0 / alpha), 3)


def get_wind_loads(
    city: str,
    height_ft: float,
    exposure: str = DEFAULT_EXPOSURE,
    occupancy: str = "office",
    floor_height_ft: float = DEFAULT_FLOOR_HEIGHT_FT,
) -> dict[str, Any]:
    """
    Return ASCE-style velocity pressure estimates by floor.
    """
    city_data = get_city_data(city)
    elevations = _story_elevations(height_ft, floor_height_ft=floor_height_ft)
    velocity = float(city_data["wind_mph"])
    Kzt = 1.0
    Kd = 0.85
    floor_pressures = []

    for index, elevation in enumerate(elevations, start=1):
        Kz = _kz(exposure, elevation)
        qz = 0.00256 * Kz * Kzt * Kd * velocity**2
        floor_pressures.append(
            {
                "floor": index,
                "elevation_ft": elevation,
                "Kz": round(Kz, 3),
                "pressure_psf": round(qz, 1),
            }
        )

    return {
        "city": normalize_city_name(city),
        "occupancy": occupancy.title(),
        "basic_wind_speed_mph": velocity,
        "exposure": exposure.upper(),
        "Kzt": Kzt,
        "Kd": Kd,
        "height_ft": round(height_ft, 1),
        "floor_pressures": floor_pressures,
        "roof_pressure_psf": floor_pressures[-1]["pressure_psf"],
        "notes": "Velocity pressure estimate qz for preliminary sizing.",
    }


@lru_cache(maxsize=128)
def _fetch_usgs_seismic(city: str, site_class: str, risk_category: str) -> dict[str, Any]:
    city_data = get_city_data(city)
    query = urlencode(
        {
            "latitude": city_data["latitude"],
            "longitude": city_data["longitude"],
            "riskCategory": risk_category,
            "siteClass": site_class,
            "title": normalize_city_name(city),
        }
    )
    url = f"{USGS_ASCE7_22_ENDPOINT}?{query}"
    with urlopen(url, timeout=15) as response:
        payload = json.loads(response.read().decode("utf-8"))
    status = payload.get("request", {}).get("status")
    if status != "success":
        raise RuntimeError(payload.get("response", {}).get("error", "USGS seismic service error"))
    return payload


def get_seismic_loads(
    city: str,
    num_floors: int,
    floor_weight_kips: float | list[float],
    site_class: str = DEFAULT_SITE_CLASS,
    occupancy: str = "office",
    system: str = DEFAULT_SYSTEM,
    floor_height_ft: float = DEFAULT_FLOOR_HEIGHT_FT,
) -> dict[str, Any]:
    """
    Return USGS seismic design parameters and floor-force distribution.
    """
    occupancy_defaults = get_occupancy_defaults(occupancy.title())
    risk_category = occupancy_defaults["risk_category"]
    importance_factor = float(occupancy_defaults["importance_factor"])
    response_factor = SYSTEM_RESPONSE_FACTORS.get(system.upper(), SYSTEM_RESPONSE_FACTORS[DEFAULT_SYSTEM])

    payload = _fetch_usgs_seismic(normalize_city_name(city), site_class, risk_category)
    data = payload.get("response", {}).get("data", {})

    if isinstance(floor_weight_kips, (int, float)):
        weights = [float(floor_weight_kips)] * int(num_floors)
    else:
        weights = [float(value) for value in floor_weight_kips]
    if len(weights) != int(num_floors):
        raise ValueError("floor_weight_kips must be a scalar or a list matching num_floors.")

    heights = [(index + 1) * floor_height_ft for index in range(int(num_floors))]
    total_weight = sum(weights)
    denominator = sum(weight * height for weight, height in zip(weights, heights))

    sds = float(data.get("sds", 0.0))
    sd1 = float(data.get("sd1", 0.0))
    ss = float(data.get("ss", 0.0))
    s1 = float(data.get("s1", 0.0))
    Cs = sds / (response_factor / importance_factor) if response_factor > 0 else 0.0
    base_shear = Cs * total_weight

    floor_forces = []
    for index, (weight, height) in enumerate(zip(weights, heights), start=1):
        Fx = base_shear * (weight * height) / denominator if denominator > 0 else 0.0
        floor_forces.append(
            {
                "floor": index,
                "height_ft": round(height, 1),
                "weight_kips": round(weight, 2),
                "force_kips": round(Fx, 2),
            }
        )

    return {
        "city": normalize_city_name(city),
        "site_class": site_class,
        "risk_category": risk_category,
        "importance_factor": importance_factor,
        "response_system": system.upper(),
        "R": response_factor,
        "ss": round(ss, 3),
        "s1": round(s1, 3),
        "sds": round(sds, 3),
        "sd1": round(sd1, 3),
        "sdc": data.get("sdc", ""),
        "base_shear_kips": round(base_shear, 2),
        "effective_weight_kips": round(total_weight, 2),
        "floor_forces": floor_forces,
        "api_url": payload.get("request", {}).get("url", ""),
    }


def get_snow_load(city: str, roof_slope_deg: float = 0) -> dict[str, Any]:
    """
    Return preliminary flat-roof snow load from city lookup values.
    """
    city_data = get_city_data(city)
    Pg = float(city_data["ground_snow_pg"])
    Ce = 1.0
    Ct = 1.0
    Is = 1.0
    slope_factor = 1.0 if roof_slope_deg <= 5 else max(0.75, 1.0 - (roof_slope_deg - 5) / 100.0)
    Ps = 0.7 * Ce * Ct * Is * Pg * slope_factor
    return {
        "city": normalize_city_name(city),
        "ground_snow_pg": round(Pg, 1),
        "roof_snow_psf": round(Ps, 1),
        "Ce": Ce,
        "Ct": Ct,
        "Is": Is,
        "roof_slope_deg": round(float(roof_slope_deg), 1),
    }


def get_governing_combination(DL: float, LL: float, W: float, E: float, S: float) -> dict[str, Any]:
    """
    Return the controlling ASCE 7 load combination.
    """
    combinations = [
        {"name": "1.4D", "value": 1.4 * DL},
        {"name": "1.2D + 1.6L + 0.5S", "value": 1.2 * DL + 1.6 * LL + 0.5 * S},
        {"name": "1.2D + 1.6S + max(L, 0.5W)", "value": 1.2 * DL + 1.6 * S + max(LL, 0.5 * W)},
        {"name": "1.2D + 1.0W + L + 0.5S", "value": 1.2 * DL + 1.0 * W + LL + 0.5 * S},
        {"name": "0.9D + 1.0W", "value": 0.9 * DL + 1.0 * W},
        {"name": "1.2D + 1.0E + L + 0.2S", "value": 1.2 * DL + 1.0 * E + LL + 0.2 * S},
        {"name": "0.9D + 1.0E", "value": 0.9 * DL + 1.0 * E},
    ]
    governing = max(combinations, key=lambda item: item["value"])
    return {
        "governing_name": governing["name"],
        "governing_value": round(governing["value"], 2),
        "combinations": [
            {"name": item["name"], "value": round(item["value"], 2)}
            for item in combinations
        ],
    }


def auto_load_brief(
    city: str,
    num_floors: int,
    floor_height_ft: float,
    bay_length_ft: float,
    bay_width_ft: float,
    occupancy: str = "office",
) -> dict[str, Any]:
    """
    Generate a complete preliminary loading brief from city, occupancy, and geometry.

    This is intentionally schematic-level and should be reviewed for a real project.
    """
    occupancy_defaults = get_occupancy_defaults(occupancy.title())
    dead_psf = float(occupancy_defaults["dead_psf"])
    live_psf = float(occupancy_defaults["live_psf"])
    total_height_ft = float(num_floors) * float(floor_height_ft)
    tributary_area_ft2 = float(bay_length_ft) * float(bay_width_ft)
    effective_floor_weight_kips = tributary_area_ft2 * (dead_psf + 0.25 * live_psf) / 1000.0

    wind_data = get_wind_loads(
        city,
        total_height_ft,
        exposure=DEFAULT_EXPOSURE,
        occupancy=occupancy,
        floor_height_ft=floor_height_ft,
    )
    snow_data = get_snow_load(city)

    seismic_error = None
    try:
        seismic_data = get_seismic_loads(
            city,
            num_floors,
            effective_floor_weight_kips,
            site_class=DEFAULT_SITE_CLASS,
            occupancy=occupancy,
            floor_height_ft=floor_height_ft,
        )
    except Exception as exc:
        seismic_error = str(exc)
        seismic_data = {
            "city": normalize_city_name(city),
            "site_class": DEFAULT_SITE_CLASS,
            "risk_category": occupancy_defaults["risk_category"],
            "importance_factor": occupancy_defaults["importance_factor"],
            "response_system": DEFAULT_SYSTEM,
            "R": SYSTEM_RESPONSE_FACTORS[DEFAULT_SYSTEM],
            "ss": 0.0,
            "s1": 0.0,
            "sds": 0.0,
            "sd1": 0.0,
            "sdc": "",
            "base_shear_kips": 0.0,
            "effective_weight_kips": round(effective_floor_weight_kips * num_floors, 2),
            "floor_forces": [
                {
                    "floor": index + 1,
                    "height_ft": round((index + 1) * float(floor_height_ft), 1),
                    "weight_kips": round(effective_floor_weight_kips, 2),
                    "force_kips": 0.0,
                }
                for index in range(int(num_floors))
            ],
            "api_url": "",
        }

    footprint_area_ft2 = max(tributary_area_ft2, 1.0)
    seismic_equivalent_psf = float(seismic_data["base_shear_kips"]) * 1000.0 / footprint_area_ft2
    governing = get_governing_combination(
        DL=dead_psf,
        LL=live_psf,
        W=float(wind_data["roof_pressure_psf"]),
        E=seismic_equivalent_psf,
        S=float(snow_data["roof_snow_psf"]),
    )
    wu_gravity = 1.2 * dead_psf + 1.6 * live_psf

    return {
        "dead_psf": dead_psf,
        "live_psf": live_psf,
        "wind_psf": round(float(wind_data["roof_pressure_psf"]), 2),
        "snow_psf": round(float(snow_data["roof_snow_psf"]), 2),
        "seismic_V_kips": round(float(seismic_data["base_shear_kips"]), 2),
        "seismic_Fx": [round(float(item.get("force_kips", 0.0)), 2) for item in seismic_data.get("floor_forces", [])],
        "governing_combo": governing["governing_name"],
        "governing_combo_value": governing["governing_value"],
        "Wu_gravity": round(wu_gravity, 2),
        "city": normalize_city_name(city),
        "occupancy": occupancy.lower(),
        "wind_data": wind_data,
        "seismic_data": seismic_data,
        "snow_data": snow_data,
        "loads_auto_calculated": True,
        "seismic_error": seismic_error,
    }
