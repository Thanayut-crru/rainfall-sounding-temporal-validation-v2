"""Coordinate provenance and descriptive sounding-to-gauge distances.

Gauge coordinates reproduce the station table supplied by the author (degrees
and minutes); the sounding coordinates reproduce NOAA IGRA station THM00048565.
These distances are descriptive metadata, NOT classifier input features.
"""
from __future__ import annotations

from math import asin, cos, isfinite, radians, sin, sqrt

SOUNDING_LATITUDE = 8.1450
SOUNDING_LONGITUDE = 98.3144
SOUNDING_STATION_ID = 'THM00048565'
EARTH_RADIUS_KM = 6371.0

GAUGE_COORDINATES = {
    'phuket': (7 + 53 / 60, 98 + 24 / 60),
    'krabi': (8 + 6 / 60, 98 + 58 / 60),
    'phangnga': (8 + 41 / 60, 98 + 15 / 60),
    'nakhon_si_thammarat': (8 + 32 / 60, 99 + 56 / 60),
}

ALIASES = {
    'takua_pa': 'phangnga', 'takua pa': 'phangnga', 'takua_pa_(phang-nga)': 'phangnga',
    'takua pa (phang-nga)': 'phangnga', 'phang-nga': 'phangnga', 'phang nga': 'phangnga',
    'nakhon si thammarat': 'nakhon_si_thammarat',
}

PROVENANCE = {
    'gauge_coordinates': 'Author-supplied TMD station table: Phuket 07 53 N,98 24 E; '
                         'Krabi 08 06 N,98 58 E; Takua Pa 08 41 N,98 15 E; '
                         'Nakhon Si Thammarat 08 32 N,99 56 E. '
                         'The supplied station table is also represented in station_terrain.csv.',
    'sounding_coordinates': 'NOAA NCEI IGRA station list: THM00048565, Phuket Airport, '
                           '8.1450 N,98.3144 E; these coordinates identify the upper-air '
                           'reference, not the rounded surface-airport gauge coordinates.',
    'sounding_source_url': 'https://www.ncei.noaa.gov/pub/data/igra/igra2-station-list.txt',
    'method': 'Haversine great-circle distance on a spherical Earth of radius 6371 km; '
              'not road distance and not a terrain-flow trajectory.',
    'role': 'Descriptive station metadata only; not used to fit or select classifiers.',
}


def _validate(lat: float, lon: float) -> None:
    if not all(isfinite(v) for v in (lat, lon)):
        raise ValueError('Coordinates must be finite')
    if not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise ValueError('Latitude/longitude must be in geographic degree bounds')


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return symmetric great-circle separation; inputs are geographic degrees."""
    lat1, lon1, lat2, lon2 = map(float, (lat1, lon1, lat2, lon2))
    _validate(lat1, lon1)
    _validate(lat2, lon2)
    a, b, c, e = map(radians, (lat1, lon1, lat2, lon2))
    half_chord = sin((c - a) / 2) ** 2 + cos(a) * cos(c) * sin((e - b) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * asin(sqrt(min(1.0, max(0.0, half_chord))))


def canonical_station(name: str) -> str:
    value = str(name).strip().lower()
    value = ALIASES.get(value, value)
    if value not in GAUGE_COORDINATES:
        raise KeyError(f'Unrecognized rainfall gauge: {name!r}')
    return value


def station_distance_km(name: str) -> float:
    """Return unrounded great-circle distance from the IGRA sounding reference."""
    latitude, longitude = GAUGE_COORDINATES[canonical_station(name)]
    return haversine_km(SOUNDING_LATITUDE, SOUNDING_LONGITUDE, latitude, longitude)


def station_metadata_records() -> list[dict]:
    return [{'station': name, 'latitude': coordinates[0], 'longitude': coordinates[1],
             'sounding_latitude': SOUNDING_LATITUDE, 'sounding_longitude': SOUNDING_LONGITUDE,
             'distance_km': station_distance_km(name)}
            for name, coordinates in GAUGE_COORDINATES.items()]
