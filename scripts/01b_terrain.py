"""Extract station-local DEM gradients and 850-hPa u dot grad(h), in m/s."""
import math
import os
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

BASE = Path(os.environ.get('RAINFALL_PROJECT_DIR', Path(__file__).resolve().parent.parent))
DEM = BASE / 'data/external/dem'
OUT = Path(os.environ.get('RAINFALL_OUTPUT_DIR', BASE / 'outputs'))
STATIONS = {
    'phuket': (7 + 53 / 60, 98 + 24 / 60, 2),
    'phangnga': (8 + 41 / 60, 98 + 15 / 60, 6),
    'krabi': (8 + 6 / 60, 98 + 58 / 60, 29),
    'nakhon_si_thammarat': (8 + 32 / 60, 99 + 56 / 60, 4),
}


def tile(lat, lon):
    return DEM / f'Copernicus_DSM_COG_10_N{math.floor(lat):02d}_00_E{math.floor(lon):03d}_00_DEM.tif'


def terrain_gradient(elevation, transform, latitude):
    """Median dh/dx (east), dh/dy (north), preserving signed raster spacing.

    North-up geographic rasters have transform.e < 0; row number increases
    SOUTH. The existing 21x21-pixel window/approximate geographic metric is
    retained as a local sensitivity proxy, not a mountain-wide vertical wind.
    """
    if transform.b != 0 or transform.d != 0:
        raise ValueError('Rotated/sheared DEM grids are not supported.')
    dx = transform.a * 111320 * math.cos(math.radians(latitude))
    dy = transform.e * 110574
    if dx == 0 or dy == 0:
        raise ValueError('DEM must have nonzero pixel spacing.')
    arr = np.asarray(elevation, dtype=float)
    gy, gx = np.gradient(arr, dy, dx)
    if not np.isfinite(gx).any() or not np.isfinite(gy).any():
        raise ValueError('No valid DEM gradients at the station.')
    return float(np.nanmedian(gx)), float(np.nanmedian(gy))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, (lat, lon, station_elev) in STATIONS.items():
        path = tile(lat, lon)
        if not path.exists():
            raise FileNotFoundError(path)
        with rasterio.open(path) as src:
            if src.crs is None or not src.crs.is_geographic:
                raise ValueError(f'Expected geographic DEM coordinates: {path}')
            r, c = src.index(lon, lat)
            arr = src.read(1, window=Window(c - 10, r - 10, 21, 21),
                           masked=True, boundless=True).astype(float).filled(np.nan)
            dzdx, dzdy = terrain_gradient(arr, src.transform, lat)
            elev = float(next(src.sample([(lon, lat)], masked=True)).filled(np.nan)[0])
        rows.append(dict(station=name, latitude=lat, longitude=lon,
                         station_elevation_m=station_elev, dem_elevation_m=elev,
                         dzdx=dzdx, dzdy=dzdy,
                         slope_deg=math.degrees(math.atan(math.hypot(dzdx, dzdy))),
                         upslope_aspect_deg=(math.degrees(math.atan2(dzdx, dzdy)) + 360) % 360,
                         dem_tile=path.name, window_pixels=21,
                         gradient_convention='east_positive_x;north_positive_y',
                         forcing_units='m s-1'))
    terrain = pd.DataFrame(rows)
    terrain.to_csv(OUT / 'station_terrain.csv', index=False)
    snd = pd.read_csv(OUT / 'sounding_indices_raw.csv', parse_dates=['date'])
    for x in rows:
        snd[f"OROG850_{x['station']}"] = snd['U850'] * x['dzdx'] + snd['V850'] * x['dzdy']
    cols = ['date'] + [f'OROG850_{s}' for s in STATIONS]
    snd[cols].to_csv(OUT / 'orographic_forcing_daily.csv', index=False)
    print(terrain.to_string(index=False))


if __name__ == '__main__':
    main()
