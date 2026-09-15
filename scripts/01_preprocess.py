"""Create 00 UTC predictors from the local NOAA IGRA v2.2 profile archive.

Revision-stage physical QC, not prespecified or tuned using model performance:
* eligibility: >=5 distinct paired p/T/Td levels, origin >=950 hPa, coverage
  through 500 hPa, no >200 hPa paired-data gap in the required layer;
* CAPE/CIN/PWAT require coverage through 100 hPa with the same gap limit;
* positive CAPE without an observed EL is censored, not a complete integral;
* indices are evaluated independently; missing upper wind does not exclude a
  profile. These study-specific checks are not claimed universal standards.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import zipfile
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
from metpy.calc import cape_cin, el, lcl, lfc, parcel_profile, precipitable_water
from metpy.units import units

BASE = Path(os.environ.get('RAINFALL_PROJECT_DIR', Path(__file__).resolve().parent.parent))
ARCHIVE = BASE / 'data/external/radiosonde/THM00048565-data.txt.zip'
OUT = Path(os.environ.get('RAINFALL_OUTPUT_DIR', BASE / 'outputs'))
MIN_THERMO_LEVELS = 5
MIN_ORIGIN_HPA = 950.0
CORE_TOP_HPA = 500.0
INTEGRAL_TOP_HPA = 100.0
MAX_GAP_HPA = 200.0
THERMO_FEATURES = ('CTOT', 'VTOT', 'TTOT', 'KINX', 'CAPE', 'CINS', 'LCLP',
                   'LCLT', 'LFCT', 'EQLV', 'PWAT', 'SHOW', 'LIFT')


def val(s, a, b):
    try:
        x = int(s[a:b])
    except ValueError:
        return None
    return None if x in (-9999, -8888) else x


def profiles(path):
    """Read fixed-width IGRA values and minor level type (1 = surface)."""
    with zipfile.ZipFile(path) as z, z.open(z.namelist()[0]) as fh:
        it = iter(fh)
        for raw in it:
            h = raw.decode('ascii', 'ignore')
            if not h.startswith('#'):
                continue
            n = val(h, 32, 36) or 0
            meta = dict(station_id=h[1:12].strip(), year=val(h, 13, 17),
                        month=val(h, 18, 20), day=val(h, 21, 23),
                        hour=val(h, 24, 26), release_time=val(h, 27, 31),
                        pressure_source=h[37:45].strip())
            lev = []
            for _ in range(n):
                s = next(it).decode('ascii', 'ignore')
                lev.append([val(s, 9, 15), val(s, 16, 21), val(s, 22, 27),
                            val(s, 28, 33), val(s, 34, 39), val(s, 40, 45),
                            val(s, 46, 51), val(s, 1, 2)])
            yield meta, lev


def interp(p, x, target, max_gap=MAX_GAP_HPA):
    """Log-pressure interpolation, without extrapolation or wide-gap bridging."""
    p, x = np.asarray(p, float), np.asarray(x, float)
    ok = np.isfinite(p) & np.isfinite(x) & (p > 0)
    p, x = p[ok], x[ok]
    if not len(p):
        return np.nan
    ix = np.argsort(p, kind='stable')
    p, x = p[ix], x[ix]
    p, ix = np.unique(p, return_index=True)
    x = x[ix]
    exact = np.flatnonzero(np.isclose(p, target, rtol=0, atol=1e-6))
    if len(exact):
        return float(x[exact[0]])
    k = np.searchsorted(p, target)
    if k == 0 or k == len(p) or p[k] - p[k - 1] > max_gap:
        return np.nan
    return float(np.interp(np.log(target), np.log(p[k - 1:k + 1]), x[k - 1:k + 1]))


def layer_covered(p, bottom, top, max_gap=MAX_GAP_HPA):
    """Check actual bracketing observations across the entire requested layer."""
    p = np.unique(np.asarray(p, float))
    p = p[np.isfinite(p) & (p > 0)]
    if len(p) < 2 or p[0] > top or p[-1] < bottom:
        return False
    # Keep brackets beyond the boundaries so clipping cannot conceal a gap.
    lo = max(0, np.searchsorted(p, top, side='right') - 1)
    hi = min(len(p) - 1, np.searchsorted(p, bottom, side='left'))
    return bool(np.all(np.diff(p[lo:hi + 1]) <= max_gap))


def mag(x, unit=None):
    if unit:
        x = x.to(unit)
    y = float(x.magnitude)
    return y if np.isfinite(y) else np.nan


def derive(meta, lev):
    a = np.asarray([[np.nan if x is None else x for x in row] for row in lev], float)
    if not len(a):
        a = np.empty((0, 8))
    p, h, t, rh, dd, wd, ws = a[:, :7].T
    level_type = a[:, 7] if a.shape[1] > 7 else np.full(len(a), np.nan)
    p, t, rh, dd, ws = p / 100, t / 10, rh / 10, dd / 10, ws / 10
    p = np.where(p > 0, p, np.nan)
    td = t - dd
    ok = np.isfinite(p) & np.isfinite(t) & np.isfinite(td)
    idx = np.flatnonzero(ok)
    idx = idx[np.argsort(-p[idx], kind='stable')]
    if len(idx):
        _, first = np.unique(p[idx], return_index=True)
        idx = idx[np.sort(first)]
    pp0, tt0, td0 = p[idx], t[idx], td[idx]
    origin = float(pp0[0]) if len(pp0) else np.nan
    is_surface = bool(len(idx) and level_type[idx[0]] == 1)
    core = bool(len(pp0) >= MIN_THERMO_LEVELS and origin >= MIN_ORIGIN_HPA
                and layer_covered(pp0, origin, CORE_TOP_HPA))
    deep = bool(core and layer_covered(pp0, origin, INTEGRAL_TOP_HPA))
    reasons = []
    if len(pp0) < MIN_THERMO_LEVELS:
        reasons.append('fewer_than_5_paired_levels')
    if not np.isfinite(origin) or origin < MIN_ORIGIN_HPA:
        reasons.append('no_near_surface_paired_level_ge950hPa')
    if not len(pp0) or pp0[-1] > CORE_TOP_HPA:
        reasons.append('paired_profile_does_not_reach_500hPa')
    elif not layer_covered(pp0, origin, CORE_TOP_HPA):
        reasons.append('paired_gap_over_200hPa_in_core_layer')
    row = {'date': date(meta['year'], meta['month'], meta['day']).isoformat(),
           **meta, 'n_levels': len(a), 'n_thermo_levels': len(pp0),
           'thermo_bottom_hpa': origin,
           'thermo_top_hpa': float(pp0[-1]) if len(pp0) else np.nan,
           'parcel_origin_hpa': origin, 'parcel_origin_is_igra_surface': is_surface,
           'parcel_origin_kind': 'IGRA_surface' if is_surface else 'lowest_usable_level',
           'qc_thermo_usable': core, 'qc_integral_column': deep,
           'qc_exclusion_reasons': ';'.join(reasons),
           'integral_top_hpa': INTEGRAL_TOP_HPA,
           'cape_status': 'incomplete_column', 'CAPE_SB': np.nan,
           'CAPE_observed_lower_bound': np.nan}
    row.update({k: np.nan for k in THERMO_FEATURES})
    errors = []

    def calculate(name, fn):
        # Catch calculation-local failures, never wipe unrelated valid indices.
        try:
            return fn()
        except (ValueError, IndexError, ZeroDivisionError, FloatingPointError, RuntimeError) as e:
            errors.append(f'{name}:{type(e).__name__}:{str(e)[:100]}')
            return None

    t850, t700, t500 = [interp(p, t, q) for q in (850, 700, 500)]
    td850, td700 = [interp(p, td, q) for q in (850, 700)]
    row.update(CTOT=td850 - t500, VTOT=t850 - t500,
               TTOT=td850 + t850 - 2 * t500,
               KINX=t850 - t500 + td850 - (t700 - td700))
    if len(pp0) and origin >= MIN_ORIGIN_HPA:
        lc = calculate('LCL', lambda: lcl(origin * units.hPa,
                                         tt0[0] * units.degC, td0[0] * units.degC))
        if lc is not None:
            row['LCLP'], row['LCLT'] = mag(lc[0], 'hPa'), mag(lc[1], 'kelvin')
    if np.all(np.isfinite([t850, td850, t500])):
        show = calculate('SHOW', lambda: parcel_profile(np.array([850, 500]) * units.hPa,
                                                        t850 * units.degC, td850 * units.degC))
        if show is not None:
            row['SHOW'] = t500 - mag(show[-1], 'degC')
    if core:
        lifted = calculate('LIFT', lambda: parcel_profile(np.array([origin, 500]) * units.hPa,
                                                          tt0[0] * units.degC, td0[0] * units.degC))
        if lifted is not None and np.isfinite(t500):
            row['LIFT'] = t500 - mag(lifted[-1], 'degC')
    if deep:
        # Use a consistent 100-hPa endpoint instead of a varying sounding top.
        keep = pp0 > INTEGRAL_TOP_HPA
        pp = np.r_[pp0[keep], INTEGRAL_TOP_HPA] * units.hPa
        tt = np.r_[tt0[keep], interp(pp0, tt0, INTEGRAL_TOP_HPA)] * units.degC
        tdd = np.r_[td0[keep], interp(pp0, td0, INTEGRAL_TOP_HPA)] * units.degC
        water = calculate('PWAT', lambda: precipitable_water(pp, tdd, bottom=pp[0], top=pp[-1]))
        if water is not None:
            row['PWAT'] = mag(water, 'mm')
        prof = calculate('parcel_profile', lambda: parcel_profile(pp, tt[0], tdd[0]))
        if prof is not None:
            fp = calculate('LFC', lambda: lfc(pp, tt, tdd, parcel_temperature_profile=prof, which='bottom'))
            ep = calculate('EL', lambda: el(pp, tt, tdd, parcel_temperature_profile=prof, which='top'))
            if fp is not None:
                row['LFCT'] = mag(fp[0], 'hPa')
            if ep is not None:
                row['EQLV'] = mag(ep[0], 'hPa')
            cc = calculate('CAPE_CIN', lambda: cape_cin(pp, tt, tdd, prof,
                                                      which_lfc='bottom', which_el='top'))
            if cc is not None:
                cp, ci = mag(cc[0], 'joule/kg'), mag(cc[1], 'joule/kg')
                row['CINS'] = ci
                if np.isfinite(cp) and cp > 0 and not np.isfinite(row['EQLV']):
                    row['CAPE_observed_lower_bound'] = cp
                    row['cape_status'] = 'positive_buoyancy_no_observed_EL_censored'
                elif np.isfinite(cp):
                    row['CAPE'] = cp
                    row['CAPE_SB'] = cp if is_surface else np.nan
                    row['cape_status'] = 'complete_nonbuoyant_zero' if cp == 0 else 'complete_observed_EL'
                else:
                    row['cape_status'] = 'calculation_nonfinite'
            else:
                row['cape_status'] = 'calculation_failed'
        else:
            row['cape_status'] = 'parcel_calculation_failed'
    r = np.deg2rad(wd)
    u, v = -ws * np.sin(r), -ws * np.cos(r)
    for q in (850, 700, 200):
        uu, vv = interp(p, u, q), interp(p, v, q)
        row[f'U{q}'], row[f'V{q}'] = uu, vv
        row[f'WS{q}'] = math.hypot(uu, vv) if np.isfinite(uu + vv) else np.nan
        row[f'qc_wind{q}_available'] = bool(np.isfinite(uu + vv))
    speed = row['WS850']
    row['WDsin850'] = -row['U850'] / speed if np.isfinite(speed) and speed > 0 else np.nan
    row['WDcos850'] = -row['V850'] / speed if np.isfinite(speed) and speed > 0 else np.nan
    row['VWS'] = math.hypot(row['U200'] - row['U850'], row['V200'] - row['V850'])
    row['VWS_lo'] = math.hypot(row['U700'] - row['U850'], row['V700'] - row['V850'])
    row['calculation_errors'] = ';'.join(errors)
    for key in THERMO_FEATURES:
        row[f'qc_{key}_available'] = bool(np.isfinite(row[key]))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2011-01-01')
    ap.add_argument('--end', default='2021-12-31')
    ap.add_argument('--hour', type=int, default=0)
    o = ap.parse_args()
    start, end = date.fromisoformat(o.start), date.fromisoformat(o.end)
    if not ARCHIVE.exists():
        raise FileNotFoundError(ARCHIVE)
    OUT.mkdir(parents=True, exist_ok=True)
    rows, failures = [], []
    for m, levels in profiles(ARCHIVE):
        if m['year'] is None or m['hour'] != o.hour:
            continue
        d = date(m['year'], m['month'], m['day'])
        if start <= d <= end:
            try:
                rows.append(derive(m, levels))
            except Exception as e:
                failures.append({'date': d.isoformat(), 'error': type(e).__name__, 'detail': str(e)[:160]})
    if not rows:
        raise ValueError('No sounding rows processed; inspect archive and date range.')
    df = pd.DataFrame(rows).sort_values('date').drop_duplicates('date', keep='last')
    df.to_csv(OUT / 'sounding_indices_raw.csv', index=False)
    pd.DataFrame(failures, columns=['date', 'error', 'detail']).to_csv(OUT / 'sounding_preprocess_failures.csv', index=False)
    annual = df.groupby('year').agg(n_soundings=('date', 'size'),
        n_core_usable=('qc_thermo_usable', 'sum'), n_integral_columns=('qc_integral_column', 'sum'),
        n_cape_available=('qc_CAPE_available', 'sum'), n_wind200_available=('qc_wind200_available', 'sum'))
    annual = annual.reindex(range(start.year, end.year + 1), fill_value=0)
    annual.to_csv(OUT / 'sounding_qc_by_year.csv')
    reasons = df.qc_exclusion_reasons.str.split(';').explode()
    reason_counts = reasons[reasons.ne('')].value_counts().to_dict()
    meta = {'source': 'NOAA NCEI IGRA v2.2', 'station': 'THM00048565', 'hour_utc': o.hour,
            'requested_period': [o.start, o.end], 'n_soundings': len(df),
            'n_usable': int(df.qc_thermo_usable.sum()), 'n_integral_columns': int(df.qc_integral_column.sum()),
            'n_archive_level_failures': len(failures), 'year_counts': annual.n_soundings.to_dict(),
            'usable_year_counts': annual.n_core_usable.to_dict(), 'exclusion_reason_counts_nonexclusive': reason_counts,
            'cape_status_counts': df.cape_status.value_counts().to_dict(),
            'feature_missing_counts_among_usable': df.loc[df.qc_thermo_usable, list(THERMO_FEATURES)].isna().sum().to_dict(),
            'qc_designation': 'Revision-stage physical coverage checks, not prespecified or AUC-tuned',
            'core_qc': {'min_distinct_paired_levels': MIN_THERMO_LEVELS, 'min_origin_hpa': MIN_ORIGIN_HPA,
                        'required_top_hpa': CORE_TOP_HPA, 'max_paired_gap_hpa': MAX_GAP_HPA},
            'integral_qc': {'required_top_hpa': INTEGRAL_TOP_HPA, 'max_paired_gap_hpa': MAX_GAP_HPA},
            'cape_note': 'Lowest-usable-level parcel; CAPE_SB only when origin is IGRA surface. Positive buoyancy without an observed EL is censored (NaN). Deep nonbuoyant profiles can retain a true zero.',
            'pwat_note': 'Bounded observed-column integral from lowest paired level to 100 hPa, in mm; incomplete columns are NaN.',
            'heating_note': 'Morning-profile mixed-layer CAPE is conceptually possible; a validated daytime heating-adjusted parcel is not supplied by this pipeline. Observed later-day Tmax must not enter an operational morning predictor.',
            'wind_note': 'Missing upper-level wind does not exclude a core-usable profile. Unsupported wind features remain NaN.',
            'units': {'CAPE': 'J kg-1', 'CINS': 'J kg-1', 'PWAT': 'mm', 'LCLT': 'K',
                      'LCLP': 'hPa', 'LFCT': 'hPa', 'EQLV': 'hPa', 'SHOW': 'K difference',
                      'LIFT': 'K difference', 'CTOT': 'K difference', 'VTOT': 'K difference',
                      'TTOT': 'K difference', 'KINX': 'conventional degC index'}}
    (OUT / 'sounding_preprocess_metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
