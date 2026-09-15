"""Validate provider-confirmed accumulation semantics; never infer them from AUC.

Without confirmation this writes PENDING. The author's confirmed primary
day is after22(D-1) through22(D); separate subdaily targets start after07.
"""
import argparse
import json
import os
from pathlib import Path

import pandas as pd


def align_three_hour(rain):
    """Input timestamps are confirmed interval ends in local time, calendar dates."""
    x = rain.copy()
    if x.duplicated(['date', 'station', 'local_hour']).any():
        raise ValueError('Duplicate station/timestamp in 3-hour rainfall')
    x['interval_end'] = pd.to_datetime(x['date']) + pd.to_timedelta(x.local_hour, unit='h')
    x['interval_start'] = x['interval_end'] - pd.Timedelta(hours=3)
    x['forecast_date'] = (x.interval_end - pd.Timedelta(hours=7) - pd.Timedelta(nanoseconds=1)).dt.normalize()
    agg = x.groupby(['forecast_date','station']).rain_mm.agg(['sum','count'])
    agg.loc[agg['count'] != 8, 'sum'] = float('nan')
    wide = agg['sum'].unstack('station').add_prefix('RF_').reset_index().rename(columns={'forecast_date':'date'})
    return wide, x


def align_daily_east(east, convention):
    x = east.copy()
    x['date'] = pd.to_datetime(x['date'])
    if convention == 'period_end_date_07':
        x['date'] -= pd.Timedelta(days=1)
    elif convention not in ['period_start_date_07','period_end_date_22']:
        raise ValueError('Unsupported daily date convention')
    if x.date.duplicated().any():
        raise ValueError('Duplicate aligned daily dates')
    return x


def daily_ending_22(rain):
    """Eight interval ends01..22 on dateD; total covers22(D-1)->22(D)."""
    x=rain.copy()
    if x.duplicated(['date','station','local_hour']).any():
        raise ValueError('Duplicate station/timestamp')
    if not x.local_hour.isin([1,4,7,10,13,16,19,22]).all():
        raise ValueError('Unexpected observation time')
    agg=x.groupby(['date','station']).rain_mm.agg(['sum','count','size'])
    agg.loc[(agg['count']!=8)|(agg['size']!=8),'sum']=float('nan')
    return agg['sum'].unstack('station').add_prefix('RF_').reset_index()


def main():
    base = Path(os.environ.get('RAINFALL_PROJECT_DIR', Path(__file__).resolve().parent.parent))
    out = Path(os.environ.get('RAINFALL_OUTPUT_DIR', base/'outputs'))
    ap = argparse.ArgumentParser()
    ap.add_argument('--confirmation', type=Path, default=base/'data/metadata/temporal_alignment.json')
    args = ap.parse_args()
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(args.confirmation.read_text(encoding='utf-8-sig')) if args.confirmation.exists() else {}
    provenance = cfg.get('provenance','')
    three = bool(provenance and cfg.get('three_hour_timestamp') == 'interval_end'
                 and cfg.get('three_hour_date_label') == 'observation_date'
                 and cfg.get('three_hour_timezone') == 'Asia/Bangkok')
    daily = bool(provenance and cfg.get('daily_east_date_label') == 'period_end_date_22')
    status = {'status':'confirmed' if three and daily else 'pending',
              'three_hour_status':'confirmed' if three else 'pending',
              'daily_east_status':'confirmed' if daily else 'pending',
              'provenance':provenance, 'confirmation_file':str(args.confirmation),
              'pending_questions':[],
              'daily_window':'after22:00(D-1) through22:00(D)',
              'diurnal_windows':['07:00–13:00(D)','13:00–19:00(D)','19:00(D)–01:00(D+1)'],
              'daily_contains_pre_sounding_rain':True,
              'analysis_interpretation':'Confirmed date-labelled daily classification (22 previous day to22 labelled day), not exclusively future rainfall; subdaily targets independently restricted to after07' if three and daily else
                  'PROVISIONAL date-labelled rainfall classification only; no validated forecast lead-time or matched cross-coast accumulation claim'}
    if three:
        rain = pd.read_csv(base/'data/raw/rainfall_tmd_3hour.csv',parse_dates=['date'])
        _, intervals = align_three_hour(rain)
        wide=daily_ending_22(rain)
        wide.to_csv(out/'rainfall_daily_aligned.csv', index=False)
        intervals.to_csv(out/'rainfall_intervals_aligned.csv', index=False)
    else:
        status['pending_questions'].append('Confirm whether time columns are interval ends, local timezone, and their observation calendar date.')
        if (out/'rainfall_daily_aligned.csv').exists() or (out/'rainfall_intervals_aligned.csv').exists():
            raise RuntimeError('Pending metadata but previous aligned outputs exist; choose a new run output directory.')
    if daily:
        east = pd.read_csv(base/'data/raw/rainfall_tmd_daily_east.csv', parse_dates=['date'])
        align_daily_east(east,cfg['daily_east_date_label']).to_csv(out/'rainfall_daily_east_aligned.csv',index=False)
    else:
        status['pending_questions'].append('Confirm the daily worksheet accumulation window and its date label.')
        if (out/'rainfall_daily_east_aligned.csv').exists():
            raise RuntimeError('Pending daily metadata but previous aligned output exists; choose a new run output directory.')
    (out/'temporal_alignment_status.json').write_text(json.dumps(status,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(status,indent=2,ensure_ascii=False))


if __name__ == '__main__':
    main()
