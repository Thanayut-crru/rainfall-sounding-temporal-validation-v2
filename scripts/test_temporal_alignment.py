"""Synthetic calendar/accumulation checks; observed skill never selects a clock."""
import importlib.util
from pathlib import Path
import unittest
import pandas as pd


def module(file):
    spec=importlib.util.spec_from_file_location(file,Path(__file__).parent/file)
    result=importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


alignment=module('00c_temporal_alignment.py')
diurnal=module('06_diurnal.py')


class TemporalAlignmentTest(unittest.TestCase):
    def fixture(self):
        dates=pd.date_range('2020-01-01',periods=3)
        return pd.DataFrame([{'date':d,'station':'phuket','local_hour':h,'rain_mm':1.0} for d in dates for h in [1,4,7,10,13,16,19,22]])

    def test_daily_and_evening_cross_midnight(self):
        daily,x=alignment.align_three_hour(self.fixture())
        self.assertEqual(daily.set_index('date').loc[pd.Timestamp('2020-01-01'),'RF_phuket'],8)
        at7=x[(x.date==pd.Timestamp('2020-01-02'))&(x.local_hour==7)].iloc[0]
        self.assertEqual(at7.forecast_date,pd.Timestamp('2020-01-01'))
        q=diurnal.aggregate_periods(x)
        q=q[q.date==pd.Timestamp('2020-01-01')]
        self.assertEqual(len(q),3)
        self.assertTrue(q.rain_mm.eq(2).all())

    def test_pre_sounding_observation_excluded(self):
        raw=self.fixture()
        raw.loc[(raw.date==pd.Timestamp('2020-01-01'))&(raw.local_hour==7),'rain_mm']=99
        _,x=alignment.align_three_hour(raw)
        q=diurnal.aggregate_periods(x)
        morning=q[(q.date==pd.Timestamp('2020-01-01'))&(q.period=='morning_07_13')]
        self.assertEqual(morning.iloc[0].rain_mm,2)

    def test_missing_interval_not_dry_or_partial(self):
        raw=self.fixture(); raw.loc[(raw.date==pd.Timestamp('2020-01-01'))&(raw.local_hour==10),'rain_mm']=float('nan')
        daily,x=alignment.align_three_hour(raw)
        self.assertTrue(pd.isna(daily.set_index('date').loc[pd.Timestamp('2020-01-01'),'RF_phuket']))
        q=diurnal.aggregate_periods(x)
        self.assertTrue(pd.isna(q[(q.date==pd.Timestamp('2020-01-01'))&(q.period=='morning_07_13')].iloc[0].rain_mm))

    def test_daily_end_label_shift_and_duplicate_guard(self):
        data=pd.DataFrame({'date':['2020-01-02'],'RF_nakhon_si_thammarat':[3.]})
        self.assertEqual(alignment.align_daily_east(data,'period_end_date_07').iloc[0].date,pd.Timestamp('2020-01-01'))
        with self.assertRaises(ValueError):
            alignment.align_three_hour(pd.concat([self.fixture(),self.fixture()]))

    def test_author_confirmed_daily_window_ends22(self):
        raw=self.fixture()
        raw.loc[(raw.date==pd.Timestamp('2020-01-01'))&(raw.local_hour==1),'rain_mm']=10
        daily=alignment.daily_ending_22(raw)
        self.assertEqual(daily.set_index('date').loc[pd.Timestamp('2020-01-01'),'RF_phuket'],17)
        east=pd.DataFrame({'date':['2020-01-01'],'RF_nakhon_si_thammarat':[17.]})
        self.assertEqual(alignment.align_daily_east(east,'period_end_date_22').iloc[0].date,pd.Timestamp('2020-01-01'))


if __name__=='__main__':
    unittest.main()
