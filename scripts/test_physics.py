"""Read-only synthetic regression tests; no project data/outputs are changed."""
import importlib.util
import math
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from affine import Affine
from metpy.calc import total_totals_index


def load(filename, name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prep = load('01_preprocess.py', 'revision_preprocess')
terrain = load('01b_terrain.py', 'revision_terrain')
META = dict(year=2018, month=1, day=1, hour=0, station_id='synthetic')


def levels(deep=True, upper_wind=True, surface=True):
    pressure = [1000, 925, 850, 700, 500, 400, 300, 200, 150, 100]
    temp = [27, 23, 18, 8, -8, -20, -35, -52, -63, -73]
    if not deep:
        pressure, temp = pressure[:5], temp[:5]
    return [[p * 100, 0, t * 10, None, 50,
             180 if upper_wind or p >= 500 else None,
             100 if upper_wind or p >= 500 else None,
             1 if i == 0 and surface else 0]
            for i, (p, t) in enumerate(zip(pressure, temp))]


class TestPhysics(unittest.TestCase):
    def test_ttot_is_cross_plus_vertical_and_matches_metpy(self):
        row = prep.derive(META, levels())
        self.assertAlmostEqual(row['TTOT'], row['CTOT'] + row['VTOT'])
        p = np.array([850, 700, 500]) * prep.units.hPa
        t = np.array([18, 8, -8]) * prep.units.degC
        td = np.array([13, 3, -13]) * prep.units.degC
        self.assertAlmostEqual(row['TTOT'], float(total_totals_index(p, t, td).m))

    def test_incomplete_column_not_zero_and_local_indices_survive(self):
        row = prep.derive(META, levels(deep=False))
        self.assertTrue(row['qc_thermo_usable'])
        self.assertFalse(row['qc_integral_column'])
        for key in ['CAPE', 'CINS', 'PWAT']:
            self.assertTrue(math.isnan(row[key]))
        for key in ['SHOW', 'LIFT', 'LCLP', 'TTOT']:
            self.assertTrue(math.isfinite(row[key]))

    def test_missing_upper_wind_does_not_discard_thermodynamics(self):
        row = prep.derive(META, levels(upper_wind=False))
        self.assertTrue(row['qc_thermo_usable'])
        self.assertTrue(row['qc_integral_column'])
        self.assertTrue(math.isnan(row['WS200']))
        self.assertTrue(math.isfinite(row['PWAT']))

    def test_individual_calculation_failure_does_not_wipe_other_indices(self):
        with patch.object(prep, 'precipitable_water', side_effect=ValueError('synthetic failure')):
            row = prep.derive(META, levels())
        self.assertTrue(math.isnan(row['PWAT']))
        self.assertTrue(math.isfinite(row['SHOW']))
        self.assertTrue(math.isfinite(row['LCLP']))
        self.assertIn('PWAT:ValueError', row['calculation_errors'])

    def test_parcel_origin_is_not_assumed_surface(self):
        row = prep.derive(META, levels(surface=False))
        self.assertFalse(row['parcel_origin_is_igra_surface'])
        self.assertEqual(row['parcel_origin_kind'], 'lowest_usable_level')
        self.assertTrue(math.isnan(row['CAPE_SB']))

    def test_censored_cape_is_not_accepted_as_complete(self):
        with patch.object(prep, 'el', return_value=(np.nan * prep.units.hPa, np.nan * prep.units.degC)), patch.object(prep, 'cape_cin', return_value=(100 * prep.units('J/kg'), -10 * prep.units('J/kg'))):
            row = prep.derive(META, levels())
        self.assertTrue(math.isnan(row['CAPE']))
        self.assertEqual(row['CAPE_observed_lower_bound'], 100)
        self.assertIn('censored', row['cape_status'])

    def test_deep_true_zero_is_retained(self):
        with patch.object(prep, 'cape_cin', return_value=(0 * prep.units('J/kg'), 0 * prep.units('J/kg'))):
            row = prep.derive(META, levels())
        self.assertEqual(row['CAPE'], 0)
        self.assertEqual(row['cape_status'], 'complete_nonbuoyant_zero')

    def test_sparse_gap_is_rejected(self):
        self.assertFalse(prep.layer_covered([1000, 925, 850, 800, 500], 1000, 500))
        self.assertTrue(math.isnan(prep.interp([850, 400], [10, -20], 500)))
        self.assertEqual(prep.interp([850, 400], [10, -20], 850), 10)
        self.assertTrue(math.isnan(prep.interp([850, 700], [10, 0], 500)))

    def test_empty_profile_is_flagged_not_crashed(self):
        row = prep.derive(META, [])
        self.assertFalse(row['qc_thermo_usable'])
        self.assertTrue(math.isnan(row['CAPE']))

    def test_plane_gradients_north_up_and_south_up(self):
        lat = 8
        for spacing_y in [-1 / 3600, 1 / 3600]:
            transform = Affine(1 / 3600, 0, 98, 0, spacing_y, lat)
            dx = transform.a * 111320 * math.cos(math.radians(lat))
            dy = transform.e * 110574
            rr, cc = np.indices((21, 21))
            height = 0.2 * cc * dx + 0.3 * rr * dy + 10
            gx, gy = terrain.terrain_gradient(height, transform, lat)
            self.assertAlmostEqual(gx, 0.2)
            self.assertAlmostEqual(gy, 0.3)
            self.assertAlmostEqual(5 * gx + 10 * gy, 4.0)

    def test_flat_gradient_zero(self):
        self.assertEqual(terrain.terrain_gradient(np.ones((21, 21)), Affine(1 / 3600, 0, 98, 0, -1 / 3600, 8), 8), (0., 0.))


if __name__ == '__main__':
    unittest.main(verbosity=2)
