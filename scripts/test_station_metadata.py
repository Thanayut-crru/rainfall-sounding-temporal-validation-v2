"""Unit checks only: no access to the manuscript or real run outputs."""
import math
import unittest

from station_metadata import EARTH_RADIUS_KM, haversine_km, station_distance_km
from refresh_distance_metadata import nondistance_cells, parse_csv, prepare_refresh


class StationDistanceTests(unittest.TestCase):
    def test_identity(self):
        self.assertEqual(haversine_km(8.1,98.3,8.1,98.3),0)

    def test_equatorial_degree(self):
        self.assertAlmostEqual(haversine_km(0,0,0,1),math.pi*EARTH_RADIUS_KM/180,places=10)

    def test_antipodes_and_symmetry(self):
        self.assertAlmostEqual(haversine_km(0,0,0,180),math.pi*EARTH_RADIUS_KM,places=8)
        self.assertAlmostEqual(haversine_km(8.1,98.3,8.7,98.2),haversine_km(8.7,98.2,8.1,98.3),places=12)

    def test_gauge_coordinates_and_alias(self):
        expected={'phuket':30.584541378343424,'krabi':71.97528769686265,
                  'phangnga':60.277630658846796,'nakhon_si_thammarat':183.27284358810957}
        for station,value in expected.items():
            self.assertAlmostEqual(station_distance_km(station),value,places=9)
        self.assertEqual(station_distance_km('Takua Pa (Phang-nga)'),station_distance_km('phangnga'))
        self.assertLess(station_distance_km('phangnga'),station_distance_km('krabi'))

    def test_unknown_or_invalid(self):
        with self.assertRaises(KeyError): station_distance_km('unknown')
        with self.assertRaises(ValueError): haversine_km(float('nan'),0,0,0)
        with self.assertRaises(ValueError): haversine_km(91,0,0,0)


class MetadataOnlyTests(unittest.TestCase):
    def test_preserves_all_other_cells_exactly(self):
        original=b'station,dist_km,AUC,missing,note\r\nphuket,26,0.12345678901234567890,,"a,b"\r\nkrabi,65,0.7000000000000001,NaN,"line1\nline2"\r\n'
        corrected,report=prepare_refresh(original)
        self.assertEqual(nondistance_cells(parse_csv(original)),nondistance_cells(parse_csv(corrected)))
        self.assertTrue(report['non_distance_cells_identical'])
        self.assertEqual(len(report['changed_cells']),2)
        self.assertEqual(parse_csv(corrected)[1][1],format(station_distance_km('phuket'),'.12f'))

    def test_idempotent_and_bom(self):
        blob=b'\xef\xbb\xbfstation,distance_km,test_auc\nphangnga,73,0.873\n'
        first,_=prepare_refresh(blob)
        second,report=prepare_refresh(first)
        self.assertEqual(first,second)
        self.assertTrue(first.startswith(b'\xef\xbb\xbf'))
        self.assertEqual(report['changed_cells'],[])

    def test_two_distance_columns(self):
        blob=b'station,dist_km,distance_km,score\nnakhon_si_thammarat,181,181,0.691\n'
        result,report=prepare_refresh(blob)
        self.assertEqual(len(report['changed_cells']),2)
        self.assertEqual(parse_csv(result)[1][-1],'0.691')

    def test_does_not_touch_non_distance_csv(self):
        self.assertIsNone(prepare_refresh(b'station,score\nphuket,0.845\n'))

    def test_fail_closed(self):
        for blob in [b'dist_km,score\n26,0.8\n',b'station,dist_km\nunknown,26\n',
                     b'station,dist_km\nphuket,26,extra\n']:
            with self.assertRaises((ValueError,KeyError)):
                prepare_refresh(blob)


if __name__=='__main__':
    unittest.main()
