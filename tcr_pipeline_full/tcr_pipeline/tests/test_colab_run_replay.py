import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('colab_run_replay', ROOT/'colab_run_replay.py')
m = importlib.util.module_from_spec(spec)
if (ROOT/'colab_run_replay.py').exists(): spec.loader.exec_module(m)


class DriverTests(unittest.TestCase):
    def test_calibration_uses_local_completed_rate(self):
        self.assertTrue(hasattr(m, 'choose_calibration'))
        result = m.choose_calibration([2,3,4,5,6], [dict(completed=4, errors=0, elapsed_s=10), dict(completed=16, errors=0, elapsed_s=20), dict(completed=32, errors=1, elapsed_s=20)])
        self.assertEqual(result['reference_capacity_jobs_s'], .8)
        self.assertEqual(result['slo_s'], 24)
        self.assertAlmostEqual(result['target_rate'], .08)
        self.assertEqual(result['capacity_definition'], 'maximum error-free completed reference requests / total trial wall time including drain')

    def test_deadline_includes_all_cells_and_archive_margin(self):
        self.assertTrue(hasattr(m, 'can_start_block'))
        self.assertFalse(m.can_start_block(now=100,deadline=900,cells=5,warmup=10,duration=45,drain=60,idle_timeout=15))
        self.assertTrue(m.can_start_block(now=100,deadline=1000,cells=5,warmup=10,duration=45,drain=60,idle_timeout=15))

if __name__ == '__main__': unittest.main()
