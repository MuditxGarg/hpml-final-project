import importlib.util
from pathlib import Path
import sys
import json
import tempfile
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

    def test_slo_horizon_extends_deadline_guard(self):
        self.assertTrue(hasattr(m, 'request_horizon'))
        horizon = m.request_horizon(92)
        self.assertEqual(horizon,97)
        self.assertGreaterEqual(horizon,92)
        self.assertEqual(m.request_horizon(24),60)
        self.assertFalse(m.can_start_block(now=100,deadline=1000,drain=horizon))
        self.assertTrue(m.can_start_block(now=100,deadline=1000,cells=4,drain=horizon))

    def test_reuse_freezes_selected_calibration_and_records_source(self):
        self.assertTrue(hasattr(m, 'load_reused_calibration'))
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'calibration.json'
            selected = dict(reference_capacity_jobs_s=.292,slo_s=92,target_rate=.0292)
            source.write_text(json.dumps(dict(selected=selected,idle=[20,23])))
            record = m.load_reused_calibration(source)
            self.assertEqual(record['selected'],selected)
            self.assertEqual(record['reused_source']['path'],str(source.resolve()))
            self.assertIn('sha256',record['reused_source'])
            self.assertEqual(record['idle'],[20,23])

if __name__ == '__main__': unittest.main()
