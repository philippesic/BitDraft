"""Behavioral checks for update-time stopping and repeated rollback accounting."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from .clock import UPDATE_BUDGET, UPDATE_POINTS, charged_seconds, pending, label, panel_label
from .accounting import reconcile_log


class ClockTests(unittest.TestCase):
    def test_long_input_wait_does_not_exhaust_update_budget(self):
        p=dict(update_seconds=2873.7484793208714,training_seconds=100000.)
        self.assertTrue(pending(p,3600))
        self.assertAlmostEqual(UPDATE_BUDGET-charged_seconds(p),25926.25152067913)

    def test_update_boundary_and_rejected_time_stop_the_segment(self):
        p=dict(update_seconds=3599.,discarded_update_seconds=1.,training_seconds=100.)
        self.assertFalse(pending(p,3600));self.assertTrue(pending(p,7200))
        p['update_seconds']=28800.1
        self.assertFalse(pending(p,UPDATE_BUDGET))

    def test_eight_hour_points_cannot_overwrite_historical_wall_panels(self):
        self.assertEqual(UPDATE_POINTS,tuple(range(3600,28801,3600)))
        self.assertEqual(label(3600),'update-3600')
        self.assertEqual(panel_label(3600),'formal-update-3600')
        self.assertNotEqual(panel_label(3600),'formal-3600')


class RecoveryTests(unittest.TestCase):
    def test_repeated_failure_before_checkpoint_keeps_all_charges_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'updates.jsonl';incidents=root/'incidents'
            checkpoint=dict(updates=1,update_seconds=1.,training_seconds=10.)
            def write(rows):path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            durable=dict(update=1,seconds=1.,training_seconds=10.)
            write([durable,dict(update=2,seconds=2.,training_seconds=13.)])
            first=copy.deepcopy(checkpoint);reconcile_log(first,path,incidents)
            self.assertEqual((first['discarded_update_seconds'],first['training_seconds']),(2.,13.))
            write([durable,dict(update=2,seconds=4.,training_seconds=18.)])
            second=copy.deepcopy(checkpoint);reconcile_log(second,path,incidents)
            self.assertEqual((second['discarded_update_seconds'],second['training_seconds']),(6.,18.))
            self.assertEqual(second['discarded_updates'],2)
            third=copy.deepcopy(checkpoint);reconcile_log(third,path,incidents)
            self.assertEqual((third['discarded_update_seconds'],third['training_seconds']),(6.,18.))
            saved=copy.deepcopy(second);reconcile_log(saved,path,incidents)
            self.assertEqual((saved['discarded_update_seconds'],saved['training_seconds']),(6.,18.))
            self.assertEqual(len(list(incidents.glob('*/uncheckpointed-updates.jsonl'))),2)
            self.assertEqual(len(path.read_text().splitlines()),1)

    def test_missing_durable_update_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'updates.jsonl';path.write_text('')
            with self.assertRaisesRegex(AssertionError,'incomplete'):
                reconcile_log(dict(updates=1,update_seconds=1.,training_seconds=1.),path,root/'incidents')


if __name__=='__main__':unittest.main()
