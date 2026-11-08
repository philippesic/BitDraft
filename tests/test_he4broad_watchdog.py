"""Watchdog decisions use fixtures; no fixture signals or restarts any process."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.he4broad import watchdog as wd


class WatchdogTests(unittest.TestCase):
    def setUp(self):
        self.sample = dict(boot_id='boot',live_mtime=1000,live=dict(status='TRAINING',updates=100,additional_update_seconds=12),
            workflow_alive=True,supervisor_alive=True,trainer_alive=True,live_owner_alive=True,
            trainers=[dict(pid=42)],children=[dict(pid=42)],lock_owned=True,oom_logs=[],
            supervisor={},workflow_result={},training_result={},completion={},final_recovery={})
        self.previous = dict(boot_id='boot',progress=[90,11],last_progress_at=900)

    def test_advancing_update_resets_stall_age_to_actual_live_timestamp(self):
        record = wd.evaluate(self.sample,self.previous,1010)
        self.assertEqual(record['status'],'HEALTHY')
        self.assertTrue(record['advanced'])
        self.assertEqual(record['no_progress_seconds'],10)

    def test_alive_but_stalled_for_twenty_minutes_is_failure(self):
        previous = dict(self.previous,progress=[100,12],last_progress_at=1000)
        self.assertIn('NO_PROGRESS_20_MINUTES',wd.evaluate(self.sample,previous,2200)['failures'])
        self.assertFalse(wd.evaluate(self.sample,previous,2199)['failures'])

    def test_unexpected_trainer_death_detected_before_stall_threshold(self):
        self.sample.update(trainer_alive=False,live_owner_alive=False,trainers=[],children=[])
        record = wd.evaluate(self.sample,self.previous,1010)
        self.assertIn('TRAINER_DIED',record['failures'])
        self.assertEqual(record['resume_attempts'],0)

    def test_lost_lock_and_oom_and_supervisor_death_are_independent_failures(self):
        for field,value,expected in [('lock_owned',False,'GPU_LOCK_LOST'),
            ('oom_logs',['train-3600.log'],'OOM_REPORTED'),('workflow_alive',False,'WORKFLOW_DIED'),
            ('supervisor_alive',False,'SUPERVISOR_DIED')]:
            with self.subTest(field=field):
                sample = dict(self.sample,**{field:value})
                self.assertIn(expected,wd.evaluate(sample,self.previous,1010)['failures'])

    def test_hourly_transition_and_native_panel_do_not_look_like_trainer_death(self):
        self.sample.update(trainer_alive=False,live_owner_alive=False,trainers=[],children=[dict(module='scripts.component_campaign.native')])
        self.sample['live']['status'] = 'RUNNING'
        self.assertEqual(wd.evaluate(self.sample,self.previous,1010)['status'],'EVALUATION_OR_FINALIZATION')

    def test_completed_job_may_release_lock_and_exit(self):
        self.sample.update(workflow_alive=False,supervisor_alive=False,trainer_alive=False,lock_owned=False,
            workflow_result=dict(status='COMPLETE'),training_result=dict(actual_update_seconds=28800),
            final_recovery=dict(status='FULL_STATE_GPU_RESTORE_AND_NATIVE_BYTE_IDENTITY_PASS'),
            completion=dict(status='COMPLETE'))
        self.assertEqual(wd.evaluate(self.sample,self.previous,1010)['status'],'COMPLETE')

    def test_finalization_failure_is_visible_and_explicit_user_stop_is_respected(self):
        self.sample.update(workflow_result=dict(status='COMPLETE'),training_result=dict(actual_update_seconds=28800),
            final_recovery=dict(status='FULL_STATE_GPU_RESTORE_AND_NATIVE_BYTE_IDENTITY_PASS'),
            completion=dict(status='FINALIZATION_BLOCKED'))
        self.assertIn('COMPLETION_CALLBACK_FAILED',wd.evaluate(self.sample,self.previous,1010)['failures'])
        self.sample['workflow_result'] = dict(status='PAUSED')
        self.sample['workflow_alive'] = False
        self.assertEqual(wd.evaluate(self.sample,self.previous,1010)['status'],'PAUSED')

    def test_progress_regression_is_not_treated_as_resume(self):
        self.previous['progress'] = [101,13]
        self.assertIn('PROGRESS_REGRESSED_OR_INCONSISTENT',wd.evaluate(self.sample,self.previous,1010)['failures'])

    def test_kernel_lock_must_match_inode_device_and_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'lock'; path.touch(); stat = path.stat()
            row = f'7: FLOCK ADVISORY WRITE 123 {wd.os.major(stat.st_dev):02x}:{wd.os.minor(stat.st_dev):02x}:{stat.st_ino} 0 EOF\n'
            self.assertEqual(wd.lock_holders(path,row),[123])
            self.assertEqual(wd.lock_holders(path,row.replace(str(stat.st_ino),str(stat.st_ino+1))),[])

    def test_alarm_survives_report_republication_and_preserves_other_text(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(wd,'REPORT',Path(directory)):
            report = Path(directory)/'report.md'; report.write_text('# Existing report\n\nKeep this evidence.\n')
            incident = dict(failures=['GPU_LOCK_LOST'],checked_at='now',progress=[100,12])
            current = dict(status='FAILURE_NO_AUTOMATIC_RESUME')
            wd.failure_note(incident,current); wd.failure_note(incident,current)
            self.assertEqual(report.read_text().count(wd.BEGIN),1)
            self.assertIn('Keep this evidence.',report.read_text())
            report.write_text('# Republished report\n\nNew checkpoint data.\n')
            wd.failure_note(incident,current)
            self.assertIn('WATCHDOG FAILURE',report.read_text())
            self.assertIn('New checkpoint data.',report.read_text())


if __name__ == '__main__': unittest.main()
