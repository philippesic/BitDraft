import copy
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from runner import execute, pin, read, validate


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan = dict(schema='contender_plan_v1', budget_seconds=100,
            candidate_budgets={'a':100}, grace_seconds=.2, inputs=[], jobs=[])

    def job(self, name='one', code=None, stage=0, depends=()):
        result = self.root / (name+'.json')
        code = code or f'import json;json.dump({{"status":"OK"}},open({str(result)!r},"x"))'
        job = dict(id=name, candidate='a', stage=stage, depends_on=list(depends), timeout_seconds=3,
            argv=[sys.executable,'-c',code], cwd=str(self.root),
            checks=[dict(path=str(result),keys=['status'],equals='OK')])
        self.plan['jobs'].append(job)
        return job

    def run_plan(self, through=9):
        path = self.root/'plan.json'
        if not path.exists():path.write_text(json.dumps(self.plan))
        return execute(path,self.root/'out',through)

    def test_stage_resume_does_not_repeat(self):
        self.job();self.job('two',stage=1,depends=['one'])
        first=self.run_plan(0);self.assertEqual(first['status'],'STAGE_COMPLETE')
        before=pin(self.root/'one.json')
        final=self.run_plan();self.assertEqual(final['status'],'COMPLETE')
        self.assertEqual(before,pin(self.root/'one.json'))
        self.assertEqual(first['jobs']['one'],final['jobs']['one'])
        self.assertGreater(final['charged_seconds'],first['charged_seconds'])

    def test_nonzero_exit_blocks_dependency(self):
        self.job(code='raise SystemExit(3)');self.job('two',stage=1,depends=['one'])
        with self.assertRaisesRegex(ValueError,'adapter exit'):self.run_plan()
        self.assertFalse((self.root/'two.json').exists())
        with self.assertRaisesRegex(ValueError,'prior attempt'):self.run_plan()

    def test_missing_result_is_failure(self):
        self.job(code='pass')
        with self.assertRaises(FileNotFoundError):self.run_plan()
        self.assertEqual(read(self.root/'out/state.json')['jobs']['one']['status'],'FAILED')

    def test_stale_result_not_accepted(self):
        self.job();(self.root/'one.json').write_text('{"status":"OK"}')
        with self.assertRaisesRegex(ValueError,'stale output'):self.run_plan()

    def test_wrong_result_not_accepted(self):
        self.job(code=f'open({str(self.root/"one.json")!r},"w").write(\'{{"status":"BAD"}}\')')
        with self.assertRaisesRegex(ValueError,'result check failed'):self.run_plan()

    def test_total_and_candidate_limits_before_launch(self):
        for field in ('budget_seconds','candidate_budgets'):
            with self.subTest(field=field):
                self.plan['jobs']=[];self.job()
                self.plan['budget_seconds']=100;self.plan['candidate_budgets']['a']=100
                if field=='budget_seconds':self.plan[field]=1
                else:self.plan[field]['a']=1
                (self.root/'plan.json').unlink(missing_ok=True)
                with self.assertRaisesRegex(ValueError,'insufficient'):self.run_plan()
                self.assertFalse((self.root/'one.json').exists())

    def test_timeout_kills_grandchild(self):
        pidpath=self.root/'pid'
        self.job(code=f'import subprocess,time; p=subprocess.Popen(["sleep","60"]);open({str(pidpath)!r},"w").write(str(p.pid));time.sleep(60)')['timeout_seconds']=.3
        with self.assertRaisesRegex(ValueError,'TIMEOUT'):self.run_plan()
        row=read(self.root/'out/state.json')['jobs']['one'];self.assertTrue(row['group_released'])
        self.assertLess(row['charged_seconds'],2)

    def test_mutated_plan_and_evidence_rejected(self):
        self.job();self.run_plan()
        path=self.root/'plan.json';original=path.read_text();path.write_text(original+' ')
        with self.assertRaisesRegex(ValueError,'plan changed'):self.run_plan()
        path.write_text(original);(self.root/'one.json').write_text('{"status":"BAD"}')
        with self.assertRaisesRegex(ValueError,'evidence changed'):self.run_plan()

    def test_input_change_rejected(self):
        source=self.root/'source';source.write_text('one');self.plan['inputs']=[pin(source)];self.job();source.write_text('two')
        with self.assertRaisesRegex(ValueError,'input changed'):self.run_plan()

    def test_malformed_plans(self):
        self.job()
        for change in (lambda p:p.update(budget_seconds=float('nan')),lambda p:p['jobs'][0].update(depends_on=['later']),lambda p:p['jobs'][0].update(checks=[]),lambda p:p['jobs'].append(copy.deepcopy(p['jobs'][0]))):
            p=copy.deepcopy(self.plan);change(p)
            with self.assertRaises(ValueError):validate(p)

    def test_signal_stop(self):
        self.job(code='import time;time.sleep(60)')
        path=self.root/'plan.json';path.write_text(json.dumps(self.plan))
        proc=subprocess.Popen([sys.executable,str(Path(__file__).with_name('runner.py')),'--plan',str(path),'--output',str(self.root/'out'),'--through','0'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        try:
            deadline=time.monotonic()+5
            while time.monotonic()<deadline:
                state=self.root/'out/state.json'
                if state.exists() and read(state)['jobs'].get('one',{}).get('identity'):break
                time.sleep(.02)
            proc.send_signal(signal.SIGTERM);proc.wait(timeout=5)
            row=read(state)['jobs']['one'];self.assertEqual(row['error'],'INTERRUPTED');self.assertTrue(row['group_released'])
        finally:
            if proc.poll() is None:proc.kill();proc.wait()

if __name__=='__main__':unittest.main()

class AdditionalRunnerTests(unittest.TestCase):
    setUp = RunnerTests.setUp
    job = RunnerTests.job
    run_plan = RunnerTests.run_plan
    def test_carry_cost_cannot_reset_budget(self):
        self.job();first=self.run_plan()
        prior=pin(self.root/'out/state.json')
        self.plan['jobs']=[];self.job('next')
        self.plan['prior_state']=prior
        self.plan['budget_seconds']=first['charged_seconds']+.01
        path=self.root/'next-plan.json';path.write_text(json.dumps(self.plan))
        with self.assertRaisesRegex(ValueError,'insufficient'):execute(path,self.root/'next-out',9)
        self.assertFalse((self.root/'next.json').exists())
    def test_log_output_bounded(self):
        self.job(code=f'import sys,json;sys.stdout.write("x"*(20*1024*1024));json.dump({{"status":"OK"}},open({str(self.root/"one.json")!r},"w"))')
        self.run_plan()
        logs=list((self.root/'out').glob('*.log*'))
        self.assertEqual(len(logs),2);self.assertLessEqual(sum(p.stat().st_size for p in logs),16*1024*1024)
    def test_lock_excludes_second_runner(self):
        from runner import locked
        self.job()
        with locked(self.root/'out/run.lock'):
            with self.assertRaises(BlockingIOError):self.run_plan()
    def test_hard_crash_recovery_and_identity_guard(self):
        from recover import recover
        from runner import group_live
        self.job(code='import time;time.sleep(60)')
        path=self.root/'plan.json';path.write_text(json.dumps(self.plan));state=self.root/'out/state.json'
        proc=subprocess.Popen([sys.executable,str(Path(__file__).with_name('runner.py')),'--plan',str(path),'--output',str(self.root/'out'),'--through','0'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        saved=None
        try:
            deadline=time.monotonic()+5
            while time.monotonic()<deadline:
                if state.exists():
                    value=read(state);saved=value['jobs'].get('one',{}).get('identity')
                    if saved:break
                time.sleep(.02)
            self.assertIsNotNone(saved);proc.kill();proc.wait(timeout=5)
            value=read(state);value['jobs']['one']['identity']['start_ticks']+=1;state.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'PID reused'):recover(state,'one',.1)
            value['jobs']['one']['identity']=saved;state.write_text(json.dumps(value))
            row=recover(state,'one',.1);self.assertEqual(row['status'],'ABANDONED_RECOVERED');self.assertFalse(group_live(saved['pgid']))
            self.assertEqual(row['charged_seconds'],8.2)
        finally:
            if proc.poll() is None:proc.kill();proc.wait()
            if saved and group_live(saved['pgid']):
                import os
                os.killpg(saved['pgid'],signal.SIGKILL)
