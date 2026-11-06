import sys,unittest,json,time
from pathlib import Path
P=Path('/home/philip/binary-eagle-decoding');sys.path[:0]=[str(P/'src'),str(P/'tests')]
import torch;torch.set_num_threads(2)
modules=['test_w1a8_modern_actual_data','test_w1a8_modern_actual_union','test_w1a8_modern_resident_rows','test_w1a8_modern_train_expanded_ce_resident','test_w1a8_generalization_recovery','test_w1a8_modern_confirm','test_w1a8_modern_natural_confirmation']
t=time.perf_counter();suite=unittest.defaultTestLoader.loadTestsFromNames(modules);result=unittest.TextTestRunner(verbosity=2).run(suite)
report=dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),seconds=time.perf_counter()-t,modules=modules,status='PASS' if result.wasSuccessful() else 'FAIL')
(P/'runs/contender-infra-20261104-01/related-tests.json').write_text(json.dumps(report,indent=2)+'\n');raise SystemExit(not result.wasSuccessful())
