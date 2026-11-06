import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('inventory', ROOT / 'scripts/milestone_inventory.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class WorkflowRepairTests(unittest.TestCase):
    def test_inventory_does_not_follow_links_or_mutate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'checkpoint.pt').write_bytes(b'valuable-state')
            (root / 'outside').symlink_to('/etc')
            result = module.inventory(root)
            self.assertTrue(result['complete'])
            self.assertEqual(len(result['files']), 2)
            self.assertEqual((root / 'checkpoint.pt').read_bytes(), b'valuable-state')
            self.assertIn('protected-link', [r['category'] for r in result['files']])

    def test_bounded_inventory_reports_incompleteness(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(3):
                (Path(tmp) / str(i)).write_text('data')
            self.assertFalse(module.inventory(tmp, 1)['complete'])



if __name__ == '__main__':
    unittest.main()
