"""Native log attribution must survive delayed live log sampling and fail closed."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from scripts.dflash_baseline.native_counters import reconcile


class NativeCounterRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cell = Path(self.temp.name)
        self.items = [dict(rep=2, arm='fixture', instrumented=True, prompt=name,
                           native={}, wall_seconds=1.25, tokens=5, bounds_ns=[1, 2])
                      for name in ('first', 'second')]
        self.log = (
            'acc per pos = (0.667, 0.000)\n'
            '#calls(b,g,a) = 1 3 3, #gen drafts = 3, #acc drafts = 2, '
            '#gen tokens = 6, #acc tokens = 2, dur(b,g,a) = 0.001, 3.000, 0.001 ms\n'
            'acc per pos = (0.500, 0.250)\n'
            '#calls(b,g,a) = 2 7 7, #gen drafts = 7, #acc drafts = 4, '
            '#gen tokens = 14, #acc tokens = 5, dur(b,g,a) = 0.002, 7.000, 0.002 ms\n'
        )
        (self.cell / 'server.log').write_text(self.log)
        for name, accepted, proposed in [('first', 2, 6), ('second', 3, 8)]:
            (self.cell / (name + '-response.json')).write_text(json.dumps(
                dict(timings=dict(draft_n_accepted=accepted, draft_n=proposed))))

    def test_delayed_capture_recovers_without_changing_timing_or_output(self):
        before = copy.deepcopy(self.items)
        corrections = reconcile(self.cell, self.items)
        self.assertEqual(len(corrections), 2)
        self.assertEqual([r['native']['accepted'] for r in self.items], [2, 3])
        self.assertEqual([r['native']['rounds'] for r in self.items], [3, 4])
        self.assertEqual(self.items[1]['native']['prefix_survival_counts'], [2, 1])
        for old, new in zip(before, self.items):
            self.assertEqual({k:v for k,v in old.items() if k != 'native'},
                             {k:v for k,v in new.items() if k != 'native'})
        self.assertEqual(reconcile(self.cell, self.items), [])

    def test_missing_boundary_fails_closed(self):
        (self.cell / 'server.log').write_text(self.log.split('acc per pos')[0])
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            reconcile(self.cell, self.items)

    def test_independent_response_disagreement_fails_closed(self):
        (self.cell / 'first-response.json').write_text(json.dumps(
            dict(timings=dict(draft_n_accepted=1, draft_n=6))))
        with self.assertRaisesRegex(ValueError, 'counter disagreement'):
            reconcile(self.cell, self.items)

    def test_nonsequential_boundaries_fail_closed(self):
        (self.cell / 'server.log').write_text(self.log.replace('= 2 7 7', '= 3 7 7'))
        with self.assertRaisesRegex(ValueError, 'nonsequential'):
            reconcile(self.cell, self.items)


if __name__ == '__main__':
    unittest.main()
