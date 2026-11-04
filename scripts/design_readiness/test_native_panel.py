import unittest
from native_panel import parse_metrics

BASE='''# HELP irrelevant ignored
llamacpp:spec_decode_num_drafts_total 12
llamacpp:spec_decode_num_draft_tokens_total 24
llamacpp:spec_decode_num_accepted_tokens_total 9
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 7
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="1"} 2
'''

class NativeCounterTests(unittest.TestCase):
    def test_position_counts_preserved(self):
        x=parse_metrics(BASE)
        self.assertEqual(x['spec_decode_num_drafts_total'],12)
        positions=[v for k,v in x.items() if 'per_pos' in k]
        self.assertEqual(sum(positions),x['spec_decode_num_accepted_tokens_total'])

    def test_missing_fractional_or_negative_counts_rejected(self):
        for text in [BASE.replace('total 12','total 1.5'),BASE.replace('total 12','total -1'),
                     BASE.replace('llamacpp:spec_decode_num_drafts_total 12\n','')]:
            with self.assertRaises(ValueError):parse_metrics(text)
