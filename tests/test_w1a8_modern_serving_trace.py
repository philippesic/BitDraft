"""Synthetic native log semantics; fixtures are not serving/model evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / 'scripts/w1a8_generalization/modern_serving_trace.py'
SPEC = importlib.util.spec_from_file_location('modern_serving_trace', PATH)
trace = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trace)


def slot(message, task=7, slot_id=0):
    return f'0.00.001 D slot   operator(): id {slot_id:2} | task {task} | {message}'


def emit(index, token):
    return slot(f'n_gen = {index}, n_remaining = 80, next token: {token:5} \'piece\'')


def candidates(tokens=(11, 12)):
    return [f'0.00.001 D spec        draft:  - seq_id 0, draft candidate {rank:3}, pos {depth:3}: {token + rank:6} (   0.500) \'piece\'' for depth, token in enumerate(tokens) for rank in range(3)]


def fixture(*, replay=False, partial_stop=False, final_done=True):
    lines = [slot('processing task, is_child = 0'), emit(1, 10), *candidates(), slot('generate_draft: id=10, #tokens=5, #draft=2, pos_next=5')]
    if replay:
        lines += [slot('accepted 1/2 draft tokens (restore checkpoint)'), slot('restoring speculative checkpoint (pos_min = 0, pos_max = 4, size = 999)'), slot('generate_draft: id=10, #tokens=5, #draft=2, pos_next=5'), slot('accepted 2/2 draft tokens')]
    labels = [10, 11] if partial_stop else [10, 11, 99 if replay else 12, 13]
    lines += [slot('add accepted tokens: sampled=13, ids.size=3, n_draft=2')]
    lines += [emit(i, token) for i, token in enumerate(labels[1:], 2)]
    accepted = 1 if replay else 2
    if final_done and not partial_stop:
        lines += [slot(f'accepted {accepted}/2 draft tokens, new n_tokens = 8')]
    lines += [slot(f'draft acceptance = 0.50000 ({accepted:5} accepted / {2:5} generated), mean len = 1.50'), slot('stop processing: n_tokens = 8, truncated = 0')]
    response = {'tokens': labels, 'tokens_predicted': len(labels), 'tokens_evaluated': 5, 'id_slot': 0, 'stop': True, 'truncated': False, 'timings': {'prompt_n': 5, 'predicted_n': len(labels), 'draft_n': 2, 'draft_n_accepted': accepted}}
    return '\n'.join(lines), response


class ServingTraceTests(unittest.TestCase):
    def test_exact_emitted_prefix_and_round_geometry(self):
        text, response = fixture()
        result = trace.audit_trace(text, response, expected_depth=2)
        self.assertEqual(result['protection_mask'], [False, True, True, False])
        self.assertEqual([p['response_index'] for p in result['protections']], [1, 2])
        self.assertEqual([p['proposal_depth'] for p in result['protections']], [0, 1])
        self.assertEqual([p['actor_position'] for p in result['protections']], [4, 5])
        self.assertFalse(result['serving_hidden_state_reconstructed'])
        json.dumps(result)

    def test_restore_is_tentative_and_replay_correction_not_protected(self):
        text, response = fixture(replay=True)
        result = trace.audit_trace(text, response)
        self.assertEqual(result['protection_mask'], [False, True, False, False])
        self.assertEqual(result['draft_verified_accepted'], 1)
        self.assertEqual(result['draft_proposed'], 2)  # never count replay as a new draft
        self.assertTrue(result['rounds'][0]['replayed'])
        self.assertEqual(len(result['rounds'][0]['attempts']), 2)

    def test_stop_inside_emit_loop_keeps_only_emitted_success(self):
        text, response = fixture(partial_stop=True)
        result = trace.audit_trace(text, response)
        self.assertEqual(result['draft_verified_accepted'], 2)
        self.assertEqual(result['emitted_draft_successes'], 1)
        self.assertEqual(result['rounds'][0]['unemitted_verified_accepted'], 1)

    def test_last_round_can_stop_before_debug_summary(self):
        text, response = fixture(final_done=False)
        self.assertEqual(trace.audit_trace(text, response)['emitted_draft_successes'], 2)

    def test_token_pieces_newline_and_ansi_are_not_token_identity(self):
        text, response = fixture()
        text = '\x1b[32m' + text.replace("'piece'", "'a\nliteral newline'") + '\x1b[0m'
        self.assertEqual(trace.audit_trace(text, response)['emitted_draft_successes'], 2)

    def test_missing_or_conflicting_evidence_fails_closed(self):
        original, response = fixture()
        mutations = [
            lambda t: '\n'.join(l for l in t.splitlines() if 'stop processing' not in l),
            lambda t: '\n'.join(l for l in t.splitlines() if 'draft acceptance' not in l),
            lambda t: '\n'.join(l for l in t.splitlines() if 'candidate   1, pos   0' not in l),
            lambda t: '\n'.join(l for l in t.splitlines() if 'add accepted tokens' not in l),
            lambda t: '\n'.join(l for l in t.splitlines() if 'generate_draft:' not in l),
            lambda t: t.replace('n_gen = 2,', 'n_gen = 3,'),
            lambda t: t.replace('next token:    11', 'next token:    88'),
            lambda t: t.replace('sampled=13, ids.size=3', 'sampled=14, ids.size=3'),
            lambda t: t.replace('accepted 2/2 draft tokens, new', 'accepted 1/2 draft tokens, new'),
            lambda t: t.replace('#tokens=5, #draft=2, pos_next=5', '#tokens=6, #draft=2, pos_next=6'),
            lambda t: t.replace('seq_id 0', 'seq_id 1'),
            lambda t: t.replace('truncated = 0', 'truncated = 1'),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                with self.assertRaises(ValueError):
                    trace.audit_trace(mutate(original), response)

    def test_response_must_have_exact_raw_ids_and_stats(self):
        text, response = fixture()
        for mutate in [lambda r: r.pop('tokens'), lambda r: r['tokens'].__setitem__(1, 88), lambda r: r['timings'].__setitem__('draft_n_accepted', 1), lambda r: r.__setitem__('truncated', True), lambda r: r.__setitem__('stop', False), lambda r: r.__setitem__('tokens_predicted', 5)]:
            changed = copy.deepcopy(response)
            mutate(changed)
            with self.assertRaises(ValueError):
                trace.audit_trace(text, changed)

    def test_concurrent_requests_are_ambiguous(self):
        text, response = fixture()
        text = text.replace(emit(1, 10), slot('processing task, is_child = 0', task=8, slot_id=1) + '\n' + emit(1, 10))
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            trace.audit_trace(text, response, task_id=7)

    def test_sequential_requests_require_explicit_task(self):
        first, response = fixture()
        second = first.replace('task 7 |', 'task 8 |')
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            trace.audit_trace(first + '\n' + second, response)
        self.assertEqual(trace.audit_trace(first + '\n' + second, response, task_id=8)['emitted_draft_successes'], 2)

    def test_restore_without_replay_or_missing_middle_summary_is_rejected(self):
        text, response = fixture(replay=True)
        text = text.replace(slot('generate_draft: id=10, #tokens=5, #draft=2, pos_next=5') + '\n' + slot('accepted 2/2 draft tokens'), slot('accepted 2/2 draft tokens'))
        with self.assertRaises(ValueError):
            trace.audit_trace(text, response)
        text, response = fixture(final_done=False)
        # Explicit malformed continuation while an emitted round lacks its summary.
        text = text.replace(slot('stop processing: n_tokens = 8, truncated = 0'), '\n'.join(candidates()) + '\n' + slot('stop processing: n_tokens = 8, truncated = 0'))
        with self.assertRaises(ValueError):
            trace.audit_trace(text, response)


if __name__ == '__main__':
    unittest.main()
