"""Audit emitted EAGLE successes from native server debug logs, without models.

Source contract: common/speculative.cpp::draft logs top3 at each depth;
server-context.cpp logs handle_last_sampled_token, checkpoint restore,
add-accepted, process_token and final stats. A restore's acceptance is tentative.
Replayed verification subtracts the replay correction from n_draft_accepted.
Only process_token IDs joined exactly to raw response.tokens become protections.
This does not reconstruct serving hidden/KV state or validate conditional replay.
Concurrent speculative sequences and context shifts are deliberately unsupported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")
SLOT = re.compile(r"\bslot\s+\S+:\s+id\s+(\d+)\s*\|\s*task\s+(-?\d+)\s*\|\s*(.*)")
CANDIDATE = re.compile(r"\bspec\s+draft:\s+- seq_id (\d+), draft candidate\s+(\d+), pos\s+(\d+):\s+(\d+)\s+\(")
GENERATE = re.compile(r"generate_draft: id=(\d+), #tokens=(\d+), #draft=(\d+), pos_next=(\d+)")
ADD = re.compile(r"add accepted tokens: sampled=(\d+), ids.size=(\d+), n_draft=(\d+)")
EMIT = re.compile(r"n_gen = (\d+), n_remaining = -?\d+, next token:\s+(\d+) ")
DONE = re.compile(r"accepted (\d+)/(\d+) draft tokens, new n_tokens = (\d+)")
INFO = re.compile(r"accepted\s+(\d+)/\s*(\d+) draft tokens( \(restore checkpoint\))?$")
TOTAL = re.compile(r"draft acceptance = [0-9.]+ \(\s*(\d+) accepted /\s*(\d+) generated\)")
RESTORE = re.compile(r"restoring speculative checkpoint \(pos_min = (-?\d+), pos_max = (-?\d+), size = (\d+)\)")
RELEASE = re.compile(r"stop processing: n_tokens = (\d+), truncated = (\d+)")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _request_lines(text, task_id, slot_id):
    """Select a bounded launched/released request; require unique attribution."""
    lines = [(i, ANSI.sub('', line)) for i, line in enumerate(text.splitlines(), 1)]
    launches = []
    for i, line in lines:
        match = SLOT.search(line)
        if match and 'processing task, is_child =' in match[3]:
            launches.append((i, int(match[1]), int(match[2])))
    choices = [x for x in launches if (task_id is None or x[2] == task_id) and (slot_id is None or x[1] == slot_id)]
    require(len(choices) == 1, 'exactly one launched request required; specify task/slot for multi-request logs')
    start, slot_id, task_id = choices[0]
    releases = [(i, line) for i, line in lines if i > start and (m := SLOT.search(line)) and int(m[1]) == slot_id and int(m[2]) == task_id and RELEASE.fullmatch(m[3])]
    require(releases, 'missing request release: incomplete/truncated trace')
    end = releases[0][0]
    require(not any(start < i < end for i, _, _ in launches), 'concurrent/overlapping requests cannot attribute untagged draft logs')
    return [(i, line) for i, line in lines if start <= i <= end], slot_id, task_id


def audit_trace(text, response, *, task_id=None, slot_id=None, expected_depth=None):
    """Return exact response-index protections or raise ValueError (fail closed).

    Response must be the raw native /completion object with return_tokens=true.
    task_id is a native server task identifier, not HTTP request ordering.
    """
    labels = response.get('tokens')
    require(isinstance(labels, list) and labels and all(type(t) is int and t >= 0 for t in labels), 'raw response integer token IDs required')
    require(response.get('stop') is True and response.get('truncated') is False, 'complete non-context-truncated response required')
    require(response.get('tokens_predicted') == len(labels), 'response token count differs')
    timings = response.get('timings', {})
    require(timings.get('predicted_n') == len(labels), 'response timing token count differs')
    prompt = response.get('tokens_evaluated')
    require(type(prompt) is int and prompt > 0 and timings.get('prompt_n') == prompt, 'unambiguous native prompt length required')
    lines, slot_id, task_id = _request_lines(text, task_id, slot_id)
    require(response.get('id_slot') == slot_id, 'response slot differs')
    emitted, rounds, protections = [], [], []
    candidates = {}
    active = None
    totals = None
    released = False
    zero_draft_attempts = 0

    def validate_candidate_table():
        require(candidates and sorted(candidates) == list(range(max(candidates) + 1)), 'missing/noncontiguous draft candidate depth')
        require(all(set(v) == {0, 1, 2} for v in candidates.values()), 'missing/duplicate top3 candidate logs')

    for line_number, line in lines:
        prefix = f'line {line_number}: '
        cand = CANDIDATE.search(line)
        if cand:
            seq, rank, depth, token = map(int, cand.groups())
            require(seq == slot_id, prefix + 'concurrent speculative sequence')
            require(active is None, prefix + 'new candidates before prior round committed')
            require(rank <= 2 and rank not in candidates.setdefault(depth, {}), prefix + 'duplicate/invalid candidate rank')
            candidates[depth][rank] = token
            continue
        match = SLOT.search(line)
        if not match:
            # Pieces may contain literal newlines; only the native header is parsed.
            continue
        sid, tid, msg = int(match[1]), int(match[2]), match[3]
        require((sid, tid) == (slot_id, task_id), prefix + 'interleaved slot/task trace')
        require('context shift' not in msg and 'truncated = 1' not in msg, prefix + 'context shift/truncation unsupported')
        if m := GENERATE.fullmatch(msg):
            sampled, ntokens, ndraft, pos = map(int, m.groups())
            require(emitted and sampled == emitted[-1], prefix + 'draft boundary sample differs from emitted response')
            require(ntokens == pos == prompt + len(emitted) - 1, prefix + 'draft boundary position/cache semantics differ')
            require(ndraft > 0 and (expected_depth is None or ndraft <= expected_depth), prefix + 'invalid/unexpected draft depth')
            if active is None:
                validate_candidate_table()
                require(len(candidates) in {ndraft, ndraft + 1}, prefix + 'candidate/depth count differs')
                active = {'round': len(rounds), 'anchor_position': pos - 1, 'parent_response_index': len(emitted) - 1, 'first_proposal_response_index': len(emitted), 'boundary_token': sampled, 'pos_next': pos, 'proposals': [candidates[d][0] for d in range(ndraft)], 'top3': [candidates[d] for d in range(ndraft)], 'proposed': ndraft, 'attempts': [], 'replayed': False, 'emitted': [], 'start_line': line_number}
                candidates = {}
                rounds.append(active)
            else:
                require(active.get('await_replay') and not candidates, prefix + 'unexpected repeated verification')
                require(pos == active['pos_next'] and ndraft <= active['attempts'][-1]['draft_count'], prefix + 'invalid checkpoint replay geometry')
                previous = active['attempts'][-1]
                if 'tentative_accepted' in previous:
                    require(ndraft == previous['tentative_accepted'] + 1, prefix + 'replayed draft differs from tentative accepted-plus-correction')
                active['await_replay'] = False
                active['replayed'] = True
            active['attempts'].append({'draft_count': ndraft, 'generate_line': line_number})
        elif INFO.fullmatch(msg):
            require(active is not None, prefix + 'acceptance without proposal')
            m = INFO.fullmatch(msg)
            count, denom = int(m[1]), int(m[2])
            require(denom == active['attempts'][-1]['draft_count'] and count <= denom, prefix + 'acceptance geometry differs')
            key = 'tentative_accepted' if m[3] else 'reported_accepted'
            require(key not in active['attempts'][-1], prefix + 'duplicate acceptance info')
            require(not m[3] or count < denom, prefix + 'full acceptance cannot restore checkpoint')
            active['attempts'][-1][key] = count
        elif m := RESTORE.fullmatch(msg):
            require(active is not None and not active['emitted'] and 'ids_size' not in active and not active.get('await_replay'), prefix + 'unexpected checkpoint restoration')
            active['await_replay'] = True
            active['attempts'][-1]['restored'] = True
            active['attempts'][-1]['checkpoint'] = {'pos_min': int(m[1]), 'pos_max': int(m[2]), 'bytes': int(m[3]), 'line': line_number}
        elif m := ADD.fullmatch(msg):
            sampled, size, ndraft = map(int, m.groups())
            require(active is not None and not active.get('await_replay') and 'ids_size' not in active, prefix + 'accepted IDs without completed verification')
            attempt = active['attempts'][-1]
            require(ndraft == attempt['draft_count'] and 1 <= size <= ndraft + 1, prefix + 'accepted ID count differs')
            if 'reported_accepted' in attempt:
                require(attempt['reported_accepted'] == size - 1, prefix + 'trace acceptance differs from ids.size')
            verified = size - 1
            if active['replayed'] and verified > 0:
                verified -= 1  # server-context.cpp excludes the replay correction
            require(verified <= active['proposed'], prefix + 'accepted original-prefix count exceeds proposal')
            active.update(ids_size=size, sampled=sampled, verified_accepted=verified, final_draft_count=ndraft, add_line=line_number)
        elif m := EMIT.match(msg):
            index, token = int(m[1]) - 1, int(m[2])
            require(index == len(emitted) and index < len(labels) and token == labels[index], prefix + 'emitted token/response index mismatch')
            if active is not None:
                require('ids_size' in active and not active.get('await_replay'), prefix + 'emission before committed verification')
                depth = len(active['emitted'])
                require(depth < active['ids_size'], prefix + 'too many tokens in verified emission')
                active['emitted'].append({'response_index': index, 'token': token, 'line': line_number})
                if depth < active['verified_accepted']:
                    require(token == active['proposals'][depth], prefix + 'accepted token differs from original EAGLE proposal')
                    protections.append({'response_index': index, 'token': token, 'round': active['round'], 'proposal_depth': depth, 'anchor_position': active['anchor_position'], 'actor_position': active['anchor_position'] + depth, 'parent_response_index': active['parent_response_index'], 'proposal_line': active['start_line'], 'emission_line': line_number})
                if depth + 1 == active['ids_size']:
                    require(token == active['sampled'], prefix + 'last emitted ID differs from sampled bonus/correction')
            else:
                require(not candidates, prefix + 'unresolved draft candidates before ordinary emission')
            emitted.append(token)
        elif m := DONE.fullmatch(msg):
            accepted, ndraft, ntokens = map(int, m.groups())
            require(active is not None and 'ids_size' in active, prefix + 'round summary without verified emission')
            require(len(active['emitted']) == active['ids_size'], prefix + 'incomplete emitted round')
            require(accepted == active['verified_accepted'] and ndraft == active['final_draft_count'] and ntokens == active['pos_next'] + active['ids_size'], prefix + 'round summary/statistics mismatch')
            active['done_line'] = line_number
            active = None
        elif m := TOTAL.search(msg):
            require(totals is None, prefix + 'duplicate final draft totals')
            totals = tuple(map(int, m.groups()))
        elif m := RELEASE.fullmatch(msg):
            require(not released and int(m[2]) == 0, prefix + 'invalid release')
            if active is not None:
                require('ids_size' in active and active['emitted'] and not active.get('await_replay'), prefix + 'unresolved verification at release')
                # process_token can stop inside the emission loop before DONE.
                require(int(m[1]) == active['pos_next'] + active['ids_size'], prefix + 'final prompt/cache count differs')
                active['stopped_during_emission'] = True
                active['unemitted_verified_accepted'] = max(0, active['verified_accepted'] - len(active['emitted']))
                active = None
            require(not candidates, prefix + 'unresolved candidate table at release')
            released = True
        elif msg.startswith('slot decode token,') and candidates:
            require(active is None, prefix + 'ordinary decode during verification')
            validate_candidate_table()
            candidates = {}  # p_min/n_min produced no actual draft; not counted
            zero_draft_attempts += 1
        elif any(key in msg for key in ['generate_draft:', 'add accepted tokens:', 'next token:', 'new n_tokens', 'restoring speculative checkpoint', 'draft acceptance =']):
            raise ValueError(prefix + 'unrecognized required trace record')
    require(released and emitted == labels and active is None and not candidates, 'incomplete token/trace join')
    verified = sum(r['verified_accepted'] for r in rounds)
    proposed = sum(r['proposed'] for r in rounds)
    if proposed:
        require(totals == (verified, proposed), 'final native draft totals differ or are missing')
        require((timings.get('draft_n_accepted'), timings.get('draft_n')) == totals, 'raw response draft totals differ/missing')
    else:
        require(totals is None and timings.get('draft_n', 0) == timings.get('draft_n_accepted', 0) == 0, 'unexpected zero-draft totals')
    protected_indices = {p['response_index'] for p in protections}
    return {'schema': 'w1a8_actual_serving_trace_audit_v1', 'status': 'PASS', 'task_id': task_id, 'slot_id': slot_id, 'prompt_tokens': prompt, 'response_tokens': len(labels), 'draft_proposed': proposed, 'draft_verified_accepted': verified, 'emitted_draft_successes': len(protections), 'protection_mask': [i in protected_indices for i in range(len(labels))], 'protections': protections, 'rounds': rounds, 'zero_draft_attempts': zero_draft_attempts, 'serving_hidden_state_reconstructed': False, 'scope': 'actual emitted native draft successes; no conditional replay equivalence or quality claim'}


def _pin(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', required=True, type=Path)
    parser.add_argument('--response', required=True, type=Path)
    parser.add_argument('--task-id', type=int)
    parser.add_argument('--slot-id', type=int)
    parser.add_argument('--expected-depth', type=int)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    log_pin, log = _pin(args.log)
    response_pin, response = _pin(args.response)
    result = audit_trace(log.decode('utf-8'), json.loads(response), task_id=args.task_id, slot_id=args.slot_id, expected_depth=args.expected_depth)
    result['inputs'] = {'log': log_pin, 'response': response_pin, 'parser': _pin(__file__)[0]}
    require(not args.out.exists(), 'refusing to replace existing audit')
    args.out.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
