"""CPU metadata admission on the data host; raw features are checked on exposure."""
import argparse
import hashlib
import json
from pathlib import Path


def checked(ref):
    path = Path(ref['path'])
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ref['sha256'], path
    assert len(raw) == ref.get('bytes', len(raw)), path
    return raw


def build(root, destination):
    raw = (root / 'supplemental/combined-manifest.json').read_bytes()
    manifest_hash = hashlib.sha256(raw).hexdigest()
    assert manifest_hash == 'd56b7ee0572fec4333ec9c271a69befe1b0fb51e6507ef878f37ad90f1901592'
    manifest = json.loads(raw)
    assert manifest['status'] == 'complete' and not manifest['failures']
    exclusions = set(json.loads(checked(manifest['exclusions']))['identities'])
    rows, ids, groups = [], set(), set()
    for row in manifest['records']:
        receipt = json.loads(checked(row['receipt']))
        metadata = json.loads(checked(receipt['metadata']))
        source_ref = next(x for x in receipt['files'] if Path(x['path']).name == 'source.json')
        source = json.loads(checked(source_ref))
        identities = {source.get(k) for k in ('id','group_id','prompt_id','original_prompt_id',
            'original_group','message_identity_sha256','normalized_message_identity_sha256',
            'rendered_sha256','rendered_prompt_sha256')}
        identities.update(source.get('identities', {}).values())
        identities.discard(None); identities.discard('')
        assert not identities.intersection(exclusions), row['id']
        assert row['id'] not in ids
        ids.add(row['id']); groups.add(receipt['group_id'])
        assert receipt['status'] == 'VERIFIED_NATIVE_HARD_LABEL_CAPTURE'
        assert receipt['greedy_vs_full_logits_argmax'] and receipt['no_post_eog_labels']
        assert receipt['taps'] == [2,6,10,14,18,22,26,30]
        assert receipt['feature_semantics'] == 'zero_based_pre_block_input_all_tokens'
        positions = metadata['logit_positions']
        assert positions == list(range(positions[0], positions[0] + len(positions)))
        # Each 15-label slice includes the last native logit; only the root token
        # is read from inputs. Mask slots need no future observed input tokens.
        anchors = [p for p in positions[:-14] if p < len(metadata['input_tokens'])]
        assert len(anchors) == receipt['windows']
        if anchors:
            assert anchors[0] == positions[0] and anchors[-1] + 14 == positions[-1]
        assert len(metadata['input_tokens']) == receipt['feature_shape'][0]
        assert len(metadata['greedy_ids']) == len(positions) == receipt['unique_label_positions']
        eog = set(metadata['eog_ids'])
        assert not any(x in eog for x in metadata['greedy_ids'][:-1])
        rows.append(dict(id=row['id'], receipt_pin=row['receipt'], receipt=receipt,
                         metadata=metadata, anchors=anchors))
    assert len(rows) == 15647
    assert sum(len(x['anchors']) for x in rows) == 1568435
    assert sum(x['receipt']['unique_label_positions_in_windows'] for x in rows) == 1786275
    result = dict(status='ADMITTED_METADATA', manifest_sha256=manifest_hash,
        target_sha256=manifest['target_sha256'], exclusions=manifest['exclusions'],
        rows=rows, sequences=len(rows), windows=1568435, unique_labels_in_windows=1786275,
        note='All metadata/receipt/source hashes and identity exclusions checked. Every array/file is hashed again on first materialization before any exposure; full logit scratch is intentionally absent.')
    assert not destination.exists(), destination
    destination.write_text(json.dumps(result, separators=(',',':'))+'\n')
    print(json.dumps(dict(status=result['status'], sequences=len(rows), windows=1568435,
        bytes=destination.stat().st_size,sha256=hashlib.sha256(destination.read_bytes()).hexdigest())), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root',type=Path); parser.add_argument('destination',type=Path)
    args=parser.parse_args(); build(args.root,args.destination)
