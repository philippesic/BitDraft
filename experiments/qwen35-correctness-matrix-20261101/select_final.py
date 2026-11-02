"""Freeze development-selected configurations before generating fresh inputs."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

from run_matrix import pin, read, write


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--development-summary', type=Path, required=True)
    parser.add_argument('--record-exclusions', action='store_true', help='Explicitly retain inadmissible groups as unavailable rather than silently dropping them')
    args = parser.parse_args()
    run = args.run.resolve()
    development = read(args.development_summary)
    manifest = read(run / 'screen-freeze.json')
    output = run / 'final-freeze.json'
    assert not output.exists(), 'Never overwrite a final freeze'
    excluded = [row for row in development['selection'] if not row['selected']]
    assert not excluded or args.record_exclusions, excluded
    # IDs repeat across targets, so select by the complete target/arm identity.
    identities = {(row['target_id'], row['selected']) for row in development['selection'] if row['selected']}
    selected_rows = [row for row in development['arms'] if (row['target_id'], row['id']) in identities]
    assert all(row['admitted'] for row in selected_rows)
    assert len(selected_rows) + len(excluded) == 58
    arm_keys = ('target_id', 'id', 'group', 'family', 'dtype', 'draft', 'spec_type', 'depth')
    manifest['arms'] = [{key: row[key] for key in arm_keys} for row in selected_rows]
    for target in manifest['targets']:
        assert any(row['target_id'] == target and row['family'] == 'target_only' for row in manifest['arms'])
        assert args.record_exclusions or any(row['target_id'] == target and row['group'] == 'eagle3-Q4_0' for row in manifest['arms'])
    manifest['repetitions'] = 5
    manifest['settings']['output_limit'] = 256
    manifest['frozen_at_utc'] = datetime.now(timezone.utc).isoformat()
    manifest['scope'] = 'Fresh final four-target correctness-mode baseline matrix; development selected, no Binary/W1A8 result'
    manifest['development_selection'] = pin(args.development_summary)
    manifest['excluded_groups'] = excluded
    manifest['sources'].append(pin(__file__))
    manifest['pins'].append(pin(__file__))
    manifest['pins'].append(pin(args.development_summary))
    write(output, manifest)
    print({'final_cells': len(manifest['arms']), 'excluded_groups': len(excluded), 'frozen_at_utc': manifest['frozen_at_utc']})


if __name__ == '__main__':
    main()
