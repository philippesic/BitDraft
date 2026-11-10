"""Philip's November 10 continuation: eight cumulative hours of GPU updates.

The original recipe/config hash and historical wall-clock panels stay frozen.
This execution policy supersedes their wall-time stop, without resetting state.
"""
UPDATE_BUDGET = 8 * 3600
UPDATE_POINTS = tuple(range(3600, UPDATE_BUDGET + 1, 3600))


def charged_seconds(progress):
    return progress['update_seconds'] + progress.get('discarded_update_seconds', 0.)


def pending(progress, until):
    return charged_seconds(progress) < min(until, UPDATE_BUDGET)


def label(point):
    assert point in UPDATE_POINTS
    return 'update-' + str(point)


def panel_label(point):
    return 'formal-' + label(point)


POLICY = dict(version=1, authorization='Philip November 10: resume to eight TOTAL hours of updates, including the existing 47.9 minutes',
    clock='synchronized_update_seconds', total_update_seconds=UPDATE_BUDGET,
    training_wall_seconds_cap=None, readout_update_seconds=list(UPDATE_POINTS),
    existing_state='Exact own-run parameters, Adam, RNG, shuffled order and cursor; no reset',
    accounting='Completed and rejected update time charged cumulatively; setup/data/checkpoint/export/native time reported separately',
    historical_panels='Original wall-clock panels retained with their original labels; update-hour panels use formal-update-*')
