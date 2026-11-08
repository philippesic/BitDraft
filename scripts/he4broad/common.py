"""Paths and fixed authorization for the source-initialized HE4broad recipe."""
from scripts.component_campaign.common import P, ROOT, R as COMPONENT, pin, save, identity, pause

R = P / 'runs/he4broad-20261108-01'
OUT = R / 'candidate'
REPORT = P / 'experiments/he4broad-20261108'
DEST = __import__('pathlib').Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/he4broad-20261108-01')
BUDGET = 28800
POINTS = list(range(0, BUDGET + 1, 3600))
SLOT_BYTES = 6 * 2**30
RUNTIME = COMPONENT / 'runtime04'


def authenticate(ref):
    actual = pin(ref['path'])
    assert actual['sha256'] == ref['sha256'], ref['path']
    if 'bytes' in ref:
        assert actual['bytes'] == ref['bytes'], ref['path']
    return actual
