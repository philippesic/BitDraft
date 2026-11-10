from pathlib import Path
from scripts.component_campaign.common import P, ROOT, R as COMPONENT, pin, save, identity, pause
from scripts.he4broad.common import authenticate

R = P / 'runs/he4broad-20261110-01'
REPORT = P / 'experiments/he4broad-20261110'
TEACHER = R / 'teacher-mount'
DEST = R / 'archive-mount'
OUT = DEST / 'candidate'
BUDGET = 43200
POINTS = [0, 60, *range(3600, BUDGET + 1, 3600)]
RUNTIME = COMPONENT / 'runtime04'
REMOTE_TEACHER = Path('/home/philip/he4-capture-2080ti')
MANIFEST_HASH = 'd56b7ee0572fec4333ec9c271a69befe1b0fb51e6507ef878f37ad90f1901592'
