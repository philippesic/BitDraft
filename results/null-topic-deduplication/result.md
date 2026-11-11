# Missing-topic deduplication dropped distinct prompts

## Outcome

A deduplication bug on empty `source_topic` merged distinct prompts. The supplied
partition is 10,000 = 96 cal-fit + 48 cal-val + 9,856 candidates, and 9,856 = 12 DEV +
23 real collisions + 5,505 wrong rejections + 4,316 admitted. With only the genuine
exclusions, 9,821 would have been admitted. The recapture of about 5,497 prompts differs
from 5,505 by eight, and that difference is unexplained. The old fixture had one null
topic and could not catch two nulls being merged.

The 2080 Ti recapture reported 10,150 sequences, 970,121 windows and 1,111,811 labels.
These are gross counts supplied by Philip, not audited and not admitted.

## Scope and limitations

bug description and counts supplied by Philip, not audited on the 2080 Ti.
No recaptured prompt is admitted.

## Decision

Retain this scoped finding for future decisions. No new experimental or deployment decision is made by this documentation migration.

## Evidence and reproduction

[scripts/capture_nine_model_train_data.py](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/scripts/capture_nine_model_train_data.py) [tests/test_dspark_full_pool_capture.py](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/tests/test_dspark_full_pool_capture.py)

Imported from the linked records; no experiment was rerun for this documentation
migration. Complete executed-source, dependency and artifact availability have not
been reverified. Missing reproduction details must be recovered before artifact
retirement; report revisions alone are not execution provenance. No original source,
checkpoint or capture was deleted.
