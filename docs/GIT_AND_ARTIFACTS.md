# Git and artifact preservation

Track maintained source, tests, benchmark definitions, setup/configuration templates,
AGENTS.md, current state, indexed history, decisions and compact result summaries.
Keep credentials, machine-specific private settings, large models/data, raw captures,
logs and regenerable build caches outside Git. Ignored does not mean disposable.

## Historical audit, 2026-11-05 (before preservation commits)

Maintained source/document/configuration directories were checked for ignored files;
only expected generated/cache entries were found. Many useful files are untracked:
Git-visible is not committed, pushed or backed up. Review and commit them explicitly
before treating another clone as a complete copy. No commits or pushes were made.

- `experiments/precision-history-20261105/` now preserves the earlier precision
  campaign's compact charts, tables and generator with original-path/hash provenance.
  It does not report the active component campaign's outcome.
- `runs/` includes live ownership/queue receipts, source freezes, checkpoints and
  recovery inputs. Preserve these while work is active; do not mirror or delete them.
- `results/` contains historical analysis scripts and numerical reports alongside
  raw output and copied environments. Preserve useful unique files before cleanup.
- `models/` and `data/` contain important ignored artifacts. Record the exact retained
  checkpoint/data identity and a recoverable copy or download source before deletion.

## At a milestone or machine handoff

Keep compact results and reusable code in ordinary Git-visible project directories.
For large valuable artifacts, record identity, location, purpose and recovery source
in the existing experiment/campaign record. Verify the recovery copy before deleting
the original. A second name, symlink, same-disk copy or an untested hash is not an
independent backup. No new per-run form or automatic cleanup is required.

This audit did not verify an independent backup of the ignored artifact trees.
Historical copies contain unique dirty/untracked work and must be reconciled before
retirement. Frozen source snapshots, nested repositories and third-party dependencies
retain their own history and exclusions; do not apply current rules over pinned runs.
Raw metadata inventories and audit limits are retained outside Git in the performing
agent's `~/.local/share/p-stack/repairs/20261105-ignore-audit/` receipt directory.

## Git maintenance update, 2026-11-06

Project AGENTS.md now authorizes milestone and handoff pushes to the established
user-owned origin. Source, project agent guidance and compact findings belong in
this repository, not in p-stack. Campaign-level authored scripts and written reports
are Git-visible even inside run directories; deeper frozen copies and large raw
artifacts remain local. This is not permission to delete them. Maintained source
directories were audited for hidden files; caches remain excluded. Git synchronization
is verified separately from runtime correctness and artifact backup.
