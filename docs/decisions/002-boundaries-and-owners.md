# 002: Boundaries and owners

Date: 2026-11-09. Status: foundation decision; later gates remain pending.

Pure contracts use stdlib only. Tools never depend on tests; product never depends on tools/tests/history/outputs. Training/native are siblings composed by interfaces. Reserved owners: ledger for roles/consumption, store for states/latest/best, lifecycle for run state, runner for raw results. Filename discovery is sorted, unique and protocol/schema/digest checked; no central ID dispatch.
