# Development

[README setup](../README.md#setup) owns environment installation. The
[architecture map](architecture.md) locates source owners and public entries;
[checks](checks.md#choosing-verification) selects validation lanes. Read the relevant
data, execution or benchmark guide for the change. [FINDINGS](../FINDINGS.md) leads to
prior conclusions and their evidence; historical source recovery is in [evidence](evidence.md).

## Repository interfaces

- Public entries are declared in `policy/commands.toml` and mapped in
  [architecture](architecture.md#entries-and-flows).
- `pyproject.toml` and `uv.lock` own dependency versions. Host numerical interpreter
  configuration and check commands are documented in [checks](checks.md#commands).
- Checker rules, rejecting/accepting cases and required-test behavior are documented
  in [checks](checks.md#adding-things). A new file or ordinary test needs no rule-registry edit.
- Unfinished multi-session work uses [tasks](../tasks/README.md); durable research
  conclusions use the [result format](results.md).

## Experiments

Use maintained [benchmark](benchmarks.md) and [training](execution.md) entries. The task
records model/data identities, the question, allocation and stopping conditions. Model
or data-role changes and new training allocations require an authorized scientific scope.
Keep recovery state, provisional acceptance leader and full-comparison incumbent separate.
Frozen execution source and owner/birth/boot checks govern recovery, not the current checkout.
Persistent launch, progress inspection and pause budgets are specified in the benchmark guide.

The shared `benchmark-design` skill is for measurement design or repair. Ordinary candidate
runs use the existing commands and reference contracts. `correct` handles recurring agent
or workflow mistakes; isolated product fixes use their maintained commands and relevant tests.

## Investigation closeout

Follow the pinned [close-results procedure](../skills/close-results/SKILL.md) and
[project specification](results.md). This project copy governs even when a shared version
is installed; the link makes discovery independent of automatic skill loading.

The specification defines extraction of compact measurements, figure inputs and source
identities, result sections, FINDINGS updates and explicit artifact disposition. The
[retirement command](results.md#retirement-command) verifies retained evidence and exact
removal paths before authorized cleanup. Needed data/model inputs remain external;
discarded states lose exact resume/replay. Routine repairs normally need no result page.
