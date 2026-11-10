import argparse
import json
import subprocess
from pathlib import Path

from w1a1_eagle.config import cpu_environment, numerical_python
from w1a1_eagle.contracts import Record, record
from w1a1_eagle.lifecycle import Journal
from w1a1_eagle.supervisor import execute


def human(report: Record) -> str:
    if report.get("kind") == "admission":
        exposure = record(report["exposure"])
        return (
            f"Starting {report['mode']} from {report['starting_name']}.\n"
            f"Mechanism: {report['mechanism']}.\n"
            f"Data: {exposure['sequences']} sequences, {exposure['distinct_positions']} distinct positions; "
            f"{exposure['label_exposures']} planned label exposures, {exposure['mean_visits']} mean additional visits.\n"
            f"Data history: {report['prior_use_warning']}\n"
            f"Limiting risk: {report['limiting_risk']}.\nStop: {report['stop_action']}.\n"
            f"Readiness: {report['readiness']}. Scientific: {report['scientific']}.\n"
            f"Scope: {report['evidence_scope']}."
        )
    if "latest" in report:
        failure = report.get("failure")
        cause = f"Confirmed failure record: {record(failure)}.\n" if failure is not None else ""
        return (
            cause + f"Operational: {report['operational']}; {report['attempts']} attempts, "
            f"{report['cumulative_attempt_seconds']} cumulative seconds.\n"
            f"Scientific: {report['scientific']}.\nObjective: {report['objective']}.\n"
            f"Selection: {report['selection']}."
        )
    return json.dumps(report, indent=2, sort_keys=True)


class Arguments(argparse.Namespace):
    command: str
    as_json: bool
    intent: Path
    output: Path
    scope: str


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m w1a1_eagle")
    parser.add_argument("--json", action="store_true", dest="as_json")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("intent", type=Path)
    run.add_argument("output", type=Path)
    run.add_argument("--scope", required=True)
    report_parser = commands.add_parser("report")
    report_parser.add_argument("output", type=Path)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("intent", type=Path)
    arguments = parser.parse_args(namespace=Arguments())
    try:
        if arguments.command == "preflight":
            result = subprocess.run(
                [
                    numerical_python(Path(__file__).resolve().parent.parent),
                    "-B",
                    "-m",
                    "w1a1_eagle.numerical.entry",
                    "--preflight",
                    str(arguments.intent.resolve()),
                ],
                env=cpu_environment(),
                check=False,
            )
            return result.returncode
        if arguments.command == "run":

            def emit(value: Record) -> None:
                print(
                    json.dumps(value, sort_keys=True) if arguments.as_json else human(value),
                    flush=True,
                )

            report = execute(arguments.intent, arguments.output, arguments.scope, emit)
        else:
            report = Journal(arguments.output).report()
        print(json.dumps(report, sort_keys=True) if arguments.as_json else human(report))
        latest = report.get("latest")
        return (
            0
            if report["operational"] == "terminal"
            and latest is not None
            and record(latest).get("outcome") == "complete"
            else 1
        )
    except (OSError, ValueError, KeyError) as error:
        refusal: Record = {
            "outcome": "refused",
            "cause": type(error).__name__,
            "detail": str(error),
        }
        print(json.dumps(refusal))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
