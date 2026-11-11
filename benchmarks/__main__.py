import argparse
import json
import subprocess
import sys
from pathlib import Path

from benchmarks.acceptance import reduce_rounds
from benchmarks.capture import capture
from benchmarks.native import build, helpers
from benchmarks.runner import compare
from benchmarks.specification import Specification
from w1a1_eagle.config import benchmark_environment, numerical_python
from w1a1_eagle.contracts import Record, sequence
from w1a1_eagle.identity import freeze_source, source_identity
from w1a1_eagle.records import read


class Arguments(argparse.Namespace):
    command: str
    input: Path
    output: Path
    frozen: bool


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks")
    parser.add_argument("--frozen", action="store_true", help=argparse.SUPPRESS)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "verify-inputs", "reduce-trace"):
        command = commands.add_parser(name)
        command.add_argument("input", type=Path)
    for name in (
        "compare",
        "curate",
        "native-build",
        "capture",
        "admit-bank",
        "native-helpers",
        "evaluate",
        "live",
        "prepare-panels",
        "suite",
        "campaign",
        "verify-sampler",
        "verify-greedy",
        "select-supplement",
    ):
        comparison = commands.add_parser(name)
        comparison.add_argument("input", type=Path)
        comparison.add_argument("output", type=Path)
    args = parser.parse_args(namespace=Arguments())
    try:
        if not args.frozen and args.command not in {"plan", "verify-inputs", "reduce-trace"}:
            root = Path(__file__).resolve().parents[1]
            code = source_identity(root)
            snapshot = freeze_source(
                root, args.output.resolve().parent / "executed-sources", code, 0
            )
            python = (
                numerical_python(root)
                if args.command in {"curate", "capture", "evaluate", "campaign"}
                else sys.executable
            )
            return subprocess.run(
                [
                    python,
                    "-B",
                    "-m",
                    "benchmarks",
                    "--frozen",
                    args.command,
                    str(args.input.resolve()),
                    str(args.output.resolve()),
                ],
                cwd=snapshot,
                env=benchmark_environment(root, snapshot),
                check=False,
            ).returncode
        if args.command == "select-supplement":
            from benchmarks.supplement import supplement

            result = supplement(args.input, args.output)
        elif args.command == "verify-sampler":
            from benchmarks.verifier import verify_sampler

            result = verify_sampler(args.input, args.output)
        elif args.command == "verify-greedy":
            from benchmarks.verifier import verify_greedy

            result = verify_greedy(args.input, args.output)
        elif args.command == "suite":
            from benchmarks.suite import suite

            result = suite(args.input, args.output)
        elif args.command == "campaign":
            from benchmarks.campaign import campaign

            result = campaign(args.input, args.output)
        elif args.command == "prepare-panels":
            from benchmarks.panels import panels

            result = panels(args.input, args.output)
        elif args.command == "live":
            from benchmarks.workflows import live

            result = live(args.input, args.output)
        elif args.command == "evaluate":
            from w1a1_eagle.numerical.evaluation import evaluate

            result = evaluate(args.input, args.output)
        elif args.command == "native-helpers":
            result = helpers(args.input, args.output)
        elif args.command == "admit-bank":
            from benchmarks.bank_admission import admit

            result = admit(args.input, args.output)
        elif args.command == "capture":
            result = capture(args.input, args.output)
        elif args.command == "native-build":
            result = build(args.input, args.output)
        elif args.command == "curate":
            from benchmarks.bank import curate

            result = curate(args.input, args.output)
        elif args.command == "compare":
            result = compare(args.input, args.output)
        elif args.command == "reduce-trace":
            result = reduce_rounds(sequence(read(args.input, 16 * 1024 * 1024)["rounds"]))
        else:
            spec = Specification.load(args.input)
            result = spec.report()
            if args.command == "verify-inputs":
                if not spec.features:
                    raise ValueError("no feature artifacts declared")
                for artifact in spec.features:
                    artifact.verify()
                result["authenticated_artifacts"] = len(spec.features)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError) as error:
        refusal: Record = {"outcome": "refused", "detail": str(error)}
        print(json.dumps(refusal))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
