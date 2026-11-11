import argparse
import json
from pathlib import Path

from benchmarks.acceptance import reduce_rounds
from benchmarks.runner import compare
from benchmarks.specification import Specification
from w1a1_eagle.contracts import Record, sequence
from w1a1_eagle.records import read


class Arguments(argparse.Namespace):
    command: str
    input: Path
    output: Path


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "verify-inputs", "reduce-trace"):
        command = commands.add_parser(name)
        command.add_argument("input", type=Path)
    for name in ("compare", "curate"):
        comparison = commands.add_parser(name)
        comparison.add_argument("input", type=Path)
        comparison.add_argument("output", type=Path)
    args = parser.parse_args(namespace=Arguments())
    try:
        if args.command == "curate":
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
