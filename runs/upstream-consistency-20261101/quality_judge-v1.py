#!/usr/bin/env python3
"""Independent development checks; these tasks are not final held-out evidence.

Send only each case's ``prompt`` to a model. The tests and expected answers are
judge inputs, not part of that prompt. Generated Python runs with a restricted
AST/builtin set in a resource-limited isolated interpreter. SQL runs only against
an in-memory database with a read-only authorizer and an instruction budget.
These restrictions are task-specific checks, not a general-purpose code sandbox.
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
from pathlib import Path
import re
import resource
import sqlite3
import subprocess
import sys
import time

MAX_ANSWER_BYTES = 24_000
WORKER_TIMEOUT_SECONDS = 4
SAFE_BUILTINS = {
    name: getattr(__builtins__, name) if not isinstance(__builtins__, dict) else __builtins__[name]
    for name in ("abs", "all", "any", "bool", "dict", "enumerate", "int", "len",
                 "list", "max", "min", "range", "reversed", "set", "sorted",
                 "str", "sum", "tuple", "zip")
}
SAFE_METHODS = {"append", "clear", "copy", "count", "extend", "get", "index",
                "insert", "items", "keys", "pop", "remove", "reverse", "sort", "values"}


def cases() -> list[dict]:
    """Return eight versioned, intentionally small development tasks."""
    return [
        {"id": "dev_json_pencils", "kind": "json", "expected": {"remaining": 295},
         "prompt": "A shop has 24 boxes with 18 pencils each and sells 137 pencils. "
                   "Return only JSON with the single integer field remaining."},
        {"id": "dev_json_duration", "kind": "json", "expected": {"seconds": 8700},
         "prompt": "A recording lasts 1 hour 37 minutes, followed by 48 minutes of discussion. "
                   "Return their combined duration as JSON with the single integer field seconds."},
        {"id": "dev_json_sensor", "kind": "json", "expected": {"sensor": "B", "corrected": 17},
         "prompt": "Records: sensor A reports 14 degrees; sensor B reports 19 degrees; "
                   "sensor C reports 23 degrees. Only B has a positive measurement bias of "
                   "2 degrees, which must be subtracted. Return only JSON with sensor set "
                   "to B and corrected set to B's corrected integer temperature."},
        {"id": "dev_json_order", "kind": "json", "expected": {"names": ["Di", "Bo", "Ava", "Cy"]},
         "prompt": "Scores: Ava=4, Bo=2, Cy=4, Di=1. Sort by ascending score, breaking ties "
                   "by ascending name. Return only JSON with names containing the ordered names."},
        {"id": "dev_json_logic", "kind": "json", "expected": {"true_statement": "R"},
         "prompt": "Exactly one of the Boolean statements P, Q, R is true. P is false. "
                   "The implication 'if Q then R' is true. Which statement is true? "
                   "Return only JSON with the single string field true_statement."},
        {"id": "dev_python_first_unique", "kind": "python", "function": "first_unique",
         "prompt": "Write only a Python function first_unique(s) that returns the first "
                   "character occurring exactly once in the string s, or None if none exists. "
                   "Treat Unicode characters normally. Use no imports or module-level code. "
                   "Return the function source only, optionally in one Python code fence.",
         "tests": [{"args": [s], "expected": answer} for s, answer in [
             ("", None), ("a", "a"), ("aabb", None), ("swiss", "w"),
             ("redivider", "v"), ("aabccbd", "d"), ("αββαγ", "γ"),
             ("😀a😀b", "a"), (" a a!", "!")]]},
        {"id": "dev_python_merge_intervals", "kind": "python", "function": "merge_intervals",
         "prompt": "Write only a Python function merge_intervals(intervals). Input is a list "
                   "of closed integer intervals [start,end], each with start <= end. Return "
                   "a sorted list of merged intervals, merging both overlapping intervals "
                   "and intervals sharing an endpoint. Empty input returns []. Do not mutate "
                   "the input. Use no imports or module-level code. Return the function source "
                   "only, optionally in one Python code fence.",
         "tests": [{"args": [x], "expected": y} for x, y in [
             ([], []), ([[1, 1]], [[1, 1]]), ([[1, 4], [4, 5]], [[1, 5]]),
             ([[5, 7], [1, 3], [2, 4]], [[1, 4], [5, 7]]),
             ([[1, 10], [2, 3], [4, 9]], [[1, 10]]),
             ([[-5, -2], [-2, 0], [3, 3]], [[-5, 0], [3, 3]]),
             ([[1, 2], [1, 2], [2, 2]], [[1, 2]]),
             ([[8, 9], [2, 4], [4, 8], [-1, 0]], [[-1, 0], [2, 9]])]]},
        {"id": "dev_sql_paid_totals", "kind": "sql",
         "prompt": "SQLite table orders(id INTEGER PRIMARY KEY, customer_id INTEGER, "
                   "status TEXT, total INTEGER) contains nonnegative integer totals. Write "
                   "one SELECT query returning customer_id and paid_total for customers "
                   "with at least two orders whose status is 'paid' and paid-order totals "
                   "summing to at least 30. Ignore all other statuses. Order by customer_id "
                   "ascending. Return SQL only, optionally in one SQL code fence.",
         "schema": "CREATE TABLE orders(id INTEGER PRIMARY KEY, customer_id INTEGER, status TEXT, total INTEGER)",
         "data": [(1, 1, "paid", 10), (2, 1, "paid", 20), (3, 1, "cancelled", 500),
                  (4, 2, "paid", 40), (5, 2, "cancelled", 50),
                  (6, 3, "paid", 15), (7, 3, "paid", 15),
                  (8, 4, "paid", 5), (9, 4, "paid", 22),
                  (10, 5, "paid", 0), (11, 5, "paid", 45), (12, 6, "pending", 80)],
         "expected": [[1, 30], [3, 30], [5, 45]]},
    ]


def _code(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        match = re.fullmatch(r"```(?:python|py|sql|sqlite|json)?[ \t]*\n(.*?)\n```", text,
                             flags=re.DOTALL | re.IGNORECASE)
        if match is None:
            raise ValueError("one complete code fence with no outside text required")
        text = match.group(1).strip()
    return text


def _same(actual, expected) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(_same(actual[k], v) for k, v in expected.items())
    if isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(_same(a, b) for a, b in zip(actual, expected))
    return actual == expected


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object field")
        result[key] = value
    return result


def _reject_constant(_value):
    raise ValueError("nonfinite JSON number")


def _python_tree(program: str, function: str) -> ast.Module:
    tree = ast.parse(program)
    body = tree.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]
    if len(body) != 1 or not isinstance(body[0], ast.FunctionDef) or body[0].name != function:
        raise ValueError("exactly the requested function and no module-level code required")
    fn = body[0]
    args = fn.args
    if (fn.decorator_list or args.defaults or args.kw_defaults or args.vararg or args.kwarg
            or args.kwonlyargs or len(args.args) + len(args.posonlyargs) != 1):
        raise ValueError("one required positional parameter and no decorators/defaults required")
    forbidden = (ast.Import, ast.ImportFrom, ast.ClassDef, ast.AsyncFunctionDef,
                 ast.Global, ast.Nonlocal, ast.With, ast.AsyncWith, ast.Try,
                 ast.Raise, ast.Delete, ast.Lambda, ast.Yield, ast.YieldFrom, ast.Await)
    for node in ast.walk(tree):
        if isinstance(node, forbidden):
            raise ValueError(f"unsupported Python construct: {type(node).__name__}")
        if isinstance(node, ast.FunctionDef) and node is not fn:
            raise ValueError("nested functions are not needed for this task")
        if isinstance(node, ast.Name) and "__" in node.id:
            raise ValueError("dunder names are forbidden")
        if isinstance(node, ast.Attribute) and node.attr not in SAFE_METHODS:
            raise ValueError("unsupported attribute access")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in SAFE_BUILTINS and node.func.id != function:
                    raise ValueError("unsupported function call")
            elif not isinstance(node.func, ast.Attribute):
                raise ValueError("indirect function calls are forbidden")
    return tree


def _python_judge(case, program):
    tree = _python_tree(program, case["function"])
    namespace = {"__builtins__": SAFE_BUILTINS}
    exec(compile(tree, "<generated-function>", "exec"), namespace, namespace)
    function = namespace[case["function"]]
    for index, test in enumerate(case["tests"]):
        args = copy.deepcopy(test["args"])
        before = copy.deepcopy(args)
        actual = function(*args)
        if not _same(actual, test["expected"]):
            return {"passed": False, "reason": f"wrong result on edge case {index}", "tests_run": index + 1}
        if args != before:
            return {"passed": False, "reason": f"input mutated on edge case {index}", "tests_run": index + 1}
    return {"passed": True, "reason": "all function edge cases passed", "tests_run": len(case["tests"])}


def _sql_judge(case, query):
    if re.match(r"\s*(SELECT|WITH)\b", query, re.IGNORECASE) is None:
        raise ValueError("a SELECT query is required")
    conn = sqlite3.connect(":memory:")
    try:
        conn.enable_load_extension(False)
        conn.execute(case["schema"])
        conn.executemany("INSERT INTO orders VALUES (?,?,?,?)", case["data"])
        conn.execute("PRAGMA query_only=ON")
        conn.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, MAX_ANSWER_BYTES)
        conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 1_000_000)
        conn.setlimit(sqlite3.SQLITE_LIMIT_EXPR_DEPTH, 100)
        conn.setlimit(sqlite3.SQLITE_LIMIT_ATTACHED, 0)
        allowed_functions = {"sum", "count", "min", "max", "avg", "total", "abs",
                             "coalesce", "ifnull", "round", "lower", "upper", "length"}

        def authorize(action, arg1, arg2, _database, _source):
            if action == sqlite3.SQLITE_SELECT:
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_READ and arg1 == "orders":
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_FUNCTION and (arg2 or arg1 or "").lower() in allowed_functions:
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY

        conn.set_authorizer(authorize)
        deadline = time.monotonic() + 1.0
        steps = 0

        def progress():
            nonlocal steps
            steps += 1
            return int(steps > 1000 or time.monotonic() > deadline)

        conn.set_progress_handler(progress, 1000)
        cursor = conn.execute(query)
        actual = [list(row) for row in cursor.fetchmany(101)]
        if len(actual) > 100:
            raise ValueError("query returned too many rows")
        passed = len(cursor.description or ()) == 2 and _same(actual, case["expected"])
        return {"passed": passed, "reason": "SQL rows match" if passed else "SQL rows or ordering differ",
                "rows": len(actual)}
    finally:
        conn.close()


def _worker(kind):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024**2, 256 * 1024**2))
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (16, 16))
    payload = json.loads(sys.stdin.read(2_000_000))
    try:
        result = (_python_judge if kind == "python" else _sql_judge)(payload["case"], payload["program"])
    except BaseException as error:
        result = {"passed": False, "reason": f"{type(error).__name__}: {str(error)[:160]}"}
    print(json.dumps(result, allow_nan=False))


def judge(case: dict, text: str) -> dict:
    """Score a single completion independently of another model's answer."""
    kind = case.get("kind")
    try:
        if not isinstance(text, str) or len(text.encode()) > MAX_ANSWER_BYTES:
            raise ValueError("answer is not bounded text")
        program = _code(text)
        if kind == "json":
            actual = json.loads(program, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
            passed = _same(actual, case["expected"])
            return {"passed": passed, "reason": "JSON answer matches" if passed else "JSON value/type/fields differ"}
        if kind not in ("python", "sql"):
            raise ValueError("unknown case kind")
        if kind == "python":
            _python_tree(program, case["function"])
        proc = subprocess.run([sys.executable, "-I", "-S", "-B", str(Path(__file__).resolve()), "--worker", kind],
                              input=json.dumps({"case": case, "program": program}),
                              text=True, capture_output=True, timeout=WORKER_TIMEOUT_SECONDS,
                              cwd="/", env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"})
        if proc.returncode:
            return {"passed": False, "reason": f"bounded worker exited {proc.returncode}"}
        if len(proc.stdout) > 16_000:
            raise ValueError("unexpected worker output size")
        result = json.loads(proc.stdout)
        if type(result.get("passed")) is not bool:
            raise ValueError("invalid worker result")
        return result
    except subprocess.TimeoutExpired:
        return {"passed": False, "reason": "bounded worker timed out"}
    except (ValueError, TypeError, SyntaxError, KeyError, OSError, RecursionError) as error:
        return {"passed": False, "reason": f"{type(error).__name__}: {str(error)[:160]}"}


def self_test() -> dict:
    correct = {
        "dev_python_first_unique": "def first_unique(s):\n    for c in s:\n        if s.count(c) == 1:\n            return c\n    return None\n",
        "dev_python_merge_intervals": "def merge_intervals(intervals):\n    out = []\n    for a, b in sorted(intervals):\n        if out and a <= out[-1][1]:\n            out[-1][1] = max(out[-1][1], b)\n        else:\n            out.append([a, b])\n    return out\n",
        "dev_sql_paid_totals": "SELECT customer_id, SUM(total) AS paid_total FROM orders WHERE status='paid' GROUP BY customer_id HAVING COUNT(*) >= 2 AND SUM(total) >= 30 ORDER BY customer_id",
    }
    wrong = {
        "dev_json_pencils": '{"remaining":195}',
        "dev_json_duration": '{"seconds":5100}',
        "dev_json_sensor": '{"sensor":"B","corrected":21}',
        "dev_json_order": '{"names":["Di","Bo","Cy","Ava"]}',
        "dev_json_logic": '{"true_statement":"Q"}',
        "dev_python_first_unique": "def first_unique(s):\n    return s[0] if s else None\n",
        "dev_python_merge_intervals": "def merge_intervals(intervals):\n    return sorted(intervals)\n",
        "dev_sql_paid_totals": "SELECT customer_id, SUM(total) FROM orders GROUP BY customer_id HAVING SUM(total)>=30 ORDER BY customer_id",
    }
    records = []
    for case in cases():
        good = correct.get(case["id"], json.dumps(case.get("expected")))
        bad = wrong.get(case["id"], json.dumps({"wrong": True}))
        positive = judge(case, good)
        negative = judge(case, bad)
        records.append({"case_id": case["id"], "correct_control": positive,
                        "known_wrong_control": negative,
                        "invalid_control_rejected": negative["passed"] is False})
    python_case = next(c for c in cases() if c["kind"] == "python")
    sql_case = next(c for c in cases() if c["kind"] == "sql")
    restrictions = {
        "python_import": judge(python_case, "import os\ndef first_unique(s):\n    return None"),
        "python_module_code": judge(python_case, "x = 1\ndef first_unique(s):\n    return None"),
        "python_dunder": judge(python_case, "def first_unique(s):\n    return s.__class__"),
        "python_infinite_loop": judge(python_case, "def first_unique(s):\n    while True:\n        pass"),
        "sql_write": judge(sql_case, "DELETE FROM orders"),
        "sql_file_read": judge(sql_case, "SELECT readfile('/etc/passwd'), 1"),
        "sql_multi_statement": judge(sql_case, "SELECT customer_id,total FROM orders; DROP TABLE orders;"),
    }
    passed = (all(r["correct_control"]["passed"] and r["invalid_control_rejected"] for r in records)
              and all(not r["passed"] for r in restrictions.values()))
    return {"scope": "development only; no held-out, model or GPU evaluation", "passed": passed,
            "cases": records, "restriction_controls": restrictions,
            "invalid_control_rejected": all(r["invalid_control_rejected"] for r in records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--worker", choices=("python", "sql"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        _worker(args.worker)
        return 0
    if not args.self_test:
        parser.error("use --self-test, or import cases() and judge(case, text)")
    result = self_test()
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
