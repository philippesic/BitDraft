from pathlib import Path

import pytest

from benchmarks.sandbox import Sandbox
from benchmarks.workflows import calls
from w1a1_eagle.contracts import record, sequence, string
from w1a1_eagle.records import read

SOLUTIONS = {
    "bugfix-boundary": 'def filter_events(events, start, end):\n    return [event for event in events if start <= event["time"] <= end]\n',
    "bugfix-quoted-csv": "import csv\ndef parse_row(line):\n    return next(csv.reader([line]))\n",
    "feature-grouping": 'def group_totals(rows):\n    result = {}\n    for row in rows:\n        key = row.get("team", "unknown")\n        result[key] = result.get(key, 0) + row["amount"]\n    return result\n',
    "feature-normalization": "def unique_names(names):\n    result = []\n    seen = set()\n    for name in names:\n        name = name.strip()\n        if name and name.lower() not in seen:\n            result.append(name)\n            seen.add(name.lower())\n    return result\n",
    "diagnosis-time-units": 'import json\nfrom pathlib import Path\ndef is_slow(milliseconds):\n    return milliseconds > json.loads(Path("config.json").read_text())["threshold_seconds"] * 1000\n',
    "diagnosis-pagination": 'def page_count(total, page_size):\n    if page_size <= 0:\n        raise ValueError("page_size")\n    return (total + page_size - 1) // page_size\n',
    "tool-workflow-config": 'import json\nfrom pathlib import Path\ndef route_for(path):\n    config = json.loads(Path("config.json").read_text())\n    return config["routes"].get(path, config["default_route"])\n',
    "tool-workflow-records": 'import json\nfrom pathlib import Path\ndef active_ids():\n    return [row["id"] for row in json.loads(Path("records.json").read_text()) if row["active"]]\n',
    "revision-sort-direction": 'def ordered(rows):\n    return sorted(rows, key=lambda row: row["priority"], reverse=True)\n',
    "revision-output-shape": 'def summarize(values):\n    return {"count": len(values), "total": sum(values)}\n',
    "docs-followup-timeout": "# Request client\n\ntimeout_seconds defaults to 30 seconds. Set 0 to disable the timeout.\n",
    "docs-followup-format": '# Record loader\n\nname is required; enabled defaults to true.\nJSON example: {"name":"demo","enabled":false}\n',
}


def test_live_fixtures_reject_original_accept_repair_and_reset(tmp_path: Path) -> None:
    definitions = read(Path("benchmarks/fixtures/live-dev.json"))
    for value in sequence(definitions["tasks"]):
        definition = record(value)
        name = string(definition["id"])
        sandbox = Sandbox(tmp_path / name, record(definition["fixture"]))
        assert sandbox.test()["passed"] is False
        path = "README.md" if name.startswith("docs-") else "service.py"
        sandbox.write(path, SOLUTIONS[name])
        assert sandbox.test()["passed"] is True
        assert sandbox.test()["passed"] is True
        with pytest.raises(ValueError, match="workspace"):
            sandbox.write("../checks/check.py", "")
        assert sandbox.invoke("list_files", {})


def test_native_tool_parser_preserves_code_and_rejects_duplicate_parameters() -> None:
    assert calls(
        "<tool_call>\n<function=write_file>\n<parameter=path>service.py</parameter>\n<parameter=content>def f(x):\n    return x < 3\n</parameter>\n</function>\n</tool_call>"
    ) == [("write_file", {"path": "service.py", "content": "def f(x):\n    return x < 3\n"})]
    assert calls('<tool_call>{"name":"run_tests","arguments":{}}</tool_call>') == [
        ("run_tests", {})
    ]
    with pytest.raises(ValueError, match="duplicate"):
        calls(
            "<tool_call><function=read_file><parameter=path>a</parameter><parameter=path>b</parameter></function></tool_call>"
        )
