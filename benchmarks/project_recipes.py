from w1a1_eagle.contracts import record, sequence, string


def replace(text: str, changes: object) -> str:
    for value in sequence(changes):
        change = record(value)
        old, new = string(change["old"]), change["new"]
        if not isinstance(new, str):
            raise ValueError("replacement must be text")
        if not old or text.count(old) != 1:
            raise ValueError("project fixture mutation must match exactly once")
        text = text.replace(old, new, 1)
    return text


def check_script(value: object) -> str:
    return (
        "import sys\nsys.path.insert(0, '/work')\n"
        + string(value)
        + "\nprint('ACCEPTANCE_CHECKS_PASSED')\n"
    )
