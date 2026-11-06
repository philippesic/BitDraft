"""Generate new confirmation inputs only after candidate and evaluator freeze."""
from pathlib import Path
import copy
import datetime
import hashlib
import json
import random
import secrets
import sys

P = Path(__file__).resolve().parent
sys.path.insert(0, str(P.parents[0] / "upstream-consistency-20261101"))
import quality_judge as judge

freeze_path = P / "final-freeze.json"
freeze = json.loads(freeze_path.read_text())
assert freeze["engine_version"] == "0.32.0"
for ref in freeze["sources"]:
    assert hashlib.sha256(Path(ref["path"]).read_bytes()).hexdigest() == ref["sha256"]
out = P / "final-inputs"
out.mkdir(exist_ok=False)
seed = secrets.token_hex(16)
rng = random.Random(int(seed, 16))


def formatted(question):
    return ("<|im_start|>system\nYou are a helpful assistant. Answer directly and accurately.<|im_end|>\n"
            "<|im_start|>user\n" + question + "<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")


topics = [
    ("code", "Write a Python function that groups adjacent equal values in a list. Return value/count pairs and show tests for empty input and one group."),
    ("code", "Write a Python bounded least-recently-used cache using only standard library collections. Explain eviction and give a small test."),
    ("code", "Write a JavaScript function that limits how often an event handler runs and preserves the latest arguments. Explain the timer behavior."),
    ("code", "Write a SQLite query using a window function to select the latest event per account, breaking timestamp ties by event ID. Give sample rows."),
    ("code", "Write a Python parser for comma-separated key=value assignments. Reject duplicate keys and malformed fields. Include edge cases."),
    ("code", "Explain how to test a function that merges closed intervals without mutating its input. Provide a compact implementation and useful tests."),
    ("reasoning", "A warehouse starts with {a} cartons containing {b} parts each, ships {c} parts and receives {d} more. Calculate the balance and check it a second way."),
    ("reasoning", "A shuttle departs every {b} minutes between 06:00 and 09:00 inclusive. Explain how to count departures, including boundary cases."),
    ("reasoning", "A team completes {a} tasks in {b} days at a constant rate. Another team works at half that rate. Explain the combined rate and the time for {c} tasks."),
    ("reasoning", "There are {a} red counters, {b} blue counters and {d} green counters. Explain the probability of drawing two red counters without replacement."),
    ("reasoning", "A journey contains {a} kilometres at {b} kilometres per hour, followed by {d} kilometres at half that speed. Compute duration and average speed carefully."),
    ("reasoning", "An account grows by {b} percent and then shrinks by the same percentage. Use a starting balance of {c} to explain why the final balance differs."),
    ("prose", "Explain the difference between eventual consistency and strong consistency using a small reservation service example."),
    ("prose", "Give a practical plan for moving a small team from manual releases to automated deployments while retaining a rollback path."),
    ("prose", "Explain why measuring latency only among successful requests can give a misleading reliability picture."),
    ("prose", "Describe how a content-addressed cache can detect stale or corrupted results, including the limits of hashes."),
    ("prose", "Explain why a faster subroutine can fail to improve an application's total running time. Use a numerical example."),
    ("prose", "Write a concise incident-review template that separates observations, hypotheses, causes and corrective actions."),
    ("grounded", "From these operational notes, produce five concrete deployment checks and identify two missing facts. Notes: {notes}"),
    ("grounded", "Use the following fictional service notes to propose an ordered migration plan, distinguishing dependencies from optional steps. Notes: {notes}"),
    ("grounded", "Summarize these fictional project notes for a new engineer and explain the most likely operational bottleneck. Notes: {notes}"),
    ("grounded", "Create a test plan from these fictional requirements, including rollback and partial-failure cases. Notes: {notes}"),
    ("grounded", "Identify contradictions or ambiguities in these fictional operational notes and propose precise clarifications. Notes: {notes}"),
    ("grounded", "Write a short technical handoff from these fictional notes, preserving limits and distinguishing completed work from plans. Notes: {notes}"),
]
prompts = []
for i, (domain, template) in enumerate(topics):
    a, b, c, d = rng.randint(21, 67), rng.randint(7, 19), rng.randint(130, 390), rng.randint(11, 43)
    notes = " ".join(f"Service {rng.choice(['Juniper', 'Cedar', 'Maple', 'Willow'])}-{j:02d} retains {rng.randint(2, 9)} backups, "
                     f"runs validation before publication, and records work item {rng.randrange(100000, 999999)}. "
                     "A rollback needs the previous artifact; deployment is planned after the audit."
                     for j in range(4 + (i % 3) * 4))
    text = template.format(a=a, b=b, c=c, d=d, notes=notes)
    text += f"\nUse the reference label CONF-{rng.randrange(1000000, 9999999)} in your answer."
    prompt = formatted(text)
    prompts.append({"id": f"confirmation-{i:02d}", "domain": domain, "prompt": prompt,
                    "sha256": hashlib.sha256(prompt.encode()).hexdigest()})
assert len(prompts) == len({p["sha256"] for p in prompts}) == 24
development = json.loads((P.parent / "spec-cuda-fix-20261031/prompts.json").read_text())
assert not {p["sha256"] for p in prompts} & {p["sha256"] for p in development}


def merge_reference(intervals):
    result = []
    for a, b in sorted(intervals):
        if result and a <= result[-1][1]:
            result[-1][1] = max(result[-1][1], b)
        else:
            result.append([a, b])
    return result


cases = []
for version in range(3):
    block = copy.deepcopy(judge.cases())
    for case in block:
        case["id"] = f"final-{version}-" + case["id"].removeprefix("dev_")
    cartons, units = rng.randint(18, 41), rng.randint(11, 24)
    sold = rng.randint(30, cartons * units - 10)
    block[0].update(expected={"remaining": cartons * units - sold},
                    prompt=f"A store has {cartons} cartons with {units} bolts each and sells {sold} bolts. Return only JSON with the single integer field remaining.")
    hours, minutes, extra = rng.randint(1, 4), rng.randint(10, 55), rng.randint(21, 99)
    block[1].update(expected={"seconds": (hours * 60 + minutes + extra) * 60},
                    prompt=f"A recording lasts {hours} hours {minutes} minutes, followed by {extra} minutes of discussion. Return their combined duration as JSON with the single integer field seconds.")
    sensor, reading, bias = "S" + str(rng.randrange(100, 999)), rng.randint(15, 44), rng.randint(1, 6)
    block[2].update(expected={"sensor": sensor, "corrected": reading - bias},
                    prompt=f"Sensor {sensor} reports {reading} degrees and has a positive measurement bias of {bias} degrees that must be subtracted. Return only JSON with sensor set to {sensor} and corrected set to its corrected integer temperature.")
    names = ["N" + str(x) for x in rng.sample(range(1000, 9999), 4)]
    assert len(set(names)) == 4
    pairs = list(zip(names, [4, 2, 4, 1]))
    block[3].update(expected={"names": [n for n, s in sorted(pairs, key=lambda x: (x[1], x[0]))]},
                    prompt="Scores: " + ", ".join(f"{n}={s}" for n, s in pairs) + ". Sort by ascending score, breaking ties by ascending name. Return only JSON with names containing the ordered names.")
    symbols = rng.sample(list("ABCDEFGHJKLMNPQRSTUVWXYZ"), 3)
    block[4].update(expected={"true_statement": symbols[2]},
                    prompt=f"Exactly one Boolean statement among {symbols[0]}, {symbols[1]}, {symbols[2]} is true. {symbols[0]} is false. The implication 'if {symbols[1]} then {symbols[2]}' is true. Return only JSON with the single string field true_statement naming the true statement.")
    for index in (5, 6):
        name = block[index]["function"]
        new_name = name + f"_v{version}"
        block[index]["prompt"] = block[index]["prompt"].replace(name, new_name)
        block[index]["function"] = new_name
    for _ in range(6):
        text = "".join(rng.choice("abcdeαβ") for _ in range(rng.randrange(1, 20)))
        expected = next((x for x in text if text.count(x) == 1), None)
        block[5]["tests"].append({"args": [text], "expected": expected})
        intervals = []
        for __ in range(rng.randrange(1, 8)):
            a = rng.randint(-30, 30)
            intervals.append([a, a + rng.randrange(0, 7)])
        block[6]["tests"].append({"args": [intervals], "expected": merge_reference(intervals)})
    threshold = rng.randint(26, 57)
    block[7]["prompt"] = block[7]["prompt"].replace("at least 30", f"at least {threshold}")
    block[7]["threshold"] = threshold
    block[7]["data"] = [(i, customer, status, total + rng.randrange(0, 12)) for i, customer, status, total in block[7]["data"]]
    totals = {}
    for _, customer, status, amount in block[7]["data"]:
        if status == "paid":
            count, total = totals.get(customer, (0, 0))
            totals[customer] = (count + 1, total + amount)
    block[7]["expected"] = [[customer, total] for customer, (count, total) in sorted(totals.items()) if count >= 2 and total >= threshold]
    cases.extend(block)

correct = {
    "first_unique": "def FUNCTION(s):\n    for x in s:\n        if s.count(x) == 1:\n            return x\n    return None\n",
    "merge_intervals": "def FUNCTION(intervals):\n    out=[]\n    for a,b in sorted(intervals, key=lambda x:x[0]):\n        if out and a <= out[-1][1]:\n            out[-1][1]=max(out[-1][1],b)\n        else:\n            out.append([a,b])\n    return out\n",
}
controls = []
for case in cases:
    if case["kind"] == "json":
        good, bad = json.dumps(case["expected"]), "{}"
    elif case["kind"] == "python":
        key = "first_unique" if case["function"].startswith("first_unique") else "merge_intervals"
        good = correct[key].replace("FUNCTION", case["function"])
        bad = f"def {case['function']}(x):\n    return None\n"
    else:
        good = f"SELECT customer_id,SUM(total) FROM orders WHERE status='paid' GROUP BY customer_id HAVING COUNT(*)>=2 AND SUM(total)>={case['threshold']} ORDER BY customer_id"
        bad = "SELECT customer_id,total FROM orders"
    assert judge.judge(case, good)["passed"], case["id"]
    assert not judge.judge(case, bad)["passed"], case["id"]
    controls.append({"case_id": case["id"], "correct_control_passed": True, "known_wrong_control_rejected": True})
for name, content in [("requests.json", prompts), ("quality-cases.json", cases), ("quality-controls.json", controls)]:
    (out / name).write_text(json.dumps(content, indent=2) + "\n")
receipt = {"created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "seed": seed,
           "freeze_sha256": hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
           "performance_prompts": 24, "quality_tasks": 24, "quality_templates": 8,
           "quality_correct_controls_passed": 24, "quality_known_wrong_controls_rejected": 24,
           "development_prompt_hash_overlap": 0, "model_requests_executed_before_freeze": 0,
           "scope": "new project confirmation inputs generated after source/settings freeze; no claim of pretraining novelty",
           "files": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in out.glob("*.json")}}
(out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({k: v for k, v in receipt.items() if k not in ("seed", "files")}, indent=2))
