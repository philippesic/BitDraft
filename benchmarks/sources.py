import ast
import json
import re
import tarfile
from pathlib import Path
from typing import Iterator, cast

from w1a1_eagle.contracts import Record, record, sequence, string
from w1a1_eagle.numerical.prompting import json_rows, parquet_rows
from w1a1_eagle.records import identity


def text_field(value: object) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("source text field must be a string")
    return value


def candidate(
    source: str,
    key: str,
    group: str,
    category: str,
    subtype: str,
    messages: list[Record],
    tools: list[Record] | None = None,
) -> Record:
    return {
        "id": source + "-" + identity({"key": key})[:24],
        "source": source,
        "source_key": key,
        "group": source + ":" + group,
        "category": category,
        "subtype": subtype,
        "messages": messages,
        "tools": tools or [],
        "prior_use": "unknown",
    }


def legacy(root: Path) -> Iterator[Record]:
    for path in sorted((*root.glob("train-*.jsonl"), *root.glob("dev-*.jsonl"))):
        if ".index." in path.name:
            continue
        index = {string(row["id"]): row for row in json_rows(path.with_suffix(".index.jsonl"))}
        for row in json_rows(path):
            key = string(row["id"])
            meta = index[key]
            source = string(meta["source_id"])
            subtype = string(meta["category"])
            category = "code" if source == "magicoder" else "reasoning"
            if source == "dolly":
                if subtype in {"summarization", "closed_qa", "information_extraction"}:
                    category = "documents"
                elif subtype in {"creative_writing", "brainstorming"}:
                    category = "writing"
                else:
                    continue
            result = candidate(
                "legacy-" + source,
                key,
                string(meta.get("topic") or meta["group"]),
                category,
                subtype,
                [record(v) for v in sequence(row["messages"])],
            )
            result["prior_use"] = (
                "historical_train" if path.name.startswith("train-") else "consumed_dev"
            )
            result["inherited_role"] = "train" if path.name.startswith("train-") else "dev"
            result["aliases"] = [
                key,
                meta["group"],
                *([meta["topic"]] if meta.get("topic") else []),
            ]
            yield result


def ultrachat(path: Path) -> Iterator[Record]:
    for row in parquet_rows(path):
        messages = [record(v) for v in sequence(row["messages"])]
        users = [i for i, message in enumerate(messages) if message["role"] == "user"]
        if not users:
            continue
        key = string(row["prompt_id"])
        first = string(messages[0]["content"])
        intro = first.split("\n", 1)[0][:400]
        lead = r"^(?:(?:can|could|would) you |please |i (?:want|need) you to )?"
        if re.search(
            lead + r"(?:translate\b|provide (?:me with )?(?:a )?translation\b)",
            intro,
            re.IGNORECASE,
        ):
            yield candidate("ultrachat", key, key, "writing", "translation", messages[:1])
            continue
        elif re.search(
            lead
            + r"(?:create|develop|design|write|make|plan|outline|provide)\b.{0,120}\b(?:plan|schedule|itinerary|strategy)\b|\b(?:riddle|puzzle|logical reasoning|logic problem|scheduling problem|constraint satisfaction)\b",
            intro,
            re.IGNORECASE,
        ):
            yield candidate("ultrachat", key, key, "reasoning", "non_arithmetic", messages[:1])
            continue
        if re.search(
            lead + r"(?:write|rewrite|revise|edit|compose|rephrase|draft)\b",
            intro,
            re.IGNORECASE,
        ):
            if not re.search(
                r"\b(?:code|python|javascript|program|function|algorithm|sql|script)\b",
                intro,
                re.IGNORECASE,
            ):
                yield candidate("ultrachat", key, key, "writing", "writing", messages[:1])
                continue
        if len(users) > 1:
            yield candidate(
                "ultrachat", key, key, "conversation", "followup", messages[: users[-1] + 1]
            )


def glaive(path: Path) -> Iterator[Record]:
    decoder = json.JSONDecoder()
    rows = sequence(cast(object, json.loads(path.read_text())))
    for index, value in enumerate(rows):
        row = record(value)
        text = string(row["system"])
        start = text.find("{")
        if start < 0:
            continue
        tools: list[Record] = []
        try:
            while start < len(text):
                definition, end = cast(tuple[object, int], decoder.raw_decode(text[start:]))
                tools.append({"type": "function", "function": record(definition)})
                start += end
                remaining = text[start:]
                next_start = remaining.find("{")
                if next_start < 0:
                    break
                start += next_start
        except (ValueError, TypeError):
            continue
        parts = cast(
            list[str],
            re.split(r"(?:^|\n{2,})(USER|ASSISTANT|FUNCTION RESPONSE):\s*", string(row["chat"])),
        )
        messages: list[Record] = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Use the supplied functions when appropriate.",
            }
        ]
        prefixes: dict[str, list[list[Record]]] = {
            "tool_request": [],
            "tool_result": [],
            "tool_abstention": [],
        }
        valid = True
        calls = 0
        for position in range(1, len(parts), 2):
            role = parts[position]
            content = parts[position + 1].replace("<|endoftext|>", "").strip()
            if not content:
                continue
            if role == "USER":
                messages.append({"role": "user", "content": content})
            elif role == "FUNCTION RESPONSE":
                if not calls:
                    valid = False
                    break
                messages.append(
                    {"role": "tool", "tool_call_id": "call_" + str(calls), "content": content}
                )
                prefixes["tool_result"].append(list(messages))
            elif content.startswith("<functioncall>"):
                try:
                    function = record(
                        cast(object, ast.literal_eval(content[len("<functioncall>") :].strip()))
                    )
                    arguments = function["arguments"]
                    parsed = (
                        record(cast(object, json.loads(arguments)))
                        if isinstance(arguments, str)
                        else record(arguments)
                    )
                    if (
                        function["name"] not in {record(tool["function"])["name"] for tool in tools}
                        or messages[-1]["role"] != "user"
                    ):
                        valid = False
                        break
                    prefixes["tool_request"].append(list(messages))
                    calls += 1
                    messages.append(
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_" + str(calls),
                                    "type": "function",
                                    "function": {
                                        "name": string(function["name"]),
                                        "arguments": parsed,
                                    },
                                }
                            ],
                        }
                    )
                except (ValueError, SyntaxError, KeyError, TypeError):
                    valid = False
                    break
            else:
                if messages[-1]["role"] == "user":
                    prefixes["tool_abstention"].append(list(messages))
                messages.append({"role": "assistant", "content": content})
        available = [key for key, values in prefixes.items() if values]
        if valid and available and tools:
            bucket = int(identity({"group": index, "seed": 8101})[:8], 16) % 10
            subtype = (
                "tool_request" if bucket < 7 else "tool_result" if bucket < 9 else "tool_abstention"
            )
            if not prefixes[subtype]:
                subtype = available[0]
            selected = prefixes[subtype][-1]
            yield candidate(
                "glaive",
                str(index),
                str(index),
                "structured",
                subtype,
                selected,
                tools,
            )


def qasper(path: Path) -> Iterator[Record]:
    yield from qasper_member(path, "qasper-train-v0.3.json", None)


def qasper_development(path: Path) -> Iterator[Record]:
    yield from qasper_member(path, "qasper-dev-v0.3.json", "dev")


def qasper_member(path: Path, member_name: str, inherited: str | None) -> Iterator[Record]:
    with tarfile.open(path, "r:gz") as archive:
        member = archive.getmember(member_name)
        source = archive.extractfile(member)
        if source is None:
            raise ValueError("missing QASPER training member")
        papers = record(cast(object, json.load(source)))
    for key, value in papers.items():
        paper = record(value)
        sections = [record(v) for v in sequence(paper["full_text"])]
        text = "\n\n".join(
            text_field(s["section_name"])
            + "\n"
            + "\n".join(text_field(p) for p in sequence(s["paragraphs"]))
            for s in sections
        )
        questions = sequence(paper["qas"])
        if not questions or not text:
            continue
        question = string(record(questions[0])["question"])
        title = string(paper["title"])
        content = (
            "Read the paper and answer the question using its evidence. State when the paper does not supply an answer.\n\n"
            + title
            + "\n\nAbstract:\n"
            + text_field(paper.get("abstract"))
            + "\n\n"
            + text
            + "\n\nQuestion: "
            + question
        )
        result = candidate(
            "qasper", key, key, "documents", "paper_qa", [{"role": "user", "content": content}]
        )
        if inherited is not None:
            result["inherited_role"] = inherited
        yield result


def commitpack(path: Path) -> Iterator[Record]:
    rows = list(json_rows(path))
    parents: dict[str, str] = {}

    def find(name: str) -> str:
        parents.setdefault(name, name)
        if parents[name] != name:
            parents[name] = find(parents[name])
        return parents[name]

    for row in rows:
        repositories = sorted(set(string(row["repos"]).split(",")))
        for name in repositories:
            left, right = find(repositories[0]), find(name)
            parents[max(left, right)] = min(left, right)
    for row in rows:
        old = text_field(row["old_contents"])
        if not old.strip():
            continue
        key = string(row["commit"])
        filename = string(row["old_file"])
        repositories = string(row["repos"]).split(",")
        group = find(repositories[0])
        content = (
            "Edit the existing Python file to implement the following change. Return the updated code.\n\nChange:\n"
            + string(row["message"])
            + "\n\nFile: "
            + filename
            + "\n```python\n"
            + old
            + "\n```"
        )
        result = candidate(
            "commitpack",
            key + ":" + filename,
            group,
            "code",
            "editing",
            [{"role": "user", "content": content}],
        )
        result["source_license"] = row["license"]
        yield result
