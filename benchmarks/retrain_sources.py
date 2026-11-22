import gzip
import json
import re
from pathlib import Path
from typing import Callable, Iterator, cast

from benchmarks.sources import candidate, glaive, qasper, text_field, ultrachat_label
from w1a1_eagle.contracts import Record, record, sequence, string
from w1a1_eagle.numerical.prompting import parquet_rows
from w1a1_eagle.records import identity

ADMITTED_LICENSES = {"cc by 4.0", "cc-by-4.0", "odc-by", "odc by"}
COMPETITION_MATH = re.compile(r"\\boxed|olympiad|AIME|AMC\b|IMO\b", re.IGNORECASE)
MULTILINGUAL_PROMPT_CHARACTERS = 4000


def prompt_messages(messages: list[Record]) -> list[Record] | None:
    kept: list[Record] = []
    for message in messages:
        role = string(message["role"])
        content = message.get("content")
        if not isinstance(content, str):
            return None
        if role == "system":
            if content.strip():
                kept.append({"role": "system", "content": content})
            continue
        if role != "user":
            break
        if not content.strip():
            return None
        kept.append({"role": "user", "content": content})
        break
    return kept if kept and kept[-1]["role"] == "user" else None


def chat_label(text: str) -> tuple[str, str]:
    label = ultrachat_label(text)
    if label is not None:
        return label
    if len(text) > 6000:
        return "documents", "pasted_document"
    return "conversation", "open_request"


def nemotron(path: Path) -> Iterator[Record]:
    for row in parquet_rows(path):
        if string(row["license"]).lower() not in ADMITTED_LICENSES:
            continue
        messages = prompt_messages([record(v) for v in sequence(row["messages"])])
        if messages is None:
            continue
        text = string(messages[-1]["content"])
        split = string(row["category"])
        if split == "chat":
            category, subtype = chat_label(text)
        elif split == "code":
            category, subtype = "code", "generation"
        elif split == "stem":
            category, subtype = "reasoning", "science"
        elif split.startswith("multilingual"):
            if COMPETITION_MATH.search(text) or len(text) > MULTILINGUAL_PROMPT_CHARACTERS:
                continue
            category, subtype = "multilingual", split.removeprefix("multilingual_") or "mixed"
        else:
            continue
        key = string(row["uuid"])
        yield candidate("nemotron-" + split, key, key, category, subtype, messages)


def wildchat(path: Path) -> Iterator[Record]:
    for row in parquet_rows(path):
        if row.get("toxic") is True or row.get("redacted") is True:
            continue
        turns = [record(v) for v in sequence(row["conversation"])]
        if not turns or turns[0]["role"] != "user":
            continue
        first = text_field(turns[0]["content"]).strip()
        if not first or len(first) > 60000:
            continue
        language = text_field(row.get("language"))
        if language and language != "English":
            category, subtype = "multilingual", language.lower()
        else:
            category, subtype = chat_label(first)
        key = string(row["conversation_hash"])
        yield candidate(
            "wildchat", key, key, category, subtype, [{"role": "user", "content": first}]
        )


def oasst2_trees(path: Path) -> Iterator[Record]:
    with gzip.open(path, "rt") as source:
        for line in source:
            tree = record(cast(object, json.loads(line)))
            prompt = record(tree["prompt"])
            if prompt.get("deleted") is True or prompt.get("review_result") is False:
                continue
            text = text_field(prompt.get("text")).strip()
            if not text:
                continue
            language = text_field(prompt.get("lang"))
            if language == "en":
                category, subtype = chat_label(text)
            else:
                category, subtype = "multilingual", language
            key = string(tree["message_tree_id"])
            yield candidate(
                "oasst2", key, key, category, subtype, [{"role": "user", "content": text}]
            )


def aya(path: Path) -> Iterator[Record]:
    for row in parquet_rows(path):
        text = text_field(row.get("inputs")).strip()
        if not text or len(text) > MULTILINGUAL_PROMPT_CHARACTERS:
            continue
        language = text_field(row.get("language_code")) or text_field(row.get("language"))
        if language in {"eng", "en", "English"}:
            continue
        key = identity({"aya": text})
        yield candidate(
            "aya", key, key, "multilingual", language.lower(), [{"role": "user", "content": text}]
        )


def gsm8k(path: Path) -> Iterator[Record]:
    for row in parquet_rows(path):
        text = text_field(row["question"]).strip()
        key = identity({"gsm8k": text})
        yield candidate(
            "gsm8k", key, key, "reasoning", "arithmetic", [{"role": "user", "content": text}]
        )


def orca_math(path: Path) -> Iterator[Record]:
    for row in parquet_rows(path):
        text = text_field(row["question"]).strip()
        if not text:
            continue
        key = identity({"orca-math": text})
        yield candidate(
            "orca-math", key, key, "reasoning", "arithmetic", [{"role": "user", "content": text}]
        )


def oss_instruct(path: Path) -> Iterator[Record]:
    for row in parquet_rows(path):
        text = text_field(row["instruction"]).strip()
        if not text:
            continue
        key = identity({"oss-instruct": text})
        yield candidate(
            "oss-instruct", key, key, "code", "generation", [{"role": "user", "content": text}]
        )


PG19_TASKS = (
    (
        "documents",
        "summary",
        "Summarize the following excerpt from a book. Cover the main events, characters and themes.\n\n",
    ),
    (
        "documents",
        "question",
        "Read the following excerpt from a book and answer: who are the main characters, and how does each change over the excerpt?\n\n",
    ),
    (
        "structured",
        "extraction",
        'Read the following excerpt and return a JSON object with keys "characters" (list of names), "places" (list) and "events" (list of one-sentence events in order). Return only JSON.\n\n',
    ),
)
PG19_CHARACTERS = (6000, 16000, 32000, 48000)


def pg19(path: Path) -> Iterator[Record]:
    for row in parquet_rows(path):
        text = text_field(row["text"])
        book = text_field(row.get("short_book_title")) or identity({"pg19": text[:200]})
        for index, characters in enumerate(PG19_CHARACTERS):
            start = len(text) // 3
            excerpt = text[start : start + characters]
            if len(excerpt) < characters // 2:
                continue
            category, subtype, instruction = PG19_TASKS[
                int(identity({"book": book, "length": characters})[:4], 16) % len(PG19_TASKS)
            ]
            key = book + ":" + str(index)
            yield candidate(
                "pg19",
                key,
                book,
                category,
                subtype,
                [{"role": "user", "content": instruction + excerpt}],
            )
            break


SHAREGPT_ROLES = {"system": "system", "human": "user", "user": "user", "gpt": "assistant"}
TOOLS_BLOCK = re.compile(r"<tools>(.*?)</tools>", re.DOTALL)


def tools_from_system(text: str) -> list[Record]:
    match = TOOLS_BLOCK.search(text)
    body = cast(str, match.group(1)) if match else text[text.find("[") :]
    decoder = json.JSONDecoder()
    tools: list[Record] = []
    body = body.strip()
    try:
        value = cast(object, json.loads(body))
        items: list[object] = (
            list(cast(list[object], value)) if isinstance(value, list) else [value]
        )
    except ValueError:
        items = []
        position = 0
        while True:
            start = body.find("{", position)
            if start < 0:
                break
            try:
                item, end = cast(tuple[object, int], decoder.raw_decode(body[start:]))
            except ValueError:
                break
            items.append(item)
            position = start + end
    for item in items:
        if isinstance(item, dict):
            entry = record(item)
            function = record(entry["function"]) if "function" in entry else entry
            if "name" in function:
                tools.append({"type": "function", "function": function})
    return tools


def sharegpt_tools(source: str, rows: list[Record]) -> Iterator[Record]:
    for row in rows:
        turns = [record(v) for v in sequence(row.get("conversations", []))]
        system = text_field(row.get("system"))
        messages: list[Record] = []
        for turn in turns:
            role = SHAREGPT_ROLES.get(string(turn.get("from", "")))
            if role == "system":
                system = text_field(turn["value"])
            elif role == "user":
                messages.append({"role": "user", "content": text_field(turn["value"])})
                break
        if not messages or not system:
            continue
        tools = tools_from_system(system)
        if not tools:
            continue
        key = identity({source: messages, "tools": tools})
        yield candidate(source, key, key, "structured", "tool_request", messages, tools)


def json_list(path: Path) -> list[Record]:
    value = cast(object, json.loads(path.read_text()))
    return [record(v) for v in sequence(value)]


def hermes(path: Path) -> Iterator[Record]:
    yield from sharegpt_tools("hermes", json_list(path))


def toolace(path: Path) -> Iterator[Record]:
    yield from sharegpt_tools("toolace", json_list(path))


AGENT_CONTEXT_CHARACTERS = (8000, 70000)


def openhands_call(value: object) -> Record:
    call = record(value)
    function = record(call["function"])
    return {
        "id": string(call.get("id", "call")),
        "type": "function",
        "function": {
            "name": string(function["name"]),
            "arguments": function.get("arguments", "{}"),
        },
    }


def swe_rebench(path: Path) -> Iterator[Record]:
    tools = [
        record(v)
        for v in sequence(cast(object, json.loads((path.parent / "tools.json").read_text())))
    ]
    for row in parquet_rows(path):
        repository = string(row["repo"])
        steps = [record(v) for v in sequence(row["trajectory"])]
        messages: list[Record] = []
        cuts: list[int] = []
        size = 0
        for step in steps:
            role = string(step["role"])
            content = text_field(step.get("content"))
            if role == "assistant":
                if AGENT_CONTEXT_CHARACTERS[0] <= size <= AGENT_CONTEXT_CHARACTERS[1]:
                    cuts.append(len(messages))
                message: Record = {"role": "assistant", "content": content}
                calls = step.get("tool_calls")
                if calls:
                    message["tool_calls"] = [openhands_call(v) for v in sequence(calls)]
                messages.append(message)
            elif role == "tool":
                messages.append(
                    {
                        "role": "tool",
                        "content": content,
                        "tool_call_id": text_field(step.get("tool_call_id")),
                    }
                )
            elif role in {"system", "user"}:
                messages.append({"role": role, "content": content})
            size += len(content)
        if not cuts:
            continue
        instance = string(row["instance_id"])
        cut = cuts[int(identity({"cut": instance})[:4], 16) % len(cuts)]
        candidate_row = candidate(
            "swe-rebench", instance, instance, "agentic", "session_turn", messages[:cut], tools
        )
        candidate_row["aliases"] = ["repository:" + repository.lower()]
        yield candidate_row


ADAPTERS: dict[str, Callable[[Path], Iterator[Record]]] = {
    "nemotron": nemotron,
    "wildchat": wildchat,
    "oasst2-trees": oasst2_trees,
    "aya": aya,
    "gsm8k": gsm8k,
    "orca-math": orca_math,
    "oss-instruct": oss_instruct,
    "pg19": pg19,
    "glaive": glaive,
    "hermes": hermes,
    "toolace": toolace,
    "qasper": qasper,
    "swe-rebench": swe_rebench,
}
