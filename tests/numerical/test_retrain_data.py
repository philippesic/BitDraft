import json
import unittest
from typing import cast

from benchmarks.conventions import apply
from benchmarks.release_gate import object_types
from benchmarks.retrain_bank import matching_quota, normalized
from benchmarks.retrain_sources import chat_label, prompt_messages, tools_from_system
from w1a1_eagle.contracts import Record, record

TOOLS: list[Record] = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Find a word",
            "parameters": {"type": "object", "properties": {"word": {"type": "string"}}},
        },
    }
]
HISTORY: list[Record] = [
    {"role": "user", "content": "Define cat"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "1",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"word": "cat"}'},
            }
        ],
    },
    {"role": "tool", "content": "a small feline", "tool_call_id": "1"},
]


class RetrainDataTest(unittest.TestCase):
    def test_native_convention_keeps_tools(self) -> None:
        messages, tools = apply("native", HISTORY, TOOLS)
        assert messages == HISTORY and tools == TOOLS

    def test_json_convention_serializes_calls_into_text(self) -> None:
        messages, tools = apply("json", HISTORY, TOOLS)
        assert tools == []
        assert messages[0]["role"] == "system" and "lookup" in str(messages[0]["content"])
        call = str(messages[2]["content"])
        payload = cast(object, json.loads(call.split("\n")[1]))
        assert call.startswith("<tool_call>")
        assert payload == {"name": "lookup", "arguments": {"word": "cat"}}
        assert "<tool_response>" in str(messages[-1]["content"])

    def test_markdown_and_react_conventions_differ(self) -> None:
        markdown, _ = apply("markdown", HISTORY, TOOLS)
        react, _ = apply("react", HISTORY, TOOLS)
        assert "```json" in str(markdown[2]["content"])
        assert str(react[2]["content"]).startswith("Action: lookup")
        assert str(react[-1]["content"]).endswith("Observation: a small feline")

    def test_markdown_accepts_tool_without_description(self) -> None:
        bare: list[Record] = [{"type": "function", "function": {"name": "ping", "description": ""}}]
        messages, tools = apply("markdown", [{"role": "user", "content": "hi"}], bare)
        assert "`ping`: " in str(messages[0]["content"]) and tools == []

    def test_prompt_messages_stop_at_first_user_turn(self) -> None:
        rows: list[Record] = [
            {"role": "system", "content": ""},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        assert prompt_messages(rows) == [{"role": "user", "content": "hello"}]
        assert prompt_messages([{"role": "assistant", "content": "x"}]) is None

    def test_tools_from_system_reads_tagged_and_bare_lists(self) -> None:
        tagged = '<tools> {"name": "a", "parameters": {}} {"name": "b", "parameters": {}} </tools>'
        names = [str(record(t["function"])["name"]) for t in tools_from_system(tagged)]
        assert names == ["a", "b"]
        prose = 'Signatures go in <tools> </tools> tags.\n<tools>[{"name": "d"}]</tools>'
        assert [str(record(t["function"])["name"]) for t in tools_from_system(prose)] == ["d"]
        bare = 'Use: [{"type": "function", "function": {"name": "c"}}]'
        assert len(tools_from_system(bare)) == 1

    def test_bfcl_types_become_json_schema(self) -> None:
        converted = object_types(
            {"type": "dict", "properties": {"x": {"type": "float"}, "y": {"type": "tuple"}}}
        )
        assert converted == {
            "type": "object",
            "properties": {"x": {"type": "number"}, "y": {"type": "array"}},
        }

    def test_quota_matching_and_normalization(self) -> None:
        quotas = {"code/generation": 3, "reasoning/*": 2}
        assert matching_quota("code/generation", quotas) == "code/generation"
        assert matching_quota("reasoning/science", quotas) == "reasoning/*"
        assert matching_quota("writing/writing", quotas) is None
        assert normalized("Hello   World") == normalized("hello world")
        assert chat_label("x" * 7000) == ("documents", "pasted_document")
