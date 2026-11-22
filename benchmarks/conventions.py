import json
from typing import cast

from w1a1_eagle.contracts import Record, record, sequence, string

CONVENTIONS = ("native", "json", "markdown", "react")


def function_of(tool: Record) -> Record:
    return record(tool["function"]) if "function" in tool else tool


def arguments_of(call: Record) -> object:
    function = record(call["function"]) if "function" in call else call
    value = function.get("arguments", {})
    if isinstance(value, str):
        try:
            return cast(object, json.loads(value))
        except ValueError:
            return value
    return value


def name_of(call: Record) -> str:
    function = record(call["function"]) if "function" in call else call
    return string(function["name"])


def listing(tools: list[Record]) -> str:
    functions: list[Record] = [function_of(tool) for tool in tools]
    return json.dumps(functions, ensure_ascii=False, indent=1)


def system_text(convention: str, tools: list[Record]) -> str:
    if convention == "json":
        return (
            "You can call the following functions. To call one, reply with a JSON object "
            'inside <tool_call></tool_call> tags: {"name": NAME, "arguments": {...}}. '
            "Results come back inside <tool_response></tool_response>.\n\nFunctions:\n"
            + listing(tools)
        )
    if convention == "markdown":
        lines = []
        for tool in tools:
            function = function_of(tool)
            lines.append(
                "- `"
                + string(function["name"])
                + "`: "
                + string(function.get("description", ""))
                + "\n  Parameters: "
                + json.dumps(function.get("parameters", {}), ensure_ascii=False)
            )
        return (
            "## Tools\n\n"
            + "\n".join(lines)
            + '\n\nTo use a tool, write a fenced ```json block holding {"tool": NAME, "args": {...}} '
            "and stop. The result arrives in the next message."
        )
    if convention == "react":
        names = ", ".join(string(function_of(tool)["name"]) for tool in tools)
        return (
            "Answer the user's request. You have access to these tools:\n\n"
            + listing(tools)
            + "\n\nUse this format:\nThought: your reasoning\nAction: one of ["
            + names
            + "]\nAction Input: the JSON arguments\nObservation: the tool result\n"
            "(repeat Thought/Action/Action Input/Observation as needed)\n"
            "Thought: I know the final answer\nFinal Answer: the answer"
        )
    raise ValueError("unsupported tool-call convention")


def call_text(convention: str, call: Record) -> str:
    name, arguments = name_of(call), arguments_of(call)
    if convention == "json":
        data: Record = {"name": name, "arguments": arguments}
        payload = json.dumps(data, ensure_ascii=False)
        return "<tool_call>\n" + payload + "\n</tool_call>"
    if convention == "markdown":
        fenced: Record = {"tool": name, "args": arguments}
        payload = json.dumps(fenced, ensure_ascii=False)
        return "```json\n" + payload + "\n```"
    if convention == "react":
        return "Action: " + name + "\nAction Input: " + json.dumps(arguments, ensure_ascii=False)
    raise ValueError("unsupported tool-call convention")


def result_text(convention: str, content: str) -> str:
    if convention == "json":
        return "<tool_response>\n" + content + "\n</tool_response>"
    if convention == "markdown":
        return "Tool result:\n```\n" + content + "\n```"
    if convention == "react":
        return "Observation: " + content
    raise ValueError("unsupported tool-call convention")


def apply(
    convention: str, messages: list[Record], tools: list[Record]
) -> tuple[list[Record], list[Record]]:
    if convention == "native" or not tools:
        return messages, tools
    if convention not in CONVENTIONS:
        raise ValueError("unsupported tool-call convention")
    instruction = system_text(convention, tools)
    rendered: list[Record] = []
    system_seen = False
    for message in messages:
        role = string(message["role"])
        content = message.get("content")
        text = content if isinstance(content, str) else ""
        if role == "system":
            rendered.append(
                {"role": "system", "content": (text + "\n\n" if text else "") + instruction}
            )
            system_seen = True
        elif role == "assistant" and message.get("tool_calls"):
            calls = [
                call_text(convention, record(value)) for value in sequence(message["tool_calls"])
            ]
            body = (text + "\n" if text else "") + "\n".join(calls)
            rendered.append({"role": "assistant", "content": body})
        elif role == "tool":
            wrapped = result_text(convention, text)
            if rendered and rendered[-1]["role"] == "user":
                rendered[-1] = {
                    "role": "user",
                    "content": string(rendered[-1]["content"]) + "\n\n" + wrapped,
                }
            else:
                rendered.append({"role": "user", "content": wrapped})
        else:
            rendered.append({"role": role, "content": text})
    if not system_seen:
        rendered.insert(0, {"role": "system", "content": instruction})
    return rendered, []
