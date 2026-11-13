import re

from w1a1_eagle.contracts import Record, integer, number, record, sequence, string


def signature(response: Record) -> Record:
    return {
        key: response[key]
        for key in ("tokens", "content", "stop_type", "stopping_word", "truncated")
    }


def valid_sample(response: Record, cap: int) -> bool:
    tokens = sequence(response["tokens"])
    return (
        bool(tokens)
        and len(tokens) <= cap
        and all(type(token) is int and integer(token) >= 0 for token in tokens)
        and response.get("truncated") is False
        and response.get("stop_type") in {"eos", "limit"}
        and (response["stop_type"] != "limit" or len(tokens) == cap)
    )


def verify_generation(response: Record, payload: Record, speculative: bool) -> None:
    settings = record(response["generation_settings"])
    for key in ("seed", "top_k", "n_predict"):
        if integer(settings[key]) != integer(payload[key]):
            raise ValueError("native generation setting differs from request: " + key)
    for key in ("temperature", "top_p", "min_p"):
        if abs(number(settings[key]) - number(payload[key])) > 1e-6:
            raise ValueError("native sampling setting differs from request: " + key)
    if settings.get("ignore_eos") is not False or settings.get("grammar") != "":
        raise ValueError(
            "native stopping/grammar settings differ from the qualified request policy"
        )
    if any(
        number(settings[key]) != expected
        for key, expected in (
            ("repeat_penalty", 1.0),
            ("frequency_penalty", 0.0),
            ("presence_penalty", 0.0),
            ("dynatemp_range", 0.0),
        )
    ):
        raise ValueError("unrequested native penalty or dynamic temperature")
    modes = string(settings["speculative.types"])
    if ("draft-dflash" in modes) != speculative:
        raise ValueError("actual native speculative mode differs from the selected arm")


def native_counts(response: Record, speculative: bool) -> Record:
    timings = record(response["timings"])
    if not speculative:
        return {"proposed": 0, "accepted": 0, "acceptance": None}
    if ("draft_n" in timings) != ("draft_n_accepted" in timings):
        raise ValueError("incomplete native draft counters")
    proposed, accepted = (
        integer(timings.get("draft_n", 0)),
        integer(timings.get("draft_n_accepted", 0)),
    )
    if not 0 <= accepted <= proposed:
        raise ValueError("native draft counters inconsistent")
    return {
        "proposed": proposed,
        "accepted": accepted,
        "acceptance": accepted / proposed if proposed else None,
    }


def totals(rows: list[Record]) -> Record:
    if not rows:
        raise ValueError("empty comparison")
    seconds = sum(number(row["wall_seconds"]) for row in rows)
    tokens = sum(integer(row["tokens"]) for row in rows)
    proposed = sum(integer(record(row["native"])["proposed"]) for row in rows)
    accepted = sum(integer(record(row["native"])["accepted"]) for row in rows)
    if seconds <= 0:
        raise ValueError("nonpositive measured request time")
    return {
        "requests": len(rows),
        "tokens": tokens,
        "wall_seconds": seconds,
        "tokens_per_second": tokens / seconds,
        "proposed": proposed,
        "accepted": accepted,
        "acceptance": accepted / proposed if proposed else None,
        "correct": all(row["correct"] is True for row in rows),
        "cap_hits": sum(row.get("stop_type") == "limit" for row in rows),
        "eos_stops": sum(row.get("stop_type") == "eos" for row in rows),
        **rounds(len(rows), tokens, proposed, accepted),
        **server_totals(rows),
    }


def server_seconds(responses: list[Record]) -> Record:
    timings = [record(response["timings"]) for response in responses]
    if any("prompt_ms" not in value or "predicted_ms" not in value for value in timings):
        return {}
    return {
        "server_prompt_seconds": sum(number(value["prompt_ms"]) for value in timings) / 1000,
        "server_prediction_seconds": sum(number(value["predicted_ms"]) for value in timings) / 1000,
    }


def server_totals(rows: list[Record]) -> Record:
    complete = all("server_prompt_seconds" in row for row in rows)
    return {
        name: sum(number(row[name]) for row in rows) if complete else None
        for name in ("server_prompt_seconds", "server_prediction_seconds")
    }


def rounds(requests: int, tokens: int, proposed: int, accepted: int) -> Record:
    derived = tokens - accepted - requests
    if not proposed or derived <= 0:
        return {
            "derived_rounds": None,
            "proposed_per_round": None,
            "accepted_per_round": None,
            "emitted_per_round": None,
        }
    return {
        "derived_rounds": derived,
        "proposed_per_round": proposed / derived,
        "accepted_per_round": accepted / derived,
        "emitted_per_round": (accepted + derived) / derived,
    }


def relative_speed(summaries: Record) -> Record:
    references = {
        name: number(record(summaries[name])["tokens_per_second"])
        for name in ("target", "baseline")
        if name in summaries
    }
    return {
        arm: {
            **record(value),
            **{
                "speed_vs_" + name: number(record(value)["tokens_per_second"]) / speed
                for name, speed in references.items()
            },
        }
        for arm, value in summaries.items()
    }


def reconcile(log: str, responses: list[Record]) -> None:
    pattern = r"#calls\(b,g,a\) =\s*(\d+)\s+(\d+)\s+(\d+), #gen drafts =\s*(\d+), #acc drafts =\s*(\d+), #gen tokens =\s*(\d+), #acc tokens =\s*(\d+)"
    matches = list(re.finditer(pattern, log))
    if len(matches) != len(responses):
        raise ValueError("native final log does not reconcile with request count")
    last_proposed, last_accepted = 0, 0
    for index, (match, response) in enumerate(zip(matches, responses, strict=True), 1):
        if int(match[1]) != index:
            raise ValueError("native request boundary is ambiguous")
        proposed, accepted = int(match[6]), int(match[7])
        timings = native_counts(response, True)
        if (proposed - last_proposed, accepted - last_accepted) != (
            timings["proposed"],
            timings["accepted"],
        ):
            raise ValueError("native response and independent final log disagree")
        last_proposed, last_accepted = proposed, accepted
