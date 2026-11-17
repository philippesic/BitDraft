from w1a1_eagle.contracts import Record, integer, sequence, string
from w1a1_eagle.records import identity

Spans = tuple[tuple[int, int], ...]


def replay_spans(metadata: Record) -> tuple[tuple[int, ...], Spans]:
    tokens = tuple(integer(value) for value in sequence(metadata["input_tokens"]))
    positions = tuple(integer(value) for value in sequence(metadata["logit_positions"]))
    spans: Spans
    if "supervised_spans" in metadata:
        spans = tuple(
            (integer(sequence(value)[0]), integer(sequence(value)[1]))
            for value in sequence(metadata["supervised_spans"])
        )
    else:
        prompt = integer(metadata["prompt_tokens"])
        if not 1 <= prompt <= len(tokens):
            raise ValueError("source record prompt length outside its token stream")
        spans = ((prompt - 1, len(tokens) - 1),)
    covered = tuple(position for first, last in spans for position in range(first, last + 1))
    if not tokens or positions != covered or covered[-1] >= len(tokens):
        raise ValueError("source record logit positions do not form its replay spans")
    return tokens, spans


def verify_tokenization(tokens: tuple[int, ...], source: Record | None) -> None:
    if source is None or "token_sha256" not in source:
        return
    count = integer(source["input_tokens"])
    if count > len(tokens) or identity({"tokens": list(tokens[:count])}) != string(
        source["token_sha256"]
    ):
        raise ValueError(
            "token stream differs from its recorded tokenization; replay never retokenises"
        )


def token_stream(
    name: str,
    tokens: tuple[int, ...],
    spans: Spans,
    prompt_tokens: int,
    target_sha256: str,
    source: Record | None,
    origin: Record,
) -> Record:
    return {
        "schema": "token-stream-v1",
        "id": name,
        "target_sha256": target_sha256,
        "template_sha256": None if source is None else source.get("template_sha256"),
        "thinking": None if source is None else source.get("thinking", False) is True,
        "tokens": list(tokens),
        "tokens_sha256": identity({"tokens": list(tokens)}),
        "prompt_tokens": prompt_tokens,
        "supervised_spans": [list(span) for span in spans],
        "origin": origin,
    }
