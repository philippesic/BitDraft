from pathlib import Path

import pytest

from benchmarks.batched import Prompt, answer_row, answered, divergence
from benchmarks.replay import banked, checked_metadata, located, request
from benchmarks.runtime_policy import arguments
from benchmarks.token_streams import replay_spans, token_stream, verify_tokenization
from w1a1_eagle.contracts import Record, record
from w1a1_eagle.data import Artifact, capture_supervision
from w1a1_eagle.records import canonical, identity, reference

PROMPT = [5, 6, 7, 8, 9, 10]
ANSWER = [11, 12, 13, 14, 15]
EOS = 19


def single_metadata() -> Record:
    tokens = PROMPT + ANSWER
    return {
        "taps": [2, 6, 10, 14, 18, 22, 26, 30],
        "feature_semantics": "zero_based_pre_block_input_all_tokens",
        "input_tokens": tokens,
        "prompt_tokens": len(PROMPT),
        "logit_positions": list(range(len(PROMPT) - 1, len(tokens))),
        "greedy_ids": ANSWER + [EOS],
        "generated_tokens": ANSWER,
        "eog_ids": [EOS],
        "stop_type": "eos",
        "output_cap": 16,
        "window_policy": "up-to-15",
        "target_sha256": "a" * 64,
    }


def source_row(tokens: list[int]) -> Record:
    return {
        "id": "row",
        "input_tokens": len(PROMPT),
        "token_sha256": identity({"tokens": tokens[: len(PROMPT)]}),
        "template_sha256": "d" * 64,
        "thinking": True,
    }


def write_corpus(root: Path, metadata: Record, source: Record, role: str = "train") -> Artifact:
    root.mkdir(parents=True, exist_ok=True)
    metadata_path = root / "training-metadata.json"
    metadata_path.write_bytes(canonical(metadata))
    source_path = root / "source.json"
    source_path.write_bytes(canonical(source))
    corpus = root / "corpus.json"
    corpus.write_bytes(
        canonical(
            {
                "schema": 1,
                "target_sha256": "a" * 64,
                "protected": ["heldout24", "secret-alias"],
                "expected_train": 1,
                "records": [
                    {
                        "id": "row",
                        "source": "fixture",
                        "group": "row-group",
                        "topic": "code",
                        "aliases": [],
                        "role": role,
                        "ancestry": "verified",
                        "prior_use": "unknown",
                        "metadata": reference(Artifact.capture(metadata_path)),
                        "files": [reference(Artifact.capture(source_path))],
                    }
                ],
            }
        )
    )
    return Artifact.capture(corpus)


def test_replay_spans_cover_single_and_session_records() -> None:
    metadata = single_metadata()
    tokens, spans = replay_spans(metadata)
    assert tokens == tuple(PROMPT + ANSWER)
    assert spans == ((len(PROMPT) - 1, len(tokens) - 1),)
    session = {
        **metadata,
        "supervised_spans": [[1, 2], [6, 8]],
        "logit_positions": [1, 2, 6, 7, 8],
        "greedy_ids": [7, 8, 12, 13, 14],
    }
    assert replay_spans(session)[1] == ((1, 2), (6, 8))
    with pytest.raises(ValueError, match="replay spans"):
        replay_spans({**metadata, "logit_positions": list(range(len(PROMPT), 11))})


def test_replay_refuses_retokenised_streams_and_records_identity(tmp_path: Path) -> None:
    metadata = single_metadata()
    tokens = tuple(PROMPT + ANSWER)
    verify_tokenization(tokens, source_row(list(tokens)))
    verify_tokenization(tokens, None)
    changed = (5, 6, 70, 8, 9, 10, *ANSWER)
    with pytest.raises(ValueError, match="never retokenises"):
        verify_tokenization(changed, source_row(list(tokens)))
    corpus = write_corpus(tmp_path / "plain", metadata, source_row(list(tokens)))
    streams, protected = located((corpus,), ("row",))
    assert "secret-alias" in protected
    item = streams["row"]
    assert item.tokens == tokens and item.prompt_tokens == len(PROMPT)
    sent = request(item, tmp_path / "out", "float16", 32)
    assert sent["tokens"] == list(tokens)
    assert sent["spans"] == [[len(PROMPT) - 1, len(tokens) - 1]]
    assert sent["replay"] == {"chunk": 32}
    helper: Record = {
        **metadata,
        "supervised_spans": [[len(PROMPT) - 1, len(tokens) - 1]],
        "replay": {"route": "token-replay-v1", "chunk": 32, "outputs": 512},
        "feature_dtype": "float16",
        "generated_tokens": [],
        "stop_type": "spans",
    }
    derived = checked_metadata(item, helper, "float16", 32, "a" * 64, {"path": "x"})
    assert record(derived["token_replay"])["tokens_sha256"] == identity({"tokens": list(tokens)})
    with pytest.raises(ValueError, match="differ from the request"):
        checked_metadata(
            item,
            {**helper, "input_tokens": list(changed)},
            "float16",
            32,
            "a" * 64,
            {"path": "x"},
        )
    with pytest.raises(ValueError, match="differ from the request"):
        checked_metadata(item, helper, "float16", 16, "a" * 64, {"path": "x"})
    retokenised = write_corpus(tmp_path / "retokenised", metadata, source_row(list(changed)))
    with pytest.raises(ValueError, match="never retokenises"):
        located((retokenised,), ("row",))
    dev = write_corpus(tmp_path / "dev", metadata, source_row(list(tokens)), "dev")
    with pytest.raises(ValueError, match="TRAIN"):
        located((dev,), ("row",))
    stream = token_stream(
        "row",
        tokens,
        ((len(PROMPT) - 1, len(tokens) - 1),),
        len(PROMPT),
        "a" * 64,
        source_row(list(tokens)),
        {"route": "test"},
    )
    assert stream["template_sha256"] == "d" * 64 and stream["thinking"] is True
    bank_record: Record = {
        "id": "row",
        "source": "fixture",
        "group": "row-group",
        "topic": "code",
        "aliases": [],
        "role": "train",
        "ancestry": "verified",
        "prior_use": "unknown",
        "window_policy": "up-to-15",
        "source_row": source_row(list(tokens)),
        "stream": stream,
    }
    bank = tmp_path / "streams.json"
    for value, accepted in ((stream, True), ({**stream, "tokens": list(changed)}, False)):
        bank.write_bytes(
            canonical(
                {
                    "schema": "token-stream-bank-v1",
                    "target_sha256": "a" * 64,
                    "protected": ["heldout24"],
                    "records": [{**bank_record, "stream": value}],
                }
            )
        )
        if accepted:
            found, _ = banked((Artifact.capture(bank),), ("row",))
            assert found["row"].tokens == tokens
        else:
            with pytest.raises(ValueError, match="identity mismatch"):
                banked((Artifact.capture(bank),), ("row",))


def test_replayed_supervision_matches_capture_and_ends_at_disagreement() -> None:
    metadata = single_metadata()
    original = capture_supervision("row", metadata)
    tokens = PROMPT + ANSWER
    replayed: Record = {
        **metadata,
        "supervised_spans": [[len(PROMPT) - 1, len(tokens) - 1]],
        "replay": {"route": "token-replay-v1", "chunk": 32, "outputs": 512},
        "generated_tokens": [],
    }
    assert capture_supervision("row", replayed) == original
    with pytest.raises(ValueError, match="inside the sequence"):
        capture_supervision(
            "row", {key: value for key, value in replayed.items() if key != "replay"}
        )
    near_tie = {**replayed, "greedy_ids": [11, 12, 99, 14, 15, EOS]}
    windows = {window.anchor: window.labels for window in capture_supervision("row", near_tie)}
    assert windows[len(PROMPT) - 1] == (11, 12, 99)
    assert windows[len(PROMPT) + 2] == (14, 15, EOS)


def test_batched_divergence_and_default_server_slots() -> None:
    assert divergence((1, 2, 3), (1, 2, 3)) is None
    assert divergence((1, 2, 4), (1, 2, 3)) == 2
    assert divergence((1, 2), (1, 2, 3)) == 2
    item = Prompt(
        {"id": "row"},
        Artifact(Path("/tmp/none"), "a" * 64, 1),
        tuple(PROMPT),
        tuple(ANSWER),
        True,
        16,
        None,
        "a" * 64,
        frozenset({EOS}),
        "up-to-15",
    )
    assert answered(item, {"tokens": ANSWER + [EOS], "stop_type": "eos"}) == (tuple(ANSWER), True)
    same = answer_row("row", item, {"tokens": ANSWER, "stop_type": "eos"}, 1.0)
    assert same["identical"] is True and same["share_before_divergence"] == 1.0
    other = answer_row("row", item, {"tokens": [11, 12, 30, 31], "stop_type": "eos"}, 1.0)
    assert other["first_divergence"] == 2 and other["share_before_divergence"] == 2 / 5
    plain = arguments(Path("server"), Path("target"), None, 4096, 3, 9000)
    assert plain == arguments(Path("server"), Path("target"), None, 4096, 3, 9000, 1)
    assert plain[plain.index("-np") + 1] == "1"
    parallel = arguments(Path("server"), Path("target"), None, 4096, 3, 9000, 16)
    assert parallel[parallel.index("-np") + 1] == "16"
