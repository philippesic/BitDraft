from dataclasses import dataclass

from w1a1_eagle.contracts import Record, integer, record, sequence


@dataclass(frozen=True)
class Round:
    proposed: tuple[int, ...]
    accepted: int
    emitted: tuple[int, ...]
    target_tokens: tuple[int, ...]

    @classmethod
    def parse(cls, value: object) -> "Round":
        raw = record(value)
        if set(raw) != {"proposed", "accepted", "emitted", "target_tokens"}:
            raise ValueError("unexpected round fields")
        result = cls(
            tuple(integer(v) for v in sequence(raw["proposed"])),
            integer(raw["accepted"]),
            tuple(integer(v) for v in sequence(raw["emitted"])),
            tuple(integer(v) for v in sequence(raw["target_tokens"])),
        )
        if any(v < 0 for v in (*result.proposed, *result.emitted, *result.target_tokens)):
            raise ValueError("negative token identity")
        if not 0 <= result.accepted <= len(result.proposed):
            raise ValueError("accepted count exceeds proposal")
        if result.emitted != result.proposed[: result.accepted] + result.target_tokens:
            raise ValueError(
                "emitted tokens disagree with accepted prefix and target correction/bonus"
            )
        if len(result.target_tokens) > 1 or not result.emitted:
            raise ValueError(
                "linear round requires progress and at most one target correction/bonus"
            )
        return result


def reduce_rounds(values: list[object]) -> Record:
    rounds = [Round.parse(v) for v in values]
    if not rounds:
        raise ValueError("no completed rounds")
    proposed = sum(len(v.proposed) for v in rounds)
    accepted = sum(v.accepted for v in rounds)
    return {
        "rounds": len(rounds),
        "proposed": proposed,
        "accepted": accepted,
        "acceptance": accepted / proposed if proposed else None,
        "emitted": sum(len(v.emitted) for v in rounds),
        "target_tokens": sum(len(v.target_tokens) for v in rounds),
        "activation": "proposals_observed" if proposed else "fallback_only",
        "scope": "structural linear trace reconciliation; not independent verifier correctness",
    }
