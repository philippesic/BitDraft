import hashlib

from w1a1_eagle.contracts import Record, sequence, string
from w1a1_eagle.records import canonical


def source_names(row: Record, rendered: str | None = None) -> set[str]:
    names = {
        string(row["id"]),
        string(row["source_key"]),
        string(row["group"]),
        *(string(v) for v in sequence(row.get("aliases", []))),
    }
    if rendered is not None:
        names.update(
            (
                hashlib.sha256(rendered.encode()).hexdigest(),
                hashlib.sha256(" ".join(rendered.split()).encode()).hexdigest(),
                hashlib.sha256(canonical(row["messages"])).hexdigest(),
            )
        )
    return names
