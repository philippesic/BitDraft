from pathlib import Path

from w1a1_eagle.contracts import Record

LIVE_TURNS = 16
LIVE_SECONDS = 900


def arguments(
    binary: Path, target: Path, drafter: Path | None, context: int, depth: int, port: int
) -> list[str]:
    argv = [
        str(binary),
        "-m",
        str(target),
        "-ngl",
        "99",
        "-c",
        str(context),
        "-np",
        "1",
        "-b",
        "512",
        "-ub",
        "512",
        "-t",
        "4",
        "-fa",
        "on",
        "--cache-type-k",
        "f16",
        "--cache-type-v",
        "f16",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--metrics",
        "--jinja",
        "--reasoning",
        "off",
        "--no-webui",
        "--no-warmup",
        "--spec-type",
        "draft-dflash" if drafter else "none",
        "-lv",
        "3",
    ]
    if drafter:
        argv += [
            "-md",
            str(drafter),
            "-ngld",
            "99",
            "--cache-type-k-draft",
            "f16",
            "--cache-type-v-draft",
            "f16",
            "--spec-draft-n-max",
            str(depth),
            "--spec-draft-n-min",
            "0",
            "--spec-draft-p-min",
            "0",
            "--spec-draft-sampling",
            "greedy",
        ]
    return argv


def payload(
    prompt: list[object],
    cap: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    seed: int,
) -> Record:
    return {
        "prompt": prompt,
        "n_predict": cap,
        "temperature": temperature,
        "seed": seed,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "cache_prompt": False,
        "return_tokens": True,
    }
