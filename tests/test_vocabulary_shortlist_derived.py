from pathlib import Path

from benchmarks.runtime_edits import SOURCE_EDITS

EDIT = SOURCE_EDITS["vocabulary_shortlist_derived_rows_v1"]
EDIT_NAMES = (
    "vocabulary_shortlist_v1",
    "vocabulary_shortlist_only_v1",
    "vocabulary_shortlist_derived_rows_v1",
)


def added(path: str) -> str:
    return "".join(new for where, _, new in EDIT if where == path)


def test_a_file_with_rows_keeps_the_v10_loader_lines_and_copy() -> None:
    loader = [(old, new) for where, old, new in EDIT if where == "src/models/dflash.cpp"]
    assert len(loader) == 1
    old, new = loader[0]
    assert old in new
    assert 'gguf_kv.find("dflash.vocab_shortlist.derived_type")' in new
    context = added("src/llama-context.cpp")
    assert (
        "    if (rows_source) {\n        ggml_backend_tensor_copy(rows_source, shortlist->rows);\n"
        in (context)
    )


def test_derived_rows_use_the_on_demand_conversion_and_log_time_and_hash() -> None:
    context = added("src/llama-context.cpp")
    assert (
        "vocab_shortlist_convert_row(convert_source, type, (llama_token) ids[i], "
        "shortlist->source_row, shortlist->values, derived.data() + (size_t) i * row_bytes);"
    ) in context
    assert "ggml_backend_tensor_set(shortlist->rows, derived.data(), 0, derived.size());" in context
    assert "in %.3f s, FNV-1a 64 %016" in context
    assert 'LLAMA_LOG_WARN("%s: derived %" PRId64' in context
    assert "std::thread" not in context
    assert "ggml_type dflash_shortlist_derived = GGML_TYPE_COUNT;" in added("src/llama-model.h")


def test_edit_applies_once_after_the_v10_edits_on_the_pinned_anchors(tmp_path: Path) -> None:
    import benchmarks.native as native

    texts: dict[str, str] = {}
    for path, old, _ in SOURCE_EDITS["vocabulary_shortlist_v1"]:
        texts[path] = texts.get(path, "head\n") + old + "middle\n"
    extra = {
        "src/models/dflash.cpp": [
            '    const auto component_tied_it = gguf_kv.find("dflash.component.tied");\n'
            "    const bool component_tied = component_tied_it != gguf_kv.end() && "
            'component_tied_it->second == "true";\n    if (component_tied) {\n',
            "    ggml_tensor * inpL = ggml_get_rows(ctx0, tok_embd, inp_tokens);\n",
        ],
        "src/llama-context.cpp": [
            '        LLAMA_LOG_ERROR("%s: embd row width %zu does not match the decoder input '
            '%zu\\n",\n                __func__, batch_inp.n_embd, batch_inp.n_embd_inp);\n'
            "        return -1;\n    }\n",
        ],
    }
    for path, blocks in extra.items():
        texts[path] = texts[path] + "".join(block + "tail\n" for block in blocks)
    for path, text in texts.items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(text)
    for name in EDIT_NAMES:
        native.apply_source_edits(tmp_path, name)
    for path, _, new in EDIT:
        assert (tmp_path / path).read_text().count(new) == 1
