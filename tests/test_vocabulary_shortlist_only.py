from pathlib import Path

from benchmarks.runtime_edits import SOURCE_EDITS

EDIT = SOURCE_EDITS["vocabulary_shortlist_only_v1"]
BASE = SOURCE_EDITS["vocabulary_shortlist_v1"]


def kept_in_order(old: str, new: str) -> bool:
    remaining = iter(new.splitlines())
    return all(any(line == candidate for candidate in remaining) for line in old.splitlines())


def added(path: str) -> str:
    return "".join(new for where, _, new in EDIT if where == path)


def test_only_four_anchors_change_existing_lines_and_each_keeps_the_v9_path() -> None:
    changed = [(path, old, new) for path, old, new in EDIT if not kept_in_order(old, new)]
    assert [path for path, _, _ in changed] == [
        "src/models/dflash.cpp",
        "src/models/dflash.cpp",
        "src/models/dflash.cpp",
        "src/llama-context.cpp",
    ]
    tied, inputs, scoring, request = (new for _, _, new in changed)
    assert tied.endswith("    } else if (component_tied) {\n")
    assert 'gguf_kv.find("dflash.vocab_shortlist.rows_only")' in tied
    assert inputs.endswith("            : ggml_get_rows(ctx0, tok_embd, inp_tokens);\n")
    assert "input_rows && input_rows->input_map" in inputs
    assert scoring == (
        "    if (shortlist && (shortlist->input_map || (shortlist->source == output && "
        "!output_s)) && !model.d2t"
    )
    guarded, default = request.split("            } else {\n")
    assert guarded.startswith("            if (shortlist.input_map) {\n")
    v9_lines = [old for path, old, _ in changed if path == "src/llama-context.cpp"][0]
    assert default == "".join("    " + line + "\n" for line in v9_lines.splitlines()) + (
        "            }\n"
    )


def test_rows_only_state_exists_only_for_a_marked_drafter() -> None:
    context = added("src/llama-context.cpp")
    assert context.count("shortlist->input_map = ") == 1
    assert context.count("vocab_shortlist_init_rows(ids, n, rows_per_seq)") == 1
    assert "    if (model.dflash_shortlist_rows) {\n        return vocab_shortlist_init_rows" in (
        context
    )
    assert (
        "    if (model.dflash_shortlist_rows == nullptr) {\n        return true;\n    }" in context
    )
    assert "decodes only with its vocabulary shortlist attached" in context
    assert "if (!vocab_shortlist_inputs(batch_inp)) {" in context
    loader = added("src/models/dflash.cpp")
    assert loader.count("dflash_shortlist_rows = create_tensor(") == 1
    assert "if (shortlist_rows_it != gguf_kv.end()) {" in loader
    assert "struct ggml_tensor * dflash_shortlist_rows = nullptr;" in added("src/llama-model.h")
    assert "ggml_tensor * input_map = nullptr;" in added("src/llama-cparams.h")


def test_rows_are_converted_with_the_offline_quantizer_call_and_never_copied_whole() -> None:
    context = added("src/llama-context.cpp")
    assert (
        "ggml_get_type_traits(source->type)->to_float(encoded.data(), values.data(), columns);"
        in (context)
    )
    assert "ggml_quantize_chunk(type, values.data(), out, 0, 1, columns, nullptr)" in context
    assert "ggml_backend_tensor_copy(rows_source, shortlist->rows);" in context
    assert "other->tok_embd" in context
    assert "ggml_nbytes(convert_source)" not in context
    assert "std::vector<uint8_t> table(" not in context


def test_edit_applies_after_the_v9_shortlist_on_a_synthetic_source(tmp_path: Path) -> None:
    import benchmarks.native as native

    texts: dict[str, str] = {}
    for path, old, _ in BASE:
        texts[path] = texts.get(path, "head\n") + old + "middle\n"
    for path, text in texts.items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(text)
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
        target = tmp_path / path
        target.write_text(target.read_text() + "".join(block + "tail\n" for block in blocks))
    native.apply_source_edits(tmp_path, "vocabulary_shortlist_v1")
    native.apply_source_edits(tmp_path, "vocabulary_shortlist_only_v1")
    for path, _, new in EDIT:
        assert new in (tmp_path / path).read_text()
    for path, _, new in BASE:
        if path not in {where for where, _, _ in EDIT}:
            assert new in (tmp_path / path).read_text()
