# Roster extension02 and immutability correction

Root requested immutable original roster/freeze after the extension had just been written in place. Original roster.json, freeze.json, prepare.py and report.md have been restored to the preceding schema/content; extension02 explicitly pins those restored originals. No capture, generation, labels, training, original messages, selected groups or token IDs changed.

Use roster02.json["rows"] or ["roster"] for aliases raw_messages/rendered_text_path and per-row tokenizer_pins (metadata/tokens/command). Exact native_prompt_tokens is present. Source/group ancestry is retained. The36groups remain24fit/12development, balanced8/4perdomain; rendered36–223tokens, at most303rows including80continuation, no replacements/truncation.512rawgroups had zero exact or normalized overlaps with44modern input artifacts representing5unique prior user inputs. Historical F16 TRAIN exposure is explicitly retained.

Window policy is frozen before labels: SHA256(seed+NUL+group_id+NUL+serving-window-v1), first64bits interpreted big-endian. Given L actual committed continuation tokens (capped80), complete17teacher-row blocks N=max(0,L-16). If N>0, select offset=integer_hash%N; teacher-logit positions are prompt_tokens-1+offset through +16. Small modulo bias is accepted. For L<17 retain the short output and account for insufficient complete blocks; never replace the group because of its outcome. Numeric or unsupported-label admission happens after sampling and retains denominators. Fixed server continuation capture does not require greedy81logit generation; build09 fixed replay can capture just these selected17positions.

Root owns GPU/capture. This worker has stopped after CPU input preparation. No processes remain.
