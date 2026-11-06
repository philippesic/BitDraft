"""Adapt released speculators metadata to the existing native DFlash2 converter."""
import json
import urllib.request
from scripts.component_campaign.common import P, ROOT, pin, save

R = P/'runs/dflash2-smoke-20261106-01'


def main():
    original = json.load(open(R/'release-audit/config.json'))
    assert original['architectures'] == ['DFlash2DraftModel']
    assert original['draft_vocab_size'] == original['transformer_layer_config']['vocab_size'] == 248320
    assert original['speculators_config']['verifier']['architectures'] == ['Qwen3_5ForConditionalGeneration']
    config = dict(original['transformer_layer_config'])
    config.update(architectures=['DFlash2DraftModel'], has_embed_tokens=False,
                  rope_is_neox_style=True, is_causal=False)
    config['dflash_config'] = {k: original[k] for k in
        ('block_size','conv_group_size','conv_kernel_size','mask_token_id','selector_rank','selector_top_k')}
    # Speculators taps index HF hidden_states (embedding at0); native metadata
    # stores that same1-based output index. The converter adds1 to layer IDs.
    config['dflash_config']['target_layer_ids'] = [i-1 for i in original['aux_hidden_state_layer_ids']]
    save(R/'models/dflash2-4b/config.json', config)
    target = R/'models/target-tokenizer'
    target.mkdir(exist_ok=True)
    api = json.load(urllib.request.urlopen('https://huggingface.co/api/models/Qwen/Qwen3.5-4B'))
    save(target/'api.json', api)
    for name in ('config.json','tokenizer.json','tokenizer_config.json'):
        data=urllib.request.urlopen('https://huggingface.co/Qwen/Qwen3.5-4B/resolve/'+api['sha']+'/'+name).read()
        (target/name).write_bytes(data)
    save(R/'release-audit/adaptation.json', dict(source=pin(R/'release-audit/config.json'),
        output=pin(R/'models/dflash2-4b/config.json'),script=pin(__file__),
        tokenizer_files=[pin(target/n) for n in ('config.json','tokenizer.json','tokenizer_config.json')],
        weights='All released tensor bytes unchanged; no selector weights invented; target supplies embeddings and output head.',
        converter=pin(ROOT/'third_party/llama.cpp/conversion/qwen.py'),
        taps=original['aux_hidden_state_layer_ids'],format='Speculators config flattened; DFlash2 conv/selector metadata nested for existing converter.'))


if __name__=='__main__': main()
