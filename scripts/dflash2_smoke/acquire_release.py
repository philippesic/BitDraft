"""Bounded acquisition of the authenticated compatible release, without remote code."""
import hashlib
import json
import shutil
import urllib.request
from scripts.component_campaign.common import pin, save
from scripts.dflash2_smoke.train import R

REPO='nchapman/Qwen3.5-4B-DFlash2'
REVISION='8d8bb62873a4fe6165e644163e9ff76e06ab78ba'
WEIGHT_SHA='c919a019f84c8be86fcfc24a3e694dd2e90f4f616ac6aa103f27f5308b5f5e6c'
WEIGHT_BYTES=1_324_107_408


def main():
    audit=R/'release-audit';audit.mkdir(parents=True,exist_ok=True)
    for name in ('config.json','README.md'):
        path=audit/name
        if not path.exists():
            path.write_bytes(urllib.request.urlopen(f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}').read())
    config=json.load(open(audit/'config.json'))
    assert config['architectures']==['DFlash2DraftModel']
    assert config['transformer_layer_config']['hidden_size']==2560
    assert config['draft_vocab_size']==248320 and config['selector_top_k']==16
    assert config['speculators_config']['verifier']['architectures']==['Qwen3_5ForConditionalGeneration']
    path=R/'models/dflash2-4b/model.safetensors';path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        if shutil.disk_usage(path.parent).free < WEIGHT_BYTES+(1<<30):
            raise RuntimeError('Release acquisition needs weight bytes plus1GiB guest headroom')
        partial=path.with_suffix('.partial')
        digest=hashlib.sha256();size=0
        with urllib.request.urlopen(f'https://huggingface.co/{REPO}/resolve/{REVISION}/model.safetensors') as source,partial.open('xb') as out:
            while block:=source.read(8<<20):
                size+=len(block)
                if size>WEIGHT_BYTES:raise ValueError('Release exceeds frozen download bound')
                out.write(block);digest.update(block)
        if size!=WEIGHT_BYTES or digest.hexdigest()!=WEIGHT_SHA:
            raise ValueError('Downloaded release identity mismatch; partial preserved')
        partial.rename(path)
    actual=pin(path)
    assert actual['sha256']==WEIGHT_SHA and actual['bytes']==WEIGHT_BYTES
    save(audit/'acquisition.json',dict(repo=REPO,revision=REVISION,model=actual,
        config=pin(audit/'config.json'),script=pin(__file__),trust_remote_code=False))
    print('AUTHENTICATED_RELEASE_READY',flush=True)


if __name__=='__main__':main()
