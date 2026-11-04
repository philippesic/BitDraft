"""Create and verify a local drafter release bundle after explicit selection.

Includes the drafter, selected settings and evidence; target weights and native
runtime remain explicit pinned dependencies. No upload or publication occurs.
"""
import argparse
import json
from pathlib import Path
import shutil
from runner import atomic,pin,read,require


def verify(root):
    root=Path(root).resolve();manifest=read(root/'bundle.json')
    require(manifest['schema']=='contender_bundle_v1','invalid bundle schema')
    for name,expected in manifest['files'].items():
        path=(root/name).resolve();require(path.is_relative_to(root),'bundle path escapes root')
        actual=pin(path);require(actual['bytes']==expected['bytes'] and actual['sha256']==expected['sha256'],'bundle file changed: '+name)
    return manifest


def nested_pins(value):
    if isinstance(value,dict):
        if {'path','bytes','sha256'} <= value.keys():
            yield {k:value[k] for k in ('path','bytes','sha256')}
        else:
            for child in value.values():yield from nested_pins(child)
    elif isinstance(value,list):
        for child in value:yield from nested_pins(child)


def create(selection_path, output):
    selection=read(selection_path)
    require(selection['schema']=='contender_release_selection_v1','invalid release selection')
    require(selection['candidate'] and selection['supported_workloads'],'candidate/workload scope required')
    require(selection['evidence'] and selection['runtime_dependencies'],'deployment evidence/runtime required')
    dependencies=list(selection['runtime_dependencies'])+list(nested_pins(selection['settings']))
    dependencies=list({(v['path'],v['sha256']):v for v in dependencies}.values())
    refs=[selection['model'],selection['target']]+dependencies+list(selection['evidence'].values())
    for ref in refs:require(pin(ref['path'])==ref,'selected dependency changed')
    with Path(selection['model']['path']).open('rb') as stream:
        require(stream.read(4)==b'GGUF','drafter is not a GGUF artifact')
    require(selection['correctness']=='greedy_exact_target','unsupported release correctness contract')
    require(selection['claim'] in ('reference_rehearsal','evaluated_contender','confirmed_winner'),'invalid claim')
    if selection['claim']=='confirmed_winner':
        from comparison import summarize
        require('confirmation_panel' in selection['evidence'],'confirmation panel missing')
        panel=read(selection['evidence']['confirmation_panel']['path']);result=summarize(panel)
        require(panel['tier']=='confirmation','winner needs independent confirmation')
        require(panel['target_pin']==selection['target'],'confirmation target differs')
        policy=selection['baseline_policy'];require(pin(policy['path'])==policy,'baseline policy changed')
        frozen=read(policy['path']);require(frozen['primary_comparison_baseline']=='DFlash Q4_K_M','wrong baseline policy')
        matched=[r for r in frozen['rows'] if r['target_pin']==selection['target']]
        require(len(matched)==1 and panel['model_pins'][panel['baseline']]==matched[0]['draft_pin'],'confirmation baseline differs from frozen DFlash')
        require(panel['selection_pin']==pin(selection['frozen_candidate_selection']),'confirmation selection mismatch')
        frozen_choice=read(selection['frozen_candidate_selection'])
        require(frozen_choice['candidate']==selection['candidate'] and frozen_choice['model']==selection['model'],'frozen candidate differs')
        row=next(r for r in result['results'] if r['arm']==selection['candidate'])
        require(row.get('finding')=='faster_on_this_panel','winner not established on confirmation')
        require(panel['model_pins'][selection['candidate']]==selection['model'],'confirmed model differs')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(selection['model']['path'],output/'draft.gguf')
    shutil.copyfile(selection_path,output/'selection.json')
    copied=pin(output/'draft.gguf');require(copied['sha256']==selection['model']['sha256'] and copied['bytes']==selection['model']['bytes'],'model copy differs')
    files={name:{k:pin(output/name)[k] for k in ('bytes','sha256')} for name in ('draft.gguf','selection.json')}
    manifest=dict(schema='contender_bundle_v1',candidate=selection['candidate'],claim=selection['claim'],files=files,target=selection['target'],runtime_dependencies=dependencies,settings=selection['settings'],supported_workloads=selection['supported_workloads'],evidence=selection['evidence'],scope='Local drafter artifact/settings bundle. Target and runtime are pinned external dependencies; installation/runtime qualification remains required on another machine.')
    atomic(output/'bundle.json',manifest);verify(output)
    atomic(output/'result.json',dict(status='LOCAL_BUNDLE_VERIFIED',bundle=pin(output/'bundle.json')))
    return manifest


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--selection',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--verify',action='store_true');a=p.parse_args()
    if a.verify:verify(a.output)
    else:
        require(a.selection is not None,'selection required');create(a.selection,a.output)
if __name__=='__main__':main()
