"""Compare complete paired greedy-output panels; never extrapolate learning curves.

Adapters must authenticate native execution separately. This layer authenticates
response bytes, exact case coverage, parity, timing and paired comparisons.
"""
import argparse
import math
from pathlib import Path
import json
import numpy as np
from runner import atomic, pin, read, require


def signature(value):
    return {k:value[k] for k in ('tokens','content','stop_type','stopping_word','truncated')}


def summarize(panel):
    require(panel['schema']=='contender_comparison_v1','unknown comparison schema')
    require(panel['tier'] in ('smoke','contender','confirmation'),'unknown tier')
    repeats=panel['repetitions'];cases=panel['cases'];arms=panel['arms'];baseline=panel['baseline']
    require(type(repeats) is int and repeats >= (1 if panel['tier']=='smoke' else 5),'insufficient repetitions')
    require(cases and len(set(cases))==len(cases),'duplicate/empty cases')
    require(arms and len(set(arms))==len(arms) and baseline in arms,'invalid arms')
    require(set(panel['model_pins'])==set(arms),'model pins do not cover arms')
    require(set(panel['runtime_pins'])==set(arms),'runtime pins do not cover arms')
    for ref in [panel['target_pin'],panel['workload_pin'],panel['admission_pin']]+list(panel['model_pins'].values())+list(panel['runtime_pins'].values()):
        require(pin(ref['path'])==ref,'pinned artifact changed')
    require(panel['semantics']=='greedy_exact_target','unsupported correctness semantics')
    if panel['tier']=='confirmation':
        require(panel['partition']=='confirmation' and panel['selection_pin'],'confirmation requires frozen selection')
        require(pin(panel['selection_pin']['path'])==panel['selection_pin'],'selection changed')
        # A reserved evaluation pool is consumed before requests start, including failed attempts.
        receipt=read(panel['consumption_receipt']['path'])
        require(pin(panel['consumption_receipt']['path'])==panel['consumption_receipt'],'consumption receipt changed')
        require(receipt['cases']==cases and receipt['selection_pin']==panel['selection_pin'],'confirmation reservation mismatch')
    tables={a:np.full((len(cases),repeats),np.nan) for a in arms}
    tokens=np.full((len(cases),repeats),np.nan);seen=set();refs={}
    for row in panel['records']:
        require(row['arm'] in arms and row['case'] in cases,'unexpected record')
        rep=row['repetition'];require(type(rep) is int and 0<=rep<repeats,'invalid repetition')
        key=(row['arm'],row['case'],rep);require(key not in seen,'duplicate record');seen.add(key)
        require(type(row['wall_seconds']) in (float,int) and math.isfinite(row['wall_seconds']) and row['wall_seconds']>0,'invalid wall time')
        for ref in (row['response'],row['reference'],row['request']):
            require(pin(ref['path'])==ref,'request/response changed')
        request=read(row['request']['path']);require(request['temperature']==0 and request['cache_prompt'] is False,'unsupported request semantics')
        refkey=row['case']
        binding=(request,signature(read(row['reference']['path'])))
        require(refkey not in refs or refs[refkey]==binding,'unmatched request/reference');refs[refkey]=binding
        response=read(row['response']['path']);reference=read(row['reference']['path'])
        require(signature(response)==signature(reference),'target parity failed')
        count=len(response['tokens']);require(count>0 and count==row['tokens'],'token count mismatch')
        i=cases.index(row['case']);prior=tokens[i,rep];require(np.isnan(prior) or prior==count,'token mismatch across arms')
        tokens[i,rep]=count;tables[row['arm']][i,rep]=row['wall_seconds']
    require(len(seen)==len(arms)*len(cases)*repeats,'incomplete panel')
    rng=np.random.default_rng(421104)
    pw=rng.multinomial(len(cases),[1/len(cases)]*len(cases),size=10000)
    rw=rng.multinomial(repeats,[1/repeats]*repeats,size=10000)
    results=[]
    for name in arms:
        wall=tables[name];base=tables[baseline]
        row=dict(arm=name,tokens=int(tokens.sum()),wall_seconds=float(wall.sum()),tokens_per_second=float(tokens.sum()/wall.sum()))
        if name!=baseline and panel['tier']!='smoke':
            samples=np.einsum('bi,ij,bj->b',pw,base,rw)/np.einsum('bi,ij,bj->b',pw,wall,rw)
            low,high=np.quantile(samples,[.025,.975]).tolist()
            row.update(speed_ratio=float(base.sum()/wall.sum()),interval95=[low,high],finding='faster_on_this_panel' if low>1 else 'slower_on_this_panel' if high<1 else 'unresolved')
        results.append(row)
    return dict(status='PAIRED_PANEL_PASS',tier=panel['tier'],results=results,scope='Exploratory unadjusted paired case/repetition bootstrap. No automatic promotion, permanent rejection or prediction of long-run training. Smoke has no rankings.')


def reserve_confirmation(ledger_path,receipt_path,cases,selection_pin):
    from runner import locked
    require(cases and len(set(cases))==len(cases),'invalid cases')
    require(pin(selection_pin['path'])==selection_pin,'selection pin changed')
    ledger_path=Path(ledger_path);receipt_path=Path(receipt_path)
    require(not receipt_path.exists(),'receipt already exists')
    with locked(str(ledger_path)+'.lock'):
        ledger=read(ledger_path) # Must already include all historical train/dev/confirmation consumption.
        require(ledger['schema']=='evaluation_consumption_v1','invalid consumption ledger')
        require(not set(cases)&set(ledger['consumed']),'evaluation cases already consumed')
        receipt=dict(cases=cases,selection_pin=selection_pin,previous_ledger=pin(ledger_path),purpose='confirmation',failed_attempts_also_consume=True)
        ledger['consumed']+=cases
        atomic(ledger_path,ledger) # Fail closed: a crash can consume cases without a receipt, never reverse.
        atomic(receipt_path,receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--panel',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();require(not args.out.exists(),'output exists');atomic(args.out,summarize(read(args.panel)))

if __name__=='__main__':main()
