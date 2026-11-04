"""Finish an interrupted chunk in a new directory, retaining authenticated paid groups."""
from pathlib import Path
import argparse,inspect,sys
import bank
capture=bank.capture

def main():
 p=argparse.ArgumentParser();p.add_argument('--prior',type=Path,required=True);p.add_argument('--roster',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--allow-cuda',action='store_true');a=p.parse_args();bank.require(a.allow_cuda,'explicit CUDA required')
 prior_receipt=a.prior/('failure.json' if (a.prior/'failure.json').exists() else 'summary.json')
 failure=bank.read(prior_receipt);oldfreeze=bank.read(a.prior/'freeze.json');roster=bank.read(a.roster)
 bank.require(bank.pin(oldfreeze['roster']['path'])==oldfreeze['roster'] and bank.read(oldfreeze['roster']['path'])['rows']==roster['rows'],'original roster changed')
 for path in [a.prior/'target-only/release.json'] + [p for p in (a.prior/'final-release.json', a.prior/'interrupted-release.json') if p.exists()]:
  value=bank.read(path);bank.require(value['owned_process_reaped'] and value['owned_process_group_absent'],'prior native group not released')
 paid={g['id']:g for g in failure['groups'] if g['status']=='ACTUAL_PARENT_TRACE_AND_PROTECTION_AUDIT_PASS'}
 for ident,g in paid.items():
  bank.require(bank.read(a.prior/ident/'result.json')==g,'paid group result changed')
  bank.require(bank.pin(g['protection']['path'])==g['protection'],'paid trace protection changed')
  release=bank.read(a.prior/ident/'release.json');bank.require(release['owned_process_reaped'] and release['owned_process_group_absent'],'paid group not released')
 bank.require(0<len(paid)<24,'partial paid chunk required')
 def reuse_targets(worker,groups,server,initial,candidate,environment):
  for ref in oldfreeze['input_pins']+oldfreeze['source_pins']:worker.checked(ref)
  (worker.output/'target-only').symlink_to((a.prior/'target-only').resolve(),target_is_directory=True)
  requests={}
  for row in groups:
   cell=a.prior/'target-only'/row['id'];request=bank.read(cell/'request.json');response=bank.read(cell/'response.json');tokenization=bank.read(cell/'tokenization.json')
   bank.require(request==dict(prompt=Path(row['rendered_text_path']).read_text(),**capture.protocol()),'old target request differs')
   bank.require(0<len(tokenization['tokens'])+80<=384 and response['timings']['cache_n']==0,'old target geometry differs');capture.response_signature(response)
   requests[row['id']]=request,response
  return requests
 def reuse_group(worker,row,destination):
  if row['id'] not in paid:return False
  result=paid[row['id']];bank.require(result['source_group']==row,'paid source row differs')
  destination.symlink_to((a.prior/row['id']).resolve(),target_is_directory=True)
  worker.groups.append(result);worker.costs.append(dict(stage='reused_'+row['id'],wall_seconds=0,prior_cost_seconds=result['wall_seconds'],status=result['status']));worker.progress();print('REUSED',row['id'],flush=True);return True
 source=inspect.getsource(capture.collect)
 needle='            cell = args.output / row["id"]\n            cell.mkdir()'
 bank.require(source.count(needle)==1,'capture loop changed')
 derived=source.replace(needle,'            cell = args.output / row["id"]\n            if reuse_group(worker, row, cell):\n                continue\n            cell.mkdir()')
 namespace=dict(vars(capture));namespace.update(target_references=reuse_targets,reuse_group=reuse_group)
 # Preserve the selector/validator indirection used by bank.collect.
 exec(compile(derived,'<paid-group-recovery>','exec'),namespace)
 def recovery(args):
  namespace.update(validate_roster=capture.validate_roster,select_rows=capture.select_rows)
  return namespace['collect'](args)
 capture.collect=recovery
 bank.collect(a.roster,a.out)
 bank.write(a.out/'recovery.json',dict(status='PAID_GROUPS_REUSED_AND_CHUNK_COMPLETED',source=bank.pin(__file__),prior_failure=bank.pin(prior_receipt),prior_freeze=bank.pin(a.prior/'freeze.json'),reused_groups=len(paid),new_groups=24-len(paid),original_outputs_unchanged=True))
if __name__=='__main__':main()
