from pathlib import Path
import json,hashlib,difflib
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/design-readiness-20261104-01';source=P/'scripts/design_readiness/fixed_objective.py';s=source.read_text()
s=s.replace('                loss = 0.\n',"                loss = 0.\n                components = {}\n")
s=s.replace('                        loss = loss + ce\n',"                        loss = loss + ce\n                        components['ce'] = components.get('ce', 0.) + ce\n")
s=s.replace('                    loss = loss + 10.*replay\n',"                    loss = loss + 10.*replay\n                    components['old_replay' if 'old_replay' not in components else 'added_replay'] = 10.*replay\n")
needle='                    loss.backward()\n'
replacement='''                    if (cursor-5760) % 31 == 0:
                        gs = {name: torch.cat([g.detach().reshape(-1) for g in torch.autograd.grad(value, params, retain_graph=True)]) for name, value in components.items()}
                        total = sum(gs.values())
                        tn = total.double().norm()
                        diagnostic = dict(cursor=cursor, values={k: float(v.detach()) for k,v in components.items()}, total_norm=float(tn), components={})
                        for name,g in gs.items():
                            gn=g.double().norm()
                            diagnostic['components'][name]=dict(norm=float(gn), cosine_to_total=float(torch.dot(g.double(), total.double())/(gn*tn)) if gn>0 and tn>0 else None)
                    loss.backward()
                    if (cursor-5760) % 31 == 0:
                        actual = torch.cat([p.grad.detach().reshape(-1) for p in params])
                        diagnostic['actual_total_norm'] = float(actual.double().norm())
                        diagnostic['component_sum_relative_error'] = float((actual-total).double().norm()/actual.double().norm().clamp_min(1e-30))
                        with (a.out/'gradient-diagnostics.jsonl').open('a') as diagnostic_stream:
                            diagnostic_stream.write(json.dumps(diagnostic)+'\\n')
'''
assert s.count(needle)==1;s=s.replace(needle,replacement)
p=R/'gradient_objective.py';p.write_text(s);(R/'gradient-observer.patch').write_text(''.join(difflib.unified_diff(source.read_text().splitlines(True),s.splitlines(True),fromfile=str(source),tofile=str(p))))
plan=json.loads((R/'gradient-diagnostic-plan.json').read_text());plan['derived_source']=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest());plan['original_source']=dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest());(R/'gradient-diagnostic-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
