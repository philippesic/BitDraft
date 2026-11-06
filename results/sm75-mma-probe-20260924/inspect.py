import pathlib,subprocess,json,hashlib
r=pathlib.Path("results/sm75-mma-probe-20260924")
b=r/"sm75_mma_probe"
for kind,args in (("sass",["cuobjdump","--dump-sass",str(b)]),("ptx",["cuobjdump","--dump-ptx",str(b)])):
 p=subprocess.run(args,capture_output=True,text=True)
 (r/(kind+".txt")).write_text(p.stdout)
 (r/(kind+".stderr.txt")).write_text(p.stderr)
 print(kind,"exit",p.returncode,"bytes",len(p.stdout),"sha256",hashlib.sha256(p.stdout.encode()).hexdigest())
 print("\n".join(x for x in p.stdout.splitlines() if any(t in x.lower() for t in ("hmma","mma.sync","sm_75","code for")))[:4000])
 if p.returncode: raise SystemExit(p.returncode)
