"""Keep detached tmux work inside a mediated devicecheckout descendant tree.

A daemonized tmux server reparents outside the broker's registered gate. A private
foreground server stays attached to that gate while its detached session retains
the project's remote_job supervision and disconnect durability.
"""
from pathlib import Path
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time


def session_shell(command,receipt,socket,channel):
    # The callback writes only its own exit receipt; no credentials enter argv.
    code="import json,os,subprocess,sys;from pathlib import Path;p=Path(sys.argv[1]);r=subprocess.run(sys.argv[2:]);t=p.with_suffix('.tmp');t.write_text(json.dumps({'returncode':r.returncode}));os.replace(t,p);raise SystemExit(r.returncode)"
    invocation=shlex.join([sys.executable,'-B','-c',code,str(receipt),*command])
    return invocation+'; rc=$?; '+shlex.join(['tmux','-L',socket,'wait-for','-S',channel])+'; exit "$rc"'


def run(command,*,socket,session,receipt,cwd,startup_seconds=10):
    for value in (socket,session):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',value):raise ValueError('safe private socket/session names required')
    if not command:raise ValueError('supervised command required')
    receipt=Path(receipt);receipt.parent.mkdir(parents=True,exist_ok=True)
    if receipt.exists():raise ValueError('retain prior session receipt; choose a new invocation')
    if subprocess.run(['tmux','-L',socket,'show-options','-s'],capture_output=True).returncode==0:raise ValueError('private socket already owned; never attach to an existing daemon')
    server=subprocess.Popen(['tmux','-L',socket,'-D'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        until=time.monotonic()+startup_seconds
        while subprocess.run(['tmux','-L',socket,'show-options','-s'],capture_output=True).returncode:
            if server.poll() is not None or time.monotonic()>=until:raise RuntimeError('owned foreground tmux server failed startup')
            time.sleep(.05)
        channel=session+'-done'
        subprocess.run(['tmux','-L',socket,'new-session','-d','-s',session,'-c',str(cwd),session_shell(command,receipt,socket,channel)],check=True)
        subprocess.run(['tmux','-L',socket,'wait-for',channel],check=True)
        if not receipt.exists():raise RuntimeError('supervised command did not publish its exit receipt')
        return json.loads(receipt.read_text())['returncode']
    finally:
        subprocess.run(['tmux','-L',socket,'kill-server'],capture_output=True)
        server.wait(timeout=10)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--socket',required=True);p.add_argument('--session',required=True);p.add_argument('--receipt',type=Path,required=True);p.add_argument('--cwd',type=Path,required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    if sys.platform!='linux':p.error('registered Linux/WSL tmux supervisor required')
    runtime=Path.home()/'.local/share/p-stack/runtime.json'
    if runtime.exists():
        installed=json.loads(runtime.read_text());sys.path.insert(0,str(runtime.parent/'runtime'/installed['source_digest']))
        if installed.get('bridge'):os.environ.setdefault('P_STACK_BROKER_COMMAND',json.dumps(installed['bridge']))
    from devicecheckout.project import before_costly_work
    before_costly_work('benchmark_quiet')
    command=a.command[1:] if a.command[:1]==['--'] else a.command
    return run(command,socket=a.socket,session=a.session,receipt=a.receipt,cwd=a.cwd)

if __name__=='__main__':raise SystemExit(main())
