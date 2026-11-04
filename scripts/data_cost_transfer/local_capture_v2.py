"""Explicit WSL capture destination adapter; unchanged native producer and byte caps."""
from pathlib import Path
import sys
import bank
import native_log_text
native_log_text.install()
ROOT=bank.P/'runs/data-cost-transfer-20261104-01/captures'
def destination(path):
 path=Path(path).resolve();bank.require(path.is_relative_to(ROOT.resolve()) and path!=ROOT.resolve() and not path.exists(),'unique campaign WSL capture directory required')
bank.capture.check_destination=destination
if __name__=='__main__':
 if '--prior' in sys.argv:
  import resume_capture_v2 as resume_capture
  resume_capture.main()
 else:
  import argparse
  p=argparse.ArgumentParser();p.add_argument('--roster',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--allow-cuda',action='store_true');a=p.parse_args();bank.require(a.allow_cuda,'explicit CUDA required');bank.collect(a.roster,a.out)
