"""Finish resource qualification, then execute the full first matched recovery window."""
import json,subprocess,sys
from .common import *

def main():
 evidence=json.load(open(R/'native-remaining-zero/result.json'))
 if evidence['status']!='COMPLETE' or not all(x['correct'] for x in evidence['rows']):raise ValueError('component native admission incomplete')
 if not (R/'checkpoint-storage-qualification.json').exists():subprocess.run([sys.executable,'-m','scripts.component_campaign.checkpoint'],cwd=ROOT,check=True)
 subprocess.run([sys.executable,'-m','scripts.component_campaign.queue','H0','H4'],cwd=ROOT,check=True)

if __name__=='__main__':main()
