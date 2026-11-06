"""Reuse qualified capture implementation on predeclared historical exposure groups."""
from pathlib import Path
import hashlib,json
ROOT=Path('/home/philip/binary-eagle-decoding');RUN=ROOT/'runs/training-transfer-diagnosis-20261027';OLD=ROOT/'runs/overnight-quality-20261027'
def main():
 original=OLD/'source/capture05.py';text=original.read_text();text=text.replace("RUN=ROOT/'runs/overnight-quality-20261027'","RUN=ROOT/'runs/training-transfer-diagnosis-20261027'")
 text=text.replace("RUN/'split.json'","RUN/'historical-split.json'").replace("RUN/'native02/ready.json'","ROOT/'runs/overnight-quality-20261027/native02/ready.json'").replace("str(RUN/'native02/bin')","str(ROOT/'runs/overnight-quality-20261027/native02/bin')")
 snap=RUN/'source/executed-historical-capture.py';snap.write_text(text);(RUN/'capture-adapter.json').write_text(json.dumps({'original':str(original),'original_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),'executed':str(snap),'executed_sha256':hashlib.sha256(snap.read_bytes()).hexdigest(),'changes':'output/split paths only; original capture runtime/protocol/target unchanged'},indent=2))
 exec(compile(text,str(snap),'exec'),{'__name__':'__main__'})
if __name__=='__main__':main()
