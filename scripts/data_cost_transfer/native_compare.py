"""Existing native comparison adapter with lossless native debug-log input."""
from pathlib import Path
import runpy,sys
import native_log_text
native_log_text.install()
P=Path('/home/philip/binary-eagle-decoding');sys.path.insert(0,str(P/'scripts/contender_pipeline'))
from runner import pin,read,atomic
args=sys.argv[1:];source=Path(args[args.index('--models')+1]);out=Path(args[args.index('--output')+1]);models=read(source)
models['source_pins'] += [pin(__file__),pin(native_log_text.__file__)]
copy=out.with_suffix('.models.json');atomic(copy,models);args[args.index('--models')+1]=str(copy)
sys.argv=[str(P/'scripts/contender_pipeline/current_adapter.py')]+args
runpy.run_path(sys.argv[0],run_name='__main__')
