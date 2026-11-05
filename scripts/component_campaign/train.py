"""Enter an immutable source closure before a component training arm starts."""
import json,os,sys

def main():
 if '--help' in sys.argv:
  from .train_impl import main as execute
  return execute()
 from .frozen_source import freeze
 spec=json.load(open(sys.argv[1]));directory=freeze(spec['name'])
 env=dict(os.environ,PYTHONPATH=str(directory)+':'+str(directory/'src'),COMPONENT_FROZEN_SOURCE=str(directory))
 os.chdir(directory)
 os.execve(sys.executable,[sys.executable,'-m','scripts.component_campaign.train_impl',*sys.argv[1:]],env)

if __name__=='__main__':main()
