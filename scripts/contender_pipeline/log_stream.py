"""Bounded child output relay; disk errors must terminate the producing job."""
import os
from pathlib import Path
import threading

class LogStream:
 def __init__(self, pipe, path, limit=8*1024**2):
  self.pipe=pipe;self.path=Path(path);self.limit=limit;self.errors=[]
  self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()
 def run(self):
  stream=None
  try:
   stream=self.path.open('xb');size=0
   while chunk:=os.read(self.pipe.fileno(),65536):
    offset=0
    while offset<len(chunk):
     if size==self.limit:
      stream.close();os.replace(self.path,self.path.with_suffix('.log.previous'));stream=self.path.open('xb');size=0
     n=min(len(chunk)-offset,self.limit-size);stream.write(chunk[offset:offset+n]);stream.flush();size+=n;offset+=n
  except BaseException as exc:self.errors.append(str(exc))
  finally:
   self.pipe.close()
   if stream is not None:stream.close()
 def finish(self):
  self.thread.join(timeout=5)
  if self.thread.is_alive():self.errors.append('log pipe remained live after cleanup')
  if self.errors:raise RuntimeError('; '.join(self.errors))
