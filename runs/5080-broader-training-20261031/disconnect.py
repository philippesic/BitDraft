import time,json,os
from pathlib import Path
p=Path(__file__).parent
(p/"cpu-start.json").write_text(json.dumps({"pid":os.getpid(),"boot":Path("/proc/sys/kernel/random/boot_id").read_text(),"start":time.time(),"interop":os.getenv("WSL_INTEROP")}))
time.sleep(95)
(p/"cpu-done.json").write_text(json.dumps({"pid":os.getpid(),"end":time.time()}))
