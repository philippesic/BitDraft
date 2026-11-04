"""Matched Q4 DSpark/DFlash state-update diagnostic; no training."""
from pathlib import Path
import argparse,copy,json,os,signal,socket,subprocess,sys,time
P=Path('/home/philip/binary-eagle-decoding')
sys.path[:0]=[str(P/'scripts/data_cost_transfer'),str(P/'scripts/design_readiness'),str(P/'scripts/w1a8_generalization'),str(P/'scripts/contender_pipeline')]
from host_profile import Profile
from native_panel import Panel
from modern_requests import Requests,references,TARGET_SHA,BASELINE,FROZEN
from modern_collect import atomic,pin,read,require,process_identity

class DSRequests(Requests):
    def cell(self, rep, arm):
        self.guard()
        cell = self.output / f"rep-{rep:02d}-{arm['id']}"
        cell.mkdir()
        begin = time.perf_counter()
        runtime = arm["runtime"]
        process = None
        failure = None
        clean = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(
                (
                    "GGML_",
                    "DSPARK_",
                    "DFLASH_",
                    "EAGLE_",
                    "W1AX_",
                    "W1A8_",
                    "LLAMA_",
                    "CUDA_",
                    "LD_",
                    "OMP_",
                )
            )
            and k != "NVIDIA_TF32_OVERRIDE"
        }
        env = {**clean, **runtime["environment"]}
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        spec = {
            "target": "none",
            "dflash": "draft-dflash",
            "dspark": "draft-dspark",
            "w1a8": "draft-eagle3",
            "eagle_q4": "draft-eagle3",
        }[arm["kind"]]
        argv = [
            runtime["server"]["path"],
            "-m",
            self.models["target"]["path"],
            "-ngl",
            "99",
            "-c",
            "4096",
            "-np",
            "1",
            "-b",
            "512",
            "-ub",
            "512",
            "-t",
            "4",
            "-fa",
            "on",
            "--cache-type-k",
            "f16",
            "--cache-type-v",
            "f16",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--metrics",
            "--no-webui",
            "--no-warmup",
            "--spec-type",
            spec,
        ]
        if arm.get("model"):
            argv += [
                "-md",
                arm["model"]["path"],
                "-ngld",
                "99",
                "--cache-type-k-draft",
                "f16",
                "--cache-type-v-draft",
                "f16",
                "--spec-draft-n-max",
                "2",
                "--spec-draft-n-min",
                "0",
                "--spec-draft-p-min",
                "0",
                "--spec-draft-sampling",
                "greedy",
            ]
        level = runtime.get("log_verbosity", 4 if arm["kind"] == "w1a8" else 3)
        argv += ["-lv", str(level)]
        atomic(cell / "command.json", {"argv": argv, "environment": runtime["environment"]})
        try:
            with (cell / "server.log").open("x") as log:
                process = subprocess.Popen(
                    argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                )
                self.active = process
                self.active_identity = process_identity(process)
                atomic(cell / "identity.json", self.active_identity)
                deadline = time.monotonic() + 120
                while time.monotonic() < deadline:
                    self.guard()
                    require(process.poll() is None, "server exited during startup")
                    try:
                        self.request(port, "health", timeout=1)
                        break
                    except Exception:
                        self.guard()
                        time.sleep(0.2)
                else:
                    raise RuntimeError("server startup timeout")
                atomic(cell / "startup.json", {"seconds": time.perf_counter() - begin})
                self.prepare_cell(port, cell, arm, rep)
                for index, item in enumerate(self.warmups):
                    self.completion(port, item, cell, index, True, arm, rep)
                self.audit_maps(cell, "warm", arm)
                for index, item in enumerate(self.rows, start=2):
                    self.completion(port, item, cell, index, False, arm, rep)
                self.audit_maps(cell, "final", arm)
                if arm["kind"] == "w1a8" and level == 4:
                    content = (cell / "server.log").read_text()
                    require(
                        "complete nine-linear fixed row W1A8" in content
                        and content.count("W1A8 packed ") == 9,
                        "packed nine-linear loader receipts missing",
                    )
        except BaseException as error:
            failure = repr(error)
            atomic(
                cell / "failure.json",
                {"error": failure, "elapsed_seconds": time.perf_counter() - begin},
            )
            raise
        finally:
            if process is not None:
                self.reap(cell / "release.json")
            atomic(
                cell / "cost.json",
                {"elapsed_seconds": time.perf_counter() - begin, "failure": failure},
            )
        print(
            json.dumps(
                {
                    "stage": "cell_complete",
                    "repetition": rep,
                    "arm": arm["id"],
                    "measured_requests": len(self.rows),
                }
            ),
            flush=True,
        )


class DSPanel(Panel,DSRequests):
    def authenticate(self):
        require(self.args.allow_cuda and self.args.repetitions==5,'GPU and five reps required')
        self.output.mkdir(parents=True,exist_ok=False)
        self.models=read(self.args.models);self.inputs=read(self.args.inputs)
        for ref in references([self.models,self.inputs]):self.checked(ref)
        require(self.models['target']['sha256']==TARGET_SHA,'fixed modern target required')
        require(self.models['baseline']['sha256']==pin(BASELINE)['sha256'],'baseline changed')
        self.arms=self.models['arms'];require([a['kind'] for a in self.arms]==['target','dspark','dflash'],'explicit three arms')
        frozen=read(FROZEN)
        reference=read(P/'runs/contender-infra-20261104-01/native-retry/models-quiet.json')
        standard=next(a['runtime'] for a in reference['arms'] if a['kind']=='dflash')
        for arm in self.arms:
            require(arm['runtime']==standard,'all arms must use identical frozen runtime')
            require(arm['runtime']['server']['sha256']==frozen['server']['sha256'],'runtime mismatch')
        self.authenticate_inputs();self.guard()
        atomic(self.output/'freeze.json',dict(models=pin(self.args.models),inputs=pin(self.args.inputs),source=pin(__file__),common_depth=2,repetitions=5,scope='Modern Qwen3.5 Q4 DSpark/DFlash; historical DSpark W1A8 is not measured; root host spans without forced sync'))
class DSProfile(Profile,DSPanel):pass

def main():
 p=argparse.ArgumentParser();p.add_argument('--models',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--library',type=Path,required=True);p.add_argument('--allow-cuda',action='store_true');a=p.parse_args();a.out.mkdir(exist_ok=False)
 panels=[]
 for name,cls in [('plain',DSPanel),('traced',DSProfile)]:
  panel=cls(argparse.Namespace(models=a.models,inputs=a.inputs,output=a.out/name,repetitions=5,allow_cuda=a.allow_cuda));panel.records=[];panel.stopping=False
  if name=='traced':panel.library=a.library.resolve()
  panel.authenticate();panels.append(panel)
 def stop(sig,frame):
  for panel in panels:panel.request_stop(sig,frame)
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 for rep in range(5):
  arms=panels[0].arms if rep%2==0 else list(reversed(panels[0].arms))
  for arm in arms:
   for panel in panels if rep%2==0 else list(reversed(panels)):
    panel.cell(rep,next(x for x in panel.arms if x['id']==arm['id']))
    atomic(panel.output/'progress.json',dict(records=panel.records),replace=True)
 plain,traced=panels;lookup={(r['arm'],r['repetition'],r['id']):r for r in plain.records};pairs=[]
 for row in traced.records:
  original=lookup.pop((row['arm'],row['repetition'],row['id']))
  require(all(row[k]==original[k] for k in ('completion_tokens','draft_proposed','draft_accepted','verification_rounds')),'instrumentation changed decisions')
  pairs.append(dict(arm=row['arm'],repetition=row['repetition'],id=row['id'],warmup=row['warmup'],plain_ms=original['full_http_wall_seconds']*1000,traced_ms=row['full_http_wall_seconds']*1000))
 require(not lookup,'missing pairs')
 for panel in panels:atomic(panel.output/'result.json',dict(records=panel.records,status='MATCHED_Q4_COST_PASS'))
 atomic(a.out/'result.json',dict(status='MATCHED_Q4_COST_PASS',pairs=pairs,responses=len(plain.records)+len(traced.records),exact_outputs_and_counters=True))
if __name__=='__main__':main()
