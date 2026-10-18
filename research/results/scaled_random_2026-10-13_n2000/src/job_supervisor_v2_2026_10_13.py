"""Detached, time-bounded job with durable new logs and an exit receipt."""
import datetime
import json
from pathlib import Path
import os
import subprocess
import sys
import time

ROOT=Path('/home/lewis/rsi-research')
RUN=ROOT/'results/scaled_random_2026-10-13_n2000'
name=sys.argv[1]
limits={'determinism':1800,'labeling':18000,'freeze_set':600,'probe':300,'hashes':300}
assert name in limits
folder=RUN/f'job_{name}_rerun_2026-10-13_01';folder.mkdir(exist_ok=False)
def publish(n,obj):
    with (folder/n).open('x') as f:json.dump(obj,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
start=time.time();mono=time.monotonic()
sys.path.insert(0,str(RUN/'src'))
from recovery_2026_10_13 import ClockMonitor
from contextlib import nullcontext
publish('started.json',dict(stage=name,supervisor_pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_epoch=start,monotonic=mono,budget_seconds=limits[name],full_training_authorized=False))
with (ClockMonitor(folder/'system_time_device_memory.jsonl') if name=='labeling' else nullcontext()) as monitor, (folder/'console.log').open('x') as log:
    command=[str(ROOT/'.venv/bin/python'),'-B','-u',str(RUN/'src/recovery_2026_10_13.py'),name]
    child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
    publish('child.json',dict(pid=child.pid,command=command))
    try:code=child.wait(timeout=limits[name]);timed_out=False
    except subprocess.TimeoutExpired:
        child.terminate()
        try:child.wait(timeout=15)
        except subprocess.TimeoutExpired:child.kill();child.wait()
        code=124;timed_out=True
if name=='labeling':publish('system_time_device_memory_summary.json',monitor.summary())
publish('completed.json',dict(stage=name,exit_code=code,timed_out=timed_out,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_elapsed_seconds=time.time()-start,monotonic_elapsed_seconds=time.monotonic()-mono,full_training_started=False))
