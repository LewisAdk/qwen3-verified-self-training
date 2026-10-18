"""Single approved sequence, fresh processes, durable logs, no retries."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys
import time
import runner as run
from recovery_2026_10_13 import ClockMonitor, clocks

def phase(stage, seed=None):
    name=stage+(f'_{seed}' if seed is not None else '')
    folder=run.FULL/'jobs'/name;folder.mkdir(parents=True,exist_ok=False)
    limit={'train':4200,'evaluate':1800,'analyze':600}[stage]
    wall=time.time();mono=time.monotonic()
    run.save(folder/'started.json',dict(stage=stage,seed=seed,supervisor_pid=os.getpid(),system=clocks(),budget_seconds=limit))
    command=[str(run.ROOT/'.venv/bin/python'),'-B','-u',str(run.FULL/'src/runner.py'),stage]
    if seed is not None:command+=['--seed',str(seed)]
    with (ClockMonitor(folder/'system_time_device_memory.jsonl') if stage!='analyze' else nullcontext()) as monitor, (folder/'console.log').open('x') as log:
        child=subprocess.Popen(command,cwd=run.ROOT,stdout=log,stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        run.save(folder/'child.json',dict(pid=child.pid,command=command))
        timed_out=False
        while True:
            remaining=limit-(time.monotonic()-mono)
            if remaining<=0:
                timed_out=True;child.terminate()
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=15)
                code=124;break
            try:code=child.wait(timeout=min(50,remaining));break
            except subprocess.TimeoutExpired:continue
    elapsed=time.monotonic()-mono
    if elapsed>limit:timed_out=True;code=124
    if monitor is not None:run.save(folder/'system_time_device_memory_summary.json',monitor.summary())
    run.save(folder/'completed.json',dict(stage=stage,seed=seed,exit_code=code,timed_out=timed_out,
        monotonic_elapsed_seconds=elapsed,wall_elapsed_seconds=time.time()-wall,system=clocks()))
    run.require(code==0 and not timed_out, f'STOP: {name} failed or exceeded budget; preserve outputs, no retry')
    print(name+' COMPLETE',flush=True)

def main():
    os.chdir(run.ROOT)
    run.sources();run.integrity('full_sequence_before_hashes_2026-10-13.json')
    run.save(run.FULL/'sequence_started_2026-10-13.json',dict(pid=os.getpid(),system=clocks(),seeds=list(run.SEEDS),
        stages='three training processes, then three evaluation processes, then frozen CPU analysis',no_retry=True,accuracy_blind_until_all_evaluations=True))
    for seed in run.SEEDS:phase('train',seed)
    for seed in run.SEEDS:phase('evaluate',seed)
    run.all_evaluations_complete()
    phase('analyze')
    final=run.integrity('full_sequence_after_hashes_2026-10-13.json')
    run.save(run.FULL/'sequence_completed_2026-10-13.json',dict(status='completed_and_stopped',system=clocks(),protected=final,all_three_training_and_evaluations_complete=True))
    print('ALL_APPROVED_STAGES_COMPLETE_STOP',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        run.save(run.FULL/f'sequence_stopped_{time.time_ns()}.json',dict(status='stopped_on_failure',type=type(e).__name__,message=str(e),system=clocks(),no_retry=True,no_settings_changed=True))
        raise
