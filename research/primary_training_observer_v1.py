#!/usr/bin/env python3
"""Observe two existing timer calls; execute the immutable trainer unchanged."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import time

EXPECTED_SOURCE='d8381ab86bb6020293121dc5dd32dac113901fbb8ddac8540fe4d777cea5a1b9'

class ObservedClock:
    def __init__(self,real_clock=time):self.real=real_clock;self.calls=0
    def perf_counter(self):
        value=self.real.perf_counter()
        self.calls+=1
        if self.calls in (1,2):
            marker='TRAINING_TIMER_START' if self.calls==1 else 'TRAINING_TIMER_END'
            print(marker+' '+json.dumps({'monotonic_seconds':value}),flush=True)
        return value
    def __getattr__(self,name):return getattr(self.real,name)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group',required=True,choices=['hard','random','easy'])
    parser.add_argument('--seed',required=True,type=int,choices=[2026,2027,2028])
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    source=Path('train_primary_lora_v1.py')
    if hashlib.sha256(source.read_bytes()).hexdigest()!=EXPECTED_SOURCE:raise RuntimeError('Immutable trainer source differs')
    spec=importlib.util.spec_from_file_location('observed_primary_runner',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.time=ObservedClock() # only this module's clock; libraries keep their clocks
    call=SimpleNamespace(phase='train',group=args.group,seed=args.seed,output=args.output,
                         probe_report='results/primary_lora_training_v1/memory_probe_seed_2026/completed.json',
                         additional_runs_confirmed=True)
    import fcntl
    with Path('results/primary_lora_training_v1.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        module.execute(call)
    if module.time.calls!=2:raise RuntimeError('Unexpected trainer timer-call count')

if __name__=='__main__':main()
