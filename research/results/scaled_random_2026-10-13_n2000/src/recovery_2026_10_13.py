"""Fresh approved rerun; preserve partial run and stop on any new failure."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import ast
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import stages as st

RUN=st.RUN
RERUN_NAME='determinism_base_batch4_rerun_2026-10-13_01'
RECEIPT=RUN/'rerun_source_manifest_2026-10-13.json'
BASELINE=RUN/'rerun_protected_baseline_2026-10-13.json'
ORIGINAL_READ=st.read
ORIGINAL_CHECK=st.check_protected

def windows_clock():
    command="[pscustomobject]@{utc=[DateTime]::UtcNow.ToString('o');stopwatch_ticks=[Diagnostics.Stopwatch]::GetTimestamp();stopwatch_frequency=[Diagnostics.Stopwatch]::Frequency;tick_ms=[Environment]::TickCount}|ConvertTo-Json -Compress"
    out=subprocess.check_output(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-NonInteractive','-Command',command],text=True,timeout=10)
    return json.loads(out.lstrip('\ufeff').strip())

def clocks():
    return dict(wall_epoch=time.time(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),monotonic_seconds=time.monotonic(),boottime_seconds=time.clock_gettime(time.CLOCK_BOOTTIME),uptime_seconds=float(Path('/proc/uptime').read_text().split()[0]),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())

def clock_flags(previous,current):
    if previous is None:return {}
    wall=current['wall_epoch']-previous['wall_epoch'];mono=current['monotonic_seconds']-previous['monotonic_seconds'];boot=current['boottime_seconds']-previous['boottime_seconds']
    return dict(wall_delta_seconds=wall,monotonic_delta_seconds=mono,boottime_delta_seconds=boot,wall_clock_jump=abs(wall-mono)>2,suspend_gap=boot-mono>2,sample_gap=mono>10,boot_id_changed=current['boot_id']!=previous['boot_id'])

class ClockMonitor(st.Monitor):
    def __enter__(self):
        def run():
            previous=None;next_windows=0
            with self.path.open('x') as f:
                while not self.stop.is_set():
                    now=clocks();row=dict(time=now['wall_epoch'],system=now,clock_flags=clock_flags(previous,now));previous=now
                    try:
                        text=subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits','-i','0'],text=True,timeout=5)
                        used,total=map(float,text.strip().split(','));row.update(used_gib=used/1024,total_gib=total/1024)
                    except Exception as e:row['error']=str(e)
                    if now['monotonic_seconds']>=next_windows:
                        try:row['windows_system_time']=windows_clock()
                        except Exception as e:row['windows_clock_error']=str(e)
                        next_windows=now['monotonic_seconds']+10
                    self.rows.append(row);f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno());self.stop.wait(1)
        self.thread=threading.Thread(target=run,daemon=True);self.thread.start();return self

    def summary(self):
        result=super().summary()
        result.update(clock_event_samples=sum(any(x.get('clock_flags',{}).get(k,False) for k in ['wall_clock_jump','suspend_gap','sample_gap','boot_id_changed']) for x in self.rows),windows_clock_samples=sum('windows_system_time' in x for x in self.rows),windows_clock_errors=sum('windows_clock_error' in x for x in self.rows),clock_fields='WSL UTC/wall, monotonic, CLOCK_BOOTTIME, uptime, boot_id; Windows UTC/Stopwatch/tick counter about every 10 seconds')
        return result

def recovery_hashes(receipt=None):
    baseline=ORIGINAL_READ(BASELINE);bad=[]
    for p,h in baseline['files']:
        if not Path(p).is_file() or st.sha(p)!=h:bad.append(p)
    st.require(not bad,'Preserved or frozen files changed: '+str(bad))
    result=dict(verified=len(baseline['files']),mismatches=[],utc=clocks()['utc'],partial_run_preserved=True)
    if receipt:st.save(RUN/receipt,result)
    return result

def check_recovery():
    st.check_sources()
    for p,h in ORIGINAL_READ(RECEIPT)['sources'].items():st.require(st.sha(p)==h,'Rerun source changed: '+p)
    st.require(ORIGINAL_READ(RUN/'rerun_CPU_validation_2026-10-13.json')['passed'],'Recovery CPU checks not passed')

def install_adapters():
    # Only the complete new rerun is allowed to satisfy the downstream gate.
    def read(path):
        if Path(path)==RUN/'determinism_base_batch4/summary.json':path=RUN/RERUN_NAME/'summary.json'
        return ORIGINAL_READ(path)
    st.read=read;st.check_protected=recovery_hashes;st.Monitor=ClockMonitor

def fresh_determinism_function():
    tree=ast.parse(Path(st.__file__).read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='deterministic_check')
    class Paths(ast.NodeTransformer):
        def visit_Constant(self,n):
            if n.value=='determinism_base_batch4':return ast.copy_location(ast.Constant(RERUN_NAME),n)
            if n.value=='before_determinism_hashes.json':return ast.copy_location(ast.Constant('before_determinism_rerun_01_hashes.json'),n)
            if n.value=='after_determinism_hashes.json':return ast.copy_location(ast.Constant('after_determinism_rerun_01_hashes.json'),n)
            return n
    node=Paths().visit(node);ns=dict(st.__dict__);ns['Monitor']=ClockMonitor;ns['check_protected']=recovery_hashes
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'<identical-determinism-new-path>','exec'),ns)
    return ns['deterministic_check']

def stage(name):
    check_recovery();install_adapters()
    if name=='determinism':fresh_determinism_function()()
    elif name=='labeling':st.labeling()
    elif name=='freeze_set':st.freeze_set()
    elif name=='probe':st.memory_probe()
    elif name=='hashes':recovery_hashes('rerun_final_hashes_2026-10-13.json')
    else:raise RuntimeError('Unapproved stage: '+name)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['determinism','labeling','freeze_set','probe','hashes']);args=parser.parse_args();os.chdir(st.ROOT)
    try:stage(args.stage)
    except BaseException as e:
        st.save(RUN/f'rerun_failure_{args.stage}_{time.time_ns()}.json',dict(stage=args.stage,type=type(e).__name__,message=str(e),system=clocks(),no_settings_changed=True,full_training_started=False));raise
