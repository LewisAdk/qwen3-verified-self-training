#!/usr/bin/env python3
"""Sequential supervisor; no training setting changes; stop before evaluation."""
from pathlib import Path
import csv
import datetime
import hashlib
import json
import os
import queue
import signal
import subprocess
import threading
import time

RUNS=[('hard',2027),('hard',2028),('random',2026),('random',2027),('random',2028),('easy',2026),('easy',2027),('easy',2028)]
LIMIT_SECONDS=10.24*60
EXPECTED_SOURCE='d8381ab86bb6020293121dc5dd32dac113901fbb8ddac8540fe4d777cea5a1b9'
ROOT=Path('/home/lewis/rsi-research')
EXPORTS=Path('/mnt/c/Users/Lewis/Documents/Codex/2026-10-12/referenced-chatgpt-conversation-this-is-an/outputs')
BASE=Path('results/primary_lora_training_v1')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def parse_memory(output):
    row=next(csv.reader(output.strip().splitlines()))
    if len(row)!=5:raise ValueError('Unexpected nvidia-smi fields')
    index=int(row[0].strip());total=float(row[2].strip());used=float(row[3].strip());free=float(row[4].strip())
    if index!=0 or total<=0 or min(used,free)<0:raise ValueError('Invalid GPU memory data')
    return dict(gpu_index=index,name=row[1].strip(),total_mib=total,used_mib=used,free_mib=free,
                used_gib=used/1024,counter_above_reported_capacity=used>total)

def memory_classification(samples):
    valid=[s for s in samples if s.get('status')=='ok']
    if not valid:return dict(status='no_valid_samples',interpretation='Memory question unresolved; no settings changed')
    peak=max(s['used_gib'] for s in valid)
    total=max(s['total_mib'] for s in valid)/1024
    if any(s['counter_above_reported_capacity'] for s in valid):
        category='reported_usage_above_capacity';meaning='Counter exceeds card capacity; do not label it a physical resident-VRAM measurement'
    elif peak>=total-.05:
        category='sampled_ceiling_reached';meaning='Reported device-wide usage reached approximately the 8 GiB ceiling; continue unchanged unless a stop condition occurs'
    elif peak<7.5:
        category='below_7_5_gib';meaning='Under the agreed rule, classify the earlier 10.62 GiB reading as allocator accounting; this does not retrospectively measure the first run'
    else:
        category='between_7_5_and_ceiling';meaning='Near-ceiling band; report observed usage and training time without changing settings'
    return dict(status=category,sample_count=len(valid),sampled_peak_device_memory_gib=peak,total_device_gib=total,
                interpretation=meaning,sampling_seconds=5,brief_peaks_may_be_missed=True,device_wide_not_process_specific=True)

def should_stop(elapsed,skips=0,nan_loss=False,oom=False):
    if oom:return 'out_of_memory'
    if nan_loss:return 'non_finite_loss'
    if skips:return 'skipped_optimizer_update'
    if elapsed is not None and elapsed>LIMIT_SECONDS:return 'training_time_exceeded_10_24_minutes'
    return None

def publish(path,obj):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2);f.write('\n')

def logger(stop,path,samples,errors):
    began=time.monotonic();tick=0
    with path.open('x',encoding='utf-8') as f:
        while not stop.is_set():
            wait=began+tick*5-time.monotonic()
            if wait>0 and stop.wait(wait):break
            sample=dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        monotonic_seconds=time.perf_counter(),sample_number=tick)
            try:
                cmd=['nvidia-smi','-i','0','--query-gpu=index,name,memory.total,memory.used,memory.free','--format=csv,noheader,nounits']
                p=subprocess.run(cmd,capture_output=True,text=True,timeout=4)
                if p.returncode:raise RuntimeError(p.stderr.strip() or 'nvidia-smi failed')
                sample.update(parse_memory(p.stdout));sample['status']='ok'
            except Exception as exc:
                sample.update(status='query_failed',message=str(exc));errors.append(str(exc))
            samples.append(sample);f.write(json.dumps(sample)+'\n');f.flush()
            tick+=1
    # Training is not stopped by logger failure or memory readings alone.

def read_stdout(proc,q):
    for line in proc.stdout:q.put(line)
    q.put(None)

def export_run(folder,name):
    for file in ['manifest.json','completed.json','failed.json','optimizer_steps.jsonl']:
        if (folder/file).exists():
            with (EXPORTS/f'primary_{name}_{file}').open('xb') as f:f.write((folder/file).read_bytes())
    if (folder/'completed.json').exists():
        metrics=json.loads((folder/'completed.json').read_text())
        target=EXPORTS/f'primary_{name}_final_adapter';target.mkdir(exist_ok=False)
        for file,expected in metrics['adapter_files_sha256'].items():
            source=folder/'adapter_final'/file
            if sha(source)!=expected:raise RuntimeError('Final adapter checksum differs')
            with (target/file).open('xb') as f:f.write(source.read_bytes())
        return metrics
    return None

def main():
    os.chdir(ROOT)
    if sha('train_primary_lora_v1.py')!=EXPECTED_SOURCE:raise RuntimeError('Trainer source changed')
    baseline=json.loads((BASE/'hard_seed_2026/completed.json').read_text())
    if baseline['status']!='completed_two_epochs':raise RuntimeError('Baseline incomplete')
    for group,seed in RUNS:
        if (BASE/f'{group}_seed_{seed}').exists():raise RuntimeError('Planned output already exists; preserving it')
    stage=BASE/'remaining_eight_supervision_v1';stage.mkdir(exist_ok=False)
    publish(stage/'plan.json',dict(runs=RUNS,training_timeout_minutes=10.24,nvidia_smi_interval_seconds=5,
                                 logger_model='hard_seed_2027',trainer_sha256=EXPECTED_SOURCE,
                                 training_settings=baseline['settings'],no_training_setting_changes=True,
                                 observer_sha256=sha('primary_training_observer_v1.py'),
                                 supervisor_sha256=sha(__file__),stop_before_evaluation=True))
    results=[];stop_reason=None;memory_summary=None
    python=str(ROOT/'.venv/bin/python')
    for number,(group,seed) in enumerate(RUNS,1):
        name=f'{group}_seed_{seed}'
        folder=BASE/name
        output_queue=queue.Queue();samples=[];errors=[]
        gpu_stop=threading.Event();monitor=None
        if number==1:
            monitor=threading.Thread(target=logger,args=(gpu_stop,stage/'hard_2027_nvidia_smi_5s.jsonl',samples,errors),daemon=True);monitor.start()
        print(f'START {number}/8: {name}',flush=True)
        proc=subprocess.Popen([python,'-B','-u','primary_training_observer_v1.py','--group',group,'--seed',str(seed),'--output',str(folder)],
                              stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        reader=threading.Thread(target=read_stdout,args=(proc,output_queue),daemon=True);reader.start()
        train_start=None;train_end=None;stream_ended=False;interrupted=False;interrupted_at=None;terminated_at=None
        with (stage/f'{name}.log').open('x',encoding='utf-8') as log:
            while not stream_ended or proc.poll() is None:
                try:
                    line=output_queue.get(timeout=.5)
                    if line is None:stream_ended=True
                    else:
                        log.write(line);log.flush()
                        if line.startswith('TRAINING_TIMER_START '):
                            train_start=json.loads(line.split(' ',1)[1])['monotonic_seconds']
                        elif line.startswith('TRAINING_TIMER_END '):
                            train_end=json.loads(line.split(' ',1)[1])['monotonic_seconds']
                        elif 'FP16 SKIP at attempt ' in line:
                            stop_reason='skipped_optimizer_update'
                        elif 'Non-finite loss at ' in line:
                            stop_reason='non_finite_loss'
                        if 'Loading weights:' not in line and (line.startswith('Epoch ') or line.startswith('FP16 SKIP') or line.startswith('TRAINING_TIMER')):
                            print(name+': '+line.strip(),flush=True)
                except queue.Empty:pass
                elapsed=(train_end if train_end is not None else time.perf_counter())-train_start if train_start is not None else None
                stop_reason=stop_reason or should_stop(elapsed)
                if stop_reason and proc.poll() is None and not interrupted:
                    proc.send_signal(signal.SIGINT);interrupted=True;interrupted_at=time.monotonic()
                    print(f'STOP ALL: {stop_reason} ({name})',flush=True)
                if interrupted and proc.poll() is None and time.monotonic()-interrupted_at>30 and terminated_at is None:
                    proc.terminate();terminated_at=time.monotonic()
                if terminated_at is not None and proc.poll() is None and time.monotonic()-terminated_at>10:
                    proc.kill()
            code=proc.wait()
        if monitor:
            gpu_stop.set();monitor.join(timeout=6)
            if monitor.is_alive():errors.append('Logger did not finish promptly')
            memory_summary=memory_classification(samples)
            memory_summary['query_errors']=errors
            if train_start is not None:
                training_samples=[s for s in samples if s.get('status')=='ok' and s['monotonic_seconds']>=train_start and (train_end is None or s['monotonic_seconds']<=train_end)]
                memory_summary['training_window']=memory_classification(training_samples)
            publish(stage/'memory_summary.json',memory_summary)
            print('MEMORY LOGGER SUMMARY: '+json.dumps(memory_summary),flush=True)
        metrics=export_run(folder,name)
        if metrics:
            if metrics['settings']!=baseline['settings'] or metrics['source_sha256']!=EXPECTED_SOURCE:raise RuntimeError('Settings or source differ')
            with (folder/'optimizer_steps.jsonl').open() as f:steps=[json.loads(s) for s in f]
            if len(steps)!=metrics['attempted_optimizer_updates'] or sum(not s['skipped'] for s in steps)!=metrics['successful_optimizer_updates']:raise RuntimeError('Step logs disagree')
            if not all(s['learning_rate']==1e-4 for s in steps):raise RuntimeError('Learning rate changed')
            stop_reason=stop_reason or should_stop(metrics['elapsed_training_minutes']*60,metrics['skipped_optimizer_updates'])
            results.append(dict(name=name,metrics=metrics,observed_training_seconds=(train_end-train_start) if train_end is not None and train_start is not None else None))
            print(f"COMPLETE {number}/8 {name}: losses {metrics['epoch_mean_losses']}, updates {metrics['attempted_optimizer_updates']}/{metrics['successful_optimizer_updates']}, skips {metrics['skipped_optimizer_updates']}, minutes {metrics['elapsed_training_minutes']:.2f}",flush=True)
        elif (folder/'failed.json').exists():
            failure=json.loads((folder/'failed.json').read_text())
            if failure.get('out_of_memory'):stop_reason='out_of_memory'
            elif 'Non-finite loss' in failure.get('message',''):stop_reason='non_finite_loss'
            stop_reason=stop_reason or 'unexpected_process_failure'
        elif code:stop_reason=stop_reason or 'unexpected_process_failure'
        if code or stop_reason:break
    receipt=dict(status='all_eight_completed' if len(results)==8 and not stop_reason else 'sequence_stopped',
                 stop_reason=stop_reason,completed_models=[r['name'] for r in results],results=results,
                 memory_summary=memory_summary,evaluation_run=False,training_settings_changed=False,
                 trainer_sha256=sha('train_primary_lora_v1.py'),completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    publish(stage/'sequence_receipt.json',receipt)
    publish(EXPORTS/'remaining_primary_models_sequence_receipt.json',receipt)
    for file in ['plan.json','memory_summary.json','hard_2027_nvidia_smi_5s.jsonl']:
        if (stage/file).exists():
            with (EXPORTS/('remaining_primary_models_'+file)).open('xb') as f:f.write((stage/file).read_bytes())
    print('SEQUENCE FINISHED: '+receipt['status']+'; evaluation has not started.',flush=True)
    if stop_reason:raise SystemExit(2)

if __name__=='__main__':main()
