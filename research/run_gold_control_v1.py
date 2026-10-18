#!/usr/bin/env python3
"""Supervise probe, three trainings, three evaluations and planned CPU analysis."""
from pathlib import Path
import json
import os
import queue
import signal
import subprocess
import threading
import time
from train_gold_control_v1 import BASE,TRAINER_SHA,sha,require

ROOT=Path('/home/lewis/rsi-research')
EXPORTS=Path('/mnt/c/Users/Lewis/Documents/Codex/2026-10-12/referenced-chatgpt-conversation-this-is-an/outputs')
EVAL=Path('results/gsm_symbolic_gold_control_v1')
TRAIN_SECONDS=614.4
EVAL_SECONDS=1560

def stop_reason(elapsed,limit,line=''):
    if 'out of memory' in line.lower() or 'OutOfMemoryError' in line:return 'out_of_memory'
    if 'Non-finite' in line or 'NaN loss' in line:return 'non_finite_loss_or_gradient'
    if 'FP16 SKIP at attempt ' in line:return 'skipped_optimizer_update'
    if elapsed is not None and elapsed>limit:return 'time_limit_exceeded'
    return None

def reader(proc,q):
    for line in proc.stdout:q.put(line)
    q.put(None)

def run_child(cmd,name,stage,limit=None):
    print('START '+name,flush=True)
    proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    q=queue.Queue();t=threading.Thread(target=reader,args=(proc,q),daemon=True);t.start()
    start=None;end=None;closed=False;reason=None;interrupted=None;terminated=None
    with (stage/(name+'.log')).open('x',encoding='utf-8') as f:
        while not closed or proc.poll() is None:
            line=''
            try:
                item=q.get(timeout=.5)
                if item is None:closed=True
                else:
                    line=item;f.write(line);f.flush()
                    if line.startswith(('TRAINING_TIMER_START ','EVAL_TIMER_START ')):start=json.loads(line.split(' ',1)[1])['monotonic_seconds']
                    if line.startswith(('TRAINING_TIMER_END ','EVAL_TIMER_END ')):end=json.loads(line.split(' ',1)[1])['monotonic_seconds']
                    if 'Loading weights:' not in line and (line.startswith(('Epoch ','TRAINING_TIMER','EVAL_TIMER','gold_seed','GOLD','Gold analysis','STOP')) or name=='analysis'):print(name+': '+line.strip(),flush=True)
            except queue.Empty:pass
            elapsed=(end if end is not None else time.perf_counter())-start if start is not None else None
            reason=reason or stop_reason(elapsed,limit if limit is not None else float('inf'),line)
            if reason and proc.poll() is None and interrupted is None:
                proc.send_signal(signal.SIGINT);interrupted=time.monotonic();print('STOP WHOLE GOLD EXPERIMENT: '+reason+' at '+name,flush=True)
            if interrupted is not None and proc.poll() is None and time.monotonic()-interrupted>30 and terminated is None:
                proc.terminate();terminated=time.monotonic()
            if terminated is not None and proc.poll() is None and time.monotonic()-terminated>10:proc.kill()
    code=proc.wait()
    require(code==0 and reason is None,f'{name} stopped: {reason or "process_failure"}; code {code}')
    return (end-start) if end is not None and start is not None else None

def save(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2);f.write('\n')

def copy_file(source,name):
    with (EXPORTS/name).open('xb') as f:f.write(Path(source).read_bytes())

def main():
    os.chdir(ROOT);require(sha('train_primary_lora_v1.py')==TRAINER_SHA,'Immutable trainer changed')
    require(not BASE.exists() and not EVAL.exists(),'Existing gold outputs preserved; select a new version to restart')
    BASE.mkdir(parents=True,exist_ok=False);stage=BASE/'supervision';stage.mkdir()
    python=str(ROOT/'.venv/bin/python');models=[];evaluations=[]
    limits=dict(date='2026-10-13',training_only_limit_seconds=TRAIN_SECONDS,evaluation_adapter_wall_limit_seconds=EVAL_SECONDS,
                training_excludes_loading_and_preflight=True,evaluation_includes_model_loading_excludes_shared_preflight=True,
                stop_on_oom_nonfinite_skip=True,immutable_trainer_sha256=TRAINER_SHA,supervised_control_not_self_training=True)
    save(stage/'execution_limits.json',limits)
    try:
        run_child([python,'-B','-u','train_gold_control_v1.py','--phase','preflight','--output',str(BASE/'unused_preflight')],'training_preflight',stage)
        probe=BASE/'memory_probe_seed_2026'
        run_child([python,'-B','-u','train_gold_control_v1.py','--phase','probe','--seed','2026','--output',str(probe)],'memory_probe',stage,TRAIN_SECONDS)
        p=json.loads((probe/'completed.json').read_text());require(p['successful_optimizer_updates']==3 and p['skipped_optimizer_updates']==0 and p['maximum_sequence_length']==847 and not p['final_adapter_saved'],'Gold probe failed')
        print(f"GOLD MEMORY PROBE PASSED: allocated {p['peak_allocated_gpu_gib']:.3f} GiB, reserved {p['peak_reserved_gpu_gib']:.3f} GiB; three successful updates, zero skips; discarded adapter.",flush=True)
        for seed in [2026,2027,2028]:
            name=f'gold_seed_{seed}';folder=BASE/name
            observed=run_child([python,'-B','-u','train_gold_control_v1.py','--phase','train','--seed',str(seed),'--output',str(folder)],name,stage,TRAIN_SECONDS)
            m=json.loads((folder/'completed.json').read_text())
            require(m['attempted_optimizer_updates']==m['successful_optimizer_updates']==150 and m['skipped_optimizer_updates']==0 and m['supervised_tokens_seen']==64412 and m['elapsed_training_minutes']*60<=TRAIN_SECONDS,'Gold schedule/stop guard failed')
            require(abs(observed-m['elapsed_training_minutes']*60)<.01,'Training timer differs')
            models.append(m)
            for file in ['manifest.json','completed.json','optimizer_steps.jsonl']:copy_file(folder/file,f'{name}_{file}')
            target=EXPORTS/f'{name}_final_adapter';target.mkdir(exist_ok=False)
            for file,digest in m['adapter_files_sha256'].items():
                require(sha(folder/'adapter_final'/file)==digest,'Adapter save checksum differs')
                with (target/file).open('xb') as f:f.write((folder/'adapter_final'/file).read_bytes())
            print(f"GOLD TRAIN COMPLETE {seed}: losses {m['epoch_mean_losses']}; 150/150 updates, zero skips; {m['elapsed_training_minutes']:.2f} min; allocated/reserved {m['peak_allocated_gpu_gib']:.3f}/{m['peak_reserved_gpu_gib']:.3f} GiB",flush=True)
        run_child([python,'-B','-u','gsm_symbolic_gold_eval_v1.py','--preflight'],'evaluation_preflight',stage)
        for seed in [2026,2027,2028]:
            name=f'gold_seed_{seed}'
            observed=run_child([python,'-B','-u','gsm_symbolic_gold_eval_v1.py','--model',name],name+'_evaluation',stage,EVAL_SECONDS)
            m=json.loads((EVAL/name/'completed.json').read_text());require(m['elapsed_adapter_seconds']<=EVAL_SECONDS,'Evaluation cutoff exceeded')
            evaluations.append(m)
        run_child([python,'-B','-u','gold_control_analysis_v1.py'],'analysis',stage)
        receipt=dict(status='three_gold_models_trained_evaluated_analyzed',limits=limits,training=models,evaluation=evaluations,
                     additional_experiment_run=False,trainer_sha256=sha('train_primary_lora_v1.py'))
        save(stage/'sequence_receipt.json',receipt);save(EXPORTS/'gold_control_sequence_receipt.json',receipt)
        for source,name in [(EVAL/'manifest.json','gold_control_evaluation_manifest.json'),(Path('results/gsm_symbolic_gold_control_v1_analysis.json'),'gold_control_analysis_2026-10-13.json'),(Path('results/gsm_symbolic_gold_control_v1_analysis.md'),'gold_control_analysis_2026-10-13.md'),(probe/'completed.json','gold_control_memory_probe_completed.json')]:copy_file(source,name)
        for model in ['base']+[f'{g}_seed_{s}' for g in ['gold','hard'] for s in [2026,2027,2028]]:
            with (EXPORTS/f'gold_control_{model}_responses.jsonl').open('x',encoding='utf-8') as f:
                for v in ['p1','p2']:
                    for path in sorted((EVAL/model/v).glob('batch_*.json')):
                        for r in json.loads(path.read_text())['records']:f.write(json.dumps(r)+'\n')
        print('GOLD CONTROL FINISHED. STOP; no further experiment started.',flush=True)
    except BaseException as exc:
        receipt=dict(status='stopped',exception_type=type(exc).__name__,message=str(exc),completed_training=models,completed_evaluations=evaluations,settings_changed=False,no_automatic_restart=True)
        save(stage/'stopped_receipt.json',receipt);save(EXPORTS/'gold_control_stopped_receipt.json',receipt)
        print(json.dumps(receipt,indent=2),flush=True);raise

if __name__=='__main__':main()
