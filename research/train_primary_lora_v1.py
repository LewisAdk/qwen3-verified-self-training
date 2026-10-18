#!/usr/bin/env python3
"""Pinned final-only LoRA training; memory probe first, one run per command."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_DATASETS_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import argparse
import ast
import datetime
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import re
import time
from decimal import Decimal

MODEL_REV='ea980cb0a6c2ae4b936e82123acc929f1cec04c1'
WEIGHT_SHA='6df85b39330e5a425ee36253d0f894e4387e4f0a15b9c53cb467d668e6b3a841'
DATA_REV='740312add88f781978c0658806c59bc2815b9866'
SETS=Path('results/training_sets_primary_tokenmatched_v1')
SEEDS=(2026,2027,2028)
GROUPS=('hard','random','easy')
FROZEN_HASHES={
 'manifest.json':'b98aef474b4b996ae95e0a4ae54c420ca44926a3ae8ad35948cbccd6ef87b295',
 'hard.jsonl':'de56bc3e952bf2963f24ddb2dce4a10a99c7cfa20565dccafbc96d8acb7cc246',
 'random.jsonl':'9262e0e622024664212317ec5337cc29d9622a68e315f8a0a6ea45554da97bbb',
 'easy.jsonl':'0e79c4d3b680068f133ee5c80c0fe5a21939b0c5b274a48e8027aa2f3061e140'}
SETTINGS=dict(epochs=2,learning_rate=1e-4,microbatch=1,gradient_accumulation=4,
 max_sequence_length=1024,lora_rank=4,lora_alpha=8,lora_dropout=0.0,
 target_modules=['q_proj','v_proj'],bias='none',dtype='float16',optimizer='AdamW',
 weight_decay=0.0,gradient_clip_norm=1.0,learning_rate_schedule='constant; no scheduler',
 gradient_checkpointing=True,use_reentrant=False,prompt_loss_mask=-100,
 gradient_scaling='torch.amp.GradScaler cuda defaults; automatic overflow scaling',
 checkpoint_selection='final after exactly two epochs only')

def require(ok,message):
    if not ok:raise RuntimeError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def json_file(path,obj):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(obj,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def check_output(out):
    resolved=out.resolve();base=Path('results').resolve()
    require(resolved!=base and resolved.is_relative_to(base),'Output must be separate and under results/')
    require(not out.exists(),'Existing run preserved. Restart from base in a NEW output directory; no intermediate checkpoints.')

def check_trainer(text):
    tree=ast.parse(text)
    constants={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['EPOCHS','LEARNING_RATE','GRAD_ACCUM','MAX_SEQUENCE_LENGTH']}
    require(constants==dict(EPOCHS=2,LEARNING_RATE=1e-4,GRAD_ACCUM=4,MAX_SEQUENCE_LENGTH=1024),'Original training constants changed')
    cfg=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='LoraConfig')
    require({k.arg:ast.literal_eval(k.value) for k in cfg.keywords if k.arg!='task_type'}==dict(r=4,lora_alpha=8,lora_dropout=0.0,target_modules=['q_proj','v_proj'],bias='none'),'Original LoRA configuration changed')
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
    optimizer=[n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='AdamW']
    require(len(optimizer)==1,'Original optimizer changed')
    opts={k.arg:k.value for k in optimizer[0].keywords}
    require(ast.literal_eval(opts['weight_decay'])==0.0 and isinstance(opts['lr'],ast.Name) and opts['lr'].id=='LEARNING_RATE','Original optimizer settings changed')
    require(not any(isinstance(n.func,ast.Attribute) and 'scheduler' in ast.unparse(n.func).lower() for n in calls),'Original now uses a scheduler')
    require(all(s in text for s in ['torch.amp.GradScaler("cuda")','dtype=torch.float16','max_norm=1.0','"use_reentrant": False','[-100] * len(prompt_ids)','loss / group_size']),'Original training contract changed')
    return constants

def prepare_context():
    from transformers import AutoTokenizer
    from datasets import Dataset
    for name,expected in FROZEN_HASHES.items():require(sha(SETS/name)==expected,f'Frozen primary input changed: {name}')
    m=read(SETS/'manifest.json')
    inputs=dict(m['input_hashes'])
    inputs.update({str(SETS/name):expected for name,expected in FROZEN_HASHES.items()})
    inputs['build_primary_tokenmatched_sets_v1.py']=m['source_sha256']
    for path,expected in inputs.items():require(sha(path)==expected,f'Protected input changed: {path}')
    check_trainer(Path('train_verified_pilot.py').read_text())
    cache=Path.home()/'.cache/huggingface'
    parent=cache/'hub/models--Qwen--Qwen3-1.7B-Base/snapshots'
    require([p.name for p in parent.iterdir() if p.is_dir()]==[MODEL_REV],'Base snapshot inventory changed')
    snapshot=parent/MODEL_REV
    require(sha(snapshot/'model.safetensors')==WEIGHT_SHA,'Pinned base weight checksum differs')
    tok=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True)
    require(tok.eos_token_id is not None,'Missing EOS')
    arrow=cache/f'datasets/openai___gsm8k/main/0.0.0/{DATA_REV}/gsm8k-train.arrow'
    data=Dataset.from_file(str(arrow)).shuffle(seed=2026,keep_in_memory=True)
    prefix='Solve the math problems step by step. Finish each solution with #### followed by the numeric answer.\n\n'
    for ex in data.select(range(2)):
        solution=re.sub(r'<<[^<>]*>>','',ex['answer'])
        prefix+=f"Question: {ex['question']}\nSolution: {solution}\n\n"
    groups={}
    pattern=re.compile(r'(?m)^####[ \t]*(-?[\d,]+(?:\.\d+)?)[ \t]*$')
    for group in GROUPS:
        with (SETS/f'{group}.jsonl').open(encoding='utf-8') as f:rows=[json.loads(s) for s in f if s.strip()]
        require(len(rows)==len({r['dataset_index'] for r in rows})==m['group_stats'][group]['examples'],'Example count differs')
        examples=[]
        for r in rows:
            require(r['question']==data[r['dataset_index']]['question'],'Dataset question differs')
            match=pattern.search(r['solution'].strip())
            require(match and Decimal(match.group(1).replace(',',''))==Decimal(r['answer']),'Numerical verification failed')
            prompt=prefix+f"Question: {r['question']}\nSolution:"
            pids=tok.encode(prompt,add_special_tokens=False)
            tids=tok.encode(' '+r['solution'].strip()+'\n',add_special_tokens=False)
            ids=pids+tids+[tok.eos_token_id]
            labels=[-100]*len(pids)+tids+[tok.eos_token_id]
            require(len(ids)==r['full_sequence_tokens']<=1024 and len(tids)+1==r['supervised_tokens'],'Token count/sequence limit differs')
            examples.append(dict(dataset_index=r['dataset_index'],input_ids=ids,labels=labels,supervised_tokens=len(tids)+1))
        require(sum(r['supervised_tokens'] for r in examples)==m['group_stats'][group]['supervised_tokens_per_epoch'],'Supervised total differs')
        groups[group]=examples
    fingerprint=hashlib.sha256(json.dumps(dict(settings=SETTINGS,sets=FROZEN_HASHES,model_revision=MODEL_REV,weights=WEIGHT_SHA),sort_keys=True).encode()).hexdigest()
    return groups,inputs,snapshot,fingerprint

class UpdateCounter:
    """Counts optimizer executions, not scaler.step() return values."""
    def __init__(self,optimizer):
        self.attempted=0;self.successful=0
        self.handle=optimizer.register_step_post_hook(self._after_step)
    def _after_step(self,*args,**kwargs):self.successful+=1
    def step(self,scaler,optimizer):
        before=self.successful
        self.attempted+=1
        scaler.step(optimizer)
        scaler.update()
        require(self.successful in (before,before+1),'Unexpected optimizer hook count')
        return self.successful==before
    @property
    def skipped(self):return self.attempted-self.successful

def longest_examples(groups):
    unique={}
    for group in GROUPS:
        for ex in groups[group]:
            key=ex['dataset_index']
            if key in unique:require(unique[key]==ex,'Overlapping groups changed solution')
            unique[key]=ex
    return sorted(unique.values(),key=lambda ex:(-len(ex['input_ids']),ex['dataset_index']))[:12]

def verify_probe(path,fingerprint,source_hash):
    p=read(path)
    require(p['status']=='memory_probe_passed' and p['successful_optimizer_updates']==3,'Memory probe has not passed')
    require(p['training_fingerprint']==fingerprint and p['source_sha256']==source_hash,'Probe/settings mismatch')
    require(p['maximum_sequence_length']==849 and p['final_adapter_saved'] is False,'Unexpected probe coverage')

def execute(args):
    groups,protected,snapshot,fingerprint=prepare_context()
    if args.phase=='preflight':
        print('PREFLIGHT PASS: frozen base/data/settings; all groups validated; no GPU training or output files.',flush=True)
        return
    out=Path(args.output)
    check_output(out)
    source_hash=sha(__file__)
    if args.phase=='train':
        require(args.probe_report is not None,'Full training requires a successful matching memory-probe report')
        verify_probe(Path(args.probe_report),fingerprint,source_hash)
        if (args.group,args.seed)!=('hard',2026):require(args.additional_runs_confirmed,'Other eight runs require explicit later confirmation')
    out.mkdir(parents=True,exist_ok=False)
    manifest=dict(phase=args.phase,group=args.group if args.phase=='train' else '12 longest distinct across all groups',seed=args.seed,
                  paired_training_seeds=list(SEEDS),settings=SETTINGS,source_sha256=source_hash,input_hashes=protected,
                  training_fingerprint=fingerprint,model_revision=MODEL_REV,model_weight_sha256=WEIGHT_SHA,
                  probe_report=args.probe_report,probe_report_sha256=sha(args.probe_report) if args.probe_report else None,
                  restart_policy='fresh from base in a new folder; no intermediate checkpoints',
                  created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  versions={p:importlib.metadata.version(p) for p in ['torch','transformers','peft','datasets','accelerate']})
    json_file(out/'manifest.json',manifest)
    import torch
    try:
        require(torch.cuda.is_available(),'CUDA unavailable; stopped without changing settings')
        random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
        free,total=torch.cuda.mem_get_info()
        from transformers import AutoModelForCausalLM
        from peft import LoraConfig,TaskType,get_peft_model
        base=AutoModelForCausalLM.from_pretrained(str(snapshot),dtype=torch.float16,device_map='cuda:0',local_files_only=True)
        base.config.use_cache=False
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        config=LoraConfig(r=4,lora_alpha=8,lora_dropout=0.0,target_modules=['q_proj','v_proj'],bias='none',task_type=TaskType.CAUSAL_LM)
        model=get_peft_model(base,config);model.train()
        parameters=[p for p in model.parameters() if p.requires_grad]
        require(parameters and all('lora_' in n for n,p in model.named_parameters() if p.requires_grad),'Unexpected trainable base parameters')
        optimizer=torch.optim.AdamW(parameters,lr=1e-4,weight_decay=0.0)
        scaler=torch.amp.GradScaler('cuda')
        counter=UpdateCounter(optimizer)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
        started=time.perf_counter()
        examples=longest_examples(groups) if args.phase=='probe' else groups[args.group]
        epoch_losses=[];all_losses=[];skipped_records=[]
        max_seen=0;exposures=0;attempt_tokens=0;successful_tokens=0
        with (out/'optimizer_steps.jsonl').open('x',encoding='utf-8') as log:
            epoch=0;probe_done=False
            while args.phase=='probe' or epoch<2:
                epoch+=1
                order=list(range(len(examples)))
                if args.phase=='train':random.Random(args.seed+epoch).shuffle(order)
                losses=[];pending=[]
                for j,index in enumerate(order):
                    ex=examples[index]
                    max_seen=max(max_seen,len(ex['input_ids']))
                    inputs=torch.tensor([ex['input_ids']],dtype=torch.long,device='cuda')
                    labels=torch.tensor([ex['labels']],dtype=torch.long,device='cuda')
                    group_start=(j//4)*4;group_size=min(4,len(order)-group_start)
                    with torch.autocast('cuda',dtype=torch.float16):
                        output=model(input_ids=inputs,attention_mask=torch.ones_like(inputs),labels=labels)
                        loss=output.loss
                    require(torch.isfinite(loss).item(),f'Non-finite loss at epoch {epoch}, example {j+1}')
                    value=loss.item();losses.append(value);all_losses.append(value)
                    pending.append(ex);exposures+=1
                    scaler.scale(loss/group_size).backward()
                    if (j+1)%4==0 or j+1==len(order):
                        scaler.unscale_(optimizer)
                        norm=torch.nn.utils.clip_grad_norm_(parameters,max_norm=1.0)
                        scale_before=scaler.get_scale()
                        skipped=counter.step(scaler,optimizer)
                        tokens=sum(e['supervised_tokens'] for e in pending)
                        attempt_tokens+=tokens
                        if not skipped:
                            successful_tokens+=tokens
                            require(torch.isfinite(norm).item(),'Non-finite gradient norm despite a non-skipped optimizer update')
                            require(all(torch.isfinite(p).all().item() for p in parameters),'Non-finite adapter parameters after optimizer step')
                        record=dict(attempt=counter.attempted,successful_update_total=counter.successful,skipped=skipped,
                                    epoch=epoch,example_positions=[j+1-len(pending)+k+1 for k in range(len(pending))],
                                    dataset_indices=[e['dataset_index'] for e in pending],supervised_tokens=tokens,
                                    scale_before=scale_before,scale_after=scaler.get_scale(),
                                    gradient_norm=float(norm.item()) if torch.isfinite(norm).item() else None,
                                    gradient_norm_finite=bool(torch.isfinite(norm).item()),learning_rate=optimizer.param_groups[0]['lr'])
                        require(record['learning_rate']==1e-4,'Learning rate drifted from original constant schedule')
                        log.write(json.dumps(record,allow_nan=False)+'\n');log.flush()
                        if skipped:skipped_records.append(record);print(f'FP16 SKIP at attempt {counter.attempted}: scale {scale_before} -> {scaler.get_scale()}',flush=True)
                        optimizer.zero_grad(set_to_none=True);pending=[]
                        if args.phase=='probe':
                            if counter.successful>=3:probe_done=True
                            elif counter.attempted>=12:raise RuntimeError('Memory probe did not achieve three successful updates within 12 attempts; stopped')
                    del output,loss,inputs,labels
                    if args.phase=='train' and ((j+1)%20==0 or j+1==len(order)):
                        print(f'Epoch {epoch}/2 | {j+1}/{len(order)} examples | mean loss {sum(losses)/len(losses):.4f} | attempted {counter.attempted}, successful {counter.successful}, skipped {counter.skipped}',flush=True)
                    if probe_done:break
                epoch_losses.append(sum(losses)/len(losses))
                if probe_done:break
        torch.cuda.synchronize()
        require(counter.attempted==counter.successful+counter.skipped,'Update counters disagree')
        if args.phase=='train':
            require(epoch==2 and exposures==2*len(examples) and counter.attempted==2*math.ceil(len(examples)/4),'Incomplete two-epoch training schedule')
        for path,expected in protected.items():require(sha(path)==expected,f'Protected input changed during training: {path}')
        metrics=dict(status='memory_probe_passed' if args.phase=='probe' else 'completed_two_epochs',
                     phase=args.phase,group=args.group if args.phase=='train' else 'longest_examples',seed=args.seed,
                     source_sha256=source_hash,training_fingerprint=fingerprint,
                     example_count=len(examples),example_exposures=exposures,maximum_sequence_length=max_seen,
                     epoch_mean_losses=epoch_losses,mean_loss=sum(all_losses)/len(all_losses),
                     attempted_optimizer_updates=counter.attempted,successful_optimizer_updates=counter.successful,
                     skipped_optimizer_updates=counter.skipped,skipped_steps=skipped_records,
                     supervised_tokens_seen=attempt_tokens,supervised_tokens_in_successful_updates=successful_tokens,
                     initial_free_gpu_gib=free/1024**3,total_gpu_gib=total/1024**3,
                     peak_reserved_gpu_gib=torch.cuda.max_memory_reserved()/1024**3,
                     peak_allocated_gpu_gib=torch.cuda.max_memory_allocated()/1024**3,
                     elapsed_training_minutes=(time.perf_counter()-started)/60,
                     gpu=torch.cuda.get_device_name(0),settings=SETTINGS,all_protected_hashes_match=True,
                     final_adapter_saved=args.phase=='train',probe_selected_indices=[e['dataset_index'] for e in examples] if args.phase=='probe' else None)
        if args.phase=='train':
            require(epoch==2,'Saving only allowed after epoch two')
            temp=out/'adapter_final.part'
            model.save_pretrained(str(temp),safe_serialization=True)
            require((temp/'adapter_model.safetensors').is_file() and (temp/'adapter_config.json').is_file(),'Final adapter save incomplete')
            adapter_config=read(temp/'adapter_config.json')
            require(adapter_config['r']==4 and adapter_config['lora_alpha']==8 and sorted(adapter_config['target_modules'])==['q_proj','v_proj'],'Final adapter configuration mismatch')
            require(not (out/'adapter_final').exists(),'Final adapter already exists')
            temp.rename(out/'adapter_final')
            metrics['adapter_files_sha256']={p.name:sha(p) for p in (out/'adapter_final').iterdir() if p.is_file()}
        json_file(out/'completed.json',metrics)
        print(json.dumps(metrics,indent=2,allow_nan=False),flush=True)
        counter.handle.remove()
        del model,base,optimizer,scaler,parameters
        import gc
        gc.collect();torch.cuda.empty_cache()
    except BaseException as exc:
        failure=dict(status='failed_or_interrupted',exception_type=type(exc).__name__,message=str(exc),
                     no_automatic_setting_changes=True,restart_policy='fresh base, new output folder',
                     out_of_memory=isinstance(exc,torch.cuda.OutOfMemoryError),final_adapter_complete=(out/'completed.json').exists())
        if torch.cuda.is_initialized():
            failure['peak_reserved_gpu_gib']=torch.cuda.max_memory_reserved()/1024**3
            failure['peak_allocated_gpu_gib']=torch.cuda.max_memory_allocated()/1024**3
        if not (out/'failed.json').exists():json_file(out/'failed.json',failure)
        print(json.dumps(failure,indent=2),flush=True)
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['preflight','probe','train'],required=True)
    p.add_argument('--group',choices=GROUPS,default='hard')
    p.add_argument('--seed',type=int,choices=SEEDS,default=2026)
    p.add_argument('--output',default='results/primary_lora_training_v1/hard_seed_2026')
    p.add_argument('--probe-report')
    p.add_argument('--additional-runs-confirmed',action='store_true')
    args=p.parse_args()
    if args.phase=='preflight':execute(args)
    else:
        import fcntl
        lock=Path('results/primary_lora_training_v1.lock')
        with lock.open('a') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            execute(args)
