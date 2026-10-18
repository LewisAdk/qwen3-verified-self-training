#!/usr/bin/env python3
"""Frozen, resumable RQ2 evaluation; saved base reused after exact validation."""
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
from pathlib import Path
import re
import time
from decimal import Decimal
import gsm_symbolic_expanded_dev_v1 as old

GROUPS=('hard','random','easy')
SEEDS=(2026,2027,2028)
MODELS=[f'{g}_seed_{s}' for g in GROUPS for s in SEEDS]
BASE_RUN=Path('results/gsm_symbolic_expanded_dev_v1')
TRAIN=Path('results/primary_lora_training_v1')
PLAN=Path('results/rq2_analysis_plan_2026-10-12.md')
MODEL_REV='ea980cb0a6c2ae4b936e82123acc929f1cec04c1'
WEIGHT_SHA='6df85b39330e5a425ee36253d0f894e4387e4f0a15b9c53cb467d668e6b3a841'
TRAINER_SHA='d8381ab86bb6020293121dc5dd32dac113901fbb8ddac8540fe4d777cea5a1b9'
require=old.require
digest=old.digest
publish=old.publish

def selected_rows(ds,variant,split):
    selected=[dict(r) for r in ds if r['id'] in old.TEMPLATES[variant] and r['instance'] in range(5,15)]
    selected.sort(key=lambda r:(r['id'],r['instance']))
    require(len(selected)==200 and {(r['id'],r['instance']) for r in selected}=={(t,i) for t in old.TEMPLATES[variant] for i in range(5,15)},'Wrong evaluation rows')
    require(all(r['original_id'] in split['development_original_ids'] for r in selected),'Reserved family selected')
    return selected

def validate_batch(obj,expected,key,variant,parse):
    old.check_batch(obj,expected,key,variant,parse)
    require(all(type(r['tokens']) is int and 1<=r['tokens']<=384 for r in obj['records']),'Invalid generated token count')

def prepare():
    import torch
    from datasets import Dataset
    from transformers import StoppingCriteria
    split,hashes,tree=old.preflight()
    manifest=json.loads((BASE_RUN/'manifest.json').read_text())
    require(manifest['source_sha256']==digest(old.__file__),'Saved evaluator source differs')
    require(manifest['split']==split and manifest['input_hashes']==hashes,'Saved base split/contract differs')
    require(manifest['model_revision']==MODEL_REV and manifest['dataset_revision']==old.REVISION and manifest['gsm_revision']==old.GSM_REVISION,'Saved base revisions differ')
    settings=dict(batch_size=8,max_new_tokens=384,do_sample=False,dtype='float16',evaluation_seed=2026)
    require(all(manifest[k]==v for k,v in settings.items()),'Saved base settings differ')
    versions={p:importlib.metadata.version(p) for p in manifest['versions']}
    require(versions==manifest['versions'],'Evaluator library versions changed')
    require(PLAN.is_file(),'RQ2 plan must be saved first')
    protected={str(BASE_RUN/'manifest.json'):digest(BASE_RUN/'manifest.json'),str(PLAN):digest(PLAN),str(Path(old.__file__).resolve()):digest(old.__file__)}
    protected.update(json.loads((TRAIN/'hard_seed_2026/manifest.json').read_text())['input_hashes'])
    require(digest('train_primary_lora_v1.py')==TRAINER_SHA,'Training source changed')
    protected['train_primary_lora_v1.py']=TRAINER_SHA
    for path,expected in protected.items():require(digest(path)==expected,f'Protected input changed: {path}')
    cache=Path.home()/'.cache/huggingface'
    snapshots=cache/'hub/models--Qwen--Qwen3-1.7B-Base/snapshots'
    require([p.name for p in snapshots.iterdir() if p.is_dir()]==[MODEL_REV],'Base snapshot inventory differs')
    snapshot=snapshots/MODEL_REV
    require(digest(snapshot/'model.safetensors')==WEIGHT_SHA,'Pinned weights differ')
    for p in snapshot.iterdir():
        if p.is_file():protected[str(p)]=digest(p)
    gsm_path=cache/f'datasets/openai___gsm8k/main/0.0.0/{old.GSM_REVISION}/gsm8k-train.arrow'
    gsm=Dataset.from_file(str(gsm_path)).shuffle(seed=2026,keep_in_memory=True)
    protected[str(gsm_path)]=digest(gsm_path)
    prompt_nodes=[n for n in tree.body if (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='prefix' for t in n.targets)) or (isinstance(n,ast.For) and any(isinstance(x,ast.AugAssign) and isinstance(x.target,ast.Name) and x.target.id=='prefix' for x in ast.walk(n)))]
    prompt_ns=dict(gsm=gsm,re=re)
    exec(compile(ast.Module(body=prompt_nodes,type_ignores=[]),'<saved-prefix>','exec'),prompt_ns)
    prefix=prompt_ns['prefix'];require(prefix==manifest['prefix'],'Prompt changed')
    ns=dict(torch=torch,re=re,Decimal=Decimal,StoppingCriteria=StoppingCriteria)
    old.exact_helpers(tree,ns)
    rows={};mapping={};base={}
    for variant in old.TEMPLATES:
        path=cache/f'datasets/apple___gsm-symbolic/{variant}/0.0.0/{old.REVISION}/gsm-symbolic-test.arrow'
        ds=Dataset.from_file(str(path));protected[str(path)]=digest(path)
        mapping[variant]={}
        for r in ds:require(mapping[variant].setdefault(r['id'],r['original_id'])==r['original_id'],'Family mapping changed')
        rows[variant]=selected_rows(ds,variant,split)
        selected_mapping={str(t):mapping[variant][t] for t in old.TEMPLATES[variant]}
        require(selected_mapping==manifest['family_mapping'][variant],'Saved family mapping differs')
        paths={BASE_RUN/'base'/variant/f'batch_{i:03d}.json' for i in range(0,200,8)}
        require(set((BASE_RUN/'base'/variant).glob('batch_*.json'))==paths,'Saved base incomplete/extra batches')
        for i in range(0,200,8):
            path=BASE_RUN/'base'/variant/f'batch_{i:03d}.json'
            obj=json.loads(path.read_text());validate_batch(obj,rows[variant][i:i+8],'base',variant,ns['parse'])
            protected[str(path)]=digest(path);base[(variant,i)]=obj
    require(hashlib.sha256(old.canonical(rows).encode()).hexdigest()==manifest['rows_sha256'],'Question row checksum differs')
    exposed={mapping[v][t] for v in old.TEMPLATES for t in old.TEMPLATES[v]}
    all_families={x for m in mapping.values() for x in m.values()}
    require(exposed==set(split['development_original_ids']) and all_families-exposed==set(split['reserved_original_ids']),'Family partition changed')
    gen_path=BASE_RUN/'base/generation_config.json'
    gen=json.loads(gen_path.read_text());protected[str(gen_path)]=digest(gen_path)
    training={}
    for key in MODELS:
        folder=TRAIN/key;m=json.loads((folder/'completed.json').read_text())
        group,seed=key.split('_seed_')
        require(m['status']=='completed_two_epochs' and m['group']==group and m['seed']==int(seed),'Wrong final training result')
        require(m['source_sha256']==TRAINER_SHA and m['final_adapter_saved'] and m['skipped_optimizer_updates']==0,'Invalid final training run')
        require(m['settings']==json.loads((TRAIN/'hard_seed_2026/completed.json').read_text())['settings'],'Training settings differ')
        for name in ['manifest.json','completed.json','optimizer_steps.jsonl']:protected[str(folder/name)]=digest(folder/name)
        for name,expected in m['adapter_files_sha256'].items():
            path=folder/'adapter_final'/name;require(digest(path)==expected,f'Final adapter checksum differs: {key}');protected[str(path)]=expected
        cfg=json.loads((folder/'adapter_final/adapter_config.json').read_text())
        require(Path(cfg['base_model_name_or_path']).resolve()==snapshot.resolve() and cfg['peft_type']=='LORA','Wrong adapter base')
        require(cfg['r']==4 and cfg['lora_alpha']==8 and cfg['lora_dropout']==0 and sorted(cfg['target_modules'])==['q_proj','v_proj'],'Adapter settings differ')
        training[key]={k:m[k] for k in ['example_count','supervised_tokens_seen','attempted_optimizer_updates','successful_optimizer_updates','skipped_optimizer_updates','training_fingerprint']}
    metadata=dict(schema=1,source_sha256=digest(__file__),input_hashes=protected,rq2_plan_sha256=digest(PLAN),model=old.MODEL,
                  model_revision=MODEL_REV,model_weight_sha256=WEIGHT_SHA,dataset_revision=old.REVISION,gsm_revision=old.GSM_REVISION,
                  prefix=prefix,split=split,family_mapping=manifest['family_mapping'],rows_sha256=manifest['rows_sha256'],
                  versions=versions,models=MODELS,training=training,base_reused_from=str(BASE_RUN),base_generation_config=gen,**settings)
    return metadata,rows,base,ns,snapshot

def run(args):
    metadata,rows,base,ns,snapshot=prepare()
    if args.preflight:
        print('RQ2 PREFLIGHT PASS: exact 400 rows, 37 development families, 63 reserved untouched, matching saved base contract, nine final adapters, pinned snapshot, dated plan. No generation.',flush=True)
        return
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,StoppingCriteriaList
    from peft import PeftModel
    require(torch.cuda.is_available(),'CUDA unavailable')
    out=Path(args.output)
    require(out.resolve().is_relative_to(Path('results').resolve()) and out.resolve()!=Path('results').resolve() and out.resolve()!=BASE_RUN.resolve(),'Output must be a new separate results folder')
    if out.exists():require((out/'manifest.json').exists(),'Existing folder has no frozen manifest; choose a new output')
    out.mkdir(parents=True,exist_ok=True);old.freeze(out/'manifest.json',metadata)
    for (variant,start),obj in base.items():old.freeze(out/'base'/variant/f'batch_{start:03d}.json',obj)
    old.freeze(out/'base/generation_config.json',metadata['base_generation_config'])
    tok=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True);tok.padding_side='left'
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    torch.manual_seed(2026)
    for key in MODELS:
        missing=[]
        for variant in rows:
            directory=out/key/variant
            expected={directory/f'batch_{i:03d}.json' for i in range(0,200,8)}
            require(set(directory.glob('batch_*.json')).issubset(expected),'Unexpected saved batches')
            for start in range(0,200,8):
                path=directory/f'batch_{start:03d}.json'
                if path.exists():validate_batch(json.loads(path.read_text()),rows[variant][start:start+8],key,variant,ns['parse'])
                else:missing.append((variant,start,path))
        if not missing:print(key+': already complete',flush=True);continue
        model=None
        try:
            torch.cuda.reset_peak_memory_stats()
            model=AutoModelForCausalLM.from_pretrained(str(snapshot),dtype=torch.float16,device_map='cuda:0',local_files_only=True)
            model=PeftModel.from_pretrained(model,str(TRAIN/key/'adapter_final'),is_trainable=False);model.eval()
            require(model.generation_config.to_dict()==metadata['base_generation_config'],'Generation configuration differs from saved base')
            old.freeze(out/key/'generation_config.json',model.generation_config.to_dict())
            for variant,start,path in missing:
                batch=rows[variant][start:start+8]
                prompts=[metadata['prefix']+f"Question: {r['question']}\nSolution:" for r in batch]
                inputs=tok(prompts,padding=True,return_tensors='pt').to('cuda');prompt_len=inputs['input_ids'].shape[1]
                stopper=ns['StopAnswers'](tok,prompt_len,len(batch));began=time.perf_counter()
                with torch.inference_mode():
                    output=model.generate(**inputs,max_new_tokens=384,do_sample=False,stopping_criteria=StoppingCriteriaList([stopper]),pad_token_id=tok.pad_token_id)
                generated=output[:,prompt_len:];records=[]
                for j,r in enumerate(batch):
                    tokens=generated[j].tolist();count=stopper.lengths[j]
                    if count is None:count=tokens.index(tok.eos_token_id)+1 if tok.eos_token_id in tokens else len(tokens)
                    response=tok.decode(tokens[:count],skip_special_tokens=True)
                    prediction,gold=ns['parse'](response),ns['parse'](r['answer'])
                    records.append(dict(model=key,variant=variant,id=r['id'],instance=r['instance'],original_id=r['original_id'],question=r['question'],gold=gold,prediction=prediction,correct=prediction==gold,tokens=count,response=response))
                obj=dict(records=records,elapsed_seconds=time.perf_counter()-began,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
                validate_batch(obj,batch,key,variant,ns['parse']);publish(path,obj)
                print(f'{key} {variant}: {start+8}/200 saved',flush=True)
                del inputs,output,generated
        except BaseException as exc:
            failure=out/key/f'failure_{time.time_ns()}.json'
            publish(failure,dict(exception_type=type(exc).__name__,message=str(exc),out_of_memory=isinstance(exc,torch.cuda.OutOfMemoryError),no_settings_changed=True,completed_batches_preserved=True))
            raise
        finally:
            if model is not None:del model
            import gc
            gc.collect();torch.cuda.empty_cache()
        print(key+': all 400 questions saved; no analysis yet',flush=True)
    for path,expected in metadata['input_hashes'].items():require(digest(path)==expected,f'Protected input changed: {path}')
    old.freeze(out/'evaluation_completed.json',dict(status='nine_adapters_complete_base_reused',models=MODELS,questions_per_model=400,source_sha256=metadata['source_sha256'],all_protected_hashes_match=True))
    print('RQ2 EVALUATION COMPLETE: 3,600 new outputs; saved base reused. Ready for planned CPU analysis.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--preflight',action='store_true');p.add_argument('--output',default='results/gsm_symbolic_rq2_primary_v1');args=p.parse_args()
    if args.preflight:run(args)
    else:
        import fcntl
        lock=Path(args.output+'.lock');lock.parent.mkdir(parents=True,exist_ok=True)
        with lock.open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);run(args)
