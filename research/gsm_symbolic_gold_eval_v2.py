#!/usr/bin/env python3
"""Gold-only generation; validated base and hard results reused without regeneration."""
import argparse
import json
from pathlib import Path
import time
import gsm_symbolic_rq2_eval_v1 as prior
from train_gold_control_v2 import GOLD_SHA,PLAN_SHA,TRAINER_SHA,sha,require

MODELS=[f'gold_seed_{s}' for s in [2026,2027,2028]]
HARD_MODELS=[f'hard_seed_{s}' for s in [2026,2027,2028]]
TRAIN=Path('results/gold_control_training_v2')
RUN=Path('results/gsm_symbolic_gold_control_v2')
HARD_RUN=Path('results/gsm_symbolic_rq2_primary_v1')

def prepare():
    metadata,rows,base,ns,snapshot=prior.prepare()
    protected=dict(metadata['input_hashes']);saved_hard={}
    gold_manifest=json.loads(Path('results/training_sets_gold_control_v1/manifest.json').read_text())
    for path,digest in gold_manifest['input_hashes'].items():require(sha(path)==digest,f'Protected file changed: {path}')
    protected.update(gold_manifest['input_hashes'])
    require(gold_manifest['gold_dataset_sha256']==GOLD_SHA and gold_manifest['plan_sha256']==PLAN_SHA,'Gold data/plan differs')
    for name in ['gold.jsonl','manifest.json','gold_control_plan_2026-10-13.md']:
        path=Path('results/training_sets_gold_control_v1')/name;protected[str(path)]=sha(path)
    for key in HARD_MODELS:
        config=HARD_RUN/key/'generation_config.json'
        require(json.loads(config.read_text())==metadata['base_generation_config'],'Saved hard generation config differs');protected[str(config)]=sha(config)
        for v in rows:
            paths={HARD_RUN/key/v/f'batch_{i:03d}.json' for i in range(0,200,8)}
            require(set((HARD_RUN/key/v).glob('batch_*.json'))==paths,'Saved hard batches incomplete/extra')
            for start in range(0,200,8):
                path=HARD_RUN/key/v/f'batch_{start:03d}.json';obj=json.loads(path.read_text())
                prior.validate_batch(obj,rows[v][start:start+8],key,v,ns['parse']);protected[str(path)]=sha(path);saved_hard[(key,v,start)]=obj
    for key in MODELS:
        folder=TRAIN/key;m=json.loads((folder/'completed.json').read_text())
        require(m['status']=='completed_two_epochs' and m['group']=='gold' and m['seed']==int(key[-4:]),'Wrong gold training result')
        require(m['source_sha256']==TRAINER_SHA and m['settings']==gold_manifest['settings'] and m['skipped_optimizer_updates']==0,'Gold training contract differs')
        require(m['attempted_optimizer_updates']==m['successful_optimizer_updates']==150 and m['supervised_tokens_seen']==64412,'Incomplete gold schedule')
        for name in ['manifest.json','completed.json','optimizer_steps.jsonl']:protected[str(folder/name)]=sha(folder/name)
        for name,digest in m['adapter_files_sha256'].items():
            path=folder/'adapter_final'/name;require(sha(path)==digest,'Gold adapter changed');protected[str(path)]=digest
        cfg=json.loads((folder/'adapter_final/adapter_config.json').read_text())
        require(Path(cfg['base_model_name_or_path']).resolve()==snapshot.resolve() and cfg['r']==4 and cfg['lora_alpha']==8 and cfg['lora_dropout']==0 and sorted(cfg['target_modules'])==['q_proj','v_proj'],'Wrong gold adapter config')
    metadata=dict(metadata,source_sha256=sha(__file__),input_hashes=protected,models=MODELS,reused_hard_models=HARD_MODELS,gold_plan_sha256=PLAN_SHA,
                  gold_dataset_sha256=GOLD_SHA,supervised_control=True,hard_reused_from=str(HARD_RUN),training={},evaluation_adapter_wall_limit_minutes=26)
    return metadata,rows,base,saved_hard,ns,snapshot

def run(args):
    metadata,rows,base,hard,ns,snapshot=prepare()
    if args.preflight:print('GOLD EVAL PREFLIGHT PASS: saved base/hard exact 400 rows; three final gold adapters; same contract; no generation.',flush=True);return
    out=Path(args.output)
    require(out.resolve().is_relative_to(Path('results').resolve()) and out.resolve()!=Path('results').resolve(),'Unsafe output')
    if out.exists():require((out/'manifest.json').exists(),'Existing output lacks frozen manifest')
    out.mkdir(parents=True,exist_ok=True);prior.old.freeze(out/'manifest.json',metadata)
    for (v,start),obj in base.items():prior.old.freeze(out/'base'/v/f'batch_{start:03d}.json',obj)
    prior.old.freeze(out/'base/generation_config.json',metadata['base_generation_config'])
    for (key,v,start),obj in hard.items():prior.old.freeze(out/key/v/f'batch_{start:03d}.json',obj)
    for key in HARD_MODELS:prior.old.freeze(out/key/'generation_config.json',metadata['base_generation_config'])
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,StoppingCriteriaList
    from peft import PeftModel
    require(torch.cuda.is_available(),'CUDA unavailable')
    torch.manual_seed(2026)
    tok=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True);tok.padding_side='left'
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    key=args.model;missing=[]
    for v in rows:
        directory=out/key/v;paths={directory/f'batch_{i:03d}.json' for i in range(0,200,8)}
        require(set(directory.glob('batch_*.json')).issubset(paths),'Unexpected saved batches')
        for start in range(0,200,8):
            path=directory/f'batch_{start:03d}.json'
            if path.exists():prior.validate_batch(json.loads(path.read_text()),rows[v][start:start+8],key,v,ns['parse'])
            else:missing.append((v,start,path))
    if not missing:print(key+': already complete',flush=True);return
    started=time.perf_counter();print('EVAL_TIMER_START '+json.dumps(dict(monotonic_seconds=started)),flush=True)
    model=None
    try:
        torch.cuda.reset_peak_memory_stats()
        model=AutoModelForCausalLM.from_pretrained(str(snapshot),dtype=torch.float16,device_map='cuda:0',local_files_only=True)
        model=PeftModel.from_pretrained(model,str(TRAIN/key/'adapter_final'),is_trainable=False);model.eval()
        require(model.generation_config.to_dict()==metadata['base_generation_config'],'Gold generation config differs')
        prior.old.freeze(out/key/'generation_config.json',model.generation_config.to_dict())
        for v,start,path in missing:
            batch=rows[v][start:start+8];prompts=[metadata['prefix']+f"Question: {r['question']}\nSolution:" for r in batch]
            inputs=tok(prompts,padding=True,return_tensors='pt').to('cuda');prompt_len=inputs['input_ids'].shape[1]
            stopper=ns['StopAnswers'](tok,prompt_len,len(batch));began=time.perf_counter()
            with torch.inference_mode():
                output=model.generate(**inputs,max_new_tokens=384,do_sample=False,stopping_criteria=StoppingCriteriaList([stopper]),pad_token_id=tok.pad_token_id)
            generated=output[:,prompt_len:];records=[]
            for j,r in enumerate(batch):
                tokens=generated[j].tolist();count=stopper.lengths[j]
                if count is None:count=tokens.index(tok.eos_token_id)+1 if tok.eos_token_id in tokens else len(tokens)
                response=tok.decode(tokens[:count],skip_special_tokens=True);prediction=ns['parse'](response);gold=ns['parse'](r['answer'])
                records.append(dict(model=key,variant=v,id=r['id'],instance=r['instance'],original_id=r['original_id'],question=r['question'],gold=gold,prediction=prediction,correct=prediction==gold,tokens=count,response=response))
            obj=dict(records=records,elapsed_seconds=time.perf_counter()-began,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
            prior.validate_batch(obj,batch,key,v,ns['parse']);prior.publish(path,obj)
            print(f'{key} {v}: {start+8}/200 saved',flush=True);del inputs,output,generated
        elapsed=time.perf_counter()-started
        print('EVAL_TIMER_END '+json.dumps(dict(monotonic_seconds=time.perf_counter())),flush=True)
        prior.old.freeze(out/key/'completed.json',dict(model=key,questions=400,elapsed_adapter_seconds=elapsed,source_sha256=metadata['source_sha256']))
    except BaseException as exc:
        prior.publish(out/key/f'failure_{time.time_ns()}.json',dict(exception_type=type(exc).__name__,message=str(exc),out_of_memory=isinstance(exc,torch.cuda.OutOfMemoryError),settings_unchanged=True));raise
    finally:
        if model is not None:del model
        import gc
        gc.collect();torch.cuda.empty_cache()
    for p,digest in metadata['input_hashes'].items():require(sha(p)==digest,f'Protected input changed: {p}')
    print(key+': evaluation complete; no analysis yet',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--preflight',action='store_true');p.add_argument('--model',choices=MODELS,default=MODELS[0]);p.add_argument('--output',default=str(RUN));args=p.parse_args()
    if args.preflight:run(args)
    else:
        import fcntl
        with Path(args.output+'.lock').open('a') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);run(args)
