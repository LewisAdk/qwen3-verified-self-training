#!/usr/bin/env python3
"""Supply frozen gold inputs to the unchanged primary trainer execution function."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_DATASETS_OFFLINE']='1'
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from primary_training_observer_v1 import ObservedClock

TRAINER_SHA='d8381ab86bb6020293121dc5dd32dac113901fbb8ddac8540fe4d777cea5a1b9'
GOLD_SHA='259f2b7bd8ad42bde45bf3f41a7a2830f8209f4078221adf7305d0bb47b371ed'
PLAN_SHA='16a7bc5c678e44f17fc1a0128660cccf92eb86f9f7aa8b85c4b9982bbb585d67'
SETS=Path('results/training_sets_gold_control_v1')
BASE=Path('results/gold_control_training_v2')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def require(ok,message):
    if not ok:raise RuntimeError(message)

def module():
    require(sha('train_primary_lora_v1.py')==TRAINER_SHA,'Immutable trainer changed')
    spec=importlib.util.spec_from_file_location('unchanged_gold_training_engine','train_primary_lora_v1.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m

def reference_helper():
    # Extract only the pure cleaner. Never execute the CPU preparer's environment assignments.
    import ast
    import re
    from decimal import Decimal
    tree=ast.parse(Path('build_gold_control_v1.py').read_text())
    nodes=[]
    for n in tree.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'ANNOTATION','ANSWER_LINE'} for t in n.targets):nodes.append(n)
        elif isinstance(n,ast.FunctionDef) and n.name=='clean_reference':nodes.append(n)
    require(len(nodes)==3,'Incomplete reference-cleaning contract')
    namespace=dict(re=re,Decimal=Decimal,require=require)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<reference-cleaner-only>','exec'),namespace)
    return namespace['clean_reference']

def gold_context(engine):
    _,original_protected,snapshot,_=engine.prepare_context()
    from datasets import Dataset
    from transformers import AutoTokenizer
    clean_reference=reference_helper()
    manifest=json.loads((SETS/'manifest.json').read_text())
    require(sha(SETS/'gold.jsonl')==GOLD_SHA==manifest['gold_dataset_sha256'],'Frozen gold data changed')
    require(sha(SETS/'gold_control_plan_2026-10-13.md')==PLAN_SHA==manifest['plan_sha256'],'Frozen plan changed')
    require(manifest['settings']==engine.SETTINGS,'Gold settings differ')
    protected={**original_protected,**manifest['input_hashes']}
    for p in [SETS/'manifest.json',SETS/'gold.jsonl',SETS/'gold_control_plan_2026-10-13.md',Path(__file__),Path('primary_training_observer_v1.py')]:protected[str(p)]=sha(p)
    for p,digest in protected.items():require(sha(p)==digest,f'Protected input changed: {p}')
    tok=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True)
    arrow=Path.home()/f".cache/huggingface/datasets/openai___gsm8k/main/0.0.0/{manifest['dataset_revision']}/gsm8k-train.arrow"
    ds=Dataset.from_file(str(arrow)).shuffle(seed=2026,keep_in_memory=True)
    gold=[json.loads(x) for x in (SETS/'gold.jsonl').read_text().splitlines()]
    hard=[json.loads(x) for x in Path(manifest['hard_source']).read_text().splitlines()]
    require(len(gold)==300 and [r['dataset_index'] for r in gold]==[r['dataset_index'] for r in hard],'Gold problem identity/order differs')
    examples=[]
    for r in gold:
        reference=ds[r['dataset_index']]
        require(r['question']==reference['question'] and r['solution']==clean_reference(reference['answer'],r['answer']),'Reference supervision differs')
        prompt=manifest['prompt']+f"Question: {r['question']}\nSolution:"
        pids=tok.encode(prompt,add_special_tokens=False);tids=tok.encode(' '+r['solution'].strip()+'\n',add_special_tokens=False)
        ids=pids+tids+[tok.eos_token_id];labels=[-100]*len(pids)+tids+[tok.eos_token_id]
        require(len(ids)==r['full_sequence_tokens']<=1024 and len(tids)+1==r['supervised_tokens'],'Gold tokenization differs')
        examples.append(dict(dataset_index=r['dataset_index'],input_ids=ids,labels=labels,supervised_tokens=len(tids)+1))
    require(sum(e['supervised_tokens'] for e in examples)==32206,'Gold token total differs')
    fingerprint=hashlib.sha256(json.dumps(dict(settings=engine.SETTINGS,gold=GOLD_SHA,plan=PLAN_SHA,entrypoint=sha(__file__),model_revision=manifest['model_revision'],weights=manifest['model_weight_sha256']),sort_keys=True).encode()).hexdigest()
    return {'gold':examples},protected,snapshot,fingerprint

def longest_gold(groups):return sorted(groups['gold'],key=lambda e:(-len(e['input_ids']),e['dataset_index']))[:12]

def verify_gold_probe(path,fingerprint,source_hash):
    p=json.loads(Path(path).read_text())
    require(p['status']=='memory_probe_passed' and p['successful_optimizer_updates']==3 and p['skipped_optimizer_updates']==0,'Gold memory probe did not pass cleanly')
    require(p['training_fingerprint']==fingerprint and p['source_sha256']==source_hash==TRAINER_SHA,'Gold probe fingerprint differs')
    require(p['maximum_sequence_length']==847 and p['final_adapter_saved'] is False,'Wrong gold probe coverage')

def configured_engine():
    m=module();original=m.prepare_context
    # Adapt input/probe validation only. execute, UpdateCounter and all training ops remain unchanged.
    class Context:
        prepare_context=staticmethod(original)
        SETTINGS=m.SETTINGS
    m.prepare_context=lambda:gold_context(Context)
    m.longest_examples=longest_gold
    m.verify_probe=verify_gold_probe
    return m

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['preflight','probe','train'],required=True);p.add_argument('--seed',type=int,choices=[2026,2027,2028],default=2026);p.add_argument('--output',required=True);args=p.parse_args()
    m=configured_engine()
    if args.phase=='preflight':
        groups,protected,snapshot,fingerprint=m.prepare_context()
        print(f'GOLD TRAINING PREFLIGHT PASS: 300 exact references; 32,206 tokens; max 847; {len(protected)} protected files; immutable training operations.',flush=True);return
    m.time=ObservedClock()
    call=SimpleNamespace(phase=args.phase,group='gold',seed=args.seed,output=args.output,
                         probe_report=str(BASE/'memory_probe_seed_2026/completed.json') if args.phase=='train' else None,additional_runs_confirmed=True)
    import fcntl
    lock=BASE.with_suffix('.lock')
    with lock.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);m.execute(call)
    require(m.time.calls==2,'Unexpected immutable training timer calls')

if __name__=='__main__':main()
