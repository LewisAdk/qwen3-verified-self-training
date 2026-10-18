#!/usr/bin/env python3
"""Audit formatting-sensitive verdicts without changing any saved results."""
import argparse
import ast
from decimal import Decimal
import json
from pathlib import Path
import re
import statistics
from gsm_symbolic_expanded_dev_v1 import ADAPTERS, TEMPLATES, canonical, digest, publish, require
from gsm_symbolic_cluster_analysis_v1 import parse, validate, summarize

def numeric_correct(record):
    prediction = record['prediction']
    require(record['gold'] is not None, 'Missing gold')
    return prediction is not None and Decimal(prediction) == Decimal(record['gold'])

def original_parser(path, function):
    tree=ast.parse(Path(path).read_text(encoding='utf-8'))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==function]
    require(len(nodes)==1,f'Missing parser {function} in {path}')
    regexes=[n for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call)
             and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name)
             and n.value.func.value.id=='re' and n.value.func.attr=='compile']
    namespace=dict(re=re,Decimal=Decimal)
    exec(compile(ast.Module(body=regexes+nodes,type_ignores=[]),str(path),'exec'),namespace)
    def extract(response):
        value=namespace[function](response)
        return None if value is None else str(value)
    return extract

def audit_rows(records, legacy_rule='string', extract=parse):
    changed = []
    numeric = []
    for ordinal, r in enumerate(records):
        ok = numeric_correct(r)
        expected=(r['prediction']==r['gold']) if legacy_rule=='string' else ok
        require(r['prediction']==extract(r['response']) and type(r['correct']) is bool
                and r['correct']==expected,'Saved parser/scoring rule does not match record')
        if ok != r['correct']:
            changed.append(dict(row_ordinal_zero_based=ordinal, **{k:r[k] for k in
                           ['variant','id','instance','index','dataset_index','candidate','gold','prediction'] if k in r}))
        numeric.append(dict(r, correct=ok))
    return {'stored_scoring_rule':legacy_rule,'n':len(records),'legacy_correct':sum(r['correct'] for r in records),
            'numeric_correct':sum(r['correct'] for r in numeric),
            'changed_verdicts':len(changed),'false_to_true':sum(not r['correct'] and n['correct'] for r,n in zip(records,numeric)),
            'true_to_false':sum(r['correct'] and not n['correct'] for r,n in zip(records,numeric)),
            'changed_rows':changed}, numeric

def flip_category(base, trained):
    if not base and trained: return 'fixed'
    if base and not trained: return 'regressed'
    return 'both_correct' if base else 'both_wrong'

def transitions(legacy, numeric):
    result={}
    for model in list(ADAPTERS)[1:]:
        for variant in ['p1','p2','pooled']:
            counts={}
            for k in legacy['base']:
                if variant!='pooled' and k[0]!=variant: continue
                old=flip_category(legacy['base'][k]['correct'],legacy[model][k]['correct'])
                new=flip_category(numeric['base'][k]['correct'],numeric[model][k]['correct'])
                name=old+' -> '+new
                counts[name]=counts.get(name,0)+1
            result[f'{variant}/{model}']=counts
    return result

def run(args):
    root=Path(args.run)
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    mapping=manifest['family_mapping']
    hashes={str(root/'manifest.json'):digest(root/'manifest.json')}
    suffix={'base':'','2026':'_lora','2027':'_lora_seed_2027','2028':'_lora_seed_2028'}
    expected_new={(v,t,i) for v in TEMPLATES for t in TEMPLATES[v] for i in range(5,15)}
    expected_old={(v,t,i) for v in TEMPLATES for t in TEMPLATES[v] for i in range(5)}
    legacy_new, numeric_new, legacy_all, numeric_all={}, {}, {}, {}
    audits={}
    for model in ADAPTERS:
        old, new=[],[]
        for v in TEMPLATES:
            p=Path('results')/f'gsm_symbolic_{v}_dev_pilot_v2{suffix[model]}.jsonl'
            hashes[str(p)]=digest(p)
            pilot=[json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
            old.extend(pilot)
            expanded=[]
            for start in range(0,200,8):
                p=root/model/v/f'batch_{start:03d}.json'
                hashes[str(p)]=digest(p)
                batch=json.loads(p.read_text(encoding='utf-8'))['records']
                require(all(r['model']==model and r['variant']==v for r in batch), 'Wrong result identity')
                expanded.extend(batch)
            new.extend(expanded)
            for name,records in [('old_0_4',pilot),('new_5_14',expanded),('combined_0_14',pilot+expanded)]:
                audits[f'{name}/{v}/{model}']=audit_rows(records)[0]
        legacy_new[model]=validate(new,expected_new,model,mapping)
        old_index=validate(old,expected_old,model,mapping)
        legacy_all[model]={**old_index,**legacy_new[model]}
        numeric_new[model]={k:dict(r,correct=numeric_correct(r)) for k,r in legacy_new[model].items()}
        numeric_all[model]={k:dict(r,correct=numeric_correct(r)) for k,r in legacy_all[model].items()}
    reports={}
    for name,legacy,numeric in [('instances_5_14_primary',legacy_new,numeric_new),('instances_0_14_secondary',legacy_all,numeric_all)]:
        reports[name]={'numeric_scoring':summarize(numeric,args.bootstrap,args.seed),
                       'paired_classification_transitions':transitions(legacy,numeric)}
    gsm8k_audit={}
    old_sources={'gsm8k_baseline_pilot.jsonl':('gsm8k_baseline.py','predicted_answer','numeric'),
                 'gsm8k_dev_baseline.jsonl':('gsm8k_dev_baseline.py','extract_answer','numeric'),
                 'gsm8k_dev_batch16.jsonl':('gsm8k_dev_batch16.py','extract_answer','string'),
                 'gsm8k_dev_batch16_lora_pilot_v1.jsonl':('gsm8k_dev_batch16_lora.py','extract_answer','string')}
    for p in sorted(Path('results').glob('gsm8k*.jsonl')):
        require(p.name in old_sources,f'Unknown saved GSM8K evaluator: {p.name}')
        source,function,rule=old_sources[p.name]
        hashes[source]=digest(source)
        hashes[str(p)]=digest(p)
        records=[json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
        gsm8k_audit[p.name]=audit_rows(records,rule,original_parser(source,function))[0]
    training_audit={}
    p=Path('results/all_candidates.jsonl')
    if p.exists():
        hashes[str(p)]=digest(p)
        records=[json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
        hashes['generate_verified.py']=digest('generate_verified.py')
        training_audit['base_generated_candidates']=audit_rows(records,'numeric',original_parser('generate_verified.py','parse_number'))[0]
    p=Path('results/verified_training.jsonl')
    if p.exists():
        hashes[str(p)]=digest(p)
        records=[json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
        hashes['generate_verified.py']=digest('generate_verified.py')
        extract=original_parser('generate_verified.py','parse_number')
        wrong=[]
        for ordinal,r in enumerate(records):
            prediction=extract(r['solution'])
            gold=str(Decimal(str(r['answer']).replace(',','')))
            if not numeric_correct(dict(prediction=prediction,gold=gold)):
                wrong.append(dict(row_ordinal_zero_based=ordinal,dataset_index=r['dataset_index'],prediction=prediction,gold=gold))
        training_audit['accepted_solutions_numeric_check']={'n':len(records),'numerically_correct':len(records)-len(wrong),
                           'numeric_incorrect_rows':wrong,'note':'No stored correct flag in this file; validation only, no training-data edits or reselection.'}
    result={'audit_source_sha256':digest(__file__),'input_hashes':hashes,
            'numeric_rule':'prediction is not None and Decimal(prediction) == Decimal(gold); answer extraction remains identical to each file\'s original evaluator/generator',
            'bootstrap_replicates':args.bootstrap,'bootstrap_seed':args.seed,'verdict_audit':audits,
            'earlier_gsm8k_evaluation_audit':gsm8k_audit,'reports':reports}
    result['saved_generation_and_training_audit']=training_audit
    p=Path(args.output)
    if p.exists(): require(canonical(json.loads(p.read_text(encoding='utf-8')))==canonical(result),'Existing audit differs')
    else: publish(p,result)
    for key,value in audits.items():
        print(f'{key}: {value["changed_verdicts"]}/{value["n"]} verdicts changed')
    print('Saved:',p)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',default='results/gsm_symbolic_expanded_dev_v1')
    p.add_argument('--output',default='results/gsm_symbolic_expanded_dev_v1_numeric_audit.json')
    p.add_argument('--bootstrap',type=int,default=10000)
    p.add_argument('--seed',type=int,default=2026)
    args=p.parse_args()
    require(args.bootstrap>=1000,'Use at least 1000 replicates')
    run(args)
