#!/usr/bin/env python3
"""Audit formatting-sensitive verdicts without changing any saved results."""
import argparse
from decimal import Decimal
import json
from pathlib import Path
import statistics
from gsm_symbolic_expanded_dev_v1 import ADAPTERS, TEMPLATES, canonical, digest, publish, require
from gsm_symbolic_cluster_analysis_v1 import parse, validate, summarize

def numeric_correct(record):
    prediction = record['prediction']
    require(record['gold'] is not None, 'Missing gold')
    return prediction is not None and Decimal(prediction) == Decimal(record['gold'])

def audit_rows(records):
    changed = []
    numeric = []
    for r in records:
        require(r['prediction'] == parse(r['response']) and
                r['correct'] == (r['prediction'] == r['gold']), 'Legacy score does not match saved response')
        ok = numeric_correct(r)
        if ok != r['correct']:
            changed.append({k:r[k] for k in ['variant','id','instance','gold','prediction'] if k in r})
        numeric.append(dict(r, correct=ok))
    return {'n':len(records),'legacy_correct':sum(r['correct'] for r in records),
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
    result={'audit_source_sha256':digest(__file__),'input_hashes':hashes,
            'numeric_rule':'prediction is not None and Decimal(prediction) == Decimal(gold); same regex and first-answer extraction',
            'bootstrap_replicates':args.bootstrap,'bootstrap_seed':args.seed,'verdict_audit':audits,'reports':reports}
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
