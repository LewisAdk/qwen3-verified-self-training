#!/usr/bin/env python3
"""Prespecified paired RQ2 family bootstrap; CPU only; no further experiments."""
import argparse
from collections import Counter
from decimal import Decimal
import json
from pathlib import Path
import random
import statistics
import numpy as np
from gsm_symbolic_cluster_analysis_v1 import percentile,parse
from gsm_symbolic_rq2_eval_v1 import GROUPS,SEEDS,MODELS,PLAN,digest,publish,require,old

ALL_MODELS=['base']+MODELS

def validate(records,expected,model,mapping):
    index={}
    for r in records:
        key=(r['variant'],r['id'],r['instance'])
        require(key in expected and key not in index,'Unexpected/duplicate result')
        require(r['model']==model and r['original_id']==mapping[r['variant']][str(r['id'])],'Model/family mismatch')
        require(parse(r['response'])==r['prediction'] and type(r['correct']) is bool and r['correct']==(r['prediction']==r['gold']),'Scoring mismatch')
        require(type(r['tokens']) is int and 1<=r['tokens']<=384,'Token count mismatch')
        index[key]=r
    require(set(index)==expected,'Incomplete evaluation results')
    return index

def summarize(indices,repetitions=10000,seed=2026):
    keys=sorted(indices['base']);families=sorted({r['original_id'] for r in indices['base'].values()})
    positions={f:i for i,f in enumerate(families)}
    clusters=np.zeros((len(families),2,len(ALL_MODELS)+1),dtype=np.int64)
    for key in keys:
        b=indices['base'][key];f=positions[b['original_id']];v=['p1','p2'].index(key[0]);clusters[f,v,0]+=1
        for i,model in enumerate(ALL_MODELS):
            r=indices[model][key]
            require((r['question'],r['gold'],r['original_id'])==(b['question'],b['gold'],b['original_id']),'Models not paired')
            clusters[f,v,i+1]+=int(r['correct'])
    def estimates(weights):
        totals=np.einsum('f,fvm->vm',weights,clusters)
        if not totals[0,0] or not totals[1,0]:return None
        totals=np.vstack([totals,totals.sum(axis=0)])
        result={}
        for variant,t in zip(['p1','p2','pooled'],totals):
            acc={m:100*float(t[i+1])/float(t[0]) for i,m in enumerate(ALL_MODELS)}
            for m,x in acc.items():
                result[f'{variant}/accuracy/{m}']=x
                if m!='base':result[f'{variant}/difference_pp/{m}_minus_base']=x-acc['base']
            for group in GROUPS:
                acc[group]=statistics.mean(acc[f'{group}_seed_{s}'] for s in SEEDS)
                result[f'{variant}/accuracy/{group}_seed_mean']=acc[group]
                result[f'{variant}/difference_pp/{group}_minus_base']=acc[group]-acc['base']
            for a,b in [('hard','random'),('hard','easy'),('random','easy')]:result[f'{variant}/difference_pp/{a}_minus_{b}']=acc[a]-acc[b]
            for s in SEEDS:result[f'{variant}/difference_pp/hard_minus_random_seed_{s}']=acc[f'hard_seed_{s}']-acc[f'random_seed_{s}']
        return result
    point=estimates(np.ones(len(families),dtype=np.int64));draws={k:[] for k in point};rng=random.Random(seed)
    while len(next(iter(draws.values())))<repetitions:
        sample=Counter(rng.choices(families,k=len(families)))
        values=estimates(np.array([sample.get(f,0) for f in families],dtype=np.int64))
        if values is not None:
            for k,value in values.items():draws[k].append(value)
    metrics={k:dict(estimate=x,ci95=[percentile(draws[k],.025),percentile(draws[k],.975)]) for k,x in point.items()}
    flips={};group_average_flips={};seed_sd={}
    for variant in ['p1','p2','pooled']:
        selected=[k for k in keys if variant=='pooled' or k[0]==variant]
        def flip(a,b):
            fixed=sum(not indices[b][k]['correct'] and indices[a][k]['correct'] for k in selected)
            regressed=sum(indices[b][k]['correct'] and not indices[a][k]['correct'] for k in selected)
            return dict(fixed=fixed,regressed=regressed,net=fixed-regressed,questions=len(selected))
        for m in MODELS:flips[f'{variant}/{m}_versus_base']=flip(m,'base')
        for s in SEEDS:flips[f'{variant}/hard_versus_random_seed_{s}']=flip(f'hard_seed_{s}',f'random_seed_{s}')
        for g in GROUPS:
            group_average_flips[f'{variant}/{g}_versus_base']={k:statistics.mean(flips[f'{variant}/{g}_seed_{s}_versus_base'][k] for s in SEEDS) for k in ['fixed','regressed','net']}
            seed_sd[f'{variant}/{g}']=statistics.stdev(point[f'{variant}/accuracy/{g}_seed_{s}'] for s in SEEDS)
    primary=metrics['pooled/difference_pp/hard_minus_random'];lo,hi=primary['ci95']
    decision='hard_wins' if lo>0 else 'random_favored' if hi<0 else 'inconclusive'
    numeric={}
    for model in ALL_MODELS:
        changed=[];correct=0
        for key,r in indices[model].items():
            verdict=r['prediction'] is not None and Decimal(r['prediction'])==Decimal(r['gold'])
            correct+=int(verdict)
            if verdict!=r['correct']:changed.append(dict(variant=key[0],id=key[1],instance=key[2],legacy_correct=r['correct'],numeric_correct=verdict,prediction=r['prediction'],gold=r['gold']))
        numeric[model]=dict(verdict_changes=len(changed),numeric_correct=correct,legacy_correct=sum(r['correct'] for r in indices[model].values()),changes=changed)
    return dict(questions=len(keys),original_families=len(families),metrics=metrics,paired_flips=flips,
                group_average_flips_descriptive=group_average_flips,training_seed_sample_sd_pp_descriptive=seed_sd,
                primary=dict(comparison='pooled hard-minus-random three-seed mean',**primary,decision=decision,
                             secondary_300_example_experiment_gate=lo>0 or hi<0,additional_training_authorized=False),numeric_scoring_diagnostic=numeric)

def analyze(args):
    root=Path(args.run);manifest=json.loads((root/'manifest.json').read_text())
    require((root/'evaluation_completed.json').is_file(),'All nine adapters must complete before analysis')
    require(manifest['models']==MODELS and manifest['rq2_plan_sha256']==digest(PLAN),'Evaluation/plan changed')
    require(manifest['source_sha256']==digest('gsm_symbolic_rq2_eval_v1.py'),'Evaluator source differs')
    require(args.bootstrap==10000 and args.seed==2026,'Prespecified bootstrap changed')
    require(manifest['split']['development_templates']==old.TEMPLATES and len(manifest['split']['reserved_original_ids'])==63,'Wrong split')
    for path,expected_hash in manifest['input_hashes'].items():require(digest(path)==expected_hash,f'Protected input changed: {path}')
    expected={(v,t,i) for v in old.TEMPLATES for t in old.TEMPLATES[v] for i in range(5,15)}
    inputs={str(root/'manifest.json'):digest(root/'manifest.json'),str(root/'evaluation_completed.json'):digest(root/'evaluation_completed.json')};indices={}
    for model in ALL_MODELS:
        records=[]
        config=root/model/'generation_config.json'
        require(json.loads(config.read_text())==manifest['base_generation_config'],'Generation config differs');inputs[str(config)]=digest(config)
        for variant in old.TEMPLATES:
            directory=root/model/variant;paths={directory/f'batch_{i:03d}.json' for i in range(0,200,8)}
            require(set(directory.glob('batch_*.json'))==paths,'Missing/extra batches')
            for path in sorted(paths):
                obj=json.loads(path.read_text());start=int(path.stem[6:])
                require([(r['id'],r['instance']) for r in obj['records']]==sorted((t,i) for t in old.TEMPLATES[variant] for i in range(5,15))[start:start+8],'Wrong batch boundaries')
                inputs[str(path)]=digest(path);records.extend(obj['records'])
        indices[model]=validate(records,expected,model,manifest['family_mapping'])
    result=summarize(indices,args.bootstrap,args.seed)
    require(result['original_families']==37 and result['questions']==400,'Unexpected evaluation coverage')
    report=dict(analysis_source_sha256=digest(__file__),rq2_plan_sha256=digest(PLAN),input_hashes=inputs,
                bootstrap_replicates=args.bootstrap,bootstrap_seed=args.seed,method='Existing paired percentile original-family cluster bootstrap; shared draws across models and P1/P2; weighted totals/weighted question counts; equal three-seed group means',
                interpretation='Selected development families, previously inspected for pilot research; intervals condition on these adapters; no population seed uncertainty or multiplicity adjustment; all comparisons except pooled hard-minus-random are secondary.',result=result)
    publish(Path(args.output),report)
    lines=['# RQ2 evaluation — October 12, 2026','',f"Primary: hard minus random = {result['primary']['estimate']:+.2f} percentage points; paired 95% family-bootstrap interval [{result['primary']['ci95'][0]:+.2f}, {result['primary']['ci95'][1]:+.2f}]. Decision: **{result['primary']['decision']}**.",'',
           '| Group | P1 mean accuracy | P2 mean accuracy | Pooled mean accuracy |','|---|---:|---:|---:|']
    for g in ['base']+list(GROUPS):
        name=g if g=='base' else g+'_seed_mean'
        vals=[result['metrics'][f'{v}/accuracy/{name}']['estimate'] for v in ['p1','p2','pooled']]
        lines.append(f'| {g} | {vals[0]:.2f}% | {vals[1]:.2f}% | {vals[2]:.2f}% |')
    lines+=['','Secondary pooled comparisons; marginal intervals, no multiplicity adjustment:','','| Comparison | Difference pp | 95% interval |','|---|---:|---:|']
    for name in ['hard_minus_easy','random_minus_easy','hard_minus_base','random_minus_base','easy_minus_base']:
        m=result['metrics']['pooled/difference_pp/'+name];lines.append(f"| {name} | {m['estimate']:+.2f} | [{m['ci95'][0]:+.2f}, {m['ci95'][1]:+.2f}] |")
    lines+=['','Per-seed results and fixes/regressions against base (400 questions each):','','| Adapter | P1 correct / 200 | P2 correct / 200 | Pooled correct / 400 | Accuracy | Delta vs base pp | Fixes | Regressions | Net |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for model in MODELS:
        counts=[sum(r['correct'] for k,r in indices[model].items() if v=='pooled' or k[0]==v) for v in ['p1','p2','pooled']]
        delta=result['metrics'][f'pooled/difference_pp/{model}_minus_base']['estimate'];flip=result['paired_flips'][f'pooled/{model}_versus_base']
        lines.append(f"| {model} | {counts[0]} | {counts[1]} | {counts[2]} | {100*counts[2]/400:.2f}% | {delta:+.2f} | {flip['fixed']} | {flip['regressed']} | {flip['net']} |")
    lines+=['','Within-seed hard versus random (fix = random wrong, hard correct):','','| Seed | Hard − random pp | Fixes | Regressions | Net |','|---|---:|---:|---:|---:|']
    for s in SEEDS:
        m=result['metrics'][f'pooled/difference_pp/hard_minus_random_seed_{s}'];f=result['paired_flips'][f'pooled/hard_versus_random_seed_{s}']
        lines.append(f"| {s} | {m['estimate']:+.2f} | {f['fixed']} | {f['regressed']} | {f['net']} |")
    lines+=['',f"Numeric-equality diagnostic verdict changes: {', '.join(m+': '+str(x['verdict_changes']) for m,x in result['numeric_scoring_diagnostic'].items())}. Legacy scoring remains primary.",'',
            'Only 37 original families supply independent clusters; percentile intervals may be optimistic, and borderline exclusion of zero should be treated cautiously. Group-mean intervals condition on these three adapters, without seed resampling. Per-variant metrics, per-seed intervals, group-average flips and descriptive seed SD are in the JSON report.', '',
            'Design: supervised-token matching within ±1%, with hard/random/easy counts 300/353/403 and updates 150/178/202. Easy received 34.67% more updates than hard. Preserve the recorded reasoning-quality limitation and unresolved memory observation in the dated plan.', '',
            f"The conditional 300-example experiment gate is {'met' if result['primary']['secondary_300_example_experiment_gate'] else 'not met'}. No additional training was started. The 63 reserved families remain untouched."]
    path=Path(args.output).with_suffix('.md')
    with path.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(result['primary'],indent=2));print('Analysis saved; no further experiment started.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',default='results/gsm_symbolic_rq2_primary_v1');p.add_argument('--output',default='results/gsm_symbolic_rq2_primary_v1_analysis.json');p.add_argument('--bootstrap',type=int,default=10000);p.add_argument('--seed',type=int,default=2026);analyze(p.parse_args())
