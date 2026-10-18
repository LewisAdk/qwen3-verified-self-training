#!/usr/bin/env python3
"""Prespecified supervised gold control, paired 37-family percentile bootstrap."""
import argparse
from collections import Counter
from decimal import Decimal
import json
from pathlib import Path
import random
import statistics
import numpy as np
from gsm_symbolic_cluster_analysis_v1 import percentile,parse
from gsm_symbolic_gold_eval_v2 import MODELS,HARD_MODELS,RUN,PLAN_SHA,sha,require,prior

ALL=['base']+MODELS+HARD_MODELS
def summarize(indices,repetitions=10000,seed=2026):
    keys=sorted(indices['base']);families=sorted({r['original_id'] for r in indices['base'].values()});positions={f:i for i,f in enumerate(families)}
    clusters=np.zeros((len(families),2,len(ALL)+1),dtype=np.int64)
    for k in keys:
        b=indices['base'][k];f=positions[b['original_id']];v=['p1','p2'].index(k[0]);clusters[f,v,0]+=1
        for i,model in enumerate(ALL):
            r=indices[model][k];require((r['question'],r['gold'],r['original_id'])==(b['question'],b['gold'],b['original_id']),'Models not paired')
            clusters[f,v,i+1]+=int(r['correct'])
    def estimates(weights):
        sums=np.einsum('f,fvm->vm',weights,clusters)
        if not sums[0,0] or not sums[1,0]:return None
        sums=np.vstack([sums,sums.sum(axis=0)]);result={}
        for v,t in zip(['p1','p2','pooled'],sums):
            acc={m:100*float(t[i+1])/float(t[0]) for i,m in enumerate(ALL)}
            for m,value in acc.items():
                result[f'{v}/accuracy/{m}']=value
                if m!='base':result[f'{v}/difference_pp/{m}_minus_base']=value-acc['base']
            for g in ['gold','hard']:
                acc[g]=statistics.mean(acc[f'{g}_seed_{s}'] for s in [2026,2027,2028]);result[f'{v}/accuracy/{g}_seed_mean']=acc[g]
            # Algebraically identical seed means; direct totals prevent signed roundoff at zero.
            gold_sum=sum(int(t[ALL.index(m)+1]) for m in MODELS);hard_sum=sum(int(t[ALL.index(m)+1]) for m in HARD_MODELS)
            result[f'{v}/difference_pp/gold_minus_base']=100*(gold_sum-3*int(t[1]))/(3*int(t[0]))
            result[f'{v}/difference_pp/gold_minus_hard']=100*(gold_sum-hard_sum)/(3*int(t[0]))
            for s in [2026,2027,2028]:result[f'{v}/difference_pp/gold_minus_hard_seed_{s}']=acc[f'gold_seed_{s}']-acc[f'hard_seed_{s}']
        return result
    point=estimates(np.ones(len(families),dtype=np.int64));draws={k:[] for k in point};rng=random.Random(seed)
    while len(next(iter(draws.values())))<repetitions:
        weights=Counter(rng.choices(families,k=len(families)))
        sample=estimates(np.array([weights.get(f,0) for f in families],dtype=np.int64))
        if sample is not None:
            for k,value in sample.items():draws[k].append(value)
    metrics={k:dict(estimate=value,ci95=[percentile(draws[k],.025),percentile(draws[k],.975)]) for k,value in point.items()}
    flips={};numeric={}
    for v in ['p1','p2','pooled']:
        selected=[k for k in keys if v=='pooled' or k[0]==v]
        for s in [2026,2027,2028]:
            a=f'gold_seed_{s}'
            for b in ['base',f'hard_seed_{s}']:
                fixed=sum(not indices[b][k]['correct'] and indices[a][k]['correct'] for k in selected)
                regressed=sum(indices[b][k]['correct'] and not indices[a][k]['correct'] for k in selected)
                flips[f'{v}/{a}_versus_{b}']=dict(fixed=fixed,regressed=regressed,net=fixed-regressed)
    for m in ALL:
        verdicts=[r['prediction'] is not None and Decimal(r['prediction'])==Decimal(r['gold']) for r in indices[m].values()]
        numeric[m]=dict(verdict_changes=sum(a!=r['correct'] for a,r in zip(verdicts,indices[m].values())),numeric_correct=sum(verdicts),legacy_correct=sum(r['correct'] for r in indices[m].values()))
    primary=metrics['pooled/difference_pp/gold_minus_base'];lo,hi=primary['ci95']
    decision='works' if lo>0 else 'worse_than_base' if hi<0 else 'inconclusive'
    return dict(questions=len(keys),original_families=len(families),metrics=metrics,paired_flips=flips,numeric_scoring_diagnostic=numeric,
                primary=dict(comparison='gold minus base, equal three-seed mean',**primary,decision=decision),
                secondary=dict(comparison='gold minus hard, equal three-seed means',**metrics['pooled/difference_pp/gold_minus_hard']))

def analyze(args):
    root=Path(args.run);manifest=json.loads((root/'manifest.json').read_text())
    require(manifest['gold_plan_sha256']==PLAN_SHA and manifest['source_sha256']==sha('gsm_symbolic_gold_eval_v2.py'),'Gold plan/evaluator differs')
    inputs={str(root/'manifest.json'):sha(root/'manifest.json')};indices={}
    expected={(v,t,i) for v in prior.old.TEMPLATES for t in prior.old.TEMPLATES[v] for i in range(5,15)}
    from gsm_symbolic_rq2_analysis_v1 import validate
    for model in ALL:
        if model in MODELS:require((root/model/'completed.json').exists(),'All gold adapters must complete before analysis')
        records=[];config=root/model/'generation_config.json'
        require(json.loads(config.read_text())==manifest['base_generation_config'],'Generation config differs');inputs[str(config)]=sha(config)
        for v in prior.old.TEMPLATES:
            directory=root/model/v;paths={directory/f'batch_{start:03d}.json' for start in range(0,200,8)}
            require(set(directory.glob('batch_*.json'))==paths,'Incomplete/extra batches')
            for path in sorted(paths):
                obj=json.loads(path.read_text());start=int(path.stem[6:]);require([(r['id'],r['instance']) for r in obj['records']]==sorted((t,i) for t in prior.old.TEMPLATES[v] for i in range(5,15))[start:start+8],'Batch boundaries differ')
                records.extend(obj['records']);inputs[str(path)]=sha(path)
        indices[model]=validate(records,expected,model,manifest['family_mapping'])
    for p,digest in manifest['input_hashes'].items():require(sha(p)==digest,f'Protected input changed: {p}')
    result=summarize(indices);require(result['questions']==400 and result['original_families']==37,'Wrong coverage')
    report=dict(supervised_control_not_self_training=True,analysis_source_sha256=sha(__file__),gold_plan_sha256=PLAN_SHA,input_hashes=inputs,
                bootstrap_replicates=10000,bootstrap_seed=2026,method='Existing paired original-family percentile bootstrap; same draws across P1/P2/models; weighted totals/weighted question count; equal three-seed means',result=result)
    prior.publish(Path(args.output),report)
    p=result['primary'];s=result['secondary']
    lines=['# Supervised gold control — October 13, 2026','',
           f"Primary gold minus base: **{p['estimate']:+.2f} pp**, 95% family-bootstrap interval **[{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}]**. Decision: **{p['decision']}**.",
           f"Secondary gold minus hard: {s['estimate']:+.2f} pp, marginal 95% interval [{s['ci95'][0]:+.2f}, {s['ci95'][1]:+.2f}].",'',
           '| Model/group | P1 accuracy | P2 accuracy | Pooled accuracy |','|---|---:|---:|---:|']
    for m in ['base','gold_seed_mean','hard_seed_mean']+MODELS:
        vals=[result['metrics'][f'{v}/accuracy/{m}']['estimate'] for v in ['p1','p2','pooled']]
        lines.append(f'| {m} | {vals[0]:.2f}% | {vals[1]:.2f}% | {vals[2]:.2f}% |')
    lines+=['','| Seed | Gold vs base fixes | Regressions | Net | Gold vs hard fixes | Regressions | Net |','|---|---:|---:|---:|---:|---:|---:|']
    for seed in [2026,2027,2028]:
        b=result['paired_flips'][f'pooled/gold_seed_{seed}_versus_base'];h=result['paired_flips'][f'pooled/gold_seed_{seed}_versus_hard_seed_{seed}']
        lines.append(f"| {seed} | {b['fixed']} | {b['regressed']} | {b['net']} | {h['fixed']} | {h['regressed']} | {h['net']} |")
    lines+=['','Secondary variant contrasts (marginal, no multiplicity adjustment):','','| Variant | Contrast | pp | 95% interval |','|---|---|---:|---:|']
    for v in ['p1','p2']:
        for comparison in ['gold_minus_base','gold_minus_hard']:
            x=result['metrics'][f'{v}/difference_pp/{comparison}'];lines.append(f"| {v} | {comparison} | {x['estimate']:+.2f} | [{x['ci95'][0]:+.2f}, {x['ci95'][1]:+.2f}] |")
    lines+=['', 'Numeric-equality verdict changes: '+', '.join(m+': '+str(x['verdict_changes']) for m,x in result['numeric_scoring_diagnostic'].items())+'. Legacy scoring remains primary.', '',
            'This is supervised learning, not self-training. Gold/hard use the same 300 problems, two epochs and 150 updates; gold has 32,206 supervised tokens/epoch versus hard 32,879 (−2.047%). Solution source, quality/style and length differ together.', '',
            'Only 37 selected development families supply clusters; the questions were inspected in prior research. Intervals condition on these three adapters, without seed resampling. Secondary intervals are unadjusted; treat borderline results cautiously. The reserved 63 families remain untouched. Memory observations remain separate from the analysis. No further experiment was started.']
    with Path(args.output).with_suffix('.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(primary=p,secondary=s),indent=2));print('Gold analysis complete. STOP.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',default=str(RUN));p.add_argument('--output',default='results/gsm_symbolic_gold_control_v2_analysis.json');analyze(p.parse_args())
