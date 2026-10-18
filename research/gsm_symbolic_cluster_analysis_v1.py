#!/usr/bin/env python3
"""Paired original-family bootstrap for expanded development results, no GPU needed."""
import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import random
import re
import statistics
from gsm_symbolic_expanded_dev_v1 import ADAPTERS, TEMPLATES, canonical, digest, publish, require

ANSWER = re.compile(r'####[ \t]*(-?[\d,]+(?:\.\d+)?)')
def parse(text):
    m = ANSWER.search(text)
    return str(Decimal(m.group(1).replace(',', ''))) if m else None

def validate(records, expected, model, mapping):
    index = {}
    for r in records:
        key = (r['variant'], r['id'], r['instance'])
        require(key in expected and key not in index, f'Unexpected/duplicate result: {model} {key}')
        family = mapping[r['variant']][str(r['id'])]
        require(r.get('original_id', family) == family, 'Wrong family')
        require(parse(r['response']) == r['prediction'] and type(r['correct']) is bool
                and r['correct'] == (r['prediction'] == r['gold']), 'Invalid scoring')
        r = dict(r, original_id=family)
        index[key] = r
    require(set(index) == expected, f'Incomplete results for {model}')
    return index

def percentile(values, q):
    values = sorted(values)
    pos = (len(values)-1)*q
    lo = int(pos)
    return values[lo] + (values[min(lo+1,len(values)-1)]-values[lo])*(pos-lo)

def summarize(indices, repetitions, seed):
    keys = sorted(indices['base'])
    families = sorted({r['original_id'] for r in indices['base'].values()})
    # Each cluster stores count and four correctness totals for each variant.
    clusters = {f: {v: [0,0,0,0,0] for v in ['p1','p2']} for f in families}
    models = list(ADAPTERS)
    for key in keys:
        base = indices['base'][key]
        for model in models:
            r = indices[model][key]
            require((r['question'],r['gold'],r['original_id']) ==
                    (base['question'],base['gold'],base['original_id']), 'Models not paired')
        slot = clusters[base['original_id']][key[0]]
        slot[0] += 1
        for i, model in enumerate(models):
            slot[i+1] += int(indices[model][key]['correct'])

    def estimates(weights):
        sums = {v: [sum(weights.get(f,0)*clusters[f][v][j] for f in families)
                     for j in range(5)] for v in ['p1','p2']}
        sums['pooled'] = [sums['p1'][j]+sums['p2'][j] for j in range(5)]
        result = {}
        for v, s in sums.items():
            if not s[0]:
                return None
            acc = [100*x/s[0] for x in s[1:]]
            for i, model in enumerate(models):
                result[f'{v}/accuracy/{model}'] = acc[i]
                if i:
                    result[f'{v}/difference_pp/{model}'] = acc[i]-acc[0]
            result[f'{v}/accuracy/seed_mean'] = statistics.mean(acc[1:])
            result[f'{v}/difference_pp/seed_mean'] = statistics.mean(acc[1:])-acc[0]
        return result

    point = estimates(dict.fromkeys(families,1))
    draws = {k: [] for k in point}
    rng = random.Random(seed)
    while len(next(iter(draws.values()))) < repetitions:
        sample = estimates(Counter(rng.choices(families,k=len(families))))
        if sample is not None:
            for k, value in sample.items():
                draws[k].append(value)
    metrics = {k: {'estimate': value, 'ci95': [percentile(draws[k],.025),percentile(draws[k],.975)]}
               for k,value in point.items()}
    flips = {}
    seed_sd = {}
    for v in ['p1','p2','pooled']:
        selected = [k for k in keys if v == 'pooled' or k[0] == v]
        seed_sd[v] = statistics.stdev([metrics[f'{v}/accuracy/{m}']['estimate'] for m in models[1:]])
        for model in models[1:]:
            fixed = sum(not indices['base'][k]['correct'] and indices[model][k]['correct'] for k in selected)
            regressed = sum(indices['base'][k]['correct'] and not indices[model][k]['correct'] for k in selected)
            flips[f'{v}/{model}'] = {'fixed': fixed, 'regressed': regressed, 'net': fixed-regressed}
    return {'questions': len(keys), 'original_families': len(families),
            'families_by_variant': {v: len({indices['base'][k]['original_id'] for k in keys if k[0]==v}) for v in ['p1','p2']},
            'metrics_percent_or_percentage_points': metrics, 'paired_flips': flips,
            'training_seed_sample_sd_pp': seed_sd}

def analyze(args):
    root = Path(args.run)
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    require(manifest['split']['development_templates'] == TEMPLATES, 'Wrong templates')
    mapping = manifest['family_mapping']
    require({f for v in mapping.values() for f in v.values()} == set(manifest['split']['development_original_ids']), 'Wrong families')
    inputs = {str(root/'manifest.json'): digest(root/'manifest.json')}
    expected = {(v,t,i) for v in TEMPLATES for t in TEMPLATES[v] for i in range(5,15)}
    expanded = {}
    for model in ADAPTERS:
        records = []
        for v in TEMPLATES:
            directory = root/model/v
            paths = {directory/f'batch_{i:03d}.json' for i in range(0,200,8)}
            require(set(directory.glob('batch_*.json')) == paths, f'Missing/unexpected batches: {directory}')
            for p in sorted(paths):
                obj = json.loads(p.read_text(encoding='utf-8'))
                require(all(r['model']==model and r['variant']==v for r in obj['records']), 'Wrong batch identity')
                require([(r['id'],r['instance']) for r in obj['records']] ==
                        sorted((t,i) for t in TEMPLATES[v] for i in range(5,15))[int(p.stem[6:]):int(p.stem[6:])+8], 'Wrong batch boundaries')
                inputs[str(p)] = digest(p)
                records.extend(obj['records'])
        expanded[model] = validate(records,expected,model,mapping)
    reports = {'instances_5_14_primary': summarize(expanded,args.bootstrap,args.seed)}
    if not args.new_only:
        combined = {}
        suffix = {'base':'','2026':'_lora','2027':'_lora_seed_2027','2028':'_lora_seed_2028'}
        old_expected = {(v,t,i) for v in TEMPLATES for t in TEMPLATES[v] for i in range(5)}
        for model in ADAPTERS:
            records = []
            for v in TEMPLATES:
                p = Path('results')/f'gsm_symbolic_{v}_dev_pilot_v2{suffix[model]}.jsonl'
                inputs[str(p)] = digest(p)
                records.extend(json.loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line.strip())
            old = validate(records,old_expected,model,mapping)
            # Check immutable question/gold identity across models in summarize.
            combined[model] = {**old,**expanded[model]}
        reports['instances_0_14_secondary'] = summarize(combined,args.bootstrap,args.seed)
    result = {'analysis_source_sha256': digest(__file__), 'input_hashes':inputs,
              'bootstrap_replicates':args.bootstrap,'bootstrap_seed':args.seed,
              'method':'Paired percentile cluster bootstrap: sample 37 original_id families with replacement; identical draws across models, P1 and P2. Ratio of weighted correct totals to weighted question totals.',
              'interpretation':'Exploratory development uncertainty across these selected families. Additional instances do not add independent families. Seed-mean intervals condition on these three trained adapters; seed SD is descriptive, not population training-seed uncertainty. Pooled accuracy weights P1/P2 questions equally; variants are not paired by template ID. Combined 0–14 results include previously inspected data. Intervals are marginal, without multiplicity correction.',
              'reports':reports}
    destination = Path(args.output)
    if destination.exists():
        saved = json.loads(destination.read_text(encoding='utf-8'))
        changed = [k for k in result if canonical(saved.get(k)) != canonical(result[k])]
        require(not changed, f'Existing analysis differs in {changed}; choose a new --output')
    else:
        publish(destination,result)
    for name,report in reports.items():
        print(f'\n{name}: {report["questions"]} questions, {report["original_families"]} families')
        for v in ['p1','p2','pooled']:
            for m in ['2026','2027','2028','seed_mean']:
                x = report['metrics_percent_or_percentage_points'][f'{v}/difference_pp/{m}']
                print(f'{v} {m}: delta {x["estimate"]:+.2f} pp; 95% CI [{x["ci95"][0]:+.2f}, {x["ci95"][1]:+.2f}]')
    print('Saved:', destination)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',default='results/gsm_symbolic_expanded_dev_v1')
    p.add_argument('--output',default='results/gsm_symbolic_expanded_dev_v1_cluster_analysis.json')
    p.add_argument('--bootstrap',type=int,default=10000)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--new-only',action='store_true')
    args = p.parse_args()
    require(args.bootstrap >= 1000,'Use at least 1000 bootstrap replicates')
    analyze(args)
