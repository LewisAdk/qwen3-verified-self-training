#!/usr/bin/env python3
"""CPU-only primary set preparation. Never trains or loads model weights."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random

PREVIOUS=Path('results/training_sets_difficulty_v1')
NOTE=Path('results/primary_token_match_decisions_2026-10-12.md')
SEED=2026
TARGET=32879

def require(ok,message):
    if not ok:raise RuntimeError(message)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def rng(label):
    seed=int.from_bytes(hashlib.sha256(f'rsi-primary-tokenmatch-v1:{SEED}:{label}'.encode()).digest()[:8],'big')
    return random.Random(seed)

def prefix_match(pool,label,target=TARGET,limit=1000):
    require(target>0 and limit>0,'Invalid matching settings')
    require(all(r['supervised_tokens']>0 for r in pool),'Nonpositive token count')
    require(len({r['dataset_index'] for r in pool})==len(pool),'Duplicate source problems')
    stream=rng(label)
    for attempt in range(1,limit+1):
        order=list(pool)
        stream.shuffle(order)
        total=0
        for end,r in enumerate(order,1):
            total+=r['supervised_tokens']
            if abs(total-target)*100<=target:
                return dict(records=order[:end],supervised_tokens=total,shuffle_attempt=attempt)
            if total*100>101*target:break
    raise RuntimeError(f'No prefix matched ±1% for {label} within {limit} shuffles; no sets published')

def stats(rows,total,attempt):
    n=len(rows)
    require(n==len({r['dataset_index'] for r in rows}),'Duplicate group problems')
    require(total==sum(r['supervised_tokens'] for r in rows),'Token total differs')
    return dict(examples=n,supervised_tokens_per_epoch=total,supervised_tokens_two_epochs=2*total,
                solution_tokens_per_epoch=sum(r['solution_tokens'] for r in rows),
                mean_solution_tokens=sum(r['solution_tokens'] for r in rows)/n,
                maximum_solution_tokens=max(r['solution_tokens'] for r in rows),
                mean_supervised_tokens=total/n,maximum_supervised_tokens=max(r['supervised_tokens'] for r in rows),
                maximum_full_sequence_tokens=max(r['full_sequence_tokens'] for r in rows),
                bucket_breakdown=dict(sorted(Counter(r['correct_out_of_four'] for r in rows).items())),
                gap_vs_hard_percent=100*(total-TARGET)/TARGET,shuffle_attempt=attempt,
                optimizer_updates_per_epoch=math.ceil(n/4),optimizer_updates_per_model=2*math.ceil(n/4),
                training_authorized=False)

def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def read_lines(path):
    with Path(path).open(encoding='utf-8') as f:return [json.loads(s) for s in f if s.strip()]

def run(out):
    require(not out.exists(),'Existing output preserved; choose a new run rather than overwrite')
    require(out.resolve().is_relative_to(Path('results').resolve()) and out.resolve()!=Path('results').resolve(),'Output must be separate and under results/')
    previous=read_json(PREVIOUS/'manifest.json')
    builder=Path('build_difficulty_training_sets_v1.py')
    require(sha(builder)==previous['source_sha256'],'Previous builder source changed')
    for path,expected in previous['input_hashes'].items():require(sha(path)==expected,f'Protected input changed: {path}')
    inputs=dict(previous['input_hashes'])
    # Freeze every prior set/review file, including the unmatched secondary sets.
    for path in sorted(PREVIOUS.iterdir()):
        if path.is_file():inputs[str(path)]=sha(path)
    inputs[str(builder)]=sha(builder)
    inputs[str(NOTE)]=sha(NOTE)
    spec=importlib.util.spec_from_file_location('prior_preparer',builder)
    prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
    # Re-tokenize with the pinned tokenizer and exact trainer formatting on CPU;
    # compare all fixed choices rather than choosing any new candidates.
    reproduced,checks,settings,lora,prefix=prior.build_context()
    pool=read_json(PREVIOUS/'fixed_solution_pool.json')
    require(reproduced==pool,'Fixed candidates or original token counts changed')
    require(len(pool)==1014,'Eligible pool changed')
    inputs.update(checks)
    hard=read_lines(PREVIOUS/'hard.jsonl')
    require(len(hard)==300 and sum(r['supervised_tokens'] for r in hard)==TARGET,'Existing hard set changed')
    require(all(r in pool and r['correct_out_of_four'] in (1,2,3) for r in hard),'Invalid hard membership')
    easy_pool=[r for r in pool if r['correct_out_of_four']==4]
    require(len(easy_pool)==575,'Easy pool changed')
    random_set=prefix_match(pool,'random')
    easy_set=prefix_match(easy_pool,'easy')
    groups={'hard':hard,'random':random_set['records'],'easy':easy_set['records']}
    details={'hard':stats(hard,TARGET,0),
             'random':stats(groups['random'],random_set['supervised_tokens'],random_set['shuffle_attempt']),
             'easy':stats(groups['easy'],easy_set['supervised_tokens'],easy_set['shuffle_attempt'])}
    require(all(abs(s['gap_vs_hard_percent'])<=1 for s in details.values()),'Token budget mismatch')
    for rows in groups.values():
        require(all(r['full_sequence_tokens']<=1024 and r['supervised_tokens']==r['solution_tokens']+1 for r in rows),'Training sequence guard failed')
    for path,expected in inputs.items():require(sha(path)==expected,f'Input changed during preparation: {path}')
    out.mkdir(parents=True,exist_ok=False)
    with (out/'hard.jsonl').open('xb') as f:f.write((PREVIOUS/'hard.jsonl').read_bytes())
    require(sha(out/'hard.jsonl')==sha(PREVIOUS/'hard.jsonl'),'Hard copy is not byte-identical')
    for name in ['random','easy']:
        with (out/f'{name}.jsonl').open('x',encoding='utf-8') as f:
            for r in groups[name]:f.write(json.dumps(r,ensure_ascii=True)+'\n')
    overlaps={f'{a}_{b}':len({r['dataset_index'] for r in groups[a]} & {r['dataset_index'] for r in groups[b]})
              for a,b in [('hard','random'),('hard','easy'),('random','easy')]}
    manifest=dict(schema=1,source_sha256=sha(__file__),input_hashes=inputs,
                  preparation_seed=SEED,rng_algorithm='first 64 SHA256 bits of rsi-primary-tokenmatch-v1:2026:<random/easy>, Python random.Random',
                  sampling='uniform shuffled pool without replacement; first prefix within ±1%; restart on overshoot/exhaustion; at most 1000 shuffles',
                  target_supervised_tokens=TARGET,integer_acceptance_bounds=[32551,33207],
                  existing_hard_preserved_byte_for_byte=True,fixed_solution_choices_unchanged=True,
                  quality_policy='no reasoning-based filtering; retain numerical verification; report targeted 20-solution review as limitation',
                  primary_contrast='three-adapter mean hard minus random; paired 95% original-family bootstrap on existing 37 development families',
                  secondary_trigger='primary hard-minus-random interval excludes zero in either direction',
                  secondary_sets=str(PREVIOUS),other_pairwise_comparisons='secondary',
                  group_stats=details,overlapping_problem_counts=overlaps,
                  training_settings=settings,lora_settings=lora,microbatch=1,
                  optimizer='AdamW',weight_decay=0.0,gradient_clip_norm=1.0,dtype='float16',
                  three_training_seeds_per_group=True,reuse_fixed_groups_across_training_seeds=True,
                  development_family_count=37,reserved_family_count=63,training_authorized=False,
                  training_performed=False,gpu_inference_performed=False)
    with (out/'manifest.json').open('x',encoding='utf-8') as f:json.dump(manifest,f,indent=2);f.write('\n')
    lines=['# Primary token-matched sets — October 12, 2026','',
           'Prepared only; no training performed or authorized. Hard is an exact copy of the existing 300-problem set. Each comparison group uses unchanged fixed solution choices and a new seeded shuffled prefix. No reasoning-based filtering.','',
           '| Group | Examples | Supervised tokens / epoch | Gap vs hard | Mean solution tokens | Max solution tokens | Optimizer updates / model |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name,s in details.items():lines.append(f"| {name} | {s['examples']} | {s['supervised_tokens_per_epoch']} | {s['gap_vs_hard_percent']:+.3f}% | {s['mean_solution_tokens']:.2f} | {s['maximum_solution_tokens']} | {s['optimizer_updates_per_model']} |")
    lines+=['','Supervised totals include EOS; solution lengths include the original target space/newline formatting but exclude EOS. Two epochs double the listed totals. Optimizer updates are proposed, not performed, using two epochs and accumulation four with an end-of-epoch flush.','',
            '| Group | 1/4 | 2/4 | 3/4 | 4/4 | Max full sequence | Shuffle attempt |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,s in details.items():
        b=s['bucket_breakdown']
        lines.append(f"| {name} | {b.get(1,0)} | {b.get(2,0)} | {b.get(3,0)} | {b.get(4,0)} | {s['maximum_full_sequence_tokens']} | {s['shuffle_attempt']} |")
    lines+=['',f'Overlap counts: {overlaps}. No duplicates within a group. Hard/easy categories are disjoint; random can overlap either.','',
            'Quality limitation: retain the review findings without removing examples (15 apparently sound, two clear cancelling-error explanations, one borderline substantive explanation, two minor wording defects). The targeted assistant review is not a pool error-rate estimate or independent human annotation.',
            '', 'Training settings and original 37-development / 63-reserved family split are preserved. Example counts and optimizer updates differ by design while supervised token budgets match within 1%.',
            '', 'Primary gating contrast: the paired 95% original-family bootstrap interval for the three-adapter mean hard minus random excludes zero, in either direction. Only then run the existing 300-example secondary sets. Other pairwise contrasts are secondary. Stop for user confirmation before any training.']
    with (out/'set_report.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(details,indent=2),flush=True)
    print('CPU preparation complete; STOP before training.',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='results/training_sets_primary_tokenmatched_v1')
    args=parser.parse_args()
    import fcntl
    with Path('results/training_sets_primary_tokenmatched_v1.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(Path(args.output))
