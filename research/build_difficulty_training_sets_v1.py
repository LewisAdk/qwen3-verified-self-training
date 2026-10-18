#!/usr/bin/env python3
"""CPU-only fixed training-set preparation and review export; never trains."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import ast
from collections import Counter
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import random
import re

SEED=2026
COUNT=300
DRAW_LIMIT=1000
TOLERANCE_PERCENT=3
RUN=Path('results/gsm8k_difficulty_label_full_v1')
TRAIN_SOURCE=Path('train_verified_pilot.py')
EXPECTED_MODEL='ea980cb0a6c2ae4b936e82123acc929f1cec04c1'
EXPECTED_DATA='740312add88f781978c0658806c59bc2815b9866'

def require(ok,message):
    if not ok:raise RuntimeError(message)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def parse_number(pattern,text):
    m=pattern.search(text)
    try:return Decimal(m.group(1).replace(',','')) if m else None
    except InvalidOperation:return None

def rng(label):
    seed=int.from_bytes(hashlib.sha256(f'rsi-training-sets-v1:{SEED}:{label}'.encode()).digest()[:8],'big')
    return random.Random(seed)

def choose_solution(records,index):
    correct=sorted((r for r in records if r['correct']),key=lambda r:r['candidate'])
    require(correct,'No verified solution')
    return rng(f'solution:{index}').choice(correct)

def matching_draw(pool,target,label,count=COUNT,limit=DRAW_LIMIT):
    require(len(pool)>=count,'Insufficient eligible problems')
    stream=rng(f'group:{label}')
    best=None
    for draw in range(1,limit+1):
        selected=stream.sample(pool,count)
        total=sum(r['supervised_tokens'] for r in selected)
        delta=total-target
        item=dict(records=selected,total=total,delta=delta,draw=draw,matched=abs(delta)*100<=TOLERANCE_PERCENT*target)
        if best is None or abs(delta)<abs(best['delta']):best=item
        if item['matched']:return item
    return best

def bounds(pool,count=COUNT):
    lengths=sorted(r['supervised_tokens'] for r in pool)
    return dict(minimum_possible=sum(lengths[:count]),maximum_possible=sum(lengths[-count:]))

def review_sample(hard):
    one=[r for r in hard if r['correct_out_of_four']==1]
    medium=[r for r in hard if r['correct_out_of_four'] in (2,3)]
    require(len(one)>=15 and len(medium)>=5,'Cannot fill requested review strata')
    return rng('review:one-of-four').sample(one,15)+rng('review:two-three-of-four').sample(medium,5)

def source_contract(text):
    tree=ast.parse(text)
    values={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body
            if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)
            and n.targets[0].id in ('EPOCHS','LEARNING_RATE','GRAD_ACCUM','MAX_SEQUENCE_LENGTH')}
    require(values==dict(EPOCHS=2,LEARNING_RATE=0.0001,GRAD_ACCUM=4,MAX_SEQUENCE_LENGTH=1024),'Pilot training settings changed')
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='LoraConfig']
    require(len(calls)==1,'Unexpected LoRA configuration')
    cfg={k.arg:ast.literal_eval(k.value) for k in calls[0].keywords if k.arg!='task_type'}
    require(cfg==dict(r=4,lora_alpha=8,lora_dropout=0.0,target_modules=['q_proj','v_proj'],bias='none'),'Pilot LoRA settings changed')
    # Exact target formatting, EOS supervision, prompt masking and sequence guard
    # remain references for the later trainer; this script never executes it.
    require('target = " " + x["solution"].strip() + "\\n"' in text,'Target formatting changed')
    require('[-100] * len(prompt_ids)' in text and '+ [tokenizer.eos_token_id]' in text,'Pilot token masking changed')
    prefix_nodes=[]
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='prefix' for t in node.targets):prefix_nodes.append(node)
        if isinstance(node,ast.For) and any(isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Name) and n.target.id=='prefix' for n in ast.walk(node)):prefix_nodes.append(node)
    require(len(prefix_nodes)==2,'Unexpected training prefix structure')
    return prefix_nodes,values,cfg

def build_context():
    from transformers import AutoTokenizer
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    manifest=json.loads((RUN/'manifest.json').read_text(encoding='utf-8'))
    require((manifest['model_revision'],manifest['dataset_revision'],manifest['start_index'],manifest['end_index_exclusive'],manifest['max_new_tokens'])==
            (EXPECTED_MODEL,EXPECTED_DATA,536,1636,384),'Unexpected labeling protocol')
    inputs={str(RUN/'manifest.json'):sha(RUN/'manifest.json'),str(RUN/'labels_and_summary.json'):sha(RUN/'labels_and_summary.json'),str(TRAIN_SOURCE):sha(TRAIN_SOURCE)}
    for path,expected in manifest['input_hashes'].items():
        require(sha(path)==expected,f'Protected input changed: {path}')
        inputs[path]=expected
    require(sha('gsm8k_difficulty_label_full_v1.py')==manifest['source_sha256'],'Labeling source changed')
    inputs['gsm8k_difficulty_label_full_v1.py']=manifest['source_sha256']
    records=[]
    batch_files=sorted((RUN/'batches').glob('batch_*.json'))
    require(len(batch_files)==550,'Unexpected labeling batches')
    for path in batch_files:
        inputs[str(path)]=sha(path)
        records.extend(json.loads(path.read_text(encoding='utf-8'))['records'])
    require(len(records)==4400 and {(r['dataset_index'],r['candidate']) for r in records}=={(i,c) for i in range(536,1636) for c in range(1,5)},'Labeling cohort incomplete/duplicated')
    parser=re.compile(manifest['answer_pattern'])
    by_problem={}
    for r in records:
        number=parse_number(parser,r['response'])
        require(r['correct']==(number is not None and number==Decimal(r['gold'])),'Saved numeric verdict mismatch')
        by_problem.setdefault(r['dataset_index'],[]).append(r)
    text=TRAIN_SOURCE.read_text(encoding='utf-8')
    nodes,settings,lora=source_contract(text)
    data=load_dataset('openai/gsm8k','main',split='train',revision=EXPECTED_DATA).shuffle(seed=SEED)
    ns=dict(re=re,data=data)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<read-only-training-prefix>','exec'),ns)
    prefix=ns['prefix']
    require(prefix==manifest['prefix'],'Training and generation prefixes differ')
    snapshot=Path(snapshot_download('Qwen/Qwen3-1.7B-Base',revision=EXPECTED_MODEL,local_files_only=True))
    require(snapshot.name==EXPECTED_MODEL and len([p for p in snapshot.parent.iterdir() if p.is_dir()])==1,'Unexpected model snapshots')
    tok=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True)
    require(tok.eos_token_id is not None,'Missing EOS token')
    pool=[]
    for index,attempts in sorted(by_problem.items()):
        correct=sum(r['correct'] for r in attempts)
        if not correct:continue
        r=choose_solution(attempts,index)
        require(r['question']==data[index]['question'],'Dataset question mismatch')
        ref=re.search(r'####\s*(-?[\d,]+(?:\.\d+)?)',data[index]['answer'])
        require(ref and Decimal(ref.group(1).replace(',',''))==Decimal(r['gold']),'Reference answer mismatch')
        solution=r['response'].strip()
        target=' '+solution+'\n'
        solution_tokens=len(tok.encode(target,add_special_tokens=False))
        supervised_tokens=solution_tokens+1
        prompt=prefix+f"Question: {r['question']}\nSolution:"
        prompt_tokens=len(tok.encode(prompt,add_special_tokens=False))
        require(prompt_tokens+supervised_tokens<=1024,f'Problem {index} exceeds the original 1024-token training limit; no truncation or silent exclusion')
        pool.append(dict(dataset_index=index,question=r['question'],answer=r['gold'],solution=solution,
                         candidate=r['candidate'],correct_out_of_four=correct,solution_tokens=solution_tokens,
                         supervised_tokens=supervised_tokens,prompt_tokens=prompt_tokens,
                         full_sequence_tokens=prompt_tokens+supervised_tokens))
    require(Counter(r['correct_out_of_four'] for r in pool)==Counter({1:80,2:135,3:224,4:575}),'Eligible pool differs from validated results')
    return pool,inputs,settings,lora,prefix

def group_stats(item,target):
    records=item['records']
    n=len(records)
    return dict(examples=n,supervised_tokens=item['total'],solution_tokens=sum(r['solution_tokens'] for r in records),
                mean_solution_tokens=sum(r['solution_tokens'] for r in records)/n,
                max_solution_tokens=max(r['solution_tokens'] for r in records),
                mean_supervised_tokens=item['total']/n,max_supervised_tokens=max(r['supervised_tokens'] for r in records),
                full_sequence_tokens=sum(r['full_sequence_tokens'] for r in records),
                max_full_sequence_tokens=max(r['full_sequence_tokens'] for r in records),
                bucket_breakdown=dict(sorted(Counter(r['correct_out_of_four'] for r in records).items())),
                matching_draw=item['draw'],within_three_percent=item['matched'],token_gap_percent=100*item['delta']/target,
                training_authorized=False)

def write_json(path,obj):
    with path.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,ensure_ascii=True);f.write('\n')

def run(out,preflight=False):
    require(not out.exists(),'Output already exists; preserving it. Use validation rather than rebuilding over saved sets.')
    require(out.resolve().is_relative_to(Path('results').resolve()) and out.resolve()!=Path('results').resolve(),'Output must be new and under results/')
    pool,inputs,settings,lora,prefix=build_context()
    hard_pool=[r for r in pool if r['correct_out_of_four'] in (1,2,3)]
    easy_pool=[r for r in pool if r['correct_out_of_four']==4]
    hard=rng('group:hard').sample(hard_pool,COUNT)
    target=sum(r['supervised_tokens'] for r in hard)
    groups={'hard':dict(records=hard,total=target,delta=0,draw=1,matched=True),
            'random':matching_draw(pool,target,'random'),'easy':matching_draw(easy_pool,target,'easy')}
    review=review_sample(hard)
    report={k:group_stats(v,target) for k,v in groups.items()}
    feasible={'random':bounds(pool),'easy':bounds(easy_pool)}
    print(json.dumps({'group_stats':report,'feasible_token_totals':feasible},indent=2),flush=True)
    if preflight:
        print('CPU PREFLIGHT PASS: preserved inputs; fixed candidates; exact training tokenizer; no truncation; no training/files written.')
        return
    for path,expected in inputs.items():require(sha(path)==expected,f'Input changed during preparation: {path}')
    out.mkdir(parents=True,exist_ok=False)
    for name,item in groups.items():
        suffix='' if item['matched'] else '_UNMATCHED_DIAGNOSTIC'
        with (out/f'{name}{suffix}.jsonl').open('x',encoding='utf-8') as f:
            for r in sorted(item['records'],key=lambda r:r['dataset_index']):f.write(json.dumps(r,ensure_ascii=True)+'\n')
    write_json(out/'fixed_solution_pool.json',pool)
    write_json(out/'review_sample.json',review)
    manifest=dict(schema=1,preparation_seed=SEED,source_sha256=sha(__file__),input_hashes=inputs,
                  model_revision=EXPECTED_MODEL,dataset_revision=EXPECTED_DATA,
                  candidate_selection='uniform correct candidate once per eligible problem, index-specific RNG; never shortest',
                  rng_algorithm='first 64 SHA256 bits of rsi-training-sets-v1:2026:<stream label>, Python random.Random',
                  group_sampling='hard random 300 first; random/easy first +/-3% supervised-token match in 1000 uniform draws; otherwise closest diagnostic only',
                  token_accounting='encode(" "+solution.strip()+"\\n", add_special_tokens=False) plus one EOS; prompt tokens excluded from loss',
                  training_settings=settings,lora_settings=lora,microbatch=1,optimizer='AdamW',weight_decay=0.0,
                  dtype='float16',gradient_clip_norm=1.0,optimizer_updates_per_model=150,
                  reuse_identical_sets_across_training_seeds=True,three_training_seeds_per_group=True,
                  training_authorized=False,review_sampling='15 random 1/4 + 5 random 2–3/4 from selected hard set',
                  hard_definition='1–3/4',group_overlap_allowed=True,pilot_excluded=True,reserved_family_count=63,
                  development_family_count=37,group_stats=report,feasible_token_totals=feasible,prefix=prefix)
    write_json(out/'manifest.json',manifest)
    lines=['# Selected hard solutions for reasoning review','',
           'Seeded review: 15 selected 1/4 problems, five selected 2–3/4 problems. These are the exact solutions proposed for the hard training set. No examples have been removed and no training has run.','']
    for n,r in enumerate(review,1):
        lines += [f"## {n}. Shuffled index {r['dataset_index']} — {r['correct_out_of_four']}/4; candidate {r['candidate']}",'',
                  '**Question**',r['question'],'',f"**Reference answer:** {r['answer']}",
                  f"**Solution tokens:** {r['solution_tokens']} (+1 EOS supervised); full sequence {r['full_sequence_tokens']}",'',
                  '**Selected model solution**','```text',r['solution'],'```','',
                  '**Human review:** sound / substantive error / uncertain. Notes:','']
    with (out/'hard_reasoning_review_20.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    stats=['# Prepared training-set diagnostics','',
           'No training has run. Unmatched sets are diagnostic only. Even matched sets await reasoning review and the remaining training decisions.','',
           '| Group | Examples | Supervised tokens | Solution tokens | Mean solution length | Max solution length | Gap vs hard | Match | Draw |',
           '|---|---:|---:|---:|---:|---:|---:|---|---:|']
    for name,s in report.items():stats.append(f"| {name} | {s['examples']} | {s['supervised_tokens']} | {s['solution_tokens']} | {s['mean_solution_tokens']:.2f} | {s['max_solution_tokens']} | {s['token_gap_percent']:+.2f}% | {s['within_three_percent']} | {s['matching_draw']} |")
    stats += ['', 'Bucket breakdowns (correct out of four):','']
    for name,s in report.items():stats.append(f"- {name}: {s['bucket_breakdown']}; max full sequence {s['max_full_sequence_tokens']}.")
    stats += ['',f'Feasible totals for random 300-problem subsets: {feasible["random"]}.',
              f'Feasible totals for easy 300-problem subsets: {feasible["easy"]}.','',
              'Matching uses target-token counts including EOS; all groups contain 300 unique problems. With the original microbatch 1, accumulation 4 and two epochs, each model would take 150 optimizer updates. No shortest-solution selection, token padding, truncation or repeated examples were used.',
              '', 'Random is conditioned on token budget if matched. Group membership can overlap; review oversamples 1/4 and is not an unbiased pool error-rate estimate. The original 37 development / 63 reserved family split is preserved.']
    with (out/'set_diagnostics.md').open('x',encoding='utf-8') as f:f.write('\n'.join(stats)+'\n')
    print('PREPARATION COMPLETE; no training performed:',out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='results/training_sets_difficulty_v1')
    p.add_argument('--preflight',action='store_true')
    args=p.parse_args()
    import fcntl
    with Path('results/training_sets_difficulty_v1.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(Path(args.output),args.preflight)
