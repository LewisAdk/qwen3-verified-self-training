#!/usr/bin/env python3
"""CPU-only gold control preparation and dated plan. Stops before training."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_DATASETS_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re

MODEL_REV='ea980cb0a6c2ae4b936e82123acc929f1cec04c1'
DATA_REV='740312add88f781978c0658806c59bc2815b9866'
WEIGHT_SHA='6df85b39330e5a425ee36253d0f894e4387e4f0a15b9c53cb467d668e6b3a841'
TRAINER_SHA='d8381ab86bb6020293121dc5dd32dac113901fbb8ddac8540fe4d777cea5a1b9'
HARD=Path('results/training_sets_primary_tokenmatched_v1/hard.jsonl')
HARD_SHA='de56bc3e952bf2963f24ddb2dce4a10a99c7cfa20565dccafbc96d8acb7cc246'
DATE='2026-10-13'
ANNOTATION=re.compile(r'<<[^<>]*>>')
ANSWER_LINE=re.compile(r'####[ \t]*(-?[\d,]+(?:\.\d+)?)[ \t]*')

def require(ok,message):
    if not ok:raise RuntimeError(message)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def clean_reference(text,expected_answer):
    cleaned=ANNOTATION.sub('',text).strip()
    require('<<' not in cleaned and '>>' not in cleaned,'Unremoved/malformed calculator annotation')
    lines=cleaned.splitlines()
    require(lines,'Empty reference solution')
    matches=[(i,ANSWER_LINE.fullmatch(line.strip())) for i,line in enumerate(lines) if '####' in line]
    require(len(matches)==1 and matches[0][0]==len(lines)-1 and matches[0][1] is not None,'Reference must end in one numeric answer line')
    answer=matches[0][1].group(1)
    require(Decimal(answer.replace(',',''))==Decimal(expected_answer.replace(',','')),'Reference answer differs from hard gold')
    # Retain wording, internal spacing and the reference's number representation.
    lines[-1]='#### '+answer
    return '\n'.join(lines)

def encode_example(tok,prefix,question,solution):
    prompt=prefix+f'Question: {question}\nSolution:'
    pids=tok.encode(prompt,add_special_tokens=False)
    tids=tok.encode(' '+solution.strip()+'\n',add_special_tokens=False)
    require(tok.eos_token_id is not None,'Missing EOS')
    return dict(prompt_tokens=len(pids),solution_tokens=len(tids),supervised_tokens=len(tids)+1,
                full_sequence_tokens=len(pids)+len(tids)+1)

def statistics(rows):
    require(len(rows)==300 and len({r['dataset_index'] for r in rows})==300,'Wrong/duplicate problem count')
    total=sum(r['supervised_tokens'] for r in rows)
    return dict(examples=300,supervised_tokens_per_epoch=total,supervised_tokens_two_epochs=2*total,
                solution_tokens_per_epoch=sum(r['solution_tokens'] for r in rows),
                mean_solution_tokens=sum(r['solution_tokens'] for r in rows)/300,
                maximum_solution_tokens=max(r['solution_tokens'] for r in rows),
                mean_supervised_tokens=total/300,maximum_supervised_tokens=max(r['supervised_tokens'] for r in rows),
                mean_full_sequence_tokens=sum(r['full_sequence_tokens'] for r in rows)/300,
                maximum_full_sequence_tokens=max(r['full_sequence_tokens'] for r in rows),
                minimum_full_sequence_tokens=min(r['full_sequence_tokens'] for r in rows),
                gap_vs_hard_tokens=total-32879,gap_vs_hard_percent=100*(total-32879)/32879,
                hard_difficulty_bucket_breakdown=dict(sorted(Counter(r['correct_out_of_four'] for r in rows).items())),
                planned_optimizer_updates_per_epoch=75,planned_optimizer_updates_per_model=150)

def write_json(path,obj):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')

def plan_text(stats,settings,folder):
    return f'''# Supervised gold positive-control plan — October 13, 2026

This is a **supervised-learning control**, not self-training. The RQ2 hard-minus-random result is already known to be inconclusive. This new control is planned before any gold training or evaluation outcomes exist. It asks whether the unchanged training setup can improve development accuracy when given GSM8K reference solutions for the same hard problems.

## Frozen dataset and planned training

Use exactly the hard set's 300 problem IDs, in the same initial order, from pinned GSM8K training revision {DATA_REV}, shuffled with seed 2026. Replace each selected model-generated solution with its GSM8K reference explanation. Remove only <<...>> calculator annotations and outer whitespace, and format its final line as #### answer, retaining the reference answer representation. Keep the references' wording and internal formatting. Do not filter, resample, shorten or token-match the reference solutions.

Prepared set: {folder}/gold.jsonl. Supervised tokens include EOS and use the exact existing prompt and target formatting. Gold has {stats['supervised_tokens_per_epoch']:,} supervised tokens per epoch versus hard's 32,879: {stats['gap_vs_hard_tokens']:+,} tokens ({stats['gap_vs_hard_percent']:+.3f}%). Two epochs give {stats['supervised_tokens_two_epochs']:,} supervised-token exposures. Maximum full sequence: {stats['maximum_full_sequence_tokens']} tokens; limit remains 1,024. Both gold and hard have 300 examples and 150 planned optimizer updates per model. This matches problems, example counts, epochs and steps, **not total supervised tokens**. Gold-minus-hard changes supervision source, style and length together; it does not isolate reasoning quality alone.

Planned seeds are 2026, 2027 and 2028, paired with the existing hard models. Reuse the immutable train_primary_lora_v1.py training operations and the identical pilot settings: LoRA rank 4, alpha 8, dropout 0, q_proj/v_proj, bias none; AdamW lr 1e-4, weight decay 0, constant learning rate with no scheduler; two epochs, microbatch 1, gradient accumulation 4 with the same end-of-epoch flush; FP16 with the same gradient scaler defaults; gradient clipping 1.0; gradient checkpointing with use_reentrant=False; prompt tokens masked; maximum length 1,024. Pin base snapshot {MODEL_REV}. Preserve final-adapter-only saving after epoch two; count attempted, successful and skipped updates separately. Do not choose a best checkpoint or resume from an intermediate training checkpoint.

The existing runner accepts only its frozen hard/random/easy inputs. A future separate gold entry point must supply these validated gold inputs while reusing the immutable training operations, without editing the original runner or silently substituting the hard dataset. Confirm that integration and the matching memory check before training. No training entry point is executed by this preparation stage. Stop on failed safety checks; do not truncate data or alter settings.

## Primary estimand and decision

Primary: pooled **gold minus base**, equal-weight mean of the three gold adapter accuracies, on the same 400 GSM-Symbolic development questions: the saved 20 templates each for P1/P2, instances 5–14, covering 37 original families. Reuse the saved base results after question/gold/contract validation. The gold control "works" only if the entire paired 95% family-bootstrap interval is above zero. An interval wholly below zero indicates worse performance; otherwise the control is inconclusive. An endpoint exactly equal to zero does not pass. Treat a borderline interval cautiously.

Reuse the existing procedure: 10,000 percentile bootstrap replicates, seed 2026; resample 37 original_id families with replacement; keep a family's P1/P2 questions together; share draws across all gold, hard and base models; divide weighted correctness totals by weighted question totals before averaging the three seeds. Retain the existing rejection of draws missing a variant and linearly interpolated 2.5th/97.5th percentiles. Intervals condition on these trained adapters; do not bootstrap training seeds or present them as population training-seed uncertainty.

## Secondary reporting and evaluation

Secondary: **gold minus hard**, comparing three-seed means with the same shared family draws. Report per-seed accuracies and paired fixes/regressions versus base and versus the same-seed hard adapter; report P1/P2 separately as secondary. All secondary intervals are marginal and unadjusted for multiple comparisons. No majority-vote group verdict. The prior 300-example experiment gate is not changed by this separate control.

Evaluation remains batch 8, FP16, greedy decoding, 384 new tokens, evaluation seed 2026, exact existing prompt/parser/stopping and legacy string-Decimal scoring. Reuse the saved base and hard outputs only after their question identities, pinned revisions, evaluator contracts and source hashes match. Retain numeric-equality rescoring as a diagnostic without replacing the primary scoring rule. Generate all three gold adapters' outputs before analysis; do not select seeds or checkpoints using evaluation results.

## Preservation, limitations and stage boundary

The same 37 development families have already been inspected in prior research; this control is a prospective follow-up on development data, not a reserved-test claim. The **63 reserved original families remain untouched**. Reference solutions are dataset supervision and are not automatically proof of flawless reasoning. The original targeted 20-solution review remains a limitation of the hard comparison set; do not modify it in response to this control.

The earlier training-memory observation remains unresolved. Report native PyTorch counters separately from any sampled device-wide usage; do not assert that the earlier 9–15 GiB reserved counters establish RAM spill. Any further memory investigation is a separate documented check, without changing experimental settings mid-run.

Use only new preparation, training, evaluation and analysis output paths. Preserve all existing scripts, results, checkpoints and adapters. At this stage, **prepare the dataset and plan, then stop before training**. After later confirmation, train all three gold seeds with validated unchanged operations, evaluate the same 400 questions, run this analysis, then stop without another experiment.
'''

def run(args):
    from datasets import Dataset
    from transformers import AutoTokenizer
    out=Path(args.output)
    require(not out.exists() and out.resolve().is_relative_to(Path('results').resolve()) and out.resolve()!=Path('results').resolve(),'Use a new separate results folder')
    require(digest(HARD)==HARD_SHA and digest('train_primary_lora_v1.py')==TRAINER_SHA,'Frozen hard set/trainer changed')
    baseline=read('results/primary_lora_training_v1/hard_seed_2026/manifest.json')
    protected=dict(baseline['input_hashes'])
    for path,expected in protected.items():require(digest(path)==expected,f'Protected input changed: {path}')
    # Protect every existing top-level research file and saved result, including RQ2.
    for path in list(Path('.').iterdir())+list(Path('results').rglob('*')):
        if path.is_file():protected[str(path)]=digest(path)
    manifest=read('results/gsm_symbolic_expanded_dev_v1/manifest.json')
    split=manifest['split'];require(len(split['development_original_ids'])==37 and len(split['reserved_original_ids'])==63 and not set(split['development_original_ids'])&set(split['reserved_original_ids']),'Family split changed')
    cache=Path.home()/'.cache/huggingface'
    parent=cache/'hub/models--Qwen--Qwen3-1.7B-Base/snapshots'
    require([p.name for p in parent.iterdir() if p.is_dir()]==[MODEL_REV],'Snapshot inventory changed')
    snapshot=parent/MODEL_REV
    require(digest(snapshot/'model.safetensors')==WEIGHT_SHA,'Pinned weights changed')
    for path in snapshot.iterdir():
        if path.is_file():protected[str(path)]=digest(path)
    arrow=cache/f'datasets/openai___gsm8k/main/0.0.0/{DATA_REV}/gsm8k-train.arrow'
    protected[str(arrow)]=digest(arrow)
    ds=Dataset.from_file(str(arrow)).shuffle(seed=2026,keep_in_memory=True)
    prefix='Solve the math problems step by step. Finish each solution with #### followed by the numeric answer.\n\n'
    for ex in ds.select(range(2)):
        prefix+=f"Question: {ex['question']}\nSolution: {ANNOTATION.sub('',ex['answer'])}\n\n"
    require(prefix==manifest['prefix'],'Prompt differs from frozen evaluator')
    tok=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True)
    hard=[json.loads(x) for x in HARD.read_text().splitlines() if x.strip()]
    require(len(hard)==300 and len({r['dataset_index'] for r in hard})==300 and sum(r['supervised_tokens'] for r in hard)==32879,'Hard membership/tokens differ')
    rows=[]
    for r in hard:
        reference=ds[r['dataset_index']];require(reference['question']==r['question'],'Shuffled index/question mismatch')
        original_counts=encode_example(tok,prefix,r['question'],r['solution'])
        require(all(r[k]==v for k,v in original_counts.items()),'Existing hard token counts differ')
        solution=clean_reference(reference['answer'],r['answer'])
        counts=encode_example(tok,prefix,r['question'],solution)
        require(counts['full_sequence_tokens']<=1024,f"Reference exceeds unchanged limit at shuffled index {r['dataset_index']}: {counts['full_sequence_tokens']}; no truncation or output publication")
        rows.append(dict(dataset_index=r['dataset_index'],question=r['question'],answer=r['answer'],solution=solution,
                         solution_source='gsm8k_reference',hard_model_candidate=r['candidate'],correct_out_of_four=r['correct_out_of_four'],
                         raw_reference_sha256=hashlib.sha256(reference['answer'].encode()).hexdigest(),**counts))
    require([r['dataset_index'] for r in rows]==[r['dataset_index'] for r in hard],'Hard problem order changed')
    stats=statistics(rows)
    for path,expected in protected.items():require(digest(path)==expected,f'Protected input changed during preparation: {path}')
    out.mkdir(parents=True,exist_ok=False)
    with (out/'gold.jsonl').open('x',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=True)+'\n')
    note=out/f'gold_control_plan_{DATE}.md'
    with note.open('x',encoding='utf-8') as f:f.write(plan_text(stats,baseline['settings'],str(out)))
    result=dict(schema=1,experiment='supervised_learning_positive_control_not_self_training',source_sha256=digest(__file__),input_hashes=protected,
                date=DATE,model_revision=MODEL_REV,model_weight_sha256=WEIGHT_SHA,dataset_revision=DATA_REV,dataset_shuffle_seed=2026,
                hard_source=str(HARD),hard_source_sha256=HARD_SHA,hard_problem_order_identical=True,reference_wording_preserved=True,
                supervised_tokens_include_eos=True,prompt=prefix,statistics=stats,settings=baseline['settings'],training_seeds=[2026,2027,2028],
                gold_dataset_sha256=digest(out/'gold.jsonl'),plan_sha256=digest(note),development_families=37,reserved_families=63,
                training_performed=False,evaluation_performed=False,gpu_used=False,stop_before_training=True,
                future_output_paths=dict(training='results/gold_control_training_v1',evaluation='results/gsm_symbolic_gold_control_v1',analysis='results/gsm_symbolic_gold_control_v1_analysis.json'))
    write_json(out/'manifest.json',result)
    lines=['# Gold supervised-control dataset — October 13, 2026','',
           'Prepared only. No model weights loaded into a GPU; no training or evaluation performed. Same 300 problems and order as the frozen hard set.','',
           '| Set | Examples | Supervised tokens / epoch | Two epochs | Planned updates / model |','|---|---:|---:|---:|---:|',
           '| Hard model-generated | 300 | 32,879 | 65,758 | 150 |',
           f"| Gold reference | 300 | {stats['supervised_tokens_per_epoch']:,} | {stats['supervised_tokens_two_epochs']:,} | 150 |",'',
           f"Gold − hard: {stats['gap_vs_hard_tokens']:+,} supervised tokens per epoch ({stats['gap_vs_hard_percent']:+.3f}%). Includes EOS; no token matching or shortening.",'',
           f"Gold solution tokens (excluding EOS): total {stats['solution_tokens_per_epoch']:,}, mean {stats['mean_solution_tokens']:.2f}, maximum {stats['maximum_solution_tokens']}.",
           f"Gold full sequence tokens: mean {stats['mean_full_sequence_tokens']:.2f}, minimum {stats['minimum_full_sequence_tokens']}, maximum {stats['maximum_full_sequence_tokens']}; unchanged limit 1,024.",
           f"Original difficulty buckets preserved: {stats['hard_difficulty_bucket_breakdown']}.",'',
           'This control replaces model-generated explanations with GSM8K references, stripping calculator annotations while preserving wording and answer values. It matches problems and optimizer steps; solution length, style and supervision source differ. It is supervised learning, not self-training.', '',
           'The dated plan defines gold-minus-base as primary (three-seed mean, same 400 questions, 37-family bootstrap; works only if its 95% interval is entirely above zero). Gold-minus-hard is secondary. Existing outputs and the 63 reserved families remain protected. STOP before training.']
    with (out/'set_report.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    for path,expected in protected.items():require(digest(path)==expected,f'Protected input changed after publication: {path}')
    print(json.dumps(stats,indent=2));print(f'Protected {len(protected)} existing files; CPU preparation complete. STOP BEFORE TRAINING.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='results/training_sets_gold_control_v1');run(p.parse_args())
