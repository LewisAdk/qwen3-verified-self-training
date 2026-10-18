#!/usr/bin/env python3
"""1,100 new GSM8K questions, four sampled attempts each at 384 tokens; no training."""
import argparse
import ast
from decimal import Decimal, InvalidOperation
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import time

MODEL = 'Qwen/Qwen3-1.7B-Base'
MODEL_REVISION = 'ea980cb0a6c2ae4b936e82123acc929f1cec04c1'
DATASET_REVISION = '740312add88f781978c0658806c59bc2815b9866'
START, COUNT, ATTEMPTS, BATCH_SIZE = 536, 1100, 4, 8
SEED, MAX_TOKENS = 2026, 384
PILOT_RUN = Path('results/gsm8k_difficulty_label_pilot_v1')
DECISION_NOTE = Path('results/research_decisions_2026-10-12_full_labeling.md')
SOURCE = Path('generate_verified.py')
BACKUP_RECEIPT = Path('results/research_backup_2026-10-12_receipt.json')
PRIOR_RUN = Path('results/gsm_symbolic_expanded_dev_v1')

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':'))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def publish(path, obj):
    """Publish completed JSON without replacing any existing file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.pending.{os.getpid()}')
    with temp.open('x', encoding='utf-8') as f:
        f.write(canonical(obj) + '\n')
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(temp, path)
    finally:
        temp.unlink()

def freeze(path, obj):
    path = Path(path)
    if path.exists():
        require(json.loads(path.read_text(encoding='utf-8')) == obj, f'Existing configuration differs: {path}')
    else:
        publish(path, obj)

def validate_output(path):
    output=Path(path).resolve()
    root=Path('results').resolve()
    require(output!=root and output.is_relative_to(root), 'Output must be a separate directory underneath results/')
    if output.exists() and any(output.iterdir()):
        require((output/'manifest.json').is_file(), 'Existing nonempty output has no matching run manifest; preserving it')
    return output

def batch_seed(index):
    return int.from_bytes(hashlib.sha256(f'gsm8k-difficulty-full-v1:{SEED}:{START}:{index}'.encode()).digest()[:4], 'big')

def question_hash(question):
    normalized = re.sub(r'\s+', ' ', question).strip().lower()
    return hashlib.sha256(normalized.encode()).hexdigest()

def safe_parse(parse,text):
    try:
        return parse(text)
    except InvalidOperation:
        # A malformed numeric fragment is an unsuccessful/unparseable attempt.
        return None

def original_helpers(text):
    """Read-only extraction: never execute the original generator's top-level code."""
    tree = ast.parse(text)
    constants = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
                 if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                 and n.targets[0].id in {'MODEL_ID','SEED','MAX_TOKENS','CANDIDATES'}}
    require(constants == dict(MODEL_ID=MODEL, SEED=SEED, MAX_TOKENS=256, CANDIDATES=2), 'Original generator constants changed')
    calls = [n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='generate']
    require(len(calls)==1, 'Unexpected original generation calls')
    keywords = {k.arg:k.value for k in calls[0].keywords}
    for key, expected in [('do_sample',True),('temperature',0.7),('top_p',0.9)]:
        require(ast.literal_eval(keywords[key]) == expected, f'Original {key} changed')
    require(ast.dump(keywords['max_new_tokens']) == ast.dump(ast.Name(id='MAX_TOKENS',ctx=ast.Load())), 'Original token budget changed')
    helpers = []
    prefix_nodes = []
    for n in tree.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'ANSWER','STOP'} for t in n.targets):
            helpers.append(n)
        if isinstance(n,ast.FunctionDef) and n.name=='parse_number':
            helpers.append(n)
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='prefix' for t in n.targets):
            prefix_nodes.append(n)
        if isinstance(n,ast.For) and any(isinstance(x,ast.AugAssign) and isinstance(x.target,ast.Name) and x.target.id=='prefix' for x in ast.walk(n)):
            prefix_nodes.append(n)
    require(len(helpers)==3 and len(prefix_nodes)==2, 'Unexpected original verifier/prompt structure')
    namespace = dict(re=re, Decimal=Decimal)
    exec(compile(ast.Module(body=helpers,type_ignores=[]),'<saved-verifier>','exec'),namespace)
    return namespace, prefix_nodes

def stopping_class(torch, StoppingCriteria, pattern):
    class StopEachAnswer(StoppingCriteria):
        def __init__(self,tok,prompt_length,count):
            self.tok=tok
            self.prompt_length=prompt_length
            self.lengths=[None]*count
        def __call__(self,input_ids,scores,**kwargs):
            n=input_ids.shape[1]-self.prompt_length
            for i in range(len(self.lengths)):
                if self.lengths[i] is not None:
                    continue
                text=self.tok.decode(input_ids[i,self.prompt_length:],skip_special_tokens=True)
                if pattern.search(text):
                    self.lengths[i]=n
            return torch.tensor([x is not None for x in self.lengths],device=input_ids.device,dtype=torch.bool)
    return StopEachAnswer

def check_unused(rows, previous_records):
    requested={r['dataset_index'] for r in rows}
    require(len(requested)==COUNT and requested==set(range(START,START+COUNT)), 'Wrong full-run indices')
    seen={question_hash(r['question']) for r in rows}
    require(len(seen)==COUNT, 'Duplicate questions in proposed pool')
    for r in previous_records:
        for field in ['dataset_index','index']:
            if field in r:
                require(r[field] not in requested, f'Previously used index: {r[field]}')
        if 'question' in r:
            require(question_hash(r['question']) not in seen, 'Proposed question matches a saved earlier question')

def build_context():
    import torch
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    require(torch.cuda.is_available(), 'CUDA unavailable')
    prior=json.loads((PRIOR_RUN/'manifest.json').read_text(encoding='utf-8'))
    require(prior['model_revision']==MODEL_REVISION, 'Prior evaluation used another model snapshot')
    inputs={str(SOURCE):sha(SOURCE),str(PRIOR_RUN/'manifest.json'):sha(PRIOR_RUN/'manifest.json')}
    for key,expected in prior['input_hashes'].items():
        path=Path('results/gsm_symbolic_pilot_split_v2.json' if key=='split' else key)
        require(sha(path)==expected, f'Protected research input changed: {path}')
        inputs[str(path)]=expected
    split=prior['split']
    require(len(set(split['reserved_original_ids']))==63 and len(set(split['development_original_ids']))==37
            and not set(split['reserved_original_ids']) & set(split['development_original_ids']), 'Family partition changed')
    backup=json.loads(BACKUP_RECEIPT.read_text(encoding='utf-8'))
    require(backup['status']=='verified' and backup['source']==str(Path.cwd()), 'Missing verified research backup')
    archive=Path(backup['archive_wsl'])
    require(archive.is_file() and archive.stat().st_size==backup['archive_bytes'], 'Verified archive unavailable or size changed')
    inputs[str(BACKUP_RECEIPT)]=sha(BACKUP_RECEIPT)
    helpers,prefix_nodes=original_helpers(SOURCE.read_text(encoding='utf-8'))
    data=load_dataset('openai/gsm8k','main',split='train',revision=DATASET_REVISION).shuffle(seed=SEED)
    prefix_ns=dict(re=re,examples=data.select(range(2)))
    exec(compile(ast.Module(body=prefix_nodes,type_ignores=[]),'<saved-prompt>','exec'),prefix_ns)
    prefix=prefix_ns['prefix']
    rows=[]
    for index in range(START,START+COUNT):
        r=data[index]
        gold_match=re.search(r'####\s*(-?[\d,]+(?:\.\d+)?)',r['answer'])
        require(gold_match is not None, 'Unparseable reference answer')
        rows.append(dict(dataset_index=index,question=r['question'],gold=str(Decimal(gold_match.group(1).replace(',','')))))
    old=[dict(dataset_index=i,question=data[i]['question']) for i in range(START)]
    for p in sorted(Path('results').glob('*.jsonl')):
        inputs[str(p)]=sha(p)
        old.extend(json.loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line.strip())
    pilot=json.loads((PILOT_RUN/'manifest.json').read_text(encoding='utf-8'))
    require((pilot['start_index'],pilot['end_index_exclusive'],pilot['max_new_tokens'],pilot['generations'])==(336,536,256,800), 'Unexpected pilot protocol')
    require(sha('gsm8k_difficulty_label_pilot_v1.py')==pilot['source_sha256'], 'Pilot source changed')
    for path,expected in pilot['input_hashes'].items():
        require(sha(path)==expected, f'Pilot protected input changed: {path}')
    for path in [Path('gsm8k_difficulty_label_pilot_v1.py'),PILOT_RUN/'manifest.json',PILOT_RUN/'labels_and_summary.json',DECISION_NOTE]:
        inputs[str(path)]=sha(path)
    pilot_records=[]
    pilot_files=sorted((PILOT_RUN/'batches').glob('batch_*.json'))
    require(len(pilot_files)==100, 'Pilot batches missing or unexpected')
    for path in pilot_files:
        inputs[str(path)]=sha(path)
        pilot_records.extend(json.loads(path.read_text(encoding='utf-8'))['records'])
    require(len(pilot_records)==800 and {(r['dataset_index'],r['candidate']) for r in pilot_records}==
            {(i,c) for i in range(336,536) for c in range(1,5)}, 'Pilot cohort incomplete or duplicated')
    old.extend(pilot_records)
    check_unused(rows,old)
    snapshot=Path(snapshot_download(MODEL,revision=MODEL_REVISION,local_files_only=True))
    require(snapshot.name==MODEL_REVISION and len([p for p in snapshot.parent.iterdir() if p.is_dir()])==1,
            'Model snapshot check failed; stop and review cached snapshots')
    tok=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True)
    require(tok.eos_token_id is not None, 'Tokenizer has no EOS token')
    tok.padding_side='left'
    tok.pad_token=tok.eos_token
    config_path=PRIOR_RUN/'base/generation_config.json'
    generation_config=json.loads(config_path.read_text(encoding='utf-8'))
    inputs[str(config_path)]=sha(config_path)
    seeds=[batch_seed(i) for i in range(COUNT*ATTEMPTS//BATCH_SIZE)]
    require(len(set(seeds))==len(seeds), 'Sampling seed collision')
    metadata={'schema':2,'token_budget_change':'384 from pilot 256, majority cut-offs in unsolved/hard problems',
              'selection_pool':'new 384-token run only; exclude pilot', 'hard_definition':'1–3/4','source_sha256':sha(__file__),'input_hashes':inputs,'model':MODEL,'model_revision':MODEL_REVISION,
              'dataset':'openai/gsm8k','dataset_revision':DATASET_REVISION,'split':'train','shuffle_seed':SEED,
              'start_index':START,'end_index_exclusive':START+COUNT,'questions':COUNT,'attempts_per_question':ATTEMPTS,
              'dataset_train_rows':len(data),'maximum_source_indices_at_or_after_start':len(data)-START,
              'generations':COUNT*ATTEMPTS,'batch_size':BATCH_SIZE,'max_new_tokens':MAX_TOKENS,
              'do_sample':True,'temperature':0.7,'top_p':0.9,'dtype':'float16','padding_side':'left',
              'eos_token_id':tok.eos_token_id,'prefix':prefix,'few_shot_indices':[0,1],
              'sampling_seed_algorithm':'first 32 SHA256 bits of gsm8k-difficulty-full-v1:2026:536:<zero-based batch index>',
              'batch_seeds':seeds,'model_generation_config':generation_config,
              'answer_pattern':helpers['ANSWER'].pattern,'stop_pattern':helpers['STOP'].pattern,
              'scoring':'numeric Decimal equality; original generator answer extractor; malformed numeric fragments are unparseable',
              'rows_sha256':hashlib.sha256(canonical(rows).encode()).hexdigest(),
              'reserved_original_ids':split['reserved_original_ids'],'reserved_family_count':63,
              'backup_archive':backup['archive_windows'],'backup_archive_sha256':backup['archive_sha256'],
              'python':platform.python_version(),'gpu':torch.cuda.get_device_name(0),
              'versions':{p:importlib.metadata.version(p) for p in ['torch','transformers','datasets','accelerate']}}
    return rows,prefix,helpers,tok,snapshot,metadata

def planned_batch(rows,index):
    first=index*(BATCH_SIZE//ATTEMPTS)
    return [(r,candidate) for r in rows[first:first+BATCH_SIZE//ATTEMPTS] for candidate in range(1,ATTEMPTS+1)]

def check_batch(obj,planned,index,parse):
    require(obj['batch_index']==index and obj['sampling_seed']==batch_seed(index), 'Saved batch identity mismatch')
    records=obj['records']
    require(len(records)==len(planned)==BATCH_SIZE, 'Incomplete saved batch')
    for r,(expected,candidate) in zip(records,planned):
        require((r['dataset_index'],r['candidate'],r['question'],r['gold'],r['sampling_seed'])==
                (expected['dataset_index'],candidate,expected['question'],expected['gold'],batch_seed(index)), 'Saved candidate mismatch')
        prediction=parse(r['response'])
        require(r['prediction']==(str(prediction) if prediction is not None else None), 'Saved extraction mismatch')
        require(type(r['correct']) is bool and r['correct']==(prediction==Decimal(r['gold'])), 'Saved numeric score mismatch')
        require(type(r['tokens']) is int and 1<=r['tokens']<=MAX_TOKENS, 'Invalid token count')
    return records

def load_batches(out,rows,parse):
    expected={out/'batches'/f'batch_{i:03d}.json' for i in range(COUNT*ATTEMPTS//BATCH_SIZE)}
    require(set((out/'batches').glob('batch_*.json'))<=expected, 'Unexpected saved batch filenames')
    completed={}
    missing=[]
    for i in range(len(expected)):
        p=out/'batches'/f'batch_{i:03d}.json'
        if p.exists():
            obj=json.loads(p.read_text(encoding='utf-8'))
            check_batch(obj,planned_batch(rows,i),i,parse)
            completed[i]=obj
        else:
            missing.append(i)
    return completed,missing

def summarize(completed):
    require(set(completed)==set(range(COUNT*ATTEMPTS//BATCH_SIZE)), 'Cannot label an incomplete run')
    by_question={}
    for i in sorted(completed):
        for r in completed[i]['records']:
            by_question.setdefault(r['dataset_index'],[]).append(r)
    require(set(by_question)==set(range(START,START+COUNT)), 'Wrong summary cohort')
    counts=[0]*5
    labels=[]
    for index in sorted(by_question):
        attempts=by_question[index]
        require(len(attempts)==4 and {r['candidate'] for r in attempts}=={1,2,3,4}, 'Duplicate/missing attempts')
        correct=sum(r['correct'] for r in attempts)
        counts[correct]+=1
        label='unsolved_in_four_attempts' if correct==0 else 'hard' if correct in [1,2,3] else 'easy'
        labels.append(dict(dataset_index=index,question=attempts[0]['question'],gold=attempts[0]['gold'],
                           correct_out_of_four=correct,parsed_out_of_four=sum(r['prediction'] is not None for r in attempts),
                           label=label,correct_candidate_ids=[r['candidate'] for r in attempts if r['correct']]))
    eligible=COUNT-counts[0]
    non_easy=sum(counts[1:4])
    def projection(n):
        return {'300_unique_questions':math.ceil(300*COUNT/n),'500_unique_questions':math.ceil(500*COUNT/n)} if n else None
    return {'questions':COUNT,'attempts':COUNT*ATTEMPTS,'counts_by_correct_out_of_four':counts,
            'questions_with_any_verified_solution':eligible,'strict_hard_1_of_4':counts[1],
            'non_easy_solvable_1_to_3_of_4':non_easy,'hard_1_to_3_of_4':non_easy,'easy_4_of_4':counts[4],
            'can_fill_300_per_group':eligible>=300 and non_easy>=300 and counts[4]>=300,'parsed_attempts':sum(r['parsed_out_of_four'] for r in labels),
            'numerically_correct_attempts':sum(i*n for i,n in enumerate(counts)),
            'timed_generation_and_decode_minutes':sum(b['elapsed_seconds'] for b in completed.values())/60,
            'peak_reserved_gib':max(b['peak_reserved_gib'] for b in completed.values()),
            'rough_pool_projections':{'any_verified':projection(eligible),'strict_hard':projection(counts[1]),'non_easy_solvable':projection(non_easy)},
            'termination_counts':{reason:sum(r['termination']==reason for b in completed.values() for r in b['records']) for reason in ['answer','eos','token_limit']},
            'token_limits_by_correct_out_of_four':[sum(r['termination']=='token_limit' for q,rs in by_question.items() if sum(x['correct'] for x in rs)==k for r in rs) for k in range(5)],
            'notes':'No training or extra attempts. Four-attempt labels are noisy and protocol-specific; 0/4 is not proof of impossibility. Identical sampled responses are retained as legitimate draws. Pool projections assume pilot fractions persist and are not guaranteed. At most one unique question per training example is assumed.',
            'labels':labels}

def run(args):
    rows,prefix,helpers,tok,snapshot,metadata=build_context()
    parse=lambda text:safe_parse(helpers['parse_number'],text)
    if args.preflight:
        print('PASS: verified outside-WSL backup; frozen base snapshot; 1100 unused indices 536–1635; pilot protected/excluded; dated decision note; original prompt/verifier; 384 tokens; 63 reserved families protected. No model inference.')
        return
    out=validate_output(args.output)
    out.mkdir(parents=True,exist_ok=True)
    if args.validate_only:
        require((out/'manifest.json').exists(), 'No saved run to validate')
    freeze(out/'manifest.json',metadata)
    completed,missing=load_batches(out,rows,parse)
    print(f'Validated {len(completed)*BATCH_SIZE}/{COUNT*ATTEMPTS} saved attempts.',flush=True)
    if args.validate_only:
        return
    if missing:
        import torch
        from transformers import AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
        torch.manual_seed(SEED)
        model=AutoModelForCausalLM.from_pretrained(str(snapshot),dtype=torch.float16,device_map='cuda:0',local_files_only=True)
        model.eval()
        require(model.generation_config.to_dict()==metadata['model_generation_config'], 'Base generation defaults differ from previous evaluation; stop and inspect')
        StopEachAnswer=stopping_class(torch,StoppingCriteria,helpers['STOP'])
        torch.cuda.reset_peak_memory_stats()
        limit=len(missing) if args.max_new_batches is None else min(len(missing),args.max_new_batches)
        for i in missing[:limit]:
            planned=planned_batch(rows,i)
            began=time.perf_counter()
            prompts=[prefix+f"Question: {r['question']}\nSolution:" for r,_ in planned]
            inputs=tok(prompts,padding=True,return_tensors='pt').to('cuda')
            prompt_length=inputs['input_ids'].shape[1]
            stopper=StopEachAnswer(tok,prompt_length,len(planned))
            torch.manual_seed(batch_seed(i))
            with torch.inference_mode():
                output=model.generate(**inputs,max_new_tokens=MAX_TOKENS,do_sample=True,temperature=0.7,top_p=0.9,
                                      stopping_criteria=StoppingCriteriaList([stopper]),pad_token_id=tok.eos_token_id)
            generated=output[:,prompt_length:]
            require(generated.shape[0]==len(planned), 'Unexpected generation count')
            records=[]
            for j,(r,candidate) in enumerate(planned):
                ids=generated[j].tolist()
                count=stopper.lengths[j]
                termination='answer'
                if count is None:
                    if tok.eos_token_id in ids:
                        count=ids.index(tok.eos_token_id)+1
                        termination='eos'
                    else:
                        count=len(ids)
                        termination='token_limit'
                response=tok.decode(ids[:count],skip_special_tokens=True)
                prediction=parse(response)
                records.append(dict(dataset_index=r['dataset_index'],candidate=candidate,question=r['question'],gold=r['gold'],
                                    prediction=str(prediction) if prediction is not None else None,correct=prediction==Decimal(r['gold']),
                                    response=response,tokens=count,termination=termination,sampling_seed=batch_seed(i)))
            obj={'batch_index':i,'sampling_seed':batch_seed(i),'records':records,'elapsed_seconds':time.perf_counter()-began,
                 'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3}
            check_batch(obj,planned,i,parse)
            publish(out/'batches'/f'batch_{i:03d}.json',obj)
            completed[i]=obj
            print(f'{len(completed)*BATCH_SIZE}/{COUNT*ATTEMPTS} attempts saved; peak {obj["peak_reserved_gib"]:.2f} GiB.',flush=True)
            del inputs,output,generated
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    if len(completed)==COUNT*ATTEMPTS//BATCH_SIZE:
        result=summarize(completed)
        freeze(out/'labels_and_summary.json',result)
        print('FULL LABELING COMPLETE. Counts for 0/4, 1/4, 2/4, 3/4, 4/4:',result['counts_by_correct_out_of_four'])
        print(f'Timed generation/decode: {result["timed_generation_and_decode_minutes"]:.2f} minutes; peak {result["peak_reserved_gib"]:.2f} GiB.')
    else:
        print('Bounded check complete. Repeat the normal command to resume; no labels published for an incomplete run.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='results/gsm8k_difficulty_label_full_v1')
    p.add_argument('--preflight',action='store_true')
    p.add_argument('--validate-only',action='store_true')
    p.add_argument('--max-new-batches',type=int)
    args=p.parse_args()
    require(args.max_new_batches is None or args.max_new_batches>=1,'Batch limit must be at least one')
    validate_output(args.output)
    import fcntl
    lock_path=Path(args.output+'.lock')
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args)
