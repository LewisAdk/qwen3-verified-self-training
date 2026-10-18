"""Approved dated stages. Full training is deliberately unavailable."""
import os
os.environ.update(HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
import argparse
import ast
from collections import Counter
from decimal import Decimal
import hashlib
import importlib.util
import importlib.metadata
import json
from pathlib import Path
import random
import re
import statistics
import subprocess
import threading
import time
from types import SimpleNamespace
from scorer_v2 import score_v2

ROOT=Path('/home/lewis/rsi-research')
RUN=ROOT/'results/scaled_random_2026-10-13_n2000'
SNAPSHOT=Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3-1.7B-Base/snapshots/ea980cb0a6c2ae4b936e82123acc929f1cec04c1'
DATA=Path.home()/'.cache/huggingface/datasets/openai___gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866/gsm8k-train.arrow'
TRAIN_SHA='d8381ab86bb6020293121dc5dd32dac113901fbb8ddac8540fe4d777cea5a1b9'
LABEL_SHA='54c0825b0199d514aff74128014677f86b6bf4a451236a57cc2628a7a4644efe'
BASE_MANIFEST=ROOT/'results/gsm_symbolic_rq2_primary_v1/manifest.json'
WIN=Path('/mnt/c/Users/Lewis/Documents/Codex/2026-10-12/referenced-chatgpt-conversation-this-is-an/outputs')

def require(ok,message):
    if not ok:raise RuntimeError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())

def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        json.dump(obj,f,indent=2,ensure_ascii=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def load(name,expected=None):
    path=ROOT/name
    if expected:require(sha(path)==expected,'Unchanged source mismatch: '+name)
    spec=importlib.util.spec_from_file_location('frozen_'+path.stem,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def check_protected(receipt=None):
    baseline=read(RUN/'protected_baseline.json');bad=[]
    for p,h in baseline['research']+baseline['windows']:
        if not Path(p).is_file() or sha(p)!=h:bad.append(p)
    require(not bad,'Protected files changed: '+str(bad))
    result=dict(verified=len(baseline['research'])+len(baseline['windows']),mismatches=[],time_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat())
    if receipt:save(RUN/receipt,result)
    return result

def check_sources():
    mf=read(RUN/'execution_source_manifest.json')
    for p,h in mf['sources'].items():require(sha(p)==h,'Frozen new source changed: '+p)
    require(sha(RUN/'frozen_spec.json')==mf['spec_sha256'],'Frozen spec changed')
    require(read(RUN/'cpu_validation.json')['passed'],'CPU validation not passed')

def sampling_seed(global_batch_index):
    return int.from_bytes(hashlib.sha256(f'gsm8k-difficulty-full-v1:2026:1636:{global_batch_index}'.encode()).digest()[:4],'big')

def stream(label):return random.Random(int.from_bytes(hashlib.sha256(f'rsi-scaled-random-v1:2026:{label}'.encode()).digest()[:8],'big'))

def choices(by_problem,N=2000):
    eligible=sorted(i for i,rs in by_problem.items() if any(r['correct'] for r in rs))
    require(len(eligible)>=N,'Insufficient verified yield')
    ids=stream(f'sample:{N}').sample(eligible,N)
    return [stream(f'solution:{i}').choice(sorted([r for r in by_problem[i] if r['correct']],key=lambda r:r['candidate'])) for i in ids]

def length_guard(rows):
    bad=[r['dataset_index'] for r in rows if r['full_sequence_tokens']>1024]
    require(not bad,'Selected sequences exceed 1024; no replacements: '+str(bad))

class Monitor:
    def __init__(self,path):self.path=path;self.stop=threading.Event();self.rows=[];self.thread=None
    def __enter__(self):
        def run():
            with self.path.open('x') as f:
                while not self.stop.is_set():
                    try:
                        text=subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits','-i','0'],text=True,timeout=5)
                        used,total=map(float,text.strip().split(','));row=dict(time=time.time(),used_gib=used/1024,total_gib=total/1024)
                    except Exception as e:row=dict(time=time.time(),error=str(e))
                    self.rows.append(row);f.write(json.dumps(row)+'\n');f.flush();self.stop.wait(1)
        self.thread=threading.Thread(target=run,daemon=True);self.thread.start();return self
    def __exit__(self,*args):self.stop.set();self.thread.join(7)
    def summary(self):
        rows=[r for r in self.rows if 'used_gib' in r]
        return dict(sampled_peak_device_wide_gib=max([r['used_gib'] for r in rows],default=None),total_device_gib=rows[0]['total_gib'] if rows else None,samples=len(rows),query_errors=len(self.rows)-len(rows),sampling_interval_seconds=1,brief_peaks_may_be_missed=True,device_wide_not_process_specific=True)

def data_tokenizer():
    from datasets import Dataset
    from transformers import AutoTokenizer
    data=Dataset.from_file(str(DATA)).shuffle(seed=2026,keep_in_memory=True)
    require(len(data)==7473,'Train row count changed')
    tok=AutoTokenizer.from_pretrained(str(SNAPSHOT),local_files_only=True)
    return data,tok

def deterministic_check():
    check_sources();check_protected('before_determinism_hashes.json')
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,StoppingCriteria,StoppingCriteriaList
    meta=read(BASE_MANIFEST)
    require({p:importlib.metadata.version(p) for p in meta['versions']}==meta['versions'],'Saved evaluator versions differ')
    base=[json.loads(l) for l in (WIN/'gold_control_base_responses.jsonl').read_text().splitlines()]
    require(len(base)==400 and len({(r['variant'],r['id'],r['instance']) for r in base})==400,'Bad base alignment')
    require({r['original_id'] for r in base}==set(meta['split']['development_original_ids']),'Wrong development families')
    old=load('gsm_symbolic_expanded_dev_v1.py')
    ns=dict(torch=torch,re=re,Decimal=Decimal,StoppingCriteria=StoppingCriteria)
    old.exact_helpers(ast.parse((ROOT/'gsm_symbolic_pilot_v2.py').read_text()),ns)
    folder=RUN/'determinism_base_batch4';require(not folder.exists(),'Determinism folder already exists');folder.mkdir()
    save(folder/'manifest.json',dict(batch_size=4,reference_batch_size=8,questions=400,prefix=meta['prefix'],saved_base_sha256=sha(WIN/'gold_control_base_responses.jsonl'),generation_config=meta['base_generation_config'],versions=meta['versions'],source_sha256=sha(__file__)))
    started=time.monotonic();tok=AutoTokenizer.from_pretrained(str(SNAPSHOT),local_files_only=True);tok.padding_side='left'
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    records=[]
    with Monitor(folder/'device_memory.jsonl') as monitor:
        torch.manual_seed(2026);torch.cuda.reset_peak_memory_stats()
        model=AutoModelForCausalLM.from_pretrained(str(SNAPSHOT),dtype=torch.float16,device_map='cuda:0',local_files_only=True);model.eval()
        require(model.generation_config.to_dict()==meta['base_generation_config'],'Generation defaults differ')
        for start in range(0,400,4):
            require(time.monotonic()-started<1800,'Determinism wall budget reached')
            batch=base[start:start+4];prompts=[meta['prefix']+f"Question: {r['question']}\nSolution:" for r in batch]
            inputs=tok(prompts,padding=True,return_tensors='pt').to('cuda');plen=inputs['input_ids'].shape[1];stopper=ns['StopAnswers'](tok,plen,len(batch));began=time.monotonic()
            with torch.inference_mode():output=model.generate(**inputs,max_new_tokens=384,do_sample=False,stopping_criteria=StoppingCriteriaList([stopper]),pad_token_id=tok.pad_token_id)
            generated=output[:,plen:];rs=[]
            for j,b in enumerate(batch):
                ids=generated[j].tolist();n=stopper.lengths[j]
                if n is None:n=ids.index(tok.eos_token_id)+1 if tok.eos_token_id in ids else len(ids)
                response=tok.decode(ids[:n],skip_special_tokens=True);prediction=ns['parse'](response)
                r={k:b[k] for k in ['variant','id','instance','original_id','question','gold']}
                r.update(model='base_batch4',response=response,tokens=n,prediction=prediction,correct=prediction==b['gold'],secondary_v2=score_v2(response,b['gold']))
                require(1<=n<=384,'Token guard');rs.append(r)
            save(folder/'batches'/f'batch_{start:03d}.json',dict(records=rs,elapsed_seconds=time.monotonic()-began));records.extend(rs)
            print(f'DETERMINISM {start+4}/400',flush=True);del inputs,output,generated
        allocated=torch.cuda.max_memory_allocated()/1024**3;reserved=torch.cuda.max_memory_reserved()/1024**3
        del model;__import__('gc').collect();torch.cuda.empty_cache()
    def summary(pairs):
        return dict(questions=len(pairs),same_parsed_answer=sum(b['prediction'] is not None and a['prediction'] is not None and a['prediction']==b['prediction'] for b,a in pairs),both_null=sum(a['prediction'] is None and b['prediction'] is None for b,a in pairs),prediction_field_changes=sum(a['prediction']!=b['prediction'] for b,a in pairs),verdict_flips=sum(a['correct']!=b['correct'] for b,a in pairs),fixes=sum(not b['correct'] and a['correct'] for b,a in pairs),regressions=sum(b['correct'] and not a['correct'] for b,a in pairs),reference_correct=sum(b['correct'] for b,a in pairs),rerun_correct=sum(a['correct'] for b,a in pairs),exact_text=sum(a['response']==b['response'] for b,a in pairs),normalized_text=sum(' '.join(a['response'].split())==' '.join(b['response'].split()) for b,a in pairs),missing_transitions=sum((a['prediction'] is None)!=(b['prediction'] is None) for b,a in pairs),reference_caps=sum(b['tokens']==384 for b,a in pairs),rerun_caps=sum(a['tokens']==384 for b,a in pairs))
    pairs=list(zip(base,records));result=dict(status='complete',pooled=summary(pairs),variants={v:summary([(b,a) for b,a in pairs if b['variant']==v]) for v in ['p1','p2']},changed_records=[dict(key=[b['variant'],b['id'],b['instance']],before_prediction=b['prediction'],after_prediction=a['prediction'],before_correct=b['correct'],after_correct=a['correct']) for b,a in pairs if b['prediction']!=a['prediction']],secondary_v2_correct=sum(a['secondary_v2']['correct'] for a in records),secondary_v2_changes=sum(a['secondary_v2']['correct']!=a['correct'] for a in records),elapsed_wall_minutes=(time.monotonic()-started)/60,peak_allocated_gib=allocated,peak_reserved_gib=reserved,device_wide=monitor.summary())
    save(folder/'summary.json',result);check_protected('after_determinism_hashes.json');print('DETERMINISM_RESULT '+json.dumps(result),flush=True)

def label_context(engine,start,count,offset):
    data,tok=data_tokenizer();oldmeta=read(ROOT/'results/gsm8k_difficulty_label_full_v1/manifest.json')
    helpers,nodes=engine.original_helpers((ROOT/'generate_verified.py').read_text())
    ns=dict(re=re,examples=data.select(range(2)));exec(compile(ast.Module(body=nodes,type_ignores=[]),'<unchanged-label-prefix>','exec'),ns)
    prefix=ns['prefix'];require(prefix==oldmeta['prefix'],'Label prefix changed')
    rows=[]
    for i in range(start,start+count):
        r=data[i];match=re.search(r'####\s*(-?[\d,]+(?:\.\d+)?)',r['answer']);require(match is not None,'Bad train reference')
        rows.append(dict(dataset_index=i,question=r['question'],gold=str(Decimal(match.group(1).replace(',','')))))
    engine.check_unused(rows,[dict(dataset_index=i,question=data[i]['question']) for i in range(start)])
    tok.padding_side='left';tok.pad_token=tok.eos_token
    protected={p:h for p,h in read(RUN/'protected_baseline.json')['research']}
    metadata=dict(schema=1,source_sha256=sha(__file__),label_engine_sha256=LABEL_SHA,spec_sha256=sha(RUN/'frozen_spec.json'),input_hashes=protected,start_index=start,end_index_exclusive=start+count,questions=count,generations=count*4,attempts_per_question=4,batch_size=8,global_batch_offset=offset,batch_seeds=[sampling_seed(offset+i) for i in range(count//2)],max_new_tokens=384,temperature=.7,top_p=.9,dtype='float16',prefix=prefix,answer_pattern=helpers['ANSWER'].pattern,stop_pattern=helpers['STOP'].pattern,model_generation_config=oldmeta['model_generation_config'],model_revision=oldmeta['model_revision'],dataset_revision=oldmeta['dataset_revision'],sampling_seed_algorithm='first 32 SHA256 bits of gsm8k-difficulty-full-v1:2026:1636:<global batch index>',rows_sha256=hashlib.sha256(engine.canonical(rows).encode()).hexdigest())
    require(len(set(metadata['batch_seeds']))==len(metadata['batch_seeds']),'Batch seed collision')
    return rows,prefix,helpers,tok,SNAPSHOT,metadata

def labeling():
    check_sources();require(read(RUN/'determinism_base_batch4/summary.json')['status']=='complete','Determinism must finish first');check_protected('before_labeling_hashes.json')
    all_batches=[];start=1636;count=2400;offset=0;started=time.monotonic();blocks=[]
    while True:
        require(start+count<=7473,'No full top-up block fits; stop')
        require(time.monotonic()-started<18000,'Cumulative labeling wall budget reached')
        engine=load('gsm8k_difficulty_label_full_v1.py',LABEL_SHA);engine.START=start;engine.COUNT=count;engine.batch_seed=lambda i:sampling_seed(offset+i)
        engine.build_context=lambda:label_context(engine,start,count,offset)
        original_check=engine.check_batch
        def checked(obj,planned,index,parse):
            require(time.monotonic()-started<18000,'Cumulative labeling wall budget reached')
            return original_check(obj,planned,index,parse)
        engine.check_batch=checked
        folder=RUN/'labeling'/f'block_{start}_{start+count-1}';require(not folder.exists(),'Existing labeling block preserved; explicit validated resume required')
        engine.run(SimpleNamespace(output=str(folder),preflight=False,validate_only=False,max_new_batches=None))
        parse=lambda text:engine.safe_parse(engine.original_helpers((ROOT/'generate_verified.py').read_text())[0]['parse_number'],text)
        # Validate with frozen source helpers and every exact expected candidate identity.
        rows,_,helpers,_,_,_=label_context(engine,start,count,offset)
        completed,missing=engine.load_batches(folder,rows,lambda t:engine.safe_parse(helpers['parse_number'],t));require(not missing,'Incomplete label block')
        all_batches.extend(completed.values());blocks.append(dict(start=start,count=count,global_batch_offset=offset,folder=str(folder),manifest_sha256=sha(folder/'manifest.json')))
        by={}
        for batch in all_batches:
            for r in batch['records']:by.setdefault(r['dataset_index'],[]).append(r)
        require(len(by)==sum(b['count'] for b in blocks) and all(len(rs)==4 and {r['candidate'] for r in rs}=={1,2,3,4} for rs in by.values()),'Duplicate/missing attempts')
        mix=Counter(sum(r['correct'] for r in rs) for rs in by.values());eligible=len(by)-mix[0]
        print(f'LABEL_BLOCK_RESULT questions={len(by)} eligible={eligible} mix={dict(mix)}',flush=True)
        if eligible>=2000:break
        offset+=count//2;start+=count;count=200
    save(RUN/'labeling/yield_summary.json',dict(status='validated',questions=len(by),attempts=4*len(by),eligible=eligible,counts_by_correct_out_of_four=[mix[i] for i in range(5)],blocks=blocks,elapsed_wall_minutes=(time.monotonic()-started)/60,generation_decode_minutes=sum(b['elapsed_seconds'] for b in all_batches)/60))
    check_protected('after_labeling_hashes.json')

def selected_records():
    summary=read(RUN/'labeling/yield_summary.json');by={}
    for block in summary['blocks']:
        for p in sorted((Path(block['folder'])/'batches').glob('batch_*.json')):
            for r in read(p)['records']:by.setdefault(r['dataset_index'],[]).append(r)
    return by,choices(by)

def freeze_set():
    check_sources();check_protected('before_set_hashes.json');by,selected=selected_records();data,tok=data_tokenizer()
    prefix=read(ROOT/'results/training_sets_primary_tokenmatched_v1/manifest.json')
    # The trainer's exact prompt is generated from its original pure AST nodes.
    tree=ast.parse((ROOT/'train_verified_pilot.py').read_text());nodes=[n for n in tree.body if (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='prefix' for t in n.targets)) or (isinstance(n,ast.For) and any(isinstance(x,ast.AugAssign) and isinstance(x.target,ast.Name) and x.target.id=='prefix' for x in ast.walk(n)))]
    ns=dict(re=re,data=data);exec(compile(ast.Module(body=nodes,type_ignores=[]),'<unchanged-training-prefix>','exec'),ns);prefix=ns['prefix']
    require(prefix==read(ROOT/'results/gsm8k_difficulty_label_full_v1/manifest.json')['prefix'],'Training prefix differs')
    folder=RUN/'frozen_set';require(not folder.exists(),'Frozen set already exists');folder.mkdir()
    rows=[]
    for r in selected:
        i=r['dataset_index'];require(r['question']==data[i]['question'],'Question changed');solution=r['response'].strip()
        pids=tok.encode(prefix+f"Question: {r['question']}\nSolution:",add_special_tokens=False);tids=tok.encode(' '+solution+'\n',add_special_tokens=False)
        rows.append(dict(dataset_index=i,question=r['question'],answer=r['gold'],solution=solution,candidate=r['candidate'],correct_out_of_four=sum(x['correct'] for x in by[i]),prompt_tokens=len(pids),solution_tokens=len(tids),supervised_tokens=len(tids)+1,full_sequence_tokens=len(pids)+len(tids)+1))
    require(len(rows)==len({r['dataset_index'] for r in rows})==2000,'N/uniqueness mismatch')
    # Freeze ALL choices before reporting a guard failure; do not replace anything.
    with (folder/'random_2000.jsonl').open('x') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=True)+'\n')
    def stats(field):
        a=[r[field] for r in rows];return dict(min=min(a),max=max(a),mean=statistics.mean(a),median=statistics.median(a),total=sum(a))
    result=dict(examples=2000,prefix=prefix,source_sha256=sha(__file__),set_sha256=sha(folder/'random_2000.jsonl'),selection_rng='first 64 SHA256 bits of rsi-scaled-random-v1:2026:sample:2000',candidate_rng='first 64 SHA256 bits of rsi-scaled-random-v1:2026:solution:<index>',difficulty_mix={str(i):sum(r['correct_out_of_four']==i for r in rows) for i in [1,2,3,4]},statistics={k:stats(k) for k in ['prompt_tokens','solution_tokens','supervised_tokens','full_sequence_tokens']},overlength_indices=[r['dataset_index'] for r in rows if r['full_sequence_tokens']>1024],train_seeds=[2026,2027,2028],expected_updates_per_seed=1000,full_training_authorized=False)
    save(folder/'manifest.json',result);length_guard(rows)
    save(folder/'length_check_passed.json',dict(passed=True,max_length=max(r['full_sequence_tokens'] for r in rows),limit=1024,no_truncation=True));check_protected('after_set_hashes.json');print('FROZEN_SET_RESULT '+json.dumps(result),flush=True)

def probe_engine():
    engine=load('train_primary_lora_v1.py',TRAIN_SHA)
    original_execute=engine.execute
    def context():
        require(read(RUN/'frozen_set/length_check_passed.json')['passed'],'Length check not passed')
        manifest=read(RUN/'frozen_set/manifest.json');p=RUN/'frozen_set/random_2000.jsonl';require(sha(p)==manifest['set_sha256'],'Frozen set changed')
        data,tok=data_tokenizer();rows=[json.loads(l) for l in p.read_text().splitlines()];length_guard(rows);examples=[]
        for r in rows:
            require(r['question']==data[r['dataset_index']]['question'],'Training question mismatch')
            match=re.search(r'(?m)^####[ \t]*(-?[\d,]+(?:\.\d+)?)[ \t]*$',r['solution'].strip());require(match and Decimal(match.group(1).replace(',',''))==Decimal(r['answer']),'Original engine target verification failed')
            pids=tok.encode(manifest['prefix']+f"Question: {r['question']}\nSolution:",add_special_tokens=False);tids=tok.encode(' '+r['solution'].strip()+'\n',add_special_tokens=False);ids=pids+tids+[tok.eos_token_id];labels=[-100]*len(pids)+tids+[tok.eos_token_id]
            require(len(ids)==r['full_sequence_tokens'] and len(tids)+1==r['supervised_tokens'],'Token count differs');examples.append(dict(dataset_index=r['dataset_index'],input_ids=ids,labels=labels,supervised_tokens=len(tids)+1))
        require(len(examples)==2000,'Wrong N')
        protected={p:h for p,h in read(RUN/'protected_baseline.json')['research']}
        for p in [RUN/'frozen_spec.json',RUN/'execution_source_manifest.json',RUN/'cpu_validation.json',RUN/'frozen_set/manifest.json',RUN/'frozen_set/random_2000.jsonl',RUN/'frozen_set/length_check_passed.json']:protected[str(p)]=sha(p)
        protected.update(read(RUN/'execution_source_manifest.json')['sources'])
        fingerprint=hashlib.sha256(json.dumps(dict(settings=engine.SETTINGS,set=manifest['set_sha256'],spec=sha(RUN/'frozen_spec.json')),sort_keys=True).encode()).hexdigest()
        return {'random':examples},protected,SNAPSHOT,fingerprint
    engine.prepare_context=context
    engine.longest_examples=lambda groups:sorted(groups['random'],key=lambda r:(-len(r['input_ids']),r['dataset_index']))[:12]
    require(engine.execute is original_execute,'Engine execute changed')
    return engine

def memory_probe():
    check_sources();check_protected('before_probe_hashes.json');engine=probe_engine()
    # Only the unchanged engine's PROBE branch is reachable in this entry point.
    args=SimpleNamespace(phase='probe',group='random',seed=2026,output=str(RUN/'longest_sequence_probe'),probe_report=None,additional_runs_confirmed=False)
    with Monitor(RUN/'probe_device_memory.jsonl') as monitor:engine.execute(args)
    result=read(RUN/'longest_sequence_probe/completed.json');require(result['status']=='memory_probe_passed' and result['successful_optimizer_updates']==3 and result['skipped_optimizer_updates']==0 and not result['final_adapter_saved'],'Probe guard failed')
    save(RUN/'probe_device_memory_summary.json',monitor.summary());check_protected('after_probe_hashes.json');print('PROBE_COMPLETE_STOP_NO_FULL_TRAINING',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['determinism','labeling','freeze_set','probe','hashes']);args=p.parse_args();os.chdir(ROOT)
    try:
        {'determinism':deterministic_check,'labeling':labeling,'freeze_set':freeze_set,'probe':memory_probe,'hashes':lambda:check_protected('final_hash_receipt.json')}[args.stage]()
    except BaseException as e:
        save(RUN/f'failure_{args.stage}_{time.time_ns()}.json',dict(stage=args.stage,error_type=type(e).__name__,message=str(e),no_settings_changed=True,full_training_started=False));raise
