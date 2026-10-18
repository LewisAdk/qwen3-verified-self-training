"""CPU-only validation before any approved GPU stage."""
import ast
from collections import Counter
from decimal import Decimal
import hashlib
import json
import random
import re
import statistics
import tempfile
from pathlib import Path
import numpy as np
import stages as st
from scorer_v2 import parse_v2,score_v2
import analyzer_wrapper as aw

passed=[]
def check(name,condition):
    assert condition,name;passed.append(name)

def rejects(fn):
    try:fn()
    except (RuntimeError,FileExistsError):return True
    return False

cases=[(' 50\n','50'),(' -12.50\n','-12.5'),(' 1,234.50\n','1234.5'),(' ¥436.50\n','436.5'),(' 80%\n','80'),(' +2\n','2'),(' 0.0\n','0'),(' -0\n','0'),(' 7 pounds\n','7'),(' £9\r\n','9')]
for raw,expected in cases:check('v2_accept_'+repr(raw),parse_v2('####'+raw)['prediction']==expected)
for raw in ['75,35','18,48','60, 40','10G / 3','790/11','9/11','2+3','50 because that is the result','\nQuestion: 50','10G / 3\n#### 50','Error','','NaN','Infinity','--1','1,23,456','1,234,','1..2','50 meters','₿50','50.','1e2']:
    check('v2_reject_'+repr(raw),parse_v2('####'+raw)['prediction'] is None)
check('v2_missing_marker',parse_v2('Question: 50')['prediction'] is None)
check('v2_continuation_retains_first',parse_v2('#### 80%\nQuestion: x\n#### 1820')['prediction']=='80' and parse_v2('#### 80%\nQuestion: x\n#### 1820')['continuation'])
check('v2_blank_tail_not_continuation',not parse_v2('#### 50\n\n')['continuation'])
check('v2_currency_no_rounding',not score_v2('#### ¥436.50\n','436')['correct'])
check('v2_decimal_equality',score_v2('#### 0.0\n','0')['correct'])
check('v2_unchanged_first_invalid',parse_v2('####\nQuestion: 4\n#### 4')['prediction'] is None)
ns=dict(re=re,Decimal=Decimal)
tree=ast.parse((st.ROOT/'gsm_symbolic_cluster_analysis_v1.py').read_text())
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['parse','percentile'] or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ANSWER' for t in n.targets)]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'<old-pure-parser>','exec'),ns)
check('v1_preserved_comma_bug',ns['parse']('#### 75,35\n')=='7535')
check('v1_preserved_expression_prefix',ns['parse']('#### 10G / 3\n')=='10')

oldmodels=['base']+[f'{g}_seed_{s}' for g in ['hard','random','easy','gold'] for s in [2026,2027,2028]]
data={}
for m in oldmodels:
    fn='gold_control_base_responses.jsonl' if m=='base' else f'{"gold_control" if m.startswith("gold") else "rq2"}_{m}_responses.jsonl'
    data[m]=[json.loads(x) for x in (st.WIN/fn).read_text().splitlines()]
    # v1 verification only. Never apply v2 to historical rows.
    check('old_v1_identity_'+m,all(ns['parse'](r['response'])==r['prediction'] for r in data[m]))

idx=lambda rs:{(r['variant'],r['id'],r['instance']):r for r in rs}
goldidx={m:idx(data[m]) for m in ['base']+[f'{g}_seed_{s}' for g in ['gold','hard'] for s in [2026,2027,2028]]}
actual=aw.original_function()(goldidx,10000,2026)
expected=st.read(st.ROOT/'results/gsm_symbolic_gold_control_v2_analysis.json')['result']
check('gold_saved_result_exact',actual==expected)

rqtree=ast.parse((st.ROOT/'gsm_symbolic_rq2_analysis_v1.py').read_text());fn=next(n for n in rqtree.body if isinstance(n,ast.FunctionDef) and n.name=='summarize')
models=[f'{g}_seed_{s}' for g in ['hard','random','easy'] for s in [2026,2027,2028]]
rqns=dict(np=np,Counter=Counter,Decimal=Decimal,statistics=statistics,random=random,percentile=ns['percentile'],GROUPS=['hard','random','easy'],SEEDS=[2026,2027,2028],MODELS=models,ALL_MODELS=['base']+models,require=st.require)
exec(compile(ast.Module(body=[fn],type_ignores=[]),'<unchanged-RQ2-analysis>','exec'),rqns)
rqactual=rqns['summarize']({m:idx(data[m]) for m in ['base']+models},10000,2026)
rqexpected=st.read(st.ROOT/'results/gsm_symbolic_rq2_primary_v1_analysis.json')['result']
check('rq2_saved_result_exact',rqactual==rqexpected)
mapped=aw.family_bootstrap(data['base'],{s:data[f'gold_seed_{s}'] for s in [2026,2027,2028]},{s:data[f'hard_seed_{s}'] for s in [2026,2027,2028]})
check('new_family_wrapper_exact',mapped==json.loads(json.dumps(expected).replace('gold','scaled_random').replace('hard','original_random')))
hier=aw.seed_family_bootstrap(data['base'],{s:data[f'gold_seed_{s}'] for s in [2026,2027,2028]},{s:data[f'hard_seed_{s}'] for s in [2026,2027,2028]})
check('seed_family_exact_prior_sensitivity',hier==[[-15.250328947368422,3.3333333333333335],[-16.904907084785133,.8108657437925723]])
same=[dict(r,correct=True,prediction=r['gold']) for r in data['base']]
zero=aw.seed_family_bootstrap(same,{s:same for s in [2026,2027,2028]},repetitions=100)
check('paired_cluster_identical_zero',zero[0]==[0,0])
check('success_both_required',aw.success([1,2],[1,2])=='success')
check('success_family_only_label',aw.success([1,2],[-1,2])=='family-only pass, not robust to seed variation')
check('success_zero_does_not_pass',aw.success([0,2],[1,2])=='not_success')
check('success_seed_only_no_success',aw.success([-1,2],[1,2])=='not_success')
check('length_1024_allowed',st.length_guard([dict(dataset_index=1,full_sequence_tokens=1024)]) is None)
check('length_1025_rejected',rejects(lambda:st.length_guard([dict(dataset_index=1,full_sequence_tokens=1025)])))
toy={i:[dict(candidate=c,correct=True,dataset_index=i) for c in [1,2]] for i in range(20)}
toy[20]=[dict(candidate=1,correct=False,dataset_index=20)]
selected=st.choices(toy,12)
check('selection_reproducible',selected==st.choices(toy,12))
check('selection_unique_verified',len({r['dataset_index'] for r in selected})==12 and all(r['correct'] for r in selected))
check('selection_insufficient_stops',rejects(lambda:st.choices(toy,21)))
check('global_seed_continuity',st.sampling_seed(1200)==int.from_bytes(hashlib.sha256(b'gsm8k-difficulty-full-v1:2026:1636:1200').digest()[:4],'big'))
check('initial_seed_distinct',len({st.sampling_seed(i) for i in range(1200)})==1200)
temporary=st.RUN/'cpu_test_fixture_v2_2026-10-13.json';st.save(temporary,dict(test=True))
check('overwrite_rejected',rejects(lambda:st.save(temporary,dict(test=False))))
engine=st.probe_engine();reference=st.load('train_primary_lora_v1.py',st.TRAIN_SHA)
check('engine_execute_bytecode_identity',engine.execute.__code__.co_code==reference.execute.__code__.co_code)
check('engine_frozen_source_sha256',st.sha(st.ROOT/'train_primary_lora_v1.py')==st.TRAIN_SHA)
# Verify CPU old training preparation and exact mask/target encoding on its real frozen data.
groups,_,_,_=reference.prepare_context()
check('original_CPU_training_contract',len(groups['random'])==353 and max(len(r['input_ids']) for r in groups['random'])==849)
check('original_EOS_and_mask',all(r['labels'][-1]==r['input_ids'][-1] and r['labels'].count(-100)>0 for r in groups['random']))
label=st.load('gsm8k_difficulty_label_full_v1.py',st.LABEL_SHA)
label.START=1636;label.COUNT=2400
rows,prefix,helpers,tok,_,meta=st.label_context(label,1636,2400,0)
check('new_label_CPU_context',len(rows)==2400 and rows[0]['dataset_index']==1636 and rows[-1]['dataset_index']==4035 and meta['batch_seeds'][-1]==st.sampling_seed(1199))
check('label_generate_unmodified',label.run.__code__.co_code==st.load('gsm8k_difficulty_label_full_v1.py',st.LABEL_SHA).run.__code__.co_code)
check('label_prompt_training_identical',prefix==st.read(st.ROOT/'results/gsm8k_difficulty_label_full_v1/manifest.json')['prefix'])
source=Path(st.__file__).read_text();check('full_training_stage_unavailable',"choices=['determinism','labeling','freeze_set','probe','hashes']" in source and "phase='train'" not in source)
import torch
check('CPU_tests_no_CUDA_initialized',not torch.cuda.is_initialized())
receipt=dict(passed=True,test_count=len(passed),tests=passed,CUDA_initialized=False,original_gold_primary=actual['primary'],original_gold_secondary=actual['secondary'],original_RQ2_primary=rqactual['primary'],source_hashes={str(p):st.sha(p) for p in sorted((st.RUN/'src').glob('*.py'))},label_context_prompt_sha256=hashlib.sha256(prefix.encode()).hexdigest(),scope='v2 synthetic only; old saved outputs v1 verification and interval equivalence only')
st.check_protected('after_CPU_hashes.json');st.save(st.RUN/'cpu_validation.json',receipt)
st.save(st.RUN/'execution_source_manifest.json',dict(sources=receipt['source_hashes'],spec_sha256=st.sha(st.RUN/'frozen_spec.json'),cpu_validation_sha256=st.sha(st.RUN/'cpu_validation.json')))
print(json.dumps(receipt,indent=2))
