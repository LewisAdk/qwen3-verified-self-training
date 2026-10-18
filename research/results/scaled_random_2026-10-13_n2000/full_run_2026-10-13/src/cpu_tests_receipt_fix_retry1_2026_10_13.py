"""CPU verification of new approval plumbing; never inspect new accuracy."""
import ast
import copy
import json
import marshal
from pathlib import Path
import sys
import runner_receipt_fix_2026_10_13 as run

tests=[]
def check(name, ok):
    assert ok,name
    tests.append(name)

def fails(fn):
    try:fn()
    except (RuntimeError,AssertionError,FileNotFoundError):return True
    return False

class Optimizer:
    def register_step_post_hook(self, hook):self.hook=hook;return object()
class Scaler:
    def __init__(self,skip=False):self.skip=skip;self.steps=0;self.updates=0
    def step(self,opt):
        self.steps+=1
        if not self.skip:opt.hook()
    def update(self):self.updates+=1
    def get_scale(self):return 32768.0 if self.skip else 65536.0

def main():
    run.recovery.check_recovery()
    check('prior_79_CPU_checks_preserved',run.read(run.RUN/'cpu_validation.json')['passed'])
    prior=run.read(run.RUN/'cpu_validation.json')['tests']
    for name in ['gold_saved_result_exact','rq2_saved_result_exact','new_family_wrapper_exact','seed_family_exact_prior_sensitivity','engine_execute_bytecode_identity','label_generate_unmodified']:
        check('prior_equivalence_'+name,name in prior)
    engine=run.training_engine(2026,cpu_preflight=True)
    old=run.st.load('train_primary_lora_v1.py',run.st.TRAIN_SHA)
    check('training_frozen_source_hash',run.sha(run.ROOT/'train_primary_lora_v1.py')==run.st.TRAIN_SHA)
    check('training_code_object_equality',engine.execute.__code__==old.execute.__code__)
    execution_attributes=['co_argcount','co_posonlyargcount','co_kwonlyargcount','co_nlocals','co_stacksize','co_flags','co_code','co_consts','co_names','co_varnames','co_filename','co_name','co_qualname','co_firstlineno','co_linetable','co_exceptiontable','co_freevars','co_cellvars']
    for attribute in execution_attributes:
        check('training_execution_attribute_'+attribute,getattr(engine.execute.__code__,attribute)==getattr(old.execute.__code__,attribute))
    check('settings_unchanged',engine.SETTINGS==old.SETTINGS==run.read(run.RUN/'longest_sequence_probe/completed.json')['settings'])
    groups,protected,snapshot,fingerprint=engine.prepare_context()
    check('frozen_context_2000_examples',len(groups['random'])==2000)
    check('frozen_exposure_tokens_369144',2*sum(r['supervised_tokens'] for r in groups['random'])==369144)
    check('original_token_mask_EOS_coverage',all(len(r['input_ids'])==len(r['labels'])<=1024 and r['labels'][-1]==r['input_ids'][-1] for r in groups['random']))
    check('probe_fingerprint_identical',fingerprint==run.read(run.RUN/'longest_sequence_probe/completed.json')['training_fingerprint'])
    engine.verify_probe(run.RUN/'longest_sequence_probe/completed.json',fingerprint,run.st.TRAIN_SHA)
    check('approved_probe_validation',True)
    original=old.UpdateCounter;events=[];strict=run.strict_counter_class(original,events.append)
    opt=Optimizer();scale=Scaler();counter=strict(opt)
    check('successful_step_exactly_once',counter.step(scale,opt) is False and counter.successful==1 and counter.attempted==1 and counter.skipped==0 and scale.steps==scale.updates==1)
    opt=Optimizer();scale=Scaler(True);counter=strict(opt)
    check('first_skip_immediately_stops',fails(lambda:counter.step(scale,opt)) and counter.attempted==1 and counter.successful==0 and counter.skipped==1)
    check('skip_preserved_without_retry',len(events)==1 and events[0]['stop_reason']=='skipped_optimizer_update' and scale.steps==scale.updates==1)
    fake=dict(status='completed_two_epochs',phase='train',group='random',seed=2026,source_sha256=run.st.TRAIN_SHA,
        settings=old.SETTINGS,attempted_optimizer_updates=1000,successful_optimizer_updates=1000,skipped_optimizer_updates=0,
        example_count=2000,example_exposures=4000,supervised_tokens_seen=369144,supervised_tokens_in_successful_updates=369144,
        maximum_sequence_length=868,final_adapter_saved=True,all_protected_hashes_match=True,training_fingerprint=fingerprint)
    run.validate_training(fake,2026,fingerprint);check('full_schedule_guard_accepts',True)
    for field,value in [('successful_optimizer_updates',999),('skipped_optimizer_updates',1),('example_exposures',3999),('supervised_tokens_seen',369143),('seed',2027),('training_fingerprint','wrong')]:
        bad=copy.deepcopy(fake);bad[field]=value
        check('full_schedule_rejects_'+field,fails(lambda:run.validate_training(bad,2026,fingerprint)))
    import gsm_symbolic_gold_eval_v2 as original_eval
    source=ast.parse((run.ROOT/'gsm_symbolic_gold_eval_v2.py').read_text())
    node=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    reference_eval=run.st.load('gsm_symbolic_gold_eval_v2.py',run.read(run.FULL/'evaluation_contract_2026-10-13.json')['generation_source_sha256']);ns={'run':reference_eval.run}
    check('evaluation_generation_AST_identity',ast.dump(node)==ast.dump(next(n for n in ast.parse((run.ROOT/'gsm_symbolic_gold_eval_v2.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run')))
    check('evaluation_generation_code_identity',original_eval.run.__code__.co_code==ns['run'].__code__.co_code and original_eval.run.__code__.co_names==ns['run'].__code__.co_names)
    contract=run.read(run.FULL/'evaluation_contract_2026-10-13.json')
    check('evaluation_exact_400_batch8_contract',len(contract['rows']['p1'])==len(contract['rows']['p2'])==200 and contract['metadata']['batch_size']==8)
    check('evaluation_FP16_greedy_384',contract['metadata']['dtype']=='float16' and not contract['metadata']['do_sample'] and contract['metadata']['max_new_tokens']==384)
    check('fresh_training_evaluation_paths',all(not (run.RUN/f'training_seed_{s}').exists() and not (run.EVAL/f'training_seed_{s}').exists() for s in run.SEEDS))
    check('accuracy_gate_blocks_incomplete',fails(run.all_evaluations_complete))
    check('v2_only_new_records',run.score_v2('#### 80%\nQuestion: next','80')['correct'] and run.score_v2('#### 80%\nQuestion: next','80')['continuation'])
    check('both_intervals_required',run.analyzer.success([0.01,1],[-0.01,1])=='family-only pass, not robust to seed variation')
    check('strict_zero_not_success',run.analyzer.success([0,1],[0.1,1])!='success')
    check('seed_then_family_frozen_source',run.sha(run.RUN/'src/analyzer_wrapper.py')==run.read(run.RUN/'execution_source_manifest.json')['sources'][str(run.RUN/'src/analyzer_wrapper.py')])
    import torch
    check('all_tests_CPU_CUDA_uninitialized',not torch.cuda.is_initialized())
    result=dict(passed=True,test_count=len(tests),tests=tests,CUDA_initialized=False,original_intervals_exact_receipt=run.sha(run.RUN/'cpu_validation.json'),frozen_set_sha256=run.read(run.RUN/'frozen_set/manifest.json')['set_sha256'])
    check('GPU_gate_blocks_before_pass_receipts',fails(run.sources))
    result['test_count']=len(tests)
    run.save(run.FULL/'cpu_validation_receipt_fix_2026-10-13.json',result)
    paths=[run.FULL/'src/runner_receipt_fix_2026_10_13.py',run.FULL/'src/supervisor_receipt_fix_2026_10_13.py',Path(__file__),run.FULL/'cpu_validation_receipt_fix_2026-10-13.json',run.FULL/'protected_baseline_receipt_fix_2026-10-13.json',run.FULL/'evaluation_contract_2026-10-13.json']
    paths += [run.RUN/'src/analyzer_wrapper.py',run.RUN/'src/scorer_v2.py',run.RUN/'src/stages.py',run.RUN/'src/recovery_2026_10_13.py']
    manifest={'sources':{str(p):run.sha(p) for p in paths},'all_CPU_checks_passed_before_receipts':True,'GPU_started':False}
    run.save(run.FULL/'source_manifest_receipt_fix_2026-10-13.json',manifest)
    receipts=[run.FULL/'cpu_validation_receipt_fix_2026-10-13.json',run.FULL/'source_manifest_receipt_fix_2026-10-13.json']
    run.save(run.FULL/'gpu_gate_receipt_fix_2026-10-13.json',{'receipt_hashes':{str(p):run.sha(p) for p in receipts},'complete_CPU_pass_required':True})
    print(json.dumps(result))

if __name__=='__main__':main()
