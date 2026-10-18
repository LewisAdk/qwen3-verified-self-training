"""Approved full stages; immutable numerical engines, safety-only plumbing."""
import os
os.environ.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
import argparse
import ast
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path('/home/lewis/rsi-research')
RUN = ROOT/'results/scaled_random_2026-10-13_n2000'
FULL = RUN/'full_run_2026-10-13'
sys.path[:0] = [str(RUN/'src'), str(ROOT)]
import recovery_2026_10_13 as recovery
import analyzer_wrapper as analyzer
from scorer_v2 import score_v2
st = recovery.st
SEEDS = (2026, 2027, 2028)
MODELS = [f'training_seed_{s}' for s in SEEDS]
EVAL = RUN/'evaluation'
ANALYSIS = RUN/'analysis'

def read(p): return json.loads(Path(p).read_text())
def save(p, obj): st.save(p, obj)
def require(ok, text): st.require(ok, text)
def sha(p): return st.sha(p)

def integrity(receipt=None):
    baseline = read(FULL/'protected_baseline_receipt_fix_2026-10-13.json')
    bad = [p for p,h in baseline['files'] if not Path(p).is_file() or sha(p) != h]
    require(not bad, 'Protected hash mismatch: '+str(bad))
    result = dict(verified=len(baseline['files']), mismatches=[], utc=recovery.clocks()['utc'])
    if receipt: save(FULL/receipt, result)
    return result

def sources():
    recovery.check_recovery()
    gate_path=FULL/'gpu_gate_receipt_fix_2026-10-13.json'
    expected=os.environ.get('SCALED_RUN_GATE_SHA256')
    require(expected and sha(gate_path)==expected, 'GPU-stage gate hash missing or mismatched')
    gate=read(gate_path)
    for p,h in gate['receipt_hashes'].items():
        require(sha(p)==h, 'GPU-stage receipt hash mismatch: '+p)
    manifest=read(FULL/'source_manifest_receipt_fix_2026-10-13.json')
    for p,h in manifest['sources'].items():
        require(sha(p)==h, 'Full-stage source mismatch: '+p)
    require(read(FULL/'cpu_validation_receipt_fix_2026-10-13.json')['passed'], 'CPU checks did not pass')

def strict_counter_class(original, publish):
    class StrictCounter(original):
        def step(self, scaler, optimizer):
            skipped = super().step(scaler, optimizer)
            if skipped:
                publish(dict(attempt=self.attempted, successful=self.successful, skipped=self.skipped,
                             scale_after=float(scaler.get_scale()), stop_reason='skipped_optimizer_update'))
                raise RuntimeError('STOP: skipped optimizer update; no retry or setting change')
            return skipped
    return StrictCounter

def validate_training(m, seed, fingerprint=None):
    require(m['status']=='completed_two_epochs' and m['phase']=='train' and m['group']=='random' and m['seed']==seed, 'Training identity/phase mismatch')
    require(m['source_sha256']==st.TRAIN_SHA and m['settings']==read(RUN/'longest_sequence_probe/completed.json')['settings'], 'Immutable engine/settings mismatch')
    require(m['attempted_optimizer_updates']==m['successful_optimizer_updates']==1000 and m['skipped_optimizer_updates']==0, 'Incomplete or skipped update schedule')
    require(m['example_count']==2000 and m['example_exposures']==4000 and m['supervised_tokens_seen']==m['supervised_tokens_in_successful_updates']==369144, 'Training exposures/tokens mismatch')
    require(m['maximum_sequence_length']==868 and m['final_adapter_saved'] and m['all_protected_hashes_match'], 'Training coverage/save guard')
    if fingerprint: require(m['training_fingerprint']==fingerprint, 'Probe/frozen training fingerprint mismatch')

def training_engine(seed, cpu_preflight=False):
    recovery.install_adapters()
    engine = st.probe_engine()
    execute = engine.execute
    original_context = engine.prepare_context
    def context():
        groups, protected, snapshot, fingerprint = original_context()
        protected.update(dict(read(FULL/'protected_baseline_receipt_fix_2026-10-13.json')['files']))
        if not cpu_preflight:
            sources()
            for p in [FULL/'source_manifest_receipt_fix_2026-10-13.json', FULL/'cpu_validation_receipt_fix_2026-10-13.json', FULL/'gpu_gate_receipt_fix_2026-10-13.json']:
                protected[str(p)]=sha(p)
        return groups, protected, snapshot, fingerprint
    def verify(path, fingerprint, source_hash):
        p = read(path)
        require(p['status']=='memory_probe_passed' and p['successful_optimizer_updates']==3 and p['skipped_optimizer_updates']==0, 'Probe not clean')
        require(p['training_fingerprint']==fingerprint and p['source_sha256']==source_hash==st.TRAIN_SHA, 'Probe fingerprint/source mismatch')
        require(p['maximum_sequence_length']==868 and not p['final_adapter_saved'] and p['example_count']==12, 'Probe coverage differs')
    engine.prepare_context = context
    engine.verify_probe = verify
    out = RUN/f'training_seed_{seed}'
    engine.UpdateCounter = strict_counter_class(engine.UpdateCounter, lambda obj:save(out/'skipped_update_stop_2026-10-13.json', obj))
    require(engine.execute is execute, 'Training execution changed')
    return engine

def train(seed):
    sources(); integrity(f'before_training_{seed}_hashes_2026-10-13.json')
    require(seed in SEEDS and not (RUN/f'training_seed_{seed}').exists(), 'Existing training path preserved')
    engine = training_engine(seed)
    engine.execute(SimpleNamespace(phase='train', group='random', seed=seed,
        output=str(RUN/f'training_seed_{seed}'), probe_report=str(RUN/'longest_sequence_probe/completed.json'), additional_runs_confirmed=True))
    m = read(RUN/f'training_seed_{seed}/completed.json')
    validate_training(m, seed, read(RUN/'longest_sequence_probe/completed.json')['training_fingerprint'])
    integrity(f'after_training_{seed}_hashes_2026-10-13.json')
    print('TRAINING_COMPLETE '+str(seed)+' updates=1000 skipped=0', flush=True)

def make_evaluation_contract():
    import gsm_symbolic_rq2_eval_v1 as prior
    metadata, rows, base, ns, snapshot = prior.prepare()
    # Only frozen development rows; no regenerated base and no accuracy aggregation.
    require(len(rows['p1'])==len(rows['p2'])==200, 'Wrong development coverage')
    require(metadata['batch_size']==8 and metadata['max_new_tokens']==384 and not metadata['do_sample'] and metadata['dtype']=='float16', 'Evaluation settings changed')
    save(FULL/'evaluation_contract_2026-10-13.json', dict(metadata=metadata, rows=rows, snapshot=str(snapshot),
         generation_source_sha256=sha(ROOT/'gsm_symbolic_gold_eval_v2.py'),
         validation_source_sha256=sha(ROOT/'gsm_symbolic_rq2_eval_v1.py')))
    import torch
    require(not torch.cuda.is_initialized(), 'CPU contract initialized CUDA')

def evaluator(seed):
    import gsm_symbolic_gold_eval_v2 as original
    import gsm_symbolic_rq2_eval_v1 as prior
    frozen = read(FULL/'evaluation_contract_2026-10-13.json')
    require(sha(ROOT/'gsm_symbolic_gold_eval_v2.py')==frozen['generation_source_sha256'], 'Evaluation generation source changed')
    require(sha(ROOT/'gsm_symbolic_rq2_eval_v1.py')==frozen['validation_source_sha256'], 'Evaluation validator changed')
    from decimal import Decimal
    import re
    import torch
    from transformers import StoppingCriteria
    ns = dict(torch=torch, re=re, Decimal=Decimal, StoppingCriteria=StoppingCriteria)
    tree = ast.parse((ROOT/'gsm_symbolic_pilot_v2.py').read_text())
    prior.old.exact_helpers(tree, ns)
    rows = frozen['rows']; metadata = frozen['metadata']; base = {}
    protected = dict(read(FULL/'protected_baseline_receipt_fix_2026-10-13.json')['files'])
    for p,h in metadata['input_hashes'].items():
        require(sha(p)==h, 'Saved evaluation input mismatch: '+p); protected[p]=h
    for s in SEEDS:
        folder=RUN/f'training_seed_{s}'; m=read(folder/'completed.json')
        validate_training(m,s,read(RUN/'longest_sequence_probe/completed.json')['training_fingerprint'])
        cfg=read(folder/'adapter_final/adapter_config.json')
        require(Path(cfg['base_model_name_or_path']).resolve()==Path(frozen['snapshot']).resolve(), 'Wrong adapter base')
        for name in ['manifest.json','completed.json','optimizer_steps.jsonl']:
            protected[str(folder/name)]=sha(folder/name)
        for name,h in m['adapter_files_sha256'].items():
            require(sha(folder/'adapter_final'/name)==h, 'Trained adapter changed'); protected[str(folder/'adapter_final'/name)]=h
    for v in rows:
        for start in range(0,200,8):
            p=ROOT/'results/gsm_symbolic_expanded_dev_v1/base'/v/f'batch_{start:03d}.json'
            obj=read(p); prior.validate_batch(obj,rows[v][start:start+8],'base',v,ns['parse']);base[(v,start)]=obj
    meta=dict(metadata,models=MODELS,input_hashes=protected,source_sha256=frozen['generation_source_sha256'],
              full_stage_wrapper_sha256=sha(__file__),evaluation_adapter_wall_limit_minutes=30,
              training={str(s):read(RUN/f'training_seed_{s}/completed.json')['training_fingerprint'] for s in SEEDS})
    original.prepare=lambda:(meta,rows,base,{},ns,Path(frozen['snapshot']))
    original.MODELS=MODELS; original.HARD_MODELS=[]; original.TRAIN=RUN
    publish=prior.publish
    def publish_secondary(path,obj):
        if 'records' in obj:
            for r in obj['records']:r['secondary_v2']=score_v2(r['response'],r['gold'])
        publish(path,obj)
    prior.publish=publish_secondary
    return original, prior, rows, ns

def evaluate(seed):
    sources(); integrity(f'before_evaluation_{seed}_hashes_2026-10-13.json')
    key=f'training_seed_{seed}'
    require(seed in SEEDS and not (EVAL/key).exists(), 'Existing evaluation preserved; no resume/retry')
    original,prior,rows,ns=evaluator(seed)
    original.run(SimpleNamespace(preflight=False,model=key,output=str(EVAL)))
    files={}
    for v in rows:
        paths={EVAL/key/v/f'batch_{i:03d}.json' for i in range(0,200,8)}
        require(set((EVAL/key/v).glob('batch_*.json'))==paths, 'Evaluation incomplete/extra batches')
        for p in sorted(paths):
            start=int(p.stem[6:]);obj=read(p);prior.validate_batch(obj,rows[v][start:start+8],key,v,ns['parse'])
            require(all(r['secondary_v2']==score_v2(r['response'],r['gold']) for r in obj['records']), 'Secondary scoring inconsistent')
            files[str(p)]=sha(p)
    for name in ['generation_config.json','completed.json']:files[str(EVAL/key/name)]=sha(EVAL/key/name)
    save(FULL/f'evaluation_{seed}_sealed_2026-10-13.json', dict(seed=seed,questions=400,files=files,accuracy_inspected=False))
    integrity(f'after_evaluation_{seed}_hashes_2026-10-13.json')
    print('EVALUATION_COMPLETE '+str(seed)+' 400 questions; accuracy remains hidden',flush=True)

def all_evaluations_complete():
    for s in SEEDS:
        job=read(FULL/f'jobs/evaluate_{s}/completed.json')
        require(job['exit_code']==0 and not job['timed_out'], 'All three evaluation jobs must complete before any accuracy analysis')
        seal=read(FULL/f'evaluation_{s}_sealed_2026-10-13.json')
        require(seal['questions']==400 and seal['seed']==s, 'Incomplete evaluation seal')
        for p,h in seal['files'].items():require(sha(p)==h,'Completed evaluation changed: '+p)

def records_from_batches(folder):
    return [r for v in ['p1','p2'] for p in sorted((Path(folder)/v).glob('batch_*.json')) for r in read(p)['records']]

def clock_summary():
    keys=['wall_clock_jump','suspend_gap','sample_gap','boot_id_changed'];phases={};flagged=[]
    for j in sorted((FULL/'jobs').iterdir()):
        path=j/'system_time_device_memory.jsonl'
        if not path.exists():continue
        rows=[json.loads(l) for l in path.read_text().splitlines()]
        phases[j.name]=dict(flags={k:sum(r.get('clock_flags',{}).get(k,False) for r in rows) for k in keys},
            samples=len(rows),query_errors=sum('error' in r for r in rows),
            windows_clock_errors=sum('windows_clock_error' in r for r in rows),
            peak_device_wide_gib=max([r['used_gib'] for r in rows if 'used_gib' in r],default=None))
        flagged.extend(dict(phase=j.name,utc=r['system']['utc'],**r['clock_flags']) for r in rows if r.get('clock_flags',{}).get('wall_clock_jump',False))
    # Include all previously approved experiment-stage logs, not just full runs.
    prior_events=[]
    for path in [RUN/'determinism_base_batch4_rerun_2026-10-13_01/device_memory.jsonl', RUN/'job_labeling_rerun_2026-10-13_01/system_time_device_memory.jsonl', RUN/'probe_device_memory.jsonl']:
        if path.exists():
            for l in path.read_text().splitlines():
                r=json.loads(l)
                if r.get('clock_flags',{}).get('wall_clock_jump',False):prior_events.append(dict(phase=str(path.relative_to(RUN)),utc=r['system']['utc'],**r['clock_flags']))
    def biggest(events):
        return max(events,key=lambda r:abs(r['wall_delta_seconds']-r['monotonic_delta_seconds']),default=None)
    def size(r):return abs(r['wall_delta_seconds']-r['monotonic_delta_seconds']) if r else 0
    full_max=biggest(flagged);all_max=biggest(flagged+prior_events)
    return dict(phases=phases,wall_clock_events=flagged,full_phase_maximum_wall_clock_flag_seconds=size(full_max),
        full_phase_maximum_event=full_max,entire_experiment_maximum_wall_clock_flag_seconds=size(all_max),
        entire_experiment_maximum_event=all_max,definition='Absolute difference between successive wall-clock and monotonic deltas, not accumulated drift; raw deltas also reported. Cause unconfirmed.')

def analyze():
    sources();all_evaluations_complete();integrity('before_analysis_hashes_2026-10-13.json')
    base=records_from_batches(ROOT/'results/gsm_symbolic_expanded_dev_v1/base')
    scaled={s:records_from_batches(EVAL/f'training_seed_{s}') for s in SEEDS}
    old={s:records_from_batches(ROOT/f'results/gsm_symbolic_rq2_primary_v1/random_seed_{s}') for s in SEEDS}
    key=lambda r:(r['variant'],r['id'],r['instance'])
    expected=[key(r) for r in base]
    for rs in list(scaled.values())+list(old.values()):
        require([key(r) for r in rs]==expected and len(rs)==400,'Analysis alignment differs')
        require(all((r['question'],r['gold'],r['original_id'])==(b['question'],b['gold'],b['original_id']) for r,b in zip(rs,base)), 'Question/reference/family pairing differs')
    result=analyzer.family_bootstrap(base,scaled,old)
    seed_ci,random_seed_ci=analyzer.seed_family_bootstrap(base,scaled,old)
    primary=result['primary'];decision=analyzer.success(primary['ci95'],seed_ci)
    primary['family_only_original_decision']=primary.pop('decision')
    primary['decision']=decision;primary['seed_and_family_ci95']=seed_ci
    diagnostics={};v2={};differences=[]
    for s in SEEDS:
        rs=scaled[s]
        diagnostics[s]={v:dict(questions=len(part),parsed=sum(r['prediction'] is not None for r in part),
            unparsed=sum(r['prediction'] is None for r in part),cap_hits=sum(r['tokens']==384 for r in part),
            cap_hits_unparsed=sum(r['tokens']==384 and r['prediction'] is None for r in part))
            for v,part in [('pooled',rs),('p1',[r for r in rs if r['variant']=='p1']),('p2',[r for r in rs if r['variant']=='p2'])]}
        v2[s]={v:dict(questions=len(part),correct=sum(r['secondary_v2']['correct'] for r in part),
            accuracy=100*sum(r['secondary_v2']['correct'] for r in part)/len(part),
            verdict_changes=sum(r['secondary_v2']['correct']!=r['correct'] for r in part),
            v1_correct_v2_wrong=sum(r['correct'] and not r['secondary_v2']['correct'] for r in part),
            v1_wrong_v2_correct=sum(not r['correct'] and r['secondary_v2']['correct'] for r in part),
            invalid_or_missing=sum(r['secondary_v2']['prediction'] is None for r in part),
            continuations=sum(r['secondary_v2']['continuation'] for r in part),
            presentation_markers=sum(r['secondary_v2']['presentation'] for r in part))
            for v,part in [('pooled',rs),('p1',[r for r in rs if r['variant']=='p1']),('p2',[r for r in rs if r['variant']=='p2'])]}
        differences += [dict(seed=s,key=key(r),v1_prediction=r['prediction'],v1_correct=r['correct'],reference=r['gold'],secondary_v2=r['secondary_v2'],response=r['response']) for r in rs if r['secondary_v2']['correct']!=r['correct']]
    clocks=clock_summary();require(not ANALYSIS.exists(),'Existing analysis preserved');ANALYSIS.mkdir()
    payload=dict(status='all_three_evaluations_complete_then_analyzed',bootstrap_replicates=10000,bootstrap_seed=2026,
        primary_estimand='Equal three-seed mean scaled-random minus saved batch-8 base, percentage points',
        family_method='Frozen original family-only weighted-count percentile method; same family draws across models and P1/P2; seeds fixed',
        seed_family_method='Frozen random.Random(2026); seed IDs resampled first then shared sorted family draw; matching seed pairs for original random',
        primary=result['primary'],family_results=result,seed_family_original_random_ci95=random_seed_ci,
        response_diagnostics=diagnostics,secondary_v2=v2,secondary_v2_verdict_changes=differences,clocks=clocks,
        source_sha256=sha(__file__),set_sha256=read(RUN/'frozen_set/manifest.json')['set_sha256'],
        training={s:read(RUN/f'training_seed_{s}/completed.json') for s in SEEDS},
        full_training_complete=True,no_reserved_or_GSM8K_test_evaluation=True)
    save(ANALYSIS/'scaled_random_results_2026-10-13.json',payload)
    lines=['# Scaled random self-training — final results, 2026-10-13','',
        f"Decision: **{decision}**. Equal three-seed mean scaled-random − saved base: **{primary['estimate']:+.4f} percentage points**.",
        f"Family-only 95% interval: **[{primary['ci95'][0]:+.6f}, {primary['ci95'][1]:+.6f}]**. Seed-and-family 95% interval: **[{seed_ci[0]:+.6f}, {seed_ci[1]:+.6f}]**.",
        'Success requires both lower endpoints strictly above zero, as frozen. All three seeds and all 400 questions are retained; no best-seed selection. No new tests or outcome-guided method changes were introduced.','',
        '## Training and preservation','',
        'The frozen 2,000-example set, candidate choices and ordering were reused unchanged. Each seed trained freshly from the pinned base with the unchanged RQ2 numerical engine and settings. The same batch-8, FP16, greedy, 384-token evaluation and v1 scorer were used. Saved base outputs were reused. Accuracy was not aggregated, inspected or reported until all three evaluations and their integrity seals completed.','',
        '| Seed | Successful updates | Skipped | Training budget minutes used | Evaluation budget minutes used | Allocated GiB | Reserved GiB | Device-wide GiB |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in SEEDS:
        m=payload['training'][s];tj=read(FULL/f'jobs/train_{s}/completed.json');ej=read(FULL/f'jobs/evaluate_{s}/completed.json')
        lines.append(f"| {s} | {m['successful_optimizer_updates']} | {m['skipped_optimizer_updates']} | {tj['monotonic_elapsed_seconds']/60:.3f}/70 | {ej['monotonic_elapsed_seconds']/60:.3f}/30 | {m['peak_allocated_gpu_gib']:.3f} | {m['peak_reserved_gpu_gib']:.3f} | {clocks['phases']['train_'+str(s)]['peak_device_wide_gib']:.3f} |")
    lines += ['', '## v1 accuracies and family intervals','', '| Model/group | P1 accuracy | P2 accuracy | Pooled accuracy |', '|---|---:|---:|---:|']
    for model in ['base','scaled_random_seed_mean','original_random_seed_mean']+[f'scaled_random_seed_{s}' for s in SEEDS]+[f'original_random_seed_{s}' for s in SEEDS]:
        vals=[result['metrics'][f'{v}/accuracy/{model}']['estimate'] for v in ['p1','p2','pooled']]
        lines.append(f'| {model} | {vals[0]:.4f}% | {vals[1]:.4f}% | {vals[2]:.4f}% |')
    lines += ['', '| Variant | Contrast | Difference pp | Family-only 95% interval |', '|---|---|---:|---|']
    for v in ['p1','p2','pooled']:
        for contrast in ['scaled_random_minus_base','scaled_random_minus_original_random']:
            x=result['metrics'][f'{v}/difference_pp/{contrast}'];lines.append(f"| {v} | {contrast} | {x['estimate']:+.6f} | [{x['ci95'][0]:+.6f}, {x['ci95'][1]:+.6f}] |")
    lines += ['', f"The optional paired seed-and-family interval against original random-353 is [{random_seed_ci[0]:+.6f}, {random_seed_ci[1]:+.6f}] pp. P1/P2 and original-random comparisons are secondary marginal analyses with no multiplicity-adjusted success claim. The random-353 comparison changes data, tokens and updates together and does not isolate their effects.", '',
        '## Fixes, regressions and answer diagnostics','',
        '| Seed | Variant | Fixes vs base | Regressions | Net | Fixes vs matching random-353 | Regressions | Net | Parsed | Unparsed | Cap hits | Unparsed cap hits |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in SEEDS:
        for v in ['pooled','p1','p2']:
            b=result['paired_flips'][f'{v}/scaled_random_seed_{s}_versus_base'];o=result['paired_flips'][f'{v}/scaled_random_seed_{s}_versus_original_random_seed_{s}'];d=diagnostics[s][v]
            lines.append(f"| {s} | {v} | {b['fixed']} | {b['regressed']} | {b['net']} | {o['fixed']} | {o['regressed']} | {o['net']} | {d['parsed']} | {d['unparsed']} | {d['cap_hits']} | {d['cap_hits_unparsed']} |")
    lines += ['', '## Secondary v2 scorer (new outputs only)','',
        '| Seed | Variant | v2 accuracy | v1→v2 verdict changes | v1 right→v2 wrong | v1 wrong→v2 right | Missing/invalid v2 | Continuation flags | Presentation flags |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in SEEDS:
        for v in ['pooled','p1','p2']:
            d=v2[s][v];lines.append(f"| {s} | {v} | {d['accuracy']:.4f}% | {d['verdict_changes']} | {d['v1_correct_v2_wrong']} | {d['v1_wrong_v2_correct']} | {d['invalid_or_missing']} | {d['continuations']} | {d['presentation_markers']} |")
    lines += ['', 'The first clean numeric answer is retained while later continuation is flagged. No old base or original-random response was v2-scored. Therefore no mixed-scorer v2 gain over base is claimed. Every changed v2 verdict and its raw response is saved in the accompanying JSON.', '',
        '## Clock and memory observations','',
        f"Maximum wall-clock flag in the full training/evaluation phases: **{clocks['full_phase_maximum_wall_clock_flag_seconds']:.6f} seconds**. Maximum across the entire scaled experiment, including earlier determinism/labeling/probe logs: **{clocks['entire_experiment_maximum_wall_clock_flag_seconds']:.6f} seconds**.",
        'Flag size is abs(successive wall delta − successive monotonic delta), not cumulative drift. Raw wall/monotonic deltas, timestamps and phase of the largest event are in the JSON. Budget use is the saved monotonic elapsed counter per supervised fresh process. Clock behavior remains unconfirmed; flags do not establish sleep or memory spill. Device-wide memory is sampled about once per second and may miss brief peaks. Allocated, reserved and whole-device counters are distinct.', '',
        '| Phase | Wall flags | Suspend gaps | Sample gaps | Boot changes | Device peak GiB | Query errors |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for phase,d in clocks['phases'].items():
        f=d['flags'];lines.append(f"| {phase} | {f['wall_clock_jump']} | {f['suspend_gap']} | {f['sample_gap']} | {f['boot_id_changed']} | {d['peak_device_wide_gib']:.3f} | {d['query_errors']} |")
    lines += ['', '## Interpretation and stopped boundary','',
        'This is the prospectively specified scaling experiment motivated by the earlier post-hoc fixes/regressions pattern; that pattern was motivation, not supporting evidence. The 400 development questions cover only 37 previously inspected families. Neither the 63 reserved families nor GSM8K test was evaluated. Verified final answers do not certify every reasoning step. The observed batch-size sensitivity remains a reproducibility limitation; saved batch-8 base stayed the comparator.', '',
        'The result applies to this combined increase in examples, supervised tokens and updates on this development set. Secondary results do not amend the two-interval success gate. All old protected files and the frozen training set are rechecked after analysis. No retry, setting change, extra seed, further training or evaluation was started. STOP.', '']
    with (ANALYSIS/'scaled_random_final_report_2026-10-13.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines))
    integrity('after_analysis_hashes_2026-10-13.json')
    print('ANALYSIS_COMPLETE_STOP',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','evaluate','analyze']);p.add_argument('--seed',type=int,choices=SEEDS);a=p.parse_args();os.chdir(ROOT)
    try:
        if a.stage=='train':train(a.seed)
        elif a.stage=='evaluate':evaluate(a.seed)
        else:analyze()
    except BaseException as e:
        save(FULL/f'failure_{a.stage}_{a.seed}_{time.time_ns()}.json',dict(type=type(e).__name__,message=str(e),system=recovery.clocks(),no_retry=True,no_settings_changed=True))
        raise

if __name__=='__main__':main()
