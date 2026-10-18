#!/usr/bin/env python3
"""Fixed GSM-Symbolic development expansion. Run from ~/rsi-research."""
import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import time

MODEL = 'Qwen/Qwen3-1.7B-Base'
REVISION = '93b5b3758d9d9841ffe81d6cd2ae2b030685b078'
GSM_REVISION = '740312add88f781978c0658806c59bc2815b9866'
ADAPTERS = {'base': None, '2026': 'lora_verified_pilot_v1',
            '2027': 'lora_verified_seed_2027', '2028': 'lora_verified_seed_2028'}
SCRIPTS = {'base': 'gsm_symbolic_pilot_v2.py', '2026': 'gsm_symbolic_pilot_v2_lora.py',
           '2027': 'gsm_symbolic_pilot_v2_lora_seed_2027.py',
           '2028': 'gsm_symbolic_pilot_v2_lora_seed_2028.py'}
TEMPLATES = {'p1': [2,8,14,15,16,18,19,21,28,37,46,54,58,59,76,82,85,93,95,97],
             'p2': [0,4,7,9,10,12,17,19,24,27,28,33,36,38,39,40,41,45,46,49]}

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':'))

def publish(path, obj):
    """Publish complete JSON without replacing any existing file (same filesystem)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.pending.{os.getpid()}')
    try:
        with temp.open('x', encoding='utf-8') as f:
            f.write(canonical(obj) + '\n')
            f.flush()
            os.fsync(f.fileno())
        os.link(temp, path)
    finally:
        if temp.exists():
            temp.unlink()

def freeze(path, obj):
    if Path(path).exists():
        require(json.loads(Path(path).read_text(encoding='utf-8')) == obj, f'Configuration mismatch: {path}')
    else:
        publish(path, obj)

def source_contract(path):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    names = {'MODEL', 'DATASET', 'BATCH_SIZE', 'MAX_TOKENS', 'SEED', 'ANSWER', 'STOP', 'prefix', 'gsm'}
    selected = []
    for node in tree.body:
        keep = isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in {'parse', 'StopAnswers'}
        if isinstance(node, ast.Assign):
            keep = any(isinstance(t, ast.Name) and t.id in names for t in node.targets)
        if isinstance(node, ast.For):
            keep = any(isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name)
                       and n.target.id == 'prefix' for n in ast.walk(node))
        if keep:
            selected.append(ast.dump(node, include_attributes=False))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                                               and t.value.id == 'tok' for t in node.targets):
            selected.append(ast.dump(node, include_attributes=False))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == 'generate':
                selected.append(ast.dump(node, include_attributes=False))
            if node.func.attr == 'from_pretrained' and isinstance(node.func.value, ast.Name) and node.func.value.id in {'AutoTokenizer','AutoModelForCausalLM'}:
                selected.append(ast.dump(node, include_attributes=False))
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {'prompts','gold'} for t in node.targets):
            selected.append(ast.dump(node, include_attributes=False))
    return selected, tree

def preflight():
    split_path = Path('results/gsm_symbolic_pilot_split_v2.json')
    split = json.loads(split_path.read_text(encoding='utf-8'))
    require(split['dataset'] == 'apple/GSM-Symbolic' and split['revision'] == REVISION, 'Wrong dataset revision')
    require(split['development_templates'] == TEMPLATES, 'Wrong development templates')
    require(split['instances_per_template'] == list(range(5)) and split['seed'] == 2026, 'Wrong pilot split')
    dev, reserved = set(split['development_original_ids']), set(split['reserved_original_ids'])
    require(len(dev) == 37 and len(reserved) == 63 and not dev & reserved, 'Invalid family partition')
    hashes = {'split': digest(split_path)}
    baseline, base_tree = source_contract(SCRIPTS['base'])
    require(len(baseline) >= 12, 'Incomplete base evaluator contract')
    constants = {n.targets[0].id: ast.literal_eval(n.value) for n in base_tree.body
                 if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                 and n.targets[0].id in {'MODEL','DATASET','BATCH_SIZE','MAX_TOKENS','SEED'}}
    require(constants == dict(MODEL=MODEL, DATASET='apple/GSM-Symbolic', BATCH_SIZE=8,
                              MAX_TOKENS=384, SEED=2026), 'Unexpected base settings')
    for key, path in SCRIPTS.items():
        contract, tree = source_contract(path)
        require(contract == baseline, f'Prompt/scoring/generation mismatch in {path}; inspect before proceeding')
        hashes[path] = digest(path)
        if key != 'base':
            calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Attribute) and n.func.attr == 'from_pretrained'
                     and isinstance(n.func.value, ast.Name) and n.func.value.id == 'PeftModel']
            require(any(len(n.args) > 1 and isinstance(n.args[1], ast.Constant)
                        and n.args[1].value == ADAPTERS[key] for n in calls), f'Unverified adapter path: {path}')
            for filename in ['adapter_config.json', 'adapter_model.safetensors']:
                p = Path(ADAPTERS[key]) / filename
                hashes[str(p)] = digest(p)
            cfg = json.loads((Path(ADAPTERS[key]) / 'adapter_config.json').read_text(encoding='utf-8'))
            require(cfg['base_model_name_or_path'] == MODEL and cfg['peft_type'] == 'LORA', 'Wrong adapter base')
    return split, hashes, base_tree

def exact_helpers(tree, namespace):
    # Compile only the saved parser, regexes and stopping class; never execute a saved evaluator.
    nodes = []
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in {'parse', 'StopAnswers'}:
            nodes.append(n)
        elif isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {'ANSWER', 'STOP'} for t in n.targets):
            nodes.append(n)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<saved-scoring>', 'exec'), namespace)

def run(args):
    split, hashes, tree = preflight()
    if args.preflight:
        print('PASS: saved split, 63 reserved families, three adapters, identical evaluator contracts.')
        return
    import torch
    from decimal import Decimal
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    require(torch.cuda.is_available(), 'CUDA unavailable')
    torch.manual_seed(2026)
    ns = dict(torch=torch, re=re, Decimal=Decimal, StoppingCriteria=StoppingCriteria)
    exact_helpers(tree, ns)
    parse, StopAnswers = ns['parse'], ns['StopAnswers']
    gsm = load_dataset('openai/gsm8k', 'main', split='train', revision=GSM_REVISION).shuffle(seed=2026)
    prefix = ('Solve the math problems step by step. '
              'Finish each solution with #### followed by the numeric answer.\n\n')
    for ex in gsm.select(range(2)):
        solution = re.sub(r'<<[^<>]*>>', '', ex['answer'])
        prefix += f"Question: {ex['question']}\nSolution: {solution}\n\n"
    # Verify the saved prefix construction is identical to this implementation.
    prompt_nodes = [n for n in tree.body if
                    (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'prefix' for t in n.targets))
                    or (isinstance(n, ast.For) and any(isinstance(x, ast.AugAssign) and isinstance(x.target, ast.Name)
                        and x.target.id == 'prefix' for x in ast.walk(n)))]
    prompt_ns = dict(gsm=gsm, re=re)
    exec(compile(ast.Module(body=prompt_nodes, type_ignores=[]), '<saved-prefix>', 'exec'), prompt_ns)
    require(prefix == prompt_ns['prefix'], 'Saved prompt differs')
    rows = {}
    mappings = {}
    all_families, exposed = set(), set()
    for variant in ['p1', 'p2']:
        ds = load_dataset('apple/GSM-Symbolic', name=variant, split='test', revision=REVISION)
        mapping = {}
        for r in ds:
            require(mapping.setdefault(r['id'], r['original_id']) == r['original_id'], 'Inconsistent family')
        mappings[variant] = mapping
        all_families.update(mapping.values())
        exposed.update(mapping[t] for t in TEMPLATES[variant])
        selected = [r for r in ds if r['id'] in TEMPLATES[variant] and r['instance'] in range(5, 15)]
        selected.sort(key=lambda r: (r['id'], r['instance']))
        require(len(selected) == 200 and {(r['id'],r['instance']) for r in selected}
                == {(t,i) for t in TEMPLATES[variant] for i in range(5,15)}, 'Wrong expansion rows')
        require(all(r['original_id'] in split['development_original_ids'] for r in selected), 'Reserved family selected')
        require(all(parse(r['answer']) is not None for r in selected), 'Unparseable gold')
        rows[variant] = selected
    require(exposed == set(split['development_original_ids']) and
            all_families - exposed == set(split['reserved_original_ids']), 'Family mapping changed')
    # Freeze the actual cached base-model snapshot; never choose a remote moving revision on resume.
    snapshot = Path(snapshot_download(MODEL, local_files_only=True))
    require(len(snapshot.name) == 40, 'Expected a cached pinned model snapshot')
    snapshots = [p for p in snapshot.parent.iterdir() if p.is_dir()]
    require(len(snapshots) == 1, 'Multiple cached model snapshots; stop and review before generation')
    tok = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rule = Path('gsm_symbolic_success_rule_2026-10-12.txt')
    require(rule.is_file(), 'Missing prespecified success rule')
    metadata = {'schema': 1, 'source_sha256': digest(__file__), 'input_hashes': hashes,
                'prespecified_rule_sha256': digest(rule),
                'model': MODEL, 'model_revision': snapshot.name, 'dataset_revision': REVISION,
                'gsm_revision': GSM_REVISION, 'prefix': prefix, 'batch_size': 8,
                'max_new_tokens': 384, 'do_sample': False, 'dtype': 'float16', 'evaluation_seed': 2026,
                'versions': {p: importlib.metadata.version(p) for p in ['torch','transformers','datasets','peft','accelerate']},
                'gpu': torch.cuda.get_device_name(0), 'split': split,
                'family_mapping': {v: {str(t): mappings[v][t] for t in TEMPLATES[v]} for v in rows},
                'rows_sha256': hashlib.sha256(canonical(rows).encode()).hexdigest()}
    freeze(out / 'manifest.json', metadata)
    for key in args.models:
        missing = []
        for v in rows:
            for start in range(0, 200, 8):
                p = out / key / v / f'batch_{start:03d}.json'
                if p.exists():
                    check_batch(json.loads(p.read_text(encoding='utf-8')), rows[v][start:start+8], key, v, parse)
                else:
                    missing.append((v, start, p))
        if not missing:
            print(f'{key}: already complete', flush=True)
            continue
        torch.cuda.reset_peak_memory_stats()
        model = AutoModelForCausalLM.from_pretrained(str(snapshot), dtype=torch.float16, device_map='cuda:0', local_files_only=True)
        if key != 'base':
            model = PeftModel.from_pretrained(model, ADAPTERS[key], is_trainable=False)
        model.eval()
        freeze(out / key / 'generation_config.json', model.generation_config.to_dict())
        for v, start, p in missing:
            batch = rows[v][start:start+8]
            prompts = [prefix + f"Question: {r['question']}\nSolution:" for r in batch]
            inputs = tok(prompts, padding=True, return_tensors='pt').to('cuda')
            prompt_len = inputs['input_ids'].shape[1]
            stopper = StopAnswers(tok, prompt_len, len(batch))
            began = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=384, do_sample=False,
                                        stopping_criteria=StoppingCriteriaList([stopper]), pad_token_id=tok.pad_token_id)
            generated = output[:, prompt_len:]
            records = []
            for j, r in enumerate(batch):
                tokens = generated[j].tolist()
                count = stopper.lengths[j]
                if count is None:
                    count = tokens.index(tok.eos_token_id) + 1 if tok.eos_token_id in tokens else len(tokens)
                response = tok.decode(tokens[:count], skip_special_tokens=True)
                prediction, gold = parse(response), parse(r['answer'])
                records.append(dict(model=key, variant=v, id=r['id'], instance=r['instance'],
                                    original_id=r['original_id'], question=r['question'], gold=gold,
                                    prediction=prediction, correct=prediction == gold, tokens=count, response=response))
            publish(p, {'records': records, 'elapsed_seconds': time.perf_counter()-began,
                        'peak_reserved_gib': torch.cuda.max_memory_reserved()/1024**3})
            print(f'{key} {v} {start+len(batch)}/200 saved; peak {torch.cuda.max_memory_reserved()/1024**3:.2f} GiB', flush=True)
            del inputs, output, generated
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()

def check_batch(obj, expected, model, variant, parse):
    records = obj['records']
    require(len(records) == len(expected), 'Incomplete saved batch')
    for r, e in zip(records, expected):
        require((r['model'],r['variant'],r['id'],r['instance'],r['original_id'],r['question'],r['gold']) ==
                (model,variant,e['id'],e['instance'],e['original_id'],e['question'],parse(e['answer'])), 'Saved row mismatch')
        require(r['prediction'] == parse(r['response']) and type(r['correct']) is bool
                and r['correct'] == (r['prediction'] == r['gold']), 'Saved scoring mismatch')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--output', default='results/gsm_symbolic_expanded_dev_v1')
    parser.add_argument('--models', nargs='+', choices=list(ADAPTERS), default=list(ADAPTERS))
    args = parser.parse_args()
    # Advisory lock is released automatically on interruption; lock files need no manual removal.
    import fcntl
    lock_path = Path(args.output + '.lock')
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)
