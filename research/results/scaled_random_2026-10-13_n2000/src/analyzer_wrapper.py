"""Preserve the saved gold-control summarize function exactly; alias names only."""
import ast
from collections import Counter
import json
from pathlib import Path
import random
import statistics
import numpy as np

ROOT = Path('/home/lewis/rsi-research')

def original_function():
    tree = ast.parse((ROOT/'gold_control_analysis_v2.py').read_text())
    summarize = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'summarize')
    helper = ast.parse((ROOT/'gsm_symbolic_cluster_analysis_v1.py').read_text())
    percentile = next(n for n in helper.body if isinstance(n, ast.FunctionDef) and n.name == 'percentile')
    ns = dict(np=np, Counter=Counter, random=random, statistics=statistics,
              Decimal=__import__('decimal').Decimal, require=lambda ok,msg: None if ok else (_ for _ in ()).throw(AssertionError(msg)),
              MODELS=[f'gold_seed_{s}' for s in [2026,2027,2028]],
              HARD_MODELS=[f'hard_seed_{s}' for s in [2026,2027,2028]])
    ns['ALL']=['base']+ns['MODELS']+ns['HARD_MODELS']
    exec(compile(ast.Module(body=[percentile,summarize],type_ignores=[]),'<unchanged-gold-family-analysis>','exec'),ns)
    return ns['summarize']

def family_bootstrap(base, scaled, original_random):
    index = lambda rows: {(r['variant'],r['id'],r['instance']):r for r in rows}
    indices={'base':index(base)}
    for s in [2026,2027,2028]:
        indices[f'gold_seed_{s}']=index(scaled[s])
        indices[f'hard_seed_{s}']=index(original_random[s])
    # "gold"/"hard" are internal aliases only; no statistical code is changed.
    result=original_function()(indices,10000,2026)
    return json.loads(json.dumps(result).replace('gold','scaled_random').replace('hard','original_random'))

def seed_family_bootstrap(base, scaled, original_random=None, repetitions=10000):
    families=sorted({r['original_id'] for r in base});seeds=[2026,2027,2028]
    size=np.array([sum(r['original_id']==f for r in base) for f in families])
    counts=lambda rows: np.array([sum(r['correct'] for r in rows if r['original_id']==f) for f in families])
    b=counts(base);a={s:counts(scaled[s]) for s in seeds}
    old={s:counts(original_random[s]) for s in seeds} if original_random is not None else None
    draws=[[],[]];rng=random.Random(2026)
    for _ in range(repetitions):
        ss=rng.choices(seeds,k=3);fs=rng.choices(list(range(len(families))),k=len(families));n=int(size[fs].sum())
        aa=sum(int(a[s][fs].sum()) for s in ss)
        draws[0].append(100*(aa-3*int(b[fs].sum()))/(3*n))
        if old is not None:draws[1].append(100*(aa-sum(int(old[s][fs].sum()) for s in ss))/(3*n))
    def pct(v,q):
        v=sorted(v);p=(len(v)-1)*q;i=int(p);return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(p-i)
    return [([pct(v,.025),pct(v,.975)] if v else None) for v in draws]

def success(family_ci, seed_family_ci):
    if family_ci[0]>0 and seed_family_ci[0]>0:return 'success'
    if family_ci[0]>0:return 'family-only pass, not robust to seed variation'
    return 'not_success'
