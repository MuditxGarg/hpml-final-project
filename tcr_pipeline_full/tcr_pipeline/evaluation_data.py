"""Read immutable rollout labels; no model downloads or GPU imports."""
from collections import Counter
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
import random

import numpy as np


@dataclass
class EvaluationData:
    ids: list
    actions: list
    success: np.ndarray
    tokens: np.ndarray
    levels: np.ndarray
    prompt_lengths: np.ndarray
    hidden: object
    layers: list
    provenance: dict


def stable_seed(text, seed):
    return int.from_bytes(hashlib.sha256(f'{seed}:{text}'.encode()).digest()[:8], 'big')


def vote_probability(keys, correct, n, seed, draws=200):
    """Empirical subset-vote probability. Exact for small grids; ties averaged.

    This estimates voting over the observed trace pool, not independent future samples.
    """
    if n == 1:
        return float(np.mean(correct))
    by_key = {}
    for key, ok in zip(keys, correct):
        if key is not None:
            if key in by_key and by_key[key] != bool(ok):
                raise ValueError('Same normalized answer has conflicting correctness labels')
            by_key[key] = bool(ok)
    k = len(keys)
    if not 1 <= n <= k:
        raise ValueError('N must be between 1 and the trace count')
    rng = random.Random(seed)
    subsets = (itertools.combinations(range(k), n) if math.comb(k, n) <= draws
               else (rng.sample(range(k), n) for _ in range(draws)))
    hits = []
    for subset in subsets:
        counts = Counter(keys[i] for i in subset if keys[i] is not None)
        if not counts:
            hits.append(0.)
            continue
        top = max(counts.values())
        winners = [key for key, count in counts.items() if count == top]
        hits.append(sum(by_key[key] for key in winners) / len(winners))
    return float(np.mean(hits))


def load_data(run, widths=(1, 2, 4), budgets=(0, 1024, 2048), cap=2048,
              hidden_path=None, seed=20261112, draws=200):
    run = Path(run)
    source_files = sorted((run / 'rollouts').glob('chunk_*.json'))
    records = [r for path in source_files for r in json.loads(path.read_text())]
    if not records:
        raise ValueError(f'No original rollout labels in {run / "rollouts"}')
    ids = [r['id'] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate question IDs')
    # Sort IDs once so fold assignment is independent of file ordering.
    records.sort(key=lambda r: r['id'])
    cap_files = []
    if cap != 1024:
        cap_files = sorted((run / 'rollouts_nothink_cap').glob('chunk_*.json'))
        corrected = {r['id']: r for f in cap_files for r in json.loads(f.read_text())}
        if set(corrected) != set(ids):
            raise ValueError('Corrected B=0 labels must cover exactly the original questions')
        for rec in records:
            samples = corrected[rec['id']]['samples']
            if len(samples) != len(rec['traces']):
                raise ValueError('Corrected B=0 trace count differs')
            for trace, sample in zip(rec['traces'], samples):
                answered = bool(sample['fin'] and sample['ft'] <= cap)
                trace['b']['0'] = dict(key=sample['key'] if answered else None,
                                       ok=bool(sample['ok'] and answered), ft=min(sample['ft'], cap))
    actions = [(int(n), int(b)) for n in widths for b in budgets]
    p, costs = [], []
    for rec in records:
        ps, ts = [], []
        for n, b in actions:
            samples = [t['b'][str(b)] for t in rec['traces']]
            ps.append(vote_probability([s['key'] for s in samples], [s['ok'] for s in samples], n,
                                       stable_seed(f'{rec["id"]}:{n}:{b}', seed), draws))
            ts.append(n * np.mean([t['b'][str(b)]['ft'] + (min(t['L'], b) if b else 0)
                                  for t in rec['traces']]))
        p.append(ps)
        costs.append(ts)
    ids = [r['id'] for r in records]
    hidden, layers = None, []
    feature_file = Path(hidden_path) if hidden_path else run / 'probe' / 'hidden.npz'
    if feature_file.exists():
        with np.load(feature_file, allow_pickle=False) as z:
            feature_ids = list(z['ids'].astype(str))
            if len(set(feature_ids)) != len(feature_ids) or set(feature_ids) != set(ids):
                raise ValueError('Hidden-state IDs must match the full evaluation dataset exactly')
            lookup = {key: i for i, key in enumerate(feature_ids)}
            hidden = z['X'][[lookup[key] for key in ids]].astype(np.float32)
            layers = z['layers'].astype(int).tolist()
            if hidden.ndim != 3 or hidden.shape[1] != len(layers):
                raise ValueError('Expected hidden states shaped [question, layer, feature]')
    elif hidden_path:
        raise FileNotFoundError(feature_file)
    sources = source_files + cap_files + ([feature_file] if hidden is not None else [])
    manifest = {str(f.relative_to(run)) if f.is_relative_to(run) else str(f):
                hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}
    return EvaluationData(ids, actions, np.asarray(p), np.asarray(costs),
                          np.array([r['level'] for r in records]),
                          np.array([r['prompt_tokens'] for r in records], dtype=float), hidden, layers,
                          dict(cap=cap, seed=seed, vote_draws=draws, files_sha256=manifest,
                               feature_status='available' if hidden is not None else 'missing; probe omitted'))
