"""Stage 3: decide the experiment values from the labeling data (no hand-picking).

Writes decisions.json (N grid, B grid, dev/test split, accuracy table) and lengths.json
(empirical per-request token counts for each thinking budget, used by the load generator).
"""
import hashlib, json, os, random, sys
sys.path.insert(0, os.environ.get('TCR_DIR', '.'))
from tcr_common import *
import numpy as np

recs = [r for f in sorted((RUN / 'rollouts').glob('chunk_*.json')) for r in json.load(open(f))]
assert recs, 'no rollouts found'
K = len(recs[0]['traces'])
BUD = CFG['budgets']
rng = random.Random(CFG['seed'])
R = 100  # bootstrap resamples per question for the majority-vote estimate
CAND = [n for n in (1, 2, 4, 8, 16) if n <= K]


def vote_acc(rec, b, N):
    keys = [t['b'][str(b)]['key'] for t in rec['traces']]
    ok = {}
    for t in rec['traces']:
        d = t['b'][str(b)]
        ok.setdefault(d['key'], d['ok'])
    if N == 1:
        return sum(t['b'][str(b)]['ok'] for t in rec['traces']) / K
    hit = 0
    for _ in range(R):
        idx = rng.sample(range(K), N)
        cnt = {}
        for i in idx:
            if keys[i] is not None:
                cnt[keys[i]] = cnt.get(keys[i], 0) + 1
        if not cnt:
            continue
        top = max(cnt.values())
        win = rng.choice([k for k, v in cnt.items() if v == top])
        hit += bool(ok[win])
    return hit / R


acc = {str(N): {str(b): float(np.mean([vote_acc(r, b, N) for r in recs])) for b in BUD} for N in CAND}

# N grid: always {1,2,4}; keep doubling while the extra gain from 2N samples is >= 1 accuracy point
# (mean over budgets >= 1024). N=2 is not used as the test because a 2-vote is nearly a coin flip.
hi = [b for b in BUD if b >= 1024] or BUD
n_max = min(4, CAND[-1])
for n in [c for c in CAND if c >= 4 and 2 * c <= CAND[-1]]:
    gain = float(np.mean([acc[str(2 * n)][str(b)] - acc[str(n)][str(b)] for b in hi]))
    if gain < 0.01:
        break
    n_max = 2 * n
N_GRID = [n for n in CAND if n <= n_max]

# B grid: keep all configured budgets, flag any that a cheaper budget beats at N=1
dominated = [b for i, b in enumerate(BUD) if any(acc['1'][str(b2)] >= acc['1'][str(b)] for b2 in BUD[:i])]

# empirical request lengths (generated tokens) per budget
lengths = {}
for b in BUD:
    xs = []
    for r in recs:
        for t in r['traces']:
            ft = t['b'][str(b)]['ft']
            xs.append(int(ft if b == 0 else min(t['L'], b) + ft))
    rng.shuffle(xs)
    lengths[str(b)] = xs[:5000]
json.dump(lengths, open(RUN / 'lengths.json', 'w'))

# dev / test split by hash of the question id (dev is for tuning, test is touched once)
ids = sorted((r['id'] for r in recs), key=lambda s: hashlib.sha1(s.encode()).hexdigest())
n_dev = max(1, int(0.2 * len(ids)))
dec = dict(n_questions=len(recs), K=K, N_GRID=N_GRID, B_GRID=BUD, dominated_B=dominated, acc_vote=acc,
           dev_ids=ids[:n_dev], n_test=len(ids) - n_dev, mean_len={b: float(np.mean(v)) for b, v in lengths.items()},
           rule='N in {1,2,4}, then doubles while the mean gain from 2N samples (budgets>=1024) is >=1 point; B grid fixed; split by sha1(id), 20% dev')
json.dump(dec, open(RUN / 'decisions.json', 'w'), indent=1)
print(f'[decide] N grid {N_GRID}, B grid {BUD}, dominated B {dominated}, dev {n_dev} / test {len(ids) - n_dev}', flush=True)
print('[decide] accuracy (majority vote) rows=N cols=B:', flush=True)
for N in CAND:
    print('[decide]  N=%-2d ' % N + ' '.join('%.3f' % acc[str(N)][str(b)] for b in BUD), flush=True)
