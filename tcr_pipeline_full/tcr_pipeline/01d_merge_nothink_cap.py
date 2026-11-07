"""CPU only. Build rollouts_v2/: same as rollouts/ but B=0 is non-thinking with a 2048-token cap (from 01b).

A B=0 sample counts as answered only if it finished (stop) with at most CAP tokens; otherwise no answer, ft = CAP.
Originals are never modified. Also writes lengths_v2.json (request lengths with the new B=0) and report/accuracy_v2.md.
"""
import glob, json, os, random, sys
sys.path.insert(0, os.environ.get('TCR_DIR', '.'))
from tcr_common import *
import numpy as np

CAP = int(os.environ.get('TCR_B0_CAP', '2048'))
src = sorted(glob.glob(str(RUN / 'rollouts' / 'chunk_*.json')))
nt = sorted(glob.glob(str(RUN / 'rollouts_nothink_cap' / 'chunk_*.json')))
assert src and len(src) == len(nt), f'need matching chunks: {len(src)} rollouts vs {len(nt)} nothink'
out = RUN / 'rollouts_v2'
out.mkdir(exist_ok=True)
BUD = CFG['budgets']
recs_all = []
for fa, fb in zip(src, nt):
    A, B_ = json.load(open(fa)), json.load(open(fb))
    assert [r['id'] for r in A] == [r['id'] for r in B_], f'question order differs in {fa}'
    for ra, rb in zip(A, B_):
        for k, t in enumerate(ra['traces']):
            s = rb['samples'][k]
            ans = bool(s['fin'] and s['ft'] <= CAP)
            t['b']['0'] = dict(pred=s['pred'] if ans else None, key=s['key'] if ans else None,
                               ok=bool(s['ok'] and ans), ft=min(s['ft'], CAP))
    atomic_write_json(out / os.path.basename(fa), A)
    recs_all += A

lens = {}
for b in BUD:
    xs = [int(t['b'][str(b)]['ft'] if b == 0 else min(t['L'], b) + t['b'][str(b)]['ft']) for r in recs_all for t in r['traces']]
    random.Random(CFG['seed']).shuffle(xs)
    lens[str(b)] = xs[:5000]
json.dump(lens, open(RUN / 'lengths_v2.json', 'w'))

K = len(recs_all[0]['traces'])
rng = random.Random(CFG['seed'])


def vote_acc(rec, b, N, R=100):
    if N == 1:
        return float(np.mean([t['b'][str(b)]['ok'] for t in rec['traces']]))
    keys = [t['b'][str(b)]['key'] for t in rec['traces']]
    okmap = {}
    for t in rec['traces']:
        okmap.setdefault(t['b'][str(b)]['key'], t['b'][str(b)]['ok'])
    hit = 0
    for _ in range(R):
        cnt = {}
        for i in rng.sample(range(K), N):
            if keys[i] is not None:
                cnt[keys[i]] = cnt.get(keys[i], 0) + 1
        if cnt:
            top = max(cnt.values())
            hit += bool(okmap[rng.choice([k for k, v in cnt.items() if v == top])])
    return hit / R


Ns = [n for n in (1, 2, 4, 8, 16) if n <= K]
L = [f'# Accuracy by (N, B) with B=0 = non-thinking, {CAP}-token cap', f'{len(recs_all)} questions x {K} traces\n',
     '| N \\ B | ' + ' | '.join(str(b) for b in BUD) + ' |', '|---|' + '---|' * len(BUD)]
for N in Ns:
    L.append(f'| {N} | ' + ' | '.join('%.3f' % np.mean([vote_acc(r, b, N) for r in recs_all]) for b in BUD) + ' |')
L.append('\nMean tokens per trace: ' + ', '.join(f'B={b}: {np.mean(lens[str(b)]):.0f}' for b in BUD))
(RUN / 'report').mkdir(exist_ok=True)
(RUN / 'report' / 'accuracy_v2.md').write_text('\n'.join(L))
print('\n'.join(L))
