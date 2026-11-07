"""CPU only. Accuracy and token cost of non-thinking mode (B=0) as a function of the token cap.

A sample counts as answered under cap C only if it finished with at most C tokens (otherwise: no answer).
Reads rollouts_nothink_cap/ (from 01b), writes report/nothink_cap_report.md.
"""
import glob, json, os, random, sys
sys.path.insert(0, os.environ.get('TCR_DIR', '.'))
from tcr_common import *
import numpy as np

recs = [r for f in sorted(glob.glob(str(RUN / 'rollouts_nothink_cap' / 'chunk_*.json'))) for r in json.load(open(f))]
assert recs, 'run 01b_relabel_nothink_cap.py first'
K = len(recs[0]['samples'])
rng = random.Random(CFG['seed'])
R = 100
CAPS = [c for c in (512, 1024, 2048, 4096) if c <= max(s['ft'] for r in recs for s in r['samples']) or c == 1024]


def acc_at(cap, N):
    tot = []
    for r in recs:
        S = r['samples']
        ans = [s['fin'] and s['ft'] <= cap for s in S]
        if N == 1:
            tot.append(np.mean([a and s['ok'] for a, s in zip(ans, S)]))
            continue
        okmap = {}
        for a, s in zip(ans, S):
            if a:
                okmap.setdefault(s['key'], s['ok'])
        hit = 0
        for _ in range(R):
            cnt = {}
            for i in rng.sample(range(K), N):
                if ans[i] and S[i]['key'] is not None:
                    cnt[S[i]['key']] = cnt.get(S[i]['key'], 0) + 1
            if cnt:
                top = max(cnt.values())
                hit += bool(okmap[rng.choice([k for k, v in cnt.items() if v == top])])
        tot.append(hit / R)
    return float(np.mean(tot))


L = ['# Non-thinking mode (B=0) vs token cap', f'{len(recs)} questions x {K} samples, same questions as the main run.\n',
     '| cap (tokens) | answered | acc among answered | acc N=1 | acc N=4 | acc N=8 | acc N=16 | mean tokens/sample |',
     '|---|---|---|---|---|---|---|---|']
for cap in CAPS:
    S = [s for r in recs for s in r['samples']]
    ans = [s for s in S if s['fin'] and s['ft'] <= cap]
    row = (len(ans) / len(S), np.mean([s['ok'] for s in ans]) if ans else float('nan'),
           acc_at(cap, 1), acc_at(cap, 4), acc_at(cap, 8), acc_at(cap, 16),
           np.mean([min(s['ft'], cap) for s in S]))
    L.append(f'| {cap} | {row[0]:.3f} | {row[1]:.3f} | {row[2]:.3f} | {row[3]:.3f} | {row[4]:.3f} | {row[5]:.3f} | {row[6]:.0f} |')
ft = np.array([s['ft'] for r in recs for s in r['samples']])
L.append(f'\nGeneration length: median {np.median(ft):.0f}, p90 {np.percentile(ft, 90):.0f}, p99 {np.percentile(ft, 99):.0f} tokens.')
L.append('Compare the 2048-thinking budget: acc N=1 = 0.766 (main report). If the cap at 1024 matches the main run '
         'figures (0.658 at N=1), the main B=0 results are reproduced.')
(RUN / 'report').mkdir(exist_ok=True)
(RUN / 'report' / 'nothink_cap_report.md').write_text('\n'.join(L))
print('\n'.join(L))
