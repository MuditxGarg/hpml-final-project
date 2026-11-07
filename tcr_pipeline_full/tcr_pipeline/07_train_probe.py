"""CPU only. Success probe + routing frontier at N=1.

1. Probe: ridge on prompt hidden states (per layer) predicts p(x,B) = P(trace correct at budget B). 5-fold
   cross-validated over questions; layer chosen on the DEV ids, all numbers reported on the TEST ids.
   Baselines: prompt length, MATH level, and ridge on [length, level].
2. Routing frontier (N=1): send the questions with the largest predicted gain p(2048)-p(0) to thinking, rest to
   non-thinking. Accuracy (expected, from the 16 traces) vs mean tokens, compared with mixing the two fixed actions at random
   (straight line), a difficulty-bin rule (MATH level) and the oracle (true gain). Uses rollouts_v2 if present.
"""
import glob, json, os, sys
sys.path.insert(0, os.environ.get('TCR_DIR', '.'))
from tcr_common import *
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rdir = RUN / ('rollouts_v2' if (RUN / 'rollouts_v2').exists() else 'rollouts')
print(f'[probe] labels from {rdir.name}', flush=True)
recs = [r for f in sorted(glob.glob(str(rdir / 'chunk_*.json'))) for r in json.load(open(f))]
z = np.load(RUN / 'probe' / 'hidden.npz')
assert [r['id'] for r in recs] == list(z['ids']), 'question order differs between rollouts and probe features'
dec = json.load(open(RUN / 'decisions.json'))
dev = np.isin(z['ids'], dec['dev_ids'])
test = ~dev
BUD = CFG['budgets']
K = len(recs[0]['traces'])
ok = np.array([[[t['b'][str(b)]['ok'] for t in r['traces']] for b in BUD] for r in recs], dtype=float)  # q,b,k
P = ok.mean(2)                                                                                           # q,b
TOK = np.array([[np.mean([(t['b'][str(b)]['ft'] if b == 0 else min(t['L'], b) + t['b'][str(b)]['ft']) for t in r['traces']])
                 for b in BUD] for r in recs])
level = np.array([r['level'] for r in recs], dtype=float)
plen = z['plen'].astype(float)
X = z['X'].astype(np.float32)
nL = X.shape[1]
bi = {b: i for i, b in enumerate(BUD)}


def oof(feat):
    """5-fold out-of-fold ridge predictions of P for every budget."""
    out = np.zeros_like(P)
    for tr, te in KFold(5, shuffle=True, random_state=CFG['seed']).split(feat):
        mu, sd = feat[tr].mean(0), feat[tr].std(0) + 1e-6
        for j in range(len(BUD)):
            m = RidgeCV(alphas=[1, 10, 100, 1e3, 1e4, 1e5]).fit((feat[tr] - mu) / sd, P[tr, j])
            out[te, j] = np.clip(m.predict((feat[te] - mu) / sd), 0, 1)
    return out


def auc(score, b, mask):
    y = ok[mask, bi[b], :].ravel()
    s = np.repeat(score[mask], K)
    return roc_auc_score(y, s)


L = [f'# Success probe (labels: {rdir.name}; dev {dev.sum()} / test {test.sum()} questions)\n',
     'AUROC of "this trace is correct", score = predicted success of the question.\n',
     '| layer index | dev AUROC B=0 | dev AUROC B=2048 | test AUROC B=0 | test AUROC B=2048 |', '|---|---|---|---|---|']
preds = {}
for li in range(nL):
    preds[li] = oof(X[:, li, :])
    L.append(f"| {li} (layer {int(z['layers'][li])}) | {auc(preds[li][:, 0], 0, dev):.3f} | {auc(preds[li][:, -1], 2048, dev):.3f} | "
             f"{auc(preds[li][:, 0], 0, test):.3f} | {auc(preds[li][:, -1], 2048, test):.3f} |")
best = max(range(nL), key=lambda l: auc(preds[l][:, 0], 0, dev) + auc(preds[l][:, -1], 2048, dev))
pr = preds[best]
base = oof(np.stack([plen, level], 1))
L.append(f"\nLayer chosen on dev: index {best} (layer {int(z['layers'][best])}).\n")
L.append('| score | test AUROC B=0 | test AUROC B=2048 |')
L.append('|---|---|---|')
L.append(f"| probe (hidden state) | {auc(pr[:, 0], 0, test):.3f} | {auc(pr[:, -1], 2048, test):.3f} |")
L.append(f"| prompt length (longer = harder) | {auc(-plen, 0, test):.3f} | {auc(-plen, 2048, test):.3f} |")
L.append(f"| MATH level (higher = harder) | {auc(-level, 0, test):.3f} | {auc(-level, 2048, test):.3f} |")
L.append(f"| ridge on [length, level] | {auc(base[:, 0], 0, test):.3f} | {auc(base[:, -1], 2048, test):.3f} |")

# ---- routing frontier at N=1: B=0 vs B=2048 ----
j0, j1 = bi[0], bi[2048]
t0, t1 = TOK[test, j0].mean(), TOK[test, j1].mean()
p0, p1 = P[test, j0], P[test, j1]
c0, c1 = TOK[test, j0], TOK[test, j1]


def frontier(score):
    order = np.argsort(-score)
    accs, toks = [p0.mean()], [c0.mean()]
    sel_p, sel_c = p0.copy(), c0.copy()
    for i in order:
        sel_p[i], sel_c[i] = p1[i], c1[i]
        accs.append(sel_p.mean()); toks.append(sel_c.mean())
    return np.array(toks), np.array(accs)


curves = {
    'probe gain (p2048 - p0)': frontier(pr[test, j1] - pr[test, j0]),
    'probe, B=2048 success only (hard first)': frontier(-pr[test, j1]),
    'MATH level (hardest first)': frontier(level[test] + 1e-3 * plen[test] / plen.max()),
    'oracle (true gain)': frontier(p1 - p0),
}
tk = np.linspace(t0, t1, 5)[1:-1]
L.append(f'\n## Routing frontier at N=1 on the {test.sum()} test questions')
L.append(f'All non-thinking: acc {p0.mean():.3f} at {t0:.0f} tokens. All thinking(2048): acc {p1.mean():.3f} at {t1:.0f} tokens.')
L.append('Accuracy at equal mean tokens (random mixing of the two fixed actions = straight line):\n')
L.append('| mean tokens | random mix | ' + ' | '.join(curves) + ' |')
L.append('|---|---|' + '---|' * len(curves))
for t in tk:
    mix = p0.mean() + (p1.mean() - p0.mean()) * (t - t0) / (t1 - t0)
    L.append(f'| {t:.0f} | {mix:.3f} | ' + ' | '.join(f'{np.interp(t, *c):.3f}' for c in curves.values()) + ' |')
(RUN / 'report' / 'probe_report.md').write_text('\n'.join(L))
print('\n'.join(L))
np.savez(RUN / 'probe' / 'oof_pred.npz', ids=z['ids'], layer=best, pred=pr, budgets=np.array(BUD), P=P, TOK=TOK)

plt.figure(figsize=(6, 4.2))
plt.plot([t0, t1], [p0.mean(), p1.mean()], 'k--', label='random mix')
for k, (tt, aa) in curves.items():
    plt.plot(tt, aa, label=k)
plt.xlabel('mean tokens per question'); plt.ylabel('expected accuracy (test)')
plt.legend(fontsize=7); plt.tight_layout()
plt.savefig(RUN / 'report' / 'routing_frontier.png', dpi=140)
