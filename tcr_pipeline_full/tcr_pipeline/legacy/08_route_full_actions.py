"""CPU only. Per-question routing over the FULL (N,B) grid under a p95 SLO, using the fitted cost model.

Setup: questions arrive at rho x capacity (capacity counted in reference requests of ~1005 tokens). A policy that spends
T tokens per question per request offers load rho*T/ref; the cost model (isotonic curve, 06) predicts background p95.
A policy is FEASIBLE at rho if predicted p95 <= SLO. Metric: best accuracy among feasible operating points.

Policies (decision uses out-of-fold predictions, tokens in decisions = mean tokens of the action):
  fixed        best single (N,B), also mixing two fixed actions at random (convex hull)
  level-bin    MATH level -> action, a*=argmax pbar(level,a)-lam*tok(a)
  probe        ridge on hidden states, a*=argmax phat(x,a)-lam*tok(a)
  oracle       true per-question success probability (upper bound)
Operating point (action / lambda) is picked on the test set for ALL policies alike (equal footing, slightly optimistic
for every policy). Paired bootstrap over questions gives 95% CIs for probe minus the best baseline.
"""
import glob, json, os, sys
sys.path.insert(0, os.environ.get('TCR_DIR', '.'))
from tcr_common import *
import numpy as np, random
from sklearn.linear_model import RidgeCV
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rdir = RUN / ('rollouts_v2' if (RUN / 'rollouts_v2').exists() else 'rollouts')
print(f'[route] labels from {rdir.name}', flush=True)
recs = [r for f in sorted(glob.glob(str(rdir / 'chunk_*.json'))) for r in json.load(open(f))]
z = np.load(RUN / 'probe' / 'hidden.npz')
assert [r['id'] for r in recs] == list(z['ids'])
dec = json.load(open(RUN / 'decisions.json'))
test = ~np.isin(z['ids'], dec['dev_ids'])
BUD, K, Q = CFG['budgets'], len(recs[0]['traces']), len(recs)
Ns = [n for n in (1, 2, 4, 8, 16) if n <= K]
A = [(n, b) for n in Ns for b in BUD]
rng = random.Random(CFG['seed'])


def vote_acc(rec, b, N, R=60):
    if N == 1:
        return float(np.mean([t['b'][str(b)]['ok'] for t in rec['traces']]))
    keys = [t['b'][str(b)]['key'] for t in rec['traces']]
    okm = {}
    for t in rec['traces']:
        okm.setdefault(t['b'][str(b)]['key'], t['b'][str(b)]['ok'])
    hit = 0
    for _ in range(R):
        c = {}
        for i in rng.sample(range(K), N):
            if keys[i] is not None:
                c[keys[i]] = c.get(keys[i], 0) + 1
        if c:
            top = max(c.values())
            hit += bool(okm[rng.choice([k for k, v in c.items() if v == top])])
    return hit / R


P = np.array([[vote_acc(r, b, n) for (n, b) in A] for r in recs])                       # q,a  true success prob
tokq = np.array([[np.mean([(t['b'][str(b)]['ft'] if b == 0 else min(t['L'], b) + t['b'][str(b)]['ft'])
                           for t in r['traces']]) for b in BUD] for r in recs])
TOK = np.array([[n * tokq[q, BUD.index(b)] for (n, b) in A] for q in range(Q)])          # q,a actual tokens
tokA = TOK.mean(0)                                                                       # per-action mean (used in decisions)
level = np.array([r['level'] for r in recs])
X = z['X'].astype(np.float32)
li = int(np.load(RUN / 'probe' / 'oof_pred.npz')['layer']) if (RUN / 'probe' / 'oof_pred.npz').exists() else X.shape[1] - 1
F = X[:, li, :]
print(f'[route] probe layer index {li}', flush=True)

# out-of-fold predictions
ph, pb = np.zeros_like(P), np.zeros_like(P)
for tr, te in KFold(5, shuffle=True, random_state=CFG['seed']).split(F):
    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6
    for j in range(len(A)):
        ph[te, j] = np.clip(RidgeCV(alphas=[1e2, 1e3, 1e4, 1e5]).fit((F[tr] - mu) / sd, P[tr, j]).predict((F[te] - mu) / sd), 0, 1)
    for lv in np.unique(level):
        m = level[tr] == lv
        pb[te[level[te] == lv]] = P[tr][m].mean(0) if m.any() else P[tr].mean(0)

# cost model
cal = json.load(open(RUN / 'load' / 'calibration.json'))
cells = [json.loads(x) for x in open(RUN / 'load' / 'ext_cells.jsonl') if x.strip()]
lens = json.load(open(RUN / 'lengths.json'))
ml = {int(b): float(np.mean(v)) for b, v in lens.items()}
TR, ref = CFG['serve']['target_rate_frac'], cal['mean_len_ref']
ag = {}
for c in cells:
    if c['frac'] > 0 and c['bg'].get('n'):
        ag.setdefault((c['frac'], c['N'], c['B']), []).append(c['bg']['lat_p95'])
xs = [fr + TR * N * ml[B] / ref for (fr, N, B) in ag]
iso = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(xs, [np.mean(v) for v in ag.values()])
XMAX = max(xs)
SLOS = {'2x idle p95': cal['slos'][0], '4x idle p95': cal['slos'][1]}


def p95(rho, T):
    return float(iso.predict([rho * T / ref])[0])


def evaluate(choice, idx):
    q = np.arange(Q)[idx]
    return P[q, choice[idx]].mean(), TOK[q, choice[idx]].mean(), P[q, choice[idx]]


LAMS = np.concatenate([[0], np.logspace(-3, 1.5, 70)])
tk1000 = tokA / 1000.0


def policy_points(S, idx):
    pts = []
    for lam in LAMS:
        ch = np.argmax(S - lam * tk1000, axis=1)
        a, t, v = evaluate(ch, idx)
        pts.append((a, t, v))
    return pts


def best_feasible(pts, rho, slo):
    ok = [p for p in pts if p95(rho, p[1]) <= slo]
    return max(ok, key=lambda p: p[0]) if ok else min(pts, key=lambda p: p[1])


fixed_pts = [(P[test, j].mean(), TOK[test, j].mean(), P[test, j]) for j in range(len(A))]
# mixing two fixed actions with weight w (random assignment): expected acc/tokens linear
mix_pts = []
for i in range(len(A)):
    for j in range(len(A)):
        for w in np.linspace(0, 1, 21):
            v = (1 - w) * P[test, i] + w * P[test, j]
            mix_pts.append((v.mean(), (1 - w) * TOK[test, i].mean() + w * TOK[test, j].mean(), v))
pol = {'level-bin': policy_points(pb, test), 'probe': policy_points(ph, test), 'oracle': policy_points(P, test)}

RHO = [0.4, 0.6, 0.8, 1.0, 1.2]
rng2 = np.random.default_rng(0)
L = [f'# Routing over the full (N,B) grid under a p95 SLO (labels: {rdir.name}, {test.sum()} test questions)\n',
     f'{len(A)} actions. rho = arrival rate as a fraction of capacity (reference requests of {ref:.0f} tokens). Predicted background p95 comes from the '
     f'cost model (06); offered loads above {XMAX:.2f} (largest measured) are clipped, so those cells are optimistic.\n']
curve = {}
for sname, slo in SLOS.items():
    L.append(f'## SLO: p95 <= {slo:.1f} s ({sname})\n')
    L.append('| rho | best fixed | fixed mix | level-bin | probe | oracle | probe - best baseline (95% CI) |')
    L.append('|---|---|---|---|---|---|---|')
    for rho in RHO:
        bf = best_feasible(fixed_pts, rho, slo)
        bm = best_feasible(mix_pts, rho, slo)
        bl = best_feasible(pol['level-bin'], rho, slo)
        bp = best_feasible(pol['probe'], rho, slo)
        bo = best_feasible(pol['oracle'], rho, slo)
        base = max([bf, bm, bl], key=lambda p: p[0])
        d = bp[2] - base[2]
        bs = [d[rng2.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        cell = lambda p: f'{p[0]:.3f} ({p[1]:.0f} tok)'
        L.append(f'| {rho} | {cell(bf)} | {cell(bm)} | {cell(bl)} | {cell(bp)} | {cell(bo)} | '
                 f'{d.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}] |')
        curve.setdefault(sname, []).append((rho, bf[0], bm[0], bl[0], bp[0], bo[0]))
    L.append('')
bf_idx = int(np.argmax([p[0] for p in fixed_pts]))
L.append(f'Unconstrained best fixed action: {A[bf_idx]} acc {fixed_pts[bf_idx][0]:.3f} at {fixed_pts[bf_idx][1]:.0f} tokens.')
L.append('Feasible = predicted p95 under SLO; if nothing is feasible the cheapest point is shown (accuracy collapses).')
(RUN / 'report').mkdir(exist_ok=True)
(RUN / 'report' / 'route_full_report.md').write_text('\n'.join(L))
print('\n'.join(L))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a_, (sname, rows) in zip(ax, curve.items()):
    r = np.array(rows)
    for k, lab in enumerate(['best fixed', 'fixed mix', 'level-bin', 'probe', 'oracle'], 1):
        a_.plot(r[:, 0], r[:, k], marker='o', label=lab)
    a_.set_title(f'SLO {sname}'); a_.set_xlabel('arrival rate (fraction of capacity)'); a_.set_ylabel('accuracy at feasible point')
ax[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(RUN / 'report' / 'route_full.png', dpi=140)
