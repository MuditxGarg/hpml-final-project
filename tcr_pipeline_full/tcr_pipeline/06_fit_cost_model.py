"""CPU only. Do all sweep cells collapse onto ONE curve of background p95 vs offered token load?

Offered load of a cell (in units of saturation token rate) = background load fraction
 + target_rate_frac * N * mean_tokens(B) / mean_tokens(reference)   (what was actually sent: original lengths.json).
Compared with two token-blind alternatives (each sample = one reference request; each request = one reference request).
Leave-one-action-out error says which load definition predicts unseen actions best.
Writes report/cost_model_report.md and report/cost_model_curve.png.
"""
import json, os, sys
sys.path.insert(0, os.environ.get('TCR_DIR', '.'))
from tcr_common import *
import numpy as np
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

cells = [json.loads(x) for x in open(RUN / 'load' / 'ext_cells.jsonl') if x.strip()]
cal = json.load(open(RUN / 'load' / 'calibration.json'))
lens = json.load(open(RUN / 'lengths.json'))
lens2 = json.load(open(RUN / 'lengths_v2.json')) if (RUN / 'lengths_v2.json').exists() else lens
TR = CFG['serve']['target_rate_frac']
ref = cal['mean_len_ref']
ml = {int(b): float(np.mean(v)) for b, v in lens.items()}
ml2 = {int(b): float(np.mean(v)) for b, v in lens2.items()}

agg = {}
for c in cells:
    if c['frac'] <= 0 or not c['bg'].get('n'):
        continue
    agg.setdefault((c['frac'], c['N'], c['B']), []).append(c)
rows = []
for (fr, N, B), cs in sorted(agg.items()):
    y = float(np.mean([c['bg']['lat_p95'] for c in cs]))
    s41 = float(np.mean([c['bg'].get(f"slo_{cal['slos'][1]:.0f}s", np.nan) for c in cs]))
    tok = fr + (TR * N * ml[B] / ref if N else 0.0)
    smp = fr + (TR * N if N else 0.0)
    req = fr + (TR if N else 0.0)
    rows.append(dict(fr=fr, N=N, B=B, y=y, s41=s41, tok=tok, smp=smp, req=req, n=len(cs)))
y = np.array([r['y'] for r in rows])
act = [(r['N'], r['B']) for r in rows]
actions = sorted(set(act))


def fit(x, yy):
    m = IsotonicRegression(increasing=True, out_of_bounds='clip')
    m.fit(x, yy)
    return m


L = ['# Cost model: background p95 vs offered load', f'{len(rows)} cells (load x action, averaged over repetitions). '
     f'Reference request = {ref:.0f} tokens; extra stream = {TR:.2f} x capacity in requests/s.\n',
     '| load definition | in-sample R^2 | Spearman | leave-one-action-out MAE (s) | worst action MAE (s) |', '|---|---|---|---|---|']
res = {}
for name, key in [('tokens (N x mean tokens(B))', 'tok'), ('samples only (N)', 'smp'), ('requests only', 'req')]:
    x = np.array([r[key] for r in rows])
    m = fit(x, y)
    pred = m.predict(x)
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    rho = spearmanr(x, y)[0]
    errs = {}
    for a in actions:
        tr = [i for i, aa in enumerate(act) if aa != a]
        te = [i for i, aa in enumerate(act) if aa == a]
        mm = fit(x[tr], y[tr])
        errs[a] = float(np.mean(np.abs(mm.predict(x[te]) - y[te])))
    mae = float(np.mean([abs(fit(x[[j for j in range(len(rows)) if act[j] != act[i]]],
                                  y[[j for j in range(len(rows)) if act[j] != act[i]]]).predict(x[[i]])[0] - y[i])
                         for i in range(len(rows))]))
    worst = max(errs, key=errs.get)
    res[key] = (m, x, pred, errs)
    L.append(f'| {name} | {r2:.3f} | {rho:.3f} | {mae:.2f} | {errs[worst]:.1f} {worst} |')

m, x, pred, errs = res['tok']
L.append('\n## Biggest residuals under the token model (observed minus fitted p95, seconds)')
for i in np.argsort(-np.abs(y - pred))[:8]:
    r = rows[i]
    L.append(f"- load {r['fr']:.1f}, action ({r['N']},{r['B']}): observed {y[i]:.1f}, fitted {pred[i]:.1f}, offered load {r['tok']:.2f}")
L.append('\n## Predicted background p95 (s) if ALL of the extra stream used action (N,B), token model; B=0 uses the new lengths if available')
L.append('Values above the largest observed offered load are clipped (marked *).')
L.append('| action | load 0.5 | load 0.8 | load 1.0 |')
L.append('|---|---|---|---|')
xmax = max(r['tok'] for r in rows)
for (N, B) in [(1, 0), (2, 0), (4, 0), (8, 0), (16, 0), (1, 2048), (2, 2048), (4, 2048)]:
    cellp = []
    for fr in (0.5, 0.8, 1.0):
        o = fr + TR * N * ml2[B] / ref
        cellp.append(f'{m.predict([o])[0]:.1f}' + ('*' if o > xmax else ''))
    L.append(f'| ({N},{B}) | ' + ' | '.join(cellp) + ' |')
L.append(f'\nNoise is about +-3 s (2 repetitions). Control cells (no extra stream) are included as N=0.')
(RUN / 'report' / 'cost_model_report.md').write_text('\n'.join(L))
print('\n'.join(L))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
mk = {0: 'o', 1024: 's', 2048: '^'}
for a_i, (key, title) in enumerate([('tok', 'offered token load (model)'), ('req', 'requests only (token-blind)')]):
    mm, xx, pp, _ = res[key]
    for (N, B) in actions:
        idx = [i for i, aa in enumerate(act) if aa == (N, B)]
        ax[a_i].scatter(xx[idx], y[idx], marker=mk.get(B, 'o'), s=18 + 8 * N, alpha=.75,
                        label=f'N={N},B={B}' if a_i == 0 else None)
    g = np.linspace(xx.min(), xx.max(), 200)
    ax[a_i].plot(g, mm.predict(g), 'k-', lw=1.5)
    ax[a_i].set_xlabel(title)
    ax[a_i].set_ylabel('background p95 (s)')
ax[0].legend(fontsize=6, ncol=2)
plt.tight_layout()
plt.savefig(RUN / 'report' / 'cost_model_curve.png', dpi=140)
