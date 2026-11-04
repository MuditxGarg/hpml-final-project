"""Analysis + go/no-go verdict for LATTE. Works on whatever data exists (partial runs are fine)."""
import json, math, os, random, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
sys.path.insert(0, os.environ.get('LATTE_DIR', '/content/latte'))
from common import CFG, RUN
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = RUN / 'report'
OUT.mkdir(exist_ok=True)
REP = []


def say(s=''):
    print(s, flush=True)
    REP.append(str(s))


def finish():
    (OUT / 'report.md').write_text('\n'.join(REP))
    sys.exit(0)


rnd = random.Random(CFG['seed'])
rng = np.random.default_rng(CFG['seed'])
TH = dict(D=3.0, regret=2.0, gain=3.0, auroc=0.03)  # thresholds (accuracy points / ratios)

# ------------------------------------------------------------------ data
recs = []
for f in sorted((RUN / 'rollouts').glob('chunk_*.json')):
    recs += json.load(open(f))
say(f"# LATTE go/no-go report ({'SMOKE' if CFG.get('smoke') else 'MAIN'} run, model {CFG['model']})")
if len(recs) < 4:
    say('Not enough rollout data yet.')
    finish()
P, K, BUD = len(recs), len(recs[0]['traces']), CFG['budgets']
nb = len(BUD)
corr = np.zeros((P, K, nb), bool)
tokn = np.zeros((P, K, nb))
keys = [[[None] * nb for _ in range(K)] for _ in range(P)]
for i, r in enumerate(recs):
    for k, t in enumerate(r['traces']):
        for bi, b in enumerate(BUD):
            d = t['b'][str(b)]
            corr[i, k, bi] = d['ok']
            keys[i][k][bi] = d['key']
            tokn[i, k, bi] = (0 if b == 0 else min(t['L'], b)) + d['ft']
say(f'{P} prompts x {K} traces x budgets {BUD}')
say(f"single-sample accuracy by budget: " + ', '.join(f'B={b}: {corr[:, :, bi].mean():.3f}' for bi, b in enumerate(BUD)))
say(f"mean tokens per trace by budget: " + ', '.join(f'B={b}: {tokn[:, :, bi].mean():.0f}' for bi, b in enumerate(BUD)))

# ------------------------------------------------------------------ cost models from the load test
idle_f, cells_f = RUN / 'load' / 'idle.json', RUN / 'load' / 'cells.jsonl'
idle = json.load(open(idle_f)) if idle_f.exists() else []
cells = [json.loads(l) for l in open(cells_f) if l.strip()] if cells_f.exists() else []
ITL_a, ITL_b, ITL_c, itl_src = 0.0134, 0.0003, 0.1, 'default (A100 pilot-like)'
if len(idle) >= 4:
    A_ = np.array([[r['T'], r['T'] * r['N'], 1.0] for r in idle])
    y_ = np.array([r['lat'] for r in idle])
    sol = np.linalg.lstsq(A_, y_, rcond=None)[0]
    if sol[0] > 0:
        ITL_a, ITL_b, ITL_c, itl_src = float(sol[0]), max(float(sol[1]), 0.0), max(float(sol[2]), 0.0), 'fit from load test'
tps_best = {}
for r in cells:
    tps_best[(r['N'], r['T'])] = max(tps_best.get((r['N'], r['T']), 0.0), r['tps'])
byT = {}
for (N, T), v in tps_best.items():
    byT.setdefault(T, []).append(v)
if byT:
    Ts = sorted(byT)
    gs = [1.0 / float(np.mean(byT[T])) for T in Ts]
    g_src = 'fit from load test'
else:
    Ts, gs, g_src = [128, 4096], [1 / 3000.0, 1 / 3000.0], 'default 3000 tok/s'
logTs = np.log(np.array(Ts, float))


def g_of(T):
    return float(np.interp(math.log(max(T, 1.0)), logTs, gs))


say(f"cost models: per-token decode latency = {ITL_a * 1000:.1f} ms + {ITL_b * 1000:.2f} ms x N ({itl_src}); "
    f"saturated GPU-seconds/token ~ {np.mean(gs) * 1000:.2f} ms ({g_src})")

# ------------------------------------------------------------------ per-prompt action tables
Ns = [n for n in (1, 2, 4, 8) if n <= K // 2]
ACT = [(N, bi) for N in Ns for bi in range(nb)]
A = len(ACT)
pools = [list(range(K // 2)), list(range(K // 2, K))]
FULL = list(range(K))
R = 48
COSTS = ('tok', 'idle', 'sat')


def vote(idx, i, bi):
    counts, first = {}, {}
    for k in idx:
        key = keys[i][k][bi]
        if key is None:
            continue
        counts[key] = counts.get(key, 0) + 1
        first.setdefault(key, k)
    if not counts:
        return False
    best = max(counts.values())
    cands = [kk for kk, c in counts.items() if c == best]
    key = cands[0] if len(cands) == 1 else rnd.choice(cands)
    return bool(corr[i, first[key], bi])


def astats(i, bi, N, pool):
    if N == 1:
        subs = [[k] for k in pool]
    elif N >= len(pool):
        subs = [list(pool)]
    else:
        subs = [rng.choice(pool, N, replace=False).tolist() for _ in range(R)]
    p = float(np.mean([vote(s, i, bi) for s in subs]))
    tk = float(np.mean([sum(tokn[i, k, bi] for k in s) for s in subs]))
    lt = float(np.mean([max(tokn[i, k, bi] for k in s) * (ITL_a + ITL_b * N) + ITL_c for s in subs]))
    st = float(np.mean([sum(tokn[i, k, bi] * g_of(tokn[i, k, bi]) for k in s) for s in subs]))
    return p, dict(tok=tk, idle=lt, sat=st)


PH = np.zeros((2, P, A))
CH = {m: np.zeros((2, P, A)) for m in COSTS}
PF = np.zeros((P, A))
CF = {m: np.zeros((P, A)) for m in COSTS}
for i in range(P):
    for a, (N, bi) in enumerate(ACT):
        for h in (0, 1):
            p, c = astats(i, bi, N, pools[h])
            PH[h, i, a] = p
            for m in COSTS:
                CH[m][h, i, a] = c[m]
        p, c = astats(i, bi, N, FULL)
        PF[i, a] = p
        for m in COSTS:
            CF[m][i, a] = c[m]
RAW = {m: float(CH[m].mean()) for m in COSTS}
for m in COSTS:  # normalise each cost model to mean 1 so one lambda grid fits all
    CH[m] /= RAW[m]
    CF[m] /= RAW[m]
LAMS = np.logspace(-3, 1.5, 70)
aname = lambda a: f'N={ACT[a][0]},B={BUD[ACT[a][1]]}'


def envelope(points):
    best, env = -1.0, []
    for c, a in sorted(points):
        if a > best + 1e-12:
            env.append((c, a))
            best = a
    return np.array(env)


def fixed_env(Pm, Cm):  # Pm,Cm shaped (...,P,A) -> mean over everything but A
    pts = [(Cm[..., a].mean(), Pm[..., a].mean()) for a in range(A)]
    return envelope(pts), pts


def route_cf(Csel, Ceval):
    pts = []
    ar = np.arange(P)
    for lam in LAMS:
        accs, cs = [], []
        for h in (0, 1):
            sel = np.argmax(PH[h] - lam * Csel[h], axis=1)
            accs.append(PH[1 - h][ar, sel].mean())
            cs.append(Ceval[1 - h][ar, sel].mean())
        pts.append((float(np.mean(cs)), float(np.mean(accs))))
    return pts


def gain_over(pts, env):
    lo, hi = env[0, 0], env[-1, 0]
    g = [(a - np.interp(c, env[:, 0], env[:, 1])) * 100 for c, a in pts if lo <= c <= hi]
    return (max(g), float(np.mean(g))) if g else (float('nan'), float('nan'))


# ------------------------------------------------------------------ test 2: headroom
say('\n## Test 2 - is there headroom for a router? (cross-fitted oracle vs best fixed action)')
ENV, ORACLE_GAIN, ORACLE_PTS, FIX_PTS = {}, {}, {}, {}
for m in COSTS:
    env, fpts = fixed_env(PH, CH[m])
    ENV[m], FIX_PTS[m] = env, fpts
    opts = route_cf(CH[m], CH[m])
    ORACLE_PTS[m] = opts
    ORACLE_GAIN[m] = gain_over(opts, env)
    say(f'- cost = {m}: oracle gain over best-fixed envelope: max {ORACLE_GAIN[m][0]:.1f} pts, mean {ORACLE_GAIN[m][1]:.1f} pts')
best_fixed = max(range(A), key=lambda a: PF[:, a].mean())
say(f'- most accurate fixed action: {aname(best_fixed)} at {PF[:, best_fixed].mean():.3f}; '
    f'cheapest: {aname(min(range(A), key=lambda a: CF["tok"][:, a].mean()))}')

# ------------------------------------------------------------------ does load-awareness matter?
say('\n## Does ignoring load cost accuracy? (regret matrix, accuracy points at equal true cost)')
say('rows = cost model the router *assumes*, columns = regime it actually runs in')
regret = {}
for e in ('idle', 'sat'):
    matched = envelope(route_cf(CH[e], CH[e]))
    for r_ in COSTS:
        pts = route_cf(CH[r_], CH[e])
        env_r = envelope(pts)
        lo, hi = matched[0, 0], matched[-1, 0]
        g = [(np.interp(c, matched[:, 0], matched[:, 1]) - a) * 100 for c, a in env_r if lo <= c <= hi]
        regret[(r_, e)] = (max(g), float(np.mean(g))) if g else (float('nan'), float('nan'))
say('| assumed \\ real | idle-latency regime | saturated-GPU regime |')
say('|---|---|---|')
for r_ in COSTS:
    say(f"| {r_} | max {regret[(r_, 'idle')][0]:.1f} / mean {regret[(r_, 'idle')][1]:.1f} | "
        f"max {regret[(r_, 'sat')][0]:.1f} / mean {regret[(r_, 'sat')][1]:.1f} |")
worst = {r_: np.nanmax([regret[(r_, 'idle')][0], regret[(r_, 'sat')][0]]) for r_ in COSTS}
for r_ in COSTS:
    say(f'- router assuming {r_} cost: worst-case regret across regimes = {worst[r_]:.1f} pts')
max_regret = float(min(worst.values()))  # even the best single (fixed) cost model loses this much somewhere
say(f'=> best single fixed cost model still loses up to {max_regret:.1f} pts in some regime')


# which action types dominate each regime's fixed-action frontier
for m in ('idle', 'sat'):
    cand = sorted(range(A), key=lambda a: CH[m][:, :, a].mean())
    pareto, best_acc = [], -1
    for a in cand:
        acc = PH[:, :, a].mean()
        if acc > best_acc + 1e-12:
            pareto.append(a)
            best_acc = acc
    say(f"- fixed Pareto actions under {m} cost: " + ', '.join(aname(a) for a in pareto))

# ------------------------------------------------------------------ test 1: load sweep
say('\n## Test 1 - does GPU cost diverge from latency as load rises? (equal-token actions)')
D_val = None
if idle and cells:
    lat = {}
    for r in idle:
        lat.setdefault((r['N'], r['T']), []).append(r['lat'])
    lat = {k: float(np.mean(v)) for k, v in lat.items()}
    rps = {}
    for r in cells:
        rps[(r['N'], r['T'])] = max(rps.get((r['N'], r['T']), 0.0), r['rps'])
    say('| action (N x T) | tokens | idle latency (s) | max tok/s | GPU-s per request at saturation |')
    say('|---|---|---|---|---|')
    for (N, T) in sorted(lat, key=lambda x: (x[0] * x[1], x[0])):
        if (N, T) in tps_best:
            say(f'| {N} x {T} | {N * T} | {lat[(N, T)]:.1f} | {tps_best[(N, T)]:.0f} | {1 / rps[(N, T)]:.2f} |')
    groups = {}
    for (N, T) in lat:
        if (N, T) in rps:
            groups.setdefault(N * T, []).append((N, T))
    Dlist = []
    for tt in sorted(groups, reverse=True):
        mem = sorted(groups[tt])
        if len(mem) < 2:
            continue
        lo_, hi_ = mem[0], mem[-1]
        r_idle = lat[hi_] / lat[lo_]
        r_sat = rps[lo_] / rps[hi_]
        Dlist.append(r_sat / r_idle)
        say(f'- {tt} tokens: wide {hi_} vs deep {lo_}: idle-latency ratio {r_idle:.2f}, saturated-cost ratio {r_sat:.2f}, '
            f'divergence D = {r_sat / r_idle:.1f}')
    D_val = max(Dlist) if Dlist else None
else:
    say('(no load-sweep data yet)')

# ------------------------------------------------------------------ test 3: probe
say('\n## Test 3 - can a prefill hidden state predict success?')
probe = {}
hid = RUN / 'probe' / 'hidden.npz'
if hid.exists():
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    H = np.load(hid, allow_pickle=True)
    id_row = {str(x): j for j, x in enumerate(H['ids'])}
    idx = np.array([i for i, r in enumerate(recs) if r['id'] in id_row])
    rows = np.array([id_row[recs[i]['id']] for i in idx])
    Xs = H['X'][rows].astype(np.float32)
    layers = list(H['layers'])
    plen = H['plen'][rows].astype(float)
    levels = np.array([recs[i]['level'] for i in idx])
    Y = PF[idx]
    nfold = min(5, len(idx))
    folds = list(KFold(nfold, shuffle=True, random_state=CFG['seed']).split(np.arange(len(idx))))

    def oof(Xf, alphas=np.logspace(0, 5, 11)):
        out = np.zeros_like(Y)
        for tr, te in folds:
            sc = StandardScaler().fit(Xf[tr])
            mdl = RidgeCV(alphas=alphas).fit(sc.transform(Xf[tr]), Y[tr])
            out[te] = np.clip(mdl.predict(sc.transform(Xf[te])), 0, 1)
        return out

    mid_bi = min((bi for bi in range(1, nb)), key=lambda bi: abs(BUD[bi] - 512)) if nb > 1 else 0
    tgt = [(ACT.index((1, mid_bi)), mid_bi), (ACT.index((1, nb - 1)), nb - 1)]

    def auroc(pred):
        res = []
        for a, bi in tgt:
            fa = []
            for tr, te in folds:
                y = corr[idx][te][:, :, bi].ravel()
                sc_ = np.repeat(pred[te, a], K)
                try:
                    fa.append(float(roc_auc_score(y, sc_)))
                except Exception:
                    pass
            res.append(float(np.mean(fa)) if fa else float('nan'))
        return float(np.nanmean(res)), res

    base = {
        'prompt length': oof(np.log(plen)[:, None], np.logspace(-2, 2, 9)),
        'MATH level (metadata)': oof(np.eye(6)[levels][:, 1:], np.logspace(-2, 2, 9)),
    }
    say(f'AUROC of predicted single-sample success (mean of B={BUD[mid_bi]} and B={BUD[-1]}; computed per fold, {nfold}-fold CV, {len(idx)} prompts):')
    for nme, pr in base.items():
        say(f'- baseline {nme}: {auroc(pr)[0]:.3f}')
    best_layer, best_auc, best_pred = None, -1, None
    for li, L_ in enumerate(layers):
        pr = oof(Xs[:, li, :])
        au = auroc(pr)[0]
        say(f'- probe layer {int(L_)}: {au:.3f}')
        if au > best_auc:
            best_layer, best_auc, best_pred = int(L_), au, pr
    probe = dict(layer=best_layer, auroc=best_auc, len_auroc=auroc(base['prompt length'])[0],
                 level_auroc=auroc(base['MATH level (metadata)'])[0])
    say(f'best probe layer {best_layer}: AUROC {best_auc:.3f} (length baseline {probe["len_auroc"]:.3f})')
    if 'prefill_ms' in H.files:
        say(f"prefill forward pass (HF, with all hidden states): median {np.median(H['prefill_ms']):.0f} ms per prompt")

    # router built on the probe's predictions
    def route_pred(pred, m):
        cbar = CF[m][idx].mean(0)
        pts, ar = [], np.arange(len(idx))
        for lam in LAMS:
            sel = np.argmax(pred - lam * cbar[None, :], axis=1)
            pts.append((float(CF[m][idx][ar, sel].mean()), float(PF[idx][ar, sel].mean())))
        return envelope(pts)

    say('\nrouter driven by predictions (gain over best fixed action at equal cost, accuracy points):')
    probe_gain = {}
    for m in COSTS:
        env_fix, _ = fixed_env(PF[idx], CF[m][idx])
        row = []
        for nme, pr in ([('probe', best_pred)] if best_pred is not None else []) + list(base.items()):
            gmax, gmean = gain_over([tuple(x) for x in route_pred(pr, m)], env_fix)
            row.append(f'{nme}: max {gmax:.1f}')
            if nme == 'probe':
                probe_gain[m] = gmax
        say(f'- cost={m}: ' + ' | '.join(row) + f" | cross-fit oracle: max {ORACLE_GAIN[m][0]:.1f}")
    probe['gain'] = probe_gain
else:
    say('(no probe features yet)')

# ------------------------------------------------------------------ figures
try:
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.4))
    for ax, m in zip(axs, COSTS):
        fp = np.array(FIX_PTS[m])
        ax.scatter(fp[:, 0], fp[:, 1], c='lightgray', s=18, label='fixed actions')
        ax.plot(ENV[m][:, 0], ENV[m][:, 1], 'k-', label='best fixed envelope')
        o = envelope(ORACLE_PTS[m])
        ax.plot(o[:, 0], o[:, 1], 'r-', label='oracle router (cross-fit)')
        ax.set_title(f'cost = {m} (normalised)')
        ax.set_xlabel('mean cost / request (x mean)')
        ax.set_ylabel('accuracy')
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(OUT / 'frontiers.png', dpi=130)
    plt.close()
    if cells:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        for key in sorted(set((r['N'], r['T']) for r in cells)):
            rr = sorted([r for r in cells if (r['N'], r['T']) == key], key=lambda r: r['C'] * r['N'])
            ax.plot([r['C'] * r['N'] for r in rr], [r['tps'] for r in rr], marker='o', label=f'{key[0]}x{key[1]}')
        ax.set_xscale('log')
        ax.set_xlabel('concurrent sequences')
        ax.set_ylabel('server tokens/s')
        ax.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(OUT / 'throughput_vs_load.png', dpi=130)
        plt.close()
except Exception as e:
    say(f'(figure error: {e})')

# ------------------------------------------------------------------ verdict
def mark(ok):
    return 'PASS' if ok else ('MISSING' if ok is None else 'FAIL')


t1 = None if D_val is None else bool(D_val >= TH['D'])
t2 = None if math.isnan(max_regret) else bool(max_regret >= TH['regret'])
g_tok = ORACLE_GAIN['tok'][0]
t3 = None if math.isnan(g_tok) else bool(g_tok >= TH['gain'])
t4 = None if not probe else bool(probe['auroc'] - probe['len_auroc'] >= TH['auroc'])
say('\n## VERDICT')
say('| check | value | threshold | result |')
say('|---|---|---|---|')
say(f"| 1. wide-vs-deep cost diverges between idle and saturated | D = {D_val if D_val is None else round(D_val, 1)} | >= {TH['D']} | {mark(t1)} |")
say(f"| 2. no fixed cost model is right at every load | best fixed model's worst regret {max_regret:.1f} pts | >= {TH['regret']} pts | {mark(t2)} |")
say(f"| 3. router headroom over best fixed (tokens) | max gain {g_tok:.1f} pts | >= {TH['gain']} pts | {mark(t3)} |")
say(f"| 4. probe beats length baseline | " + (f"{probe['auroc']:.3f} vs {probe['len_auroc']:.3f}" if probe else 'n/a') + f" | +{TH['auroc']} AUROC | {mark(t4)} |")
if t3 is False:
    say('\n**NO-GO**: even an oracle router barely beats the best fixed strategy here. Rethink the project (or test harder prompts / a bigger model before giving up).')
elif t3 and t1 and t2:
    say('\n**GO (full LATTE)**: headroom exists and ignoring load costs accuracy. Keep the load-aware cost model as the core contribution.')
elif t3:
    say('\n**GO (plain scoped router)**: headroom exists, but the load-aware angle is not (yet) supported. Submit the scoped Proposal 1 with accuracy per compute as the headline; keep load-awareness as an exploratory add-on.')
else:
    say('\nNot enough completed stages for a verdict - see MISSING rows.')
say('\nCaveats: one model and one benchmark; small sample (see CIs by re-running with more prompts); the cross-fit oracle is noisy and can even sit below a probe-driven router; load costs come from a synthetic closed-loop test, not real traffic. Thresholds were fixed before the run.')
json.dump(dict(D=D_val, max_regret=max_regret, oracle_gain=ORACLE_GAIN, probe=probe, checks=dict(t1=t1, t2=t2, t3=t3, t4=t4),
               itl=[ITL_a, ITL_b, ITL_c]), open(OUT / 'summary.json', 'w'), default=float)
finish()
