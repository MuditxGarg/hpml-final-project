"""Outer-fold evaluation with training-only policy selection.

    Independent measured cost curve is held fixed. Results are model-based;
    uncertainty in the curve and fresh rollout sampling is NOT bootstrapped.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class CostCurve:
    x: object
    y: object
    reference_tokens: float
    target_fraction: float

    def __post_init__(self):
        self.model = IsotonicRegression(increasing=True, out_of_bounds='nan').fit(self.x, self.y)
        self.low, self.high = float(min(self.x)), float(max(self.x))

    def offered_load(self, load, tokens):
        # Same semantics as original measurement: background plus an extra target stream.
        return float(load + self.target_fraction * tokens / self.reference_tokens)

    def predict(self, load, tokens):
        x = self.offered_load(load, tokens)
        if x < self.low or x > self.high:
            return None
        return float(self.model.predict([x])[0])


def load_cost_curve(run, target_fraction=.05):
    """Exclude censored cells: their stored latency percentile isn't a full p95."""
    run = Path(run)
    cal = json.loads((run / 'load/calibration.json').read_text())
    lengths = json.loads((run / 'lengths.json').read_text())
    cells = [json.loads(line) for line in (run / 'load/ext_cells.jsonl').read_text().splitlines() if line.strip()]
    groups, excluded = {}, []
    for c in cells:
        bg = c['bg']
        if c['frac'] <= 0 or not bg.get('n'):
            continue
        if bg.get('n_cens', 0) or bg.get('n_fail', 0):
            excluded.append([c['rep'], c['frac'], c['N'], c['B']])
            continue
        groups.setdefault((c['frac'], c['N'], c['B']), []).append(bg['lat_p95'])
    if not groups:
        raise ValueError('No uncensored successful calibration cells')
    xs, ys = [], []
    for (fr, n, b), values in sorted(groups.items()):
        xs.append(fr + target_fraction * n * np.mean(lengths[str(b)]) / cal['mean_len_ref'])
        ys.append(float(np.mean(values)))
    curve = CostCurve(xs, ys, cal['mean_len_ref'], target_fraction)
    return curve, dict(excluded_cells=excluded, calibration=cal, observed_load_domain=[curve.low, curve.high],
                       calibration_label_version='original lengths.json; original B=0 cap=1024',
                       limitations=['Token-only surrogate; not validated for policy mixtures or new N values.',
                                    'Aggregate source logs cannot recover arrival drops or censored tail latencies.',
                                    'Excluding censored cells may bias high-load fits downward; real replay required.'])


def paired_bootstrap(a, b, seed=0, repetitions=2000):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if not len(d) or not np.isfinite(d).all():
        raise ValueError('Paired bootstrap requires nonempty finite matched observations')
    rng = np.random.default_rng(seed)
    means = np.mean(d[rng.integers(0, len(d), (repetitions, len(d)))], axis=1)
    return dict(delta=float(d.mean()), low=float(np.percentile(means, 2.5)),
                high=float(np.percentile(means, 97.5)), n=len(d))


def select_point(points, cost, load, slo):
    valid = [p for p in points if (v := cost.predict(load, p['tokens'])) is not None and v <= slo]
    return max(valid, key=lambda p: (p['accuracy'], -p['tokens'])) if valid else None


def empirical_oracle(p, tokens, cost, load, slo):
    """Hindsight reference over a finite lambda grid, NOT a certified upper bound."""
    candidates = [_point(_weights(p, tokens.mean(0), lam), p, tokens, float(lam))
                  for lam in np.r_[0., np.logspace(-3, 2, 51)]]
    best = select_point(candidates, cost, load, slo)
    return dict(uses_test_outcomes=True, load=load, slo=slo,
                status='hindsight_reference' if best else 'no_feasible_hindsight_point',
                accuracy=best['accuracy'] if best else None,
                tokens=best['tokens'] if best else None)


def _ridge_predictions(features, p, tr, te, alpha):
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(features[tr], p[tr])
    return np.clip(model.predict(features[te]), 0., 1.)


def _level_predictions(levels, p, tr, te):
    overall = p[tr].mean(0)
    return np.array([p[tr][levels[tr] == level].mean(0) if np.any(levels[tr] == level) else overall
                     for level in levels[te]])


def _inner_predictions(data, tr, te, inner_folds, seed):
    splits = list(KFold(inner_folds, shuffle=True, random_state=seed).split(tr))
    p = data.success
    inner = np.zeros((len(tr), p.shape[1]))
    for a, b in splits:
        inner[b] = _level_predictions(data.levels, p, tr[a], tr[b])
    predictions = {'level': (inner, _level_predictions(data.levels, p, tr, te))}
    hyper = {}
    feature_sets = {'length': [(None, data.prompt_lengths[:, None])]}
    if data.hidden is not None:
        feature_sets['probe'] = [(li, data.hidden[:, li]) for li in range(data.hidden.shape[1])]
    for name, candidates in feature_sets.items():
        best = None
        for layer, feat in candidates:
            for alpha in (100., 1000., 10000.):
                out = np.zeros_like(inner)
                for a, b in splits:
                    out[b] = _ridge_predictions(feat, p, tr[a], tr[b], alpha)
                mse = float(np.mean((out - p[tr]) ** 2))
                if best is None or mse < best[0]:
                    best = (mse, layer, alpha, feat, out)
        mse, layer, alpha, feat, out = best
        predictions[name] = (out, _ridge_predictions(feat, p, tr, te, alpha))
        hyper[name] = dict(layer_index=layer, alpha=alpha, inner_mse=mse)
    return predictions, hyper


def _weights(scores, mean_tokens, lam):
    actions = np.argmax(scores - lam * mean_tokens[None, :] / 1000., axis=1)
    return np.eye(scores.shape[1])[actions]


def _point(weights, p, tokens, parameter):
    return dict(accuracy=float(np.sum(weights * p, axis=1).mean()),
                tokens=float(np.sum(weights * tokens, axis=1).mean()), parameter=parameter)


def fit_fold(data, tr, te, cost, loads, slos, reference_load, inner_folds=3, seed=0):
    """Fit and select using tr only; te supplies features, never outcomes or token labels."""
    p, tokens = data.success[tr], data.tokens[tr]
    mean_tokens = tokens.mean(0)
    predictions, hyper = _inner_predictions(data, tr, te, inner_folds, seed)
    n_actions = p.shape[1]
    lambdas = np.r_[0., np.logspace(-3, 2, 51)]
    points = {}
    points['fixed'] = [_point(np.tile(np.eye(n_actions)[j], (len(tr), 1)), p, tokens, j)
                       for j in range(n_actions)]
    points['fixed-mix'] = []
    for a in range(n_actions):
        for b in range(a, n_actions):
            for w in (0., .25, .5, .75, 1.):
                weights = np.tile((1-w)*np.eye(n_actions)[a] + w*np.eye(n_actions)[b], (len(tr), 1))
                points['fixed-mix'].append(_point(weights, p, tokens, [a, b, w]))
    for name, (scores, _) in predictions.items():
        points[name] = [_point(_weights(scores, mean_tokens, lam), p, tokens, float(lam)) for lam in lambdas]
    selections, weights_out = [], {}
    for slo in slos:
        frozen = {name: select_point(points[name], cost, reference_load, slo) for name in predictions}
        for load in loads:
            policies = [('fixed', 'fixed'), ('fixed-mix', 'fixed-mix')]
            policies += [(name+'-load-aware', name) for name in predictions]
            policies += [(name+'-fixed-cost', name) for name in predictions]
            for policy, name in policies:
                chosen = frozen[name] if policy.endswith('-fixed-cost') else select_point(points[name], cost, load, slo)
                key = f'{slo:g}/{load:g}/{policy}'
                selection = dict(key=key, policy=policy, load=float(load), slo=float(slo),
                                 parameter=None if chosen is None else chosen['parameter'],
                                 selection_load=float(reference_load if policy.endswith('-fixed-cost') else load),
                                 train_accuracy=None if chosen is None else chosen['accuracy'],
                                 train_tokens=None if chosen is None else chosen['tokens'])
                selections.append(selection)
                if chosen is None:
                    continue
                par = chosen['parameter']
                if name == 'fixed':
                    w = np.tile(np.eye(n_actions)[par], (len(te), 1))
                elif name == 'fixed-mix':
                    a, b, mix = par
                    w = np.tile((1-mix)*np.eye(n_actions)[a] + mix*np.eye(n_actions)[b], (len(te), 1))
                else:
                    w = _weights(predictions[name][1], mean_tokens, par)
                weights_out[key] = w
    return dict(selection=selections, weights=weights_out, hyperparameters=hyper,
                train_ids=[data.ids[i] for i in tr], test_ids=[data.ids[i] for i in te],
                mean_tokens=mean_tokens.tolist())


def evaluate_fold(fitted, data, te, cost):
    rows = []
    for selected in fitted['selection']:
        row = dict(selected)
        key = selected['key']
        if key not in fitted['weights']:
            row.update(status='no_feasible_validation_point', accuracy=None, tokens=None, predicted_p95=None,
                       question_accuracy=None, question_tokens=None)
        else:
            w = fitted['weights'][key]
            acc = np.sum(w * data.success[te], axis=1)
            tokens = np.sum(w * data.tokens[te], axis=1)
            pred = cost.predict(row['load'], float(tokens.mean()))
            status = ('out_of_calibration_domain' if pred is None else
                      'predicted_feasible' if pred <= row['slo'] else 'predicted_slo_violation')
            row.update(status=status, accuracy=float(acc.mean()), tokens=float(tokens.mean()), predicted_p95=pred,
                       question_accuracy=acc.tolist(), question_tokens=tokens.tolist())
        rows.append(row)
    return rows
