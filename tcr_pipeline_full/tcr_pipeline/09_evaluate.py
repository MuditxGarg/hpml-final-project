"""CPU-only evaluation. Run --help; original labels and reports are never modified."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import sklearn
from sklearn.model_selection import KFold

from evaluation import evaluate_fold, fit_fold, load_cost_curve, paired_bootstrap, empirical_oracle
from evaluation_data import load_data


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=Path(__file__).parent / 'tcr_results/main')
    p.add_argument('--output', type=Path, required=True, help='New directory; refuses overwriting a prior analysis')
    p.add_argument('--hidden', type=Path)
    p.add_argument('--require-hidden', action='store_true')
    p.add_argument('--widths', type=int, nargs='+', default=[1, 2, 4])
    p.add_argument('--budgets', type=int, nargs='+', default=[0, 1024, 2048])
    p.add_argument('--b0-cap', type=int, choices=[1024, 2048, 4096], default=2048)
    p.add_argument('--loads', type=float, nargs='+', default=[.5, .8, 1., 1.3])
    p.add_argument('--slos', type=float, nargs='+', help='Defaults to independently measured calibration targets')
    p.add_argument('--reference-load', type=float, default=.5)
    p.add_argument('--target-fraction', type=float, default=.05, help='Must match source sweep configuration')
    p.add_argument('--outer-folds', type=int, default=5)
    p.add_argument('--inner-folds', type=int, default=3)
    p.add_argument('--seed', type=int, default=20261112)
    p.add_argument('--vote-draws', type=int, default=200)
    return p


def aggregate(folds, ids, cost, seed):
    keys = sorted({row['key'] for f in folds for row in f['evaluation']})
    summary, outcomes = [], {}
    for key in keys:
        rows = [next(row for row in f['evaluation'] if row['key'] == key) for f in folds]
        first = rows[0]
        s = {k: first[k] for k in ('key', 'policy', 'load', 'slo')}
        s['fold_statuses'] = [row['status'] for row in rows]
        s['fold_violations'] = s['fold_statuses'].count('predicted_slo_violation')
        s['fold_out_of_domain'] = s['fold_statuses'].count('out_of_calibration_domain')
        s['fold_unavailable'] = s['fold_statuses'].count('no_feasible_validation_point')
        if any(row['question_accuracy'] is None for row in rows):
            s.update(status='unavailable_in_some_folds', accuracy=None, mean_tokens=None, predicted_p95=None)
        else:
            per_id = {}
            for f, row in zip(folds, rows):
                per_id.update(zip(f['test_ids'], zip(row['question_accuracy'], row['question_tokens'])))
            acc = np.array([per_id[q][0] for q in ids])
            tok = np.array([per_id[q][1] for q in ids])
            latency = cost.predict(s['load'], tok.mean())
            status = ('out_of_calibration_domain' if latency is None else
                      'predicted_feasible' if latency <= s['slo'] else 'predicted_slo_violation')
            if status == 'predicted_feasible' and (s['fold_violations'] or s['fold_out_of_domain']):
                status = 'pooled_feasible_with_fold_failures'
            s.update(status=status, accuracy=float(acc.mean()), mean_tokens=float(tok.mean()), predicted_p95=latency)
            outcomes[key] = acc
        summary.append(s)
    comparisons = []
    for s in summary:
        if s['policy'].endswith('-load-aware'):
            baselines = [s['policy'].replace('-load-aware', '-fixed-cost'), 'fixed', 'fixed-mix']
            if s['policy'] == 'probe-load-aware':
                baselines += ['level-load-aware', 'length-load-aware']
            for baseline in baselines:
                base_key = s['key'].rsplit('/', 1)[0] + '/' + baseline
                if s['key'] in outcomes and base_key in outcomes:
                    comparisons.append(dict(load=s['load'], slo=s['slo'], policy=s['policy'], baseline=baseline,
                                            **paired_bootstrap(outcomes[s['key']], outcomes[base_key], seed=seed)))
    return summary, comparisons


def report_text(result):
    lines = ['# Nested evaluation: conditional model-based estimates', '',
             f"Questions: {result['questions']}; outer folds: {result['settings']['outer_folds']}; "
             f"inner folds: {result['settings']['inner_folds']}; B=0 cap: {result['settings']['b0_cap']}.",
             f"Features: {result['provenance']['feature_status']}.", '',
             'Policies, layers and lambda are selected without outer-test outcomes. Fixed-cost policies freeze the '
             'lambda selected at the reference load; load-aware policies select it separately for each background load. '
             'Both use the same training-estimated token costs. This isolates load-dependent operating-point selection, '
             'not a learned nonlinear per-action GPU cost.', '',
             '**These are offline expected accuracies and predicted background latencies, not live routing gains or '
             'correct answers per GPU-second.** Mixture behavior and corrected-cap request lengths need live validation. '
             'The empirical trace pool supplies labels, not ground-truth future success probabilities.', '',
             'Pooled p95 applies the curve to all outer-fold mean tokens. Fold failures remain visible: '
             'V = violation, O = outside calibration domain, U = unavailable.', '',
             '| SLO | background load | policy | accuracy | mean tokens | pooled predicted p95 | status | folds V/O/U |',
             '|---|---|---|---|---|---|---|---|']
    def fmt(v):
        return '—' if v is None else f'{v:.3f}'
    for s in result['summary']:
        lines.append(f"| {s['slo']} | {s['load']} | {s['policy']} | {fmt(s['accuracy'])} | "
                     f"{fmt(s['mean_tokens'])} | {fmt(s['predicted_p95'])} | {s['status']} | "
                     f"{s['fold_violations']}/{s['fold_out_of_domain']}/{s['fold_unavailable']} |")
    lines += ['', '## Paired question bootstrap: prespecified baseline comparisons', '',
              'Conditional on fitted folds, finite trace labels and the fitted cost curve. Does not cover model '
              'retraining, trace-generation or cost-calibration uncertainty. A positive accuracy difference is not '
              'an SLO-qualified win when either policy violates the constraint.', '',
              'Each baseline is reported separately; none is chosen as best using test outcomes. Intervals are '
              'pointwise, not adjusted for multiple comparisons.', '',
              '| SLO | load | policy | baseline | delta | 95% interval |', '|---|---|---|---|---|---|']
    for c in result['comparisons']:
        lines.append(f"| {c['slo']} | {c['load']} | {c['policy']} | {c['baseline']} | {c['delta']:+.4f} | [{c['low']:+.4f}, {c['high']:+.4f}] |")
    lines += ['', '## Calibration limits', '', *['- '+x for x in result['cost_provenance']['limitations']],
              f"- {len(result['cost_provenance']['excluded_cells'])} censored/error cells excluded; "
              f"calibrated offered-load range {result['cost_provenance']['observed_load_domain']}.",
              '- Per-fold choices/status and file hashes are in results.json. No infeasible policy is silently replaced.',
              '- Widths beyond N=4, if explicitly requested, are extrapolations in action shape even within token-load range.',
              '- Existing full-dataset explorations remain prior knowledge; nested evaluation removes computational '
              'selection leakage but does not make this a previously untouched benchmark.']
    lines += ['', '## Empirical hindsight reference (not a deployable policy)', '',
              'Uses test outcomes to choose actions and lambda on a finite grid. Not a certified upper bound; '
              'excluded from policy-selection and paired baseline comparisons.', '',
              '| SLO | load | accuracy | status |', '|---|---|---|---|']
    for r in result['hindsight']:
        lines.append(f"| {r['slo']} | {r['load']} | {fmt(r['accuracy'])} | {r['status']} |")
    return '\n'.join(lines) + '\n'


def main(argv=None):
    args = parser().parse_args(argv)
    if args.output.exists():
        raise SystemExit(f'Output already exists; choose a new directory: {args.output}')
    data = load_data(args.run, args.widths, args.budgets, args.b0_cap, args.hidden, args.seed, args.vote_draws)
    if args.require_hidden and data.hidden is None:
        raise SystemExit('Full-dataset hidden.npz required. Supply --hidden; pilot features cannot substitute.')
    cost, cost_meta = load_cost_curve(args.run, args.target_fraction)
    slos = args.slos or cost_meta['calibration']['slos']
    for rel in ['load/calibration.json', 'load/ext_cells.jsonl', 'lengths.json']:
        data.provenance['files_sha256'][rel] = hashlib.sha256((args.run / rel).read_bytes()).hexdigest()
    folds = []
    for i, (tr, te) in enumerate(KFold(args.outer_folds, shuffle=True, random_state=args.seed).split(data.ids)):
        print(f'Outer fold {i+1}/{args.outer_folds}: train={len(tr)} test={len(te)}', flush=True)
        fitted = fit_fold(data, tr, te, cost, args.loads, slos, args.reference_load, args.inner_folds, args.seed+i)
        rows = evaluate_fold(fitted, data, te, cost)
        # Keep per-question probabilities/actions for exact audit and paired comparisons.
        audit = {k: v.tolist() for k, v in fitted.pop('weights').items()}
        folds.append(dict(fold=i, **fitted, evaluation=rows, action_weights=audit))
    summary, comparisons = aggregate(folds, data.ids, cost, args.seed)
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).parent, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision = None
    result = dict(questions=len(data.ids), ids=data.ids, actions=data.actions,
                  settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  provenance=data.provenance, cost_provenance=cost_meta, folds=folds,
                  summary=summary, comparisons=comparisons,
                  hindsight=[empirical_oracle(data.success, data.tokens, cost, load, slo)
                             for slo in slos for load in args.loads],
                  environment=dict(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__,
                                   git_revision=revision))
    result['provenance']['evaluation_code_sha256'] = {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                                                       for name in ['09_evaluate.py','evaluation.py','evaluation_data.py']}
    args.output.mkdir(parents=True)
    (args.output / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    (args.output / 'report.md').write_text(report_text(result))
    print(f'Saved {args.output / "report.md"}; {data.provenance["feature_status"]}')


if __name__ == '__main__':
    main()
