import copy
import importlib

import numpy as np

from evaluation import CostCurve, fit_fold, evaluate_fold, paired_bootstrap, select_point, empirical_oracle
from evaluation_data import EvaluationData, vote_probability


def fixture():
    # Two difficulty bins prefer opposite actions. Held-out values are never needed to fit.
    levels = np.array([1, 2] * 12)
    p = np.array([[.9, .5] if x == 1 else [.1, .9] for x in levels])
    tokens = np.tile([100., 500.], (24, 1))
    return EvaluationData([str(i) for i in range(24)], [(1, 0), (1, 2048)], p,
                          tokens, levels, np.arange(24.) + 20, None, [], {})


def curve():
    return CostCurve([.1, 1., 2.], [1., 10., 50.], reference_tokens=100., target_fraction=.1)


def test_heldout_labels_and_tokens_cannot_change_selection():
    data = fixture()
    tr, te = np.arange(16), np.arange(16, 24)
    first = fit_fold(data, tr, te, curve(), loads=[.5, 1.], slos=[12.], reference_load=.5,
                     inner_folds=2, seed=7)
    changed = copy.deepcopy(data)
    changed.success[te] = 1 - changed.success[te]
    changed.tokens[te] *= 100
    second = fit_fold(changed, tr, te, curve(), loads=[.5, 1.], slos=[12.], reference_load=.5,
                      inner_folds=2, seed=7)
    assert first['selection'] == second['selection']
    for name in first['weights']:
        np.testing.assert_array_equal(first['weights'][name], second['weights'][name])
    assert evaluate_fold(first, data, te, curve()) != evaluate_fold(second, changed, te, curve())


def test_hidden_layer_and_alpha_selected_without_outer_labels():
    data = fixture()
    data.hidden = np.stack([data.levels[:, None], np.ones((24, 1))], axis=1)
    data.layers = [4, 8]
    tr, te = np.arange(16), np.arange(16, 24)
    first = fit_fold(data, tr, te, curve(), [.5], [12.], .5, inner_folds=2, seed=7)
    changed = copy.deepcopy(data)
    changed.success[te] = 0
    second = fit_fold(changed, tr, te, curve(), [.5], [12.], .5, inner_folds=2, seed=7)
    assert first['hyperparameters'] == second['hyperparameters']
    assert first['selection'] == second['selection']
    assert any('probe' in k for k in first['weights'])


def test_no_feasible_point_is_not_replaced_with_cheapest():
    pts = [{'accuracy': .9, 'tokens': 500., 'parameter': 0.}]
    assert select_point(pts, curve(), load=1., slo=1.) is None


def test_out_of_domain_is_unknown_and_workload_includes_background():
    c = curve()
    assert c.offered_load(.5, 500.) == 1.
    assert c.predict(.5, 500.) == 10.
    assert c.predict(2., 500.) is None
    assert c.predict(0., 0.) is None


def test_fixed_cost_lambda_does_not_change_with_load():
    fitted = fit_fold(fixture(), np.arange(16), np.arange(16, 24), curve(),
                      [.5, 1.], [12.], .5, inner_folds=2, seed=7)
    choices = [v for v in fitted['selection'] if v['policy'] == 'level-fixed-cost']
    assert len(choices) == 2
    assert choices[0]['parameter'] == choices[1]['parameter']
    np.testing.assert_array_equal(fitted['weights'][choices[0]['key']], fitted['weights'][choices[1]['key']])


def test_bootstrap_resamples_paired_differences():
    result = paired_bootstrap(np.array([.2, .8, .4]), np.array([.1, .7, .3]), seed=3)
    np.testing.assert_allclose([result['delta'], result['low'], result['high']], [.1, .1, .1])


def test_vote_ignores_missing_answers_and_averages_ties():
    assert vote_probability(['a', 'b', None], [True, False, False], 3, seed=2) == .5
    assert vote_probability([None, None], [False, False], 2, seed=2) == 0.
    assert vote_probability(['a', 'b'], [True, False], 1, seed=2) == .5


def test_oracle_is_separate_hindsight_reference_not_feasible_fallback():
    d = fixture()
    result = empirical_oracle(d.success, d.tokens, curve(), 1., .1)
    assert result['status'] == 'no_feasible_hindsight_point'
    assert result['accuracy'] is None
    loose = empirical_oracle(d.success, d.tokens, curve(), .5, 50.)
    assert loose['accuracy'] == .9
    assert loose['uses_test_outcomes'] is True


def test_aggregate_does_not_hide_fold_slo_violations():
    cli = importlib.import_module('09_evaluate')
    rows = []
    for q, tokens, status in [('a', 100., 'predicted_feasible'), ('b', 1000., 'predicted_slo_violation')]:
        rows.append(dict(test_ids=[q], evaluation=[dict(key='15/0.5/fixed', policy='fixed', load=.5, slo=15.,
                         status=status, question_accuracy=[.8], question_tokens=[tokens])]))
    summary, _ = cli.aggregate(rows, ['a', 'b'], curve(), 0)
    assert summary[0]['status'] == 'pooled_feasible_with_fold_failures'
    assert summary[0]['fold_violations'] == 1
