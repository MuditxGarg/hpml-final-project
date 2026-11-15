import json
import types

import pytest

import load_policy as lp
import matched_real_answers as m
from sustained_calibration import KV, RUNNING, WAITING
from test_matched_real_answers import questions
from test_real_answer_validation import Tokenizer
from test_sustained_calibration import fake_server  # noqa: F401  (pytest fixture)

LADDER = dict(kind='threshold', name='kv-ladder', fallback='1:0',
              rules=[dict(action='1:2048', kv_below=.5, waiting_at_most=0), dict(action='4:0', kv_below=.8)])


def signal(kv, waiting=0, age=.5):
    return dict(kv=kv, waiting=waiting, running=10, age_s=age)


def test_threshold_takes_the_first_rule_that_holds_and_falls_back_without_a_signal():
    policy = lp.Threshold(LADDER)
    assert policy.choose(signal(.3), 1, 'q') == ((1, 2048), 'rule 0')
    assert policy.choose(signal(.3, waiting=2), 1, 'q') == ((4, 0), 'rule 1')  # a queue rules out depth
    assert policy.choose(signal(.5), 1, 'q') == ((4, 0), 'rule 1')  # limits are strict
    assert policy.choose(signal(.9), 1, 'q') == ((1, 0), 'fallback')
    assert policy.choose(None, 1, 'q') == ((1, 0), 'no fresh signal')
    assert policy.choose(signal(.1, age=9), 1, 'q') == ((1, 0), 'no fresh signal')
    with pytest.raises(ValueError):
        lp.Threshold(dict(LADDER, rules=[dict(action='1:2048', gpu_below=1)]))
    with pytest.raises(ValueError):
        lp.Threshold(dict(LADDER, rules=[dict(action='1:2048')]))


def test_signal_is_the_newest_successful_sample():
    samples = [dict(at_s=1., metrics={KV[0]: .2, WAITING: 0., RUNNING: 3.}), dict(at_s=2., metrics={KV[1]: .6, WAITING: 4., RUNNING: 9.}),
               dict(at_s=3., error='timeout')]
    assert lp.read_signal(samples, 3.5) == dict(kv=.6, waiting=4., running=9., age_s=1.5)
    assert lp.read_signal(samples[2:], 3.5) is None


def test_mixture_is_load_blind_reproducible_and_close_to_its_shares():
    mix = lp.Mixture(dict(kind='mixture', name='mix', shares={'1:0': .25, '1:2048': .75}))
    picks = [mix.choose(signal(i % 2), 7, f'q{i}')[0] for i in range(2000)]
    assert picks == [mix.choose(None, 7, f'q{i}')[0] for i in range(2000)]
    assert picks != [mix.choose(None, 8, f'q{i}')[0] for i in range(2000)]
    assert .72 < picks.count((1, 2048)) / 2000 < .78 and set(picks) == {(1, 0), (1, 2048)}
    with pytest.raises(ValueError):
        lp.Mixture(dict(name='bad', shares={'1:0': .5}))


def test_policy_names_must_be_distinct(tmp_path):
    path = tmp_path / 'p.json'
    path.write_text(json.dumps([LADDER, LADDER]))
    with pytest.raises(ValueError):
        lp.load_policies(path)


def test_policy_cells_run_and_score_against_fake_server(fake_server, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(m, 'load_tokenizer', lambda: Tokenizer())
    cohort = questions()
    policies = [dict(kind='threshold', name='always-deep', fallback='1:0', rules=[dict(action='2:400', kv_below=2)]),
                dict(kind='threshold', name='never', fallback='1:0', rules=[dict(action='2:400', kv_below=-1)]),
                dict(kind='mixture', name='mix', shares={'1:0': .5, '2:400': .5})]
    files = dict(plan=dict(questions=cohort), prompts=cohort + [dict(id='b0', thinking='bg', nothinking='bg')],
                 lengths={'200': [200]}, hardware={'device': 'fake CPU HTTP server'}, policies=policies,
                 calibration=dict(status='complete', reference_budget=200, latency_targets_s=[.4, 8.],
                                  light_rate_rps=3., contended_rate_rps=10.),
                 gold={q['id']: '1' for q in cohort})
    for name, value in files.items():
        (tmp_path / f'{name}.json').write_text(json.dumps(value))
    out = tmp_path / 'run'
    argv = ['run', '--url', fake_server, '--model', 'fake', '--out', str(out), '--no-gpu', '--seeds', '5',
            '--actions', '1:0', '--rates', 'mid=6', '--target-fraction', '.25', '--warmup', '.5', '--timeout', '30',
            '--idle-timeout', '30', '--idle-stable', '.1', '--metrics-interval', '.05',
            '--hardware-json', str(tmp_path / 'hardware.json')]
    for name in ('plan', 'calibration', 'lengths', 'prompts', 'policies'):
        argv += ['--' + name, str(tmp_path / f'{name}.json')]
    m.main(argv)

    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['status'] == 'complete' and manifest['policies'] == policies
    order = json.loads((out / 'seed-5-mid' / 'order.json').read_text())
    assert sorted(map(str, order)) == sorted(map(str, [[0, 0], [1, 0], 'always-deep', 'never', 'mix']))
    cells = json.loads((out / 'seed-5-mid' / 'cells.json').read_text())
    deep = next(c for c in cells if c['policy'] == 'always-deep')
    assert deep['n'] is None and deep['cell'].endswith('P-always-deep')
    rows = [json.loads(line) for line in (out / 'seed-5-mid' / f'{deep["cell"]}.jsonl').read_text().splitlines()]
    jobs = [r for r in rows if r['stream'] == 'target']
    assert len(jobs) == 5 and all((r['n'], r['budget'], r['policy_reason']) == (2, 400, 'rule 0') for r in jobs)
    assert all(0 <= r['policy_signal']['kv'] <= 1 and r['policy_signal']['age_s'] < 1 for r in jobs)
    assert all('policy' not in r for r in rows if r['stream'] == 'background')

    m.score(types.SimpleNamespace(out=str(out), gold=str(tmp_path / 'gold.json'), draws=50), verifier=lambda gold, pred: gold == pred)
    summary = json.loads((out / 'summary.json').read_text())
    by_name = {a['policy'] or (a['n'], a['budget']): a for a in summary['actions']}
    assert list(by_name) == [(1, 0), 'always-deep', 'mix', 'never']  # fixed actions first, then policies by name
    assert by_name['always-deep']['chosen'] == {'2:400': 5} and by_name['never']['chosen'] == {'1:0': 5}
    assert by_name['never']['reasons'] == {'fallback': 5} and sum(by_name['mix']['chosen'].values()) == 5
    assert all(a['jobs'] == 5 and a['accuracy']['value'] == 1 for a in summary['actions'])
    assert by_name['always-deep']['target_gpu_s'] > by_name['never']['target_gpu_s']
    names = {(str(d['action']), str(d['minus'])) for d in summary['paired_differences']}
    assert ('[1, 0]', 'always-deep') in names and ('always-deep', 'never') in names and len(names) == 6
    assert 'policies' in summary['caveat']
