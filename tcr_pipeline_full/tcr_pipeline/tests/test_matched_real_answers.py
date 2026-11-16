import json
import types

import pytest

import matched_real_answers as m
from test_real_answer_validation import Tokenizer
from test_sustained_calibration import fake_server  # noqa: F401  (pytest fixture)


def questions(count=5):
    return [dict(id=f'q{i}', level=i % 5 + 1, thinking=f'think {i}', nothinking=f'answer {i}') for i in range(count)]


def test_cohort_is_stratified_and_fixed_by_seed():
    rows = [dict(id=f'q{i:03d}', level=i % 5 + 1) for i in range(100)]
    cohort = m.select_questions(rows, 20, 7)
    assert sorted(r['level'] for r in cohort) == sorted([1, 2, 3, 4, 5] * 4)
    assert len({r['id'] for r in cohort}) == 20
    assert cohort == m.select_questions(list(reversed(rows)), 20, 7)
    assert cohort != m.select_questions(rows, 20, 8)
    with pytest.raises(ValueError):
        m.select_questions(rows, 12, 7)
    uneven = [dict(id=f'u{i:03d}', level=1 if i < 10 else 2 if i < 40 else 3 if i < 60 else 4 if i < 75 else 5) for i in range(100)]
    mix = m.select_questions(uneven, 21, 7, proportional=True)  # exact shares 2.1, 6.3, 4.2, 3.15, 5.25
    assert [sum(r['level'] == level for r in mix) for level in (1, 2, 3, 4, 5)] == [2, 7, 4, 3, 5] and len({r['id'] for r in mix}) == 21


def test_target_schedule_is_identical_across_background_rates():
    background = [dict(id='b', thinking='bg', nothinking='bg')]
    light = m.make_trace(questions(), background, [20, 30], 8, 3, 2., 1., 10.)
    heavy = m.make_trace(questions(), background, [20, 30], 8, 3, 9., 1., 10.)
    targets = lambda trace: [e for e in trace['events'] if e['stream'] == 'target']
    assert targets(light) == targets(heavy)
    assert sorted(e['question_id'] for e in targets(light)) == [f'q{i}' for i in range(5)]
    assert [e['at_s'] for e in targets(light)] == [2., 4., 6., 8., 10.]
    assert len(heavy['events']) > len(light['events'])
    assert [e['question_id'] for e in targets(light)] != [e['question_id'] for e in targets(m.make_trace(questions(), background, [20], 8, 4, 2., 1., 10.))]
    assert light == m.make_trace(questions(), background, [20, 30], 8, 3, 2., 1., 10.)


def test_bootstrap_resamples_questions_and_ratio_keeps_failures():
    record = lambda ok, gpu: dict(job=1, correct_loose=ok, gpu_s=gpu)
    by_question = {'a': [record(1, 2.), record(1, 2.)], 'b': [record(0, 4.), record(0, 4.)]}
    groups = list(by_question.values())
    assert m.ratio(groups, 'correct_loose', 'job') == .5
    assert m.ratio(groups, 'correct_loose', 'gpu_s') == pytest.approx(2 / 12)
    assert m.ratio([], 'correct_loose', 'job') is None
    low, high = m.bootstrap(by_question, lambda g: m.ratio(g, 'correct_loose', 'job'), 400, 'seed')
    assert (low, high) == (0, 1)  # with two questions, resamples range from all-wrong to all-right
    assert m.bootstrap(by_question, lambda g: m.ratio(g, 'correct_loose', 'job'), 400, 'seed') == [low, high]


def test_run_and_score_against_fake_server(fake_server, tmp_path, monkeypatch):  # noqa: F811
    # Fake vLLM answers every real request with \boxed{1}; one gold answer differs, so 4 of 5 are correct.
    monkeypatch.setattr(m, 'load_tokenizer', lambda: Tokenizer())
    cohort = questions()
    prompts = cohort + [dict(id=f'b{i}', thinking='bg', nothinking='bg') for i in range(3)]
    files = dict(plan=dict(questions=cohort), prompts=prompts, lengths={'200': [200]}, hardware={'device': 'fake CPU HTTP server'},
                 calibration=dict(status='complete', reference_budget=200, latency_targets_s=[.4, 8.],
                                  light_rate_rps=3., contended_rate_rps=10.),
                 gold={q['id']: '2' if q['id'] == 'q0' else '1' for q in cohort})
    for name, value in files.items():
        (tmp_path / f'{name}.json').write_text(json.dumps(value))
    out = tmp_path / 'run'
    argv = ['run', '--url', fake_server, '--model', 'fake', '--out', str(out), '--no-gpu', '--seeds', '5',
            '--actions', '1:0', '2:400', '--target-fraction', '.5', '--warmup', '.3', '--timeout', '30',
            '--idle-timeout', '30', '--idle-stable', '.1', '--metrics-interval', '.05',
            '--hardware-json', str(tmp_path / 'hardware.json')]
    for name in ('plan', 'calibration', 'lengths', 'prompts'):
        argv += ['--' + name, str(tmp_path / f'{name}.json')]
    m.main(argv)

    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['status'] == 'complete' and sorted(manifest['completed']) == ['seed-5-contended', 'seed-5-light']
    assert manifest['window_s'] == 1. and manifest['background_prompts'] == 3  # target questions never serve as background
    for block in manifest['completed']:
        cells = json.loads((out / block / 'cells.json').read_text())
        order = json.loads((out / block / 'order.json').read_text())
        assert [[c['n'], c['budget']] for c in cells] == order and sorted(map(tuple, order)) == [(0, 0), (1, 0), (2, 400)]
        assert len({c['streams']['background']['scheduled'] for c in cells}) == 1
        assert all(c['streams']['background']['completion_rate'] == 1 and c['generation_tokens'] > 0 for c in cells)
    rows = [json.loads(line) for line in (out / 'seed-5-light' / 'cell-{}-N2-B400.jsonl'.format(
        json.loads((out / 'seed-5-light' / 'order.json').read_text()).index([2, 400]))).read_text().splitlines()]
    job = next(r for r in rows if r['stream'] == 'target')
    assert job['pred'] == '1' and len(job['parts']) == 2 and len(job['parts'][0]['stages']) == 2
    assert job['parts'][0]['thinking_stopped_naturally'] and job['completion_tokens'] == 2 * (300 + 4)
    assert 'prompt tokens' in job['parts'][0]['stages'][0]['request']['prompt']

    m.score(types.SimpleNamespace(out=str(out), gold=str(tmp_path / 'gold.json'), draws=200), verifier=lambda gold, pred: gold == pred)
    summary = json.loads((out / 'summary.json').read_text())
    scored = [json.loads(line) for line in (out / 'scored.jsonl').read_text().splitlines()]
    assert len(scored) == 2 * 2 * 5 and len(summary['cells']) == 6 and len(summary['actions']) == 4
    for action in summary['actions']:
        assert action['jobs'] == action['questions'] == 5 and action['statuses']['completed'] == 5
        assert action['accuracy']['value'] == .8 and action['correct_within_loose']['value'] == .8
        assert 0 <= action['accuracy']['ci95'][0] <= .8 <= action['accuracy']['ci95'][1] <= 1
        assert action['correct_within_loose_per_gpu_s']['value'] > 0 and action['target_gpu_s'] > 0
    by_key = {(a['rate'], a['n'], a['budget']): a for a in summary['actions']}
    # Two thinking samples cost more attributed GPU time than one short answer.
    assert by_key[('light', 2, 400)]['target_gpu_s'] > by_key[('light', 1, 0)]['target_gpu_s']
    for cell in summary['cells']:
        assert cell['background_jobs'] > 0 and cell['goodput_loose_per_gpu_s'] > 0
        assert cell['target_jobs'] == (0 if cell['n'] == 0 else 5)
        assert cell['target_correct'] == (0 if cell['n'] == 0 else 4)
    gaps = [d for d in summary['paired_differences'] if d['metric'] == 'correct_within_loose']
    assert len(gaps) == 2 and all(d['value'] == 0 and d['questions'] == 5 for d in gaps)
    assert sum(d['metric'] == 'accuracy' for d in summary['paired_differences']) == 2
    assert all(c['gpu_power_mean_w'] is None for c in summary['cells'])  # the run used --no-gpu
    with pytest.raises(FileExistsError):
        m.main(argv)


def test_rate_override_keeps_the_target_schedule(fake_server, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(m, 'load_tokenizer', lambda: Tokenizer())
    cohort = questions()
    files = dict(plan=dict(questions=cohort), prompts=cohort + [dict(id='b0', thinking='bg', nothinking='bg')],
                 lengths={'200': [200]}, hardware={'device': 'fake CPU HTTP server'},
                 calibration=dict(status='complete', reference_budget=200, latency_targets_s=[.4, 8.],
                                  light_rate_rps=3., contended_rate_rps=10.),
                 gold={q['id']: '1' for q in cohort})
    for name, value in files.items():
        (tmp_path / f'{name}.json').write_text(json.dumps(value))
    out = tmp_path / 'run'
    argv = ['run', '--url', fake_server, '--model', 'fake', '--out', str(out), '--no-gpu', '--seeds', '5',
            '--actions', '1:0', '--rates', 'mid=6', '--target-fraction', '.5', '--warmup', '.3', '--timeout', '30',
            '--idle-timeout', '30', '--idle-stable', '.1', '--metrics-interval', '.05',
            '--hardware-json', str(tmp_path / 'hardware.json')]
    for name in ('plan', 'calibration', 'lengths', 'prompts'):
        argv += ['--' + name, str(tmp_path / f'{name}.json')]
    m.main(argv)
    manifest = json.loads((out / 'manifest.json').read_text())
    # The window still follows the calibrated contended rate (5 questions at 0.5 * 10 jobs/s).
    assert manifest['completed'] == ['seed-5-mid'] and manifest['rates_rps'] == {'mid': 6.} and manifest['window_s'] == 1.
    m.score(types.SimpleNamespace(out=str(out), gold=str(tmp_path / 'gold.json'), draws=50), verifier=lambda gold, pred: gold == pred)
    summary = json.loads((out / 'summary.json').read_text())
    assert [a['rate'] for a in summary['actions']] == ['mid'] and summary['actions'][0]['accuracy']['value'] == 1
    with pytest.raises(SystemExit, match='distinct'):
        m.main(argv[:argv.index('--rates') + 1] + ['a=1', 'a=2'] + argv[argv.index('--rates') + 2:])


def test_run_refuses_an_incomplete_calibration(tmp_path):
    (tmp_path / 'calibration.json').write_text(json.dumps(dict(status='no contended rung')))
    argv = ['run', '--out', str(tmp_path / 'run'), '--calibration', str(tmp_path / 'calibration.json')]
    for name in ('plan', 'lengths', 'prompts', 'hardware-json'):
        argv += ['--' + name, str(tmp_path / 'missing.json')]
    with pytest.raises(SystemExit, match='not complete'):
        m.main(argv)
    assert not (tmp_path / 'run').exists()
