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
    check = summary['power_check']  # without GPU samples every cell is reported as unverified, never as matched
    assert check['tolerance'] == .1 and check['tolerance_source'] == 'run configuration'
    assert (check['flagged'], check['unknown'], len(check['cells'])) == (0, 4, 4) and summary['warnings'] == check['warnings']
    assert len(summary['warnings']) == 1 and 'not verified' in summary['warnings'][0]
    assert all(a['gpu_power_mean_w'] is None for a in summary['actions'])
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


def test_assemble_gathers_blocks_of_several_runs_for_one_score(fake_server, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(m, 'load_tokenizer', lambda: Tokenizer())
    cohort = questions()
    files = dict(plan=dict(questions=cohort), prompts=cohort + [dict(id='b0', thinking='bg', nothinking='bg')],
                 lengths={'200': [200]}, hardware={'device': 'fake CPU HTTP server'},
                 calibration=dict(status='complete', reference_budget=200, latency_targets_s=[.4, 8.],
                                  light_rate_rps=3., contended_rate_rps=10.),
                 gold={q['id']: '1' for q in cohort})
    for name, value in files.items():
        (tmp_path / f'{name}.json').write_text(json.dumps(value))
    common = ['--url', fake_server, '--model', 'fake', '--no-gpu', '--actions', '1:0', '--rates', 'mid=6', '--target-fraction', '.5',
              '--warmup', '.3', '--timeout', '30', '--idle-timeout', '30', '--idle-stable', '.1', '--metrics-interval', '.05',
              '--hardware-json', str(tmp_path / 'hardware.json')]
    for name in ('plan', 'calibration', 'lengths', 'prompts'):
        common += ['--' + name, str(tmp_path / f'{name}.json')]
    m.main(['run', '--out', str(tmp_path / 'first'), '--seeds', '5', '6'] + common)
    # The repeated block comes from a run that also ran another rate; only the rates of the blocks taken are kept.
    two_rates = common[:common.index('mid=6') + 1] + ['other=7'] + common[common.index('mid=6') + 1:]
    m.main(['run', '--out', str(tmp_path / 'second'), '--seeds', '6'] + two_rates)
    out = tmp_path / 'assembled'
    m.main(['assemble', '--out', str(out), '--blocks', f'{tmp_path}/first/seed-5-mid', f'{tmp_path}/second/seed-6-mid'])
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['completed'] == ['seed-5-mid', 'seed-6-mid'] and manifest['status'] == 'complete' and manifest['window_s'] == 1.
    assert manifest['rates_rps'] == {'mid': 6.}
    assert [(a['run'], a['block']) for a in manifest['assembled_from']] == [(f'{tmp_path}/first', 'seed-5-mid'), (f'{tmp_path}/second', 'seed-6-mid')]
    assert (out / 'seed-6-mid' / 'cells.json').read_text() == (tmp_path / 'second' / 'seed-6-mid' / 'cells.json').read_text()
    m.score(types.SimpleNamespace(out=str(out), gold=str(tmp_path / 'gold.json'), draws=50), verifier=lambda gold, pred: gold == pred)
    summary = json.loads((out / 'summary.json').read_text())
    assert summary['actions'][0]['seeds'] == [5, 6] and summary['actions'][0]['jobs'] == 10 and len(summary['cells']) == 4
    with pytest.raises(SystemExit, match='given twice'):
        m.main(['assemble', '--out', str(tmp_path / 'x'), '--blocks', f'{tmp_path}/first/seed-5-mid', f'{tmp_path}/first/seed-5-mid'])
    with pytest.raises(SystemExit, match='not a complete block'):
        m.main(['assemble', '--out', str(tmp_path / 'y'), '--blocks', f'{tmp_path}/first/seed-7-mid'])
    other = json.loads((tmp_path / 'second' / 'manifest.json').read_text())
    (tmp_path / 'second' / 'manifest.json').write_text(json.dumps(dict(other, window_s=2.)))
    with pytest.raises(SystemExit, match='differs'):
        m.main(['assemble', '--out', str(tmp_path / 'z'), '--blocks', f'{tmp_path}/first/seed-5-mid', f'{tmp_path}/second/seed-6-mid'])
    (tmp_path / 'second' / 'manifest.json').write_text(json.dumps(dict(other, rates_rps={'mid': 5., 'other': 7.})))
    with pytest.raises(SystemExit, match='rates_rps for mid'):
        m.main(['assemble', '--out', str(tmp_path / 'z'), '--blocks', f'{tmp_path}/first/seed-5-mid', f'{tmp_path}/second/seed-6-mid'])
    assert not (tmp_path / 'z').exists()


def test_power_check_flags_cells_against_the_control_of_their_block_segment():
    cell = lambda block, n, power, segment=0, policy=None: dict(block=block, cell=f'{block}-{n}-{policy}-{segment}', n=n, budget=0, policy=policy,
                                                                gpu_power_mean_w=power, segment=segment)
    cells = [cell('a', 0, 100.), cell('a', 1, 105.), cell('a', None, 115., policy='ladder'), cell('a', 4, 89.),
             cell('a', 0, 120., segment=1), cell('a', 1, 118., segment=1),  # a resumed segment has its own control
             cell('b', 0, None), cell('b', 1, 100.), cell('c', 1, None)]
    check = m.power_check(cells, .1)
    status = {(r['block'], r['segment'], r['entry']): (r['status'], r['relative_gap']) for r in check['cells']}
    assert status[('a', 0, '1:0')] == ('ok', pytest.approx(.05)) and status[('a', 0, 'ladder')] == ('flagged', pytest.approx(.15))
    assert status[('a', 0, '4:0')][0] == 'flagged' and status[('a', 1, '1:0')] == ('ok', pytest.approx(118 / 120 - 1))
    assert status[('b', 0, '1:0')] == ('unknown', None) and status[('c', 0, '1:0')] == ('unknown', None)  # no control power, no control
    assert (check['flagged'], check['unknown'], len(check['cells'])) == (2, 2, 6)
    assert len(check['warnings']) == 2 and 'a ladder +15.0%' in check['warnings'][0] and 'b 1:0, c 1:0' in check['warnings'][1]
    assert m.power_check([cell('a', 0, 100.), cell('a', 1, 109.)], .1)['warnings'] == []


def test_compare_warns_when_entries_of_separate_runs_ran_at_different_power(tmp_path):
    def run(name, entry, power, warnings=()):
        n, budget = map(int, entry.split(':'))
        d = tmp_path / name
        d.mkdir()
        cells = [dict(rate='light', n=0, budget=0, policy=None, gpu_power_mean_w=power),
                 dict(rate='light', n=n, budget=budget, policy=None, gpu_power_mean_w=power)]
        (d / 'summary.json').write_text(json.dumps(dict(latency_targets_s=dict(tight=1., loose=2.), cells=cells, warnings=list(warnings),
                                                        power_check=dict(tolerance=.1))))
        rows = [dict(rate='light', n=n, budget=budget, policy=None, seed=1, question=f'q{i}', status='completed', job=1, correct=1,
                     correct_tight=1, correct_loose=1, tokens=10, gpu_s=1.) for i in range(3)]
        (d / 'scored.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
        return d
    out = tmp_path / 'comparison.json'
    m.main(['compare', '--runs', str(run('a', '1:0', 100.)), str(run('b', '4:0', 125., ['cell x flagged'])), '--output', str(out), '--draws', '20'])
    comparison = json.loads(out.read_text())
    assert [e['gpu_power_mean_w'] for e in comparison['entries']] == [100., 125.] and comparison['power_tolerance'] == .1
    assert len(comparison['warnings']) == 2 and '1:0 ran at 100.0 W and 4:0 at 125.0 W' in comparison['warnings'][0]
    assert comparison['warnings'][1] == 'b: cell x flagged'
    m.main(['compare', '--runs', str(run('c', '1:0', 100.)), str(run('d', '4:0', 105.)), '--output', str(tmp_path / 'ok.json'), '--draws', '20'])
    assert json.loads((tmp_path / 'ok.json').read_text())['warnings'] == []


def run_inputs(tmp_path, url):
    """Input files and the common `run` arguments of a small two-block run (seed 5 at the frozen light and contended rates)."""
    cohort = questions()
    files = dict(plan=dict(questions=cohort), prompts=cohort + [dict(id=f'b{i}', thinking='bg', nothinking='bg') for i in range(3)],
                 lengths={'200': [200]}, hardware={'device': 'fake CPU HTTP server'},
                 calibration=dict(status='complete', reference_budget=200, latency_targets_s=[.4, 8.], light_rate_rps=3., contended_rate_rps=10.),
                 gold={q['id']: '1' for q in cohort})
    for name, value in files.items():
        (tmp_path / f'{name}.json').write_text(json.dumps(value))
    argv = ['--url', url, '--model', 'fake', '--no-gpu', '--seeds', '5', '--actions', '1:0', '2:400', '--target-fraction', '.5', '--warmup', '.3',
            '--timeout', '30', '--idle-timeout', '30', '--idle-stable', '.1', '--metrics-interval', '.05', '--hardware-json', str(tmp_path / 'hardware.json')]
    for name in ('plan', 'calibration', 'lengths', 'prompts'):
        argv += ['--' + name, str(tmp_path / f'{name}.json')]
    return argv


def block_cells(out):
    manifest = json.loads((out / 'manifest.json').read_text())
    return {name: json.loads((out / name / 'cells.json').read_text()) for name in manifest['blocks'] if (out / name / 'cells.json').exists()}


@pytest.mark.parametrize('stop_at', [5, 6])  # the second or the third cell of the second block
def test_a_run_stopped_mid_cell_resumes_with_the_same_cells(fake_server, tmp_path, monkeypatch, stop_at):  # noqa: F811
    import os
    import signal
    monkeypatch.setattr(m, 'load_tokenizer', lambda: Tokenizer())
    common = run_inputs(tmp_path, fake_server)
    whole, cut = tmp_path / 'whole', tmp_path / 'cut'
    m.main(['run', '--out', str(whole)] + common)
    real, calls = m.run_cell, []

    def stopped(*args, **kwargs):  # the cell is stopped as a job limit or a spot shutdown would stop it
        calls.append(1)
        if len(calls) == stop_at:
            os.kill(os.getpid(), signal.SIGTERM)
        return real(*args, **kwargs)
    monkeypatch.setattr(m, 'run_cell', stopped)
    with pytest.raises(SystemExit):
        m.main(['run', '--out', str(cut)] + common)
    manifest = json.loads((cut / 'manifest.json').read_text())
    first, second = manifest['blocks']
    kept = stop_at - 4  # cells of the second block recorded before the stop
    assert manifest['status'] == 'interrupted' and manifest['completed'] == [first] and manifest['segments'][0]['status'] == 'interrupted'
    stopped_cell = manifest['current_cell']['cell']
    assert manifest['current_cell']['block'] == second and len(block_cells(cut)[second]) == kept
    (cut / second / f'{stopped_cell}.jsonl').write_text('{"partial": true}\n')  # a write the stop cut short
    monkeypatch.setattr(m, 'run_cell', real)
    m.main(['resume', '--out', str(cut)])

    manifest = json.loads((cut / 'manifest.json').read_text())
    assert manifest['status'] == 'complete' and manifest['completed'] == [first, second] and manifest['current_cell'] is None
    assert [(s['index'], s['kind'], s['status']) for s in manifest['segments']] == [(0, 'start', 'interrupted'), (1, 'resume', 'complete')]
    assert [(d['block'], d['cell'], d['files']) for d in manifest['discarded']] == [(second, stopped_cell, [f'{stopped_cell}.jsonl'])]
    assert (cut / second / 'discarded' / 'segment-1' / f'{stopped_cell}.jsonl').read_text() == '{"partial": true}\n'
    assert 'partial' not in (cut / second / f'{stopped_cell}.jsonl').read_text()  # rerun from the start, not continued
    original, resumed = block_cells(whole), block_cells(cut)
    assert [(c['cell'], c['segment']) for c in resumed[first]] == [(c['cell'], 0) for c in original[first]]
    names = [c['cell'] for c in original[second]]
    control_pending = any(c['control'] for c in original[second][kept:])
    # A block that continues in a new segment gets its own control there: the original one if it had not run yet, else a repeat first.
    expected = [(n, 0) for n in names[:kept]] + ([] if control_pending else [('cell-s1-control-N0-B0', 1)]) + [(n, 1) for n in names[kept:]]
    assert [(c['cell'], c['segment']) for c in resumed[second]] == expected
    assert sum(c['control'] and c['segment'] == 1 for c in resumed[second]) == 1
    for out in (whole, cut):
        m.score(types.SimpleNamespace(out=str(out), gold=str(tmp_path / 'gold.json'), draws=50), verifier=lambda gold, pred: gold == pred)
    jobs = lambda out: sorted((a['rate'], a['n'], a['budget'], a['jobs'], a['accuracy']['value']) for a in json.loads((out / 'summary.json').read_text())['actions'])
    assert jobs(cut) == jobs(whole)  # the same scored jobs: the discarded partial cell is never scored
    check = json.loads((cut / 'summary.json').read_text())['power_check']
    assert sorted(r['segment'] for r in check['cells'] if r['block'] == second) == [0] * (kept - (not control_pending)) + [1] * (2 - kept + (not control_pending))
    with pytest.raises(SystemExit, match='already complete'):
        m.main(['resume', '--out', str(cut)])


def test_resume_after_a_deadline_refuses_changed_inputs_and_new_code(fake_server, tmp_path, monkeypatch):  # noqa: F811
    import time
    monkeypatch.setattr(m, 'load_tokenizer', lambda: Tokenizer())
    out = tmp_path / 'run'
    m.main(['run', '--out', str(out), '--deadline-epoch', str(time.time() + 1)] + run_inputs(tmp_path, fake_server))
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['status'] == 'partial_deadline' and manifest['skipped'] == manifest['blocks'] and block_cells(out) == {}
    calibration = (tmp_path / 'calibration.json').read_text()
    (tmp_path / 'calibration.json').write_text(json.dumps(dict(json.loads(calibration), light_rate_rps=4.)))
    with pytest.raises(SystemExit, match='inputs changed.*calibration_sha256.*rates_rps'):
        m.main(['resume', '--out', str(out)])
    (tmp_path / 'calibration.json').write_text(calibration)
    real = m.environment
    monkeypatch.setattr(m, 'environment', lambda url: dict(real(url), git_commit='0' * 40))
    with pytest.raises(SystemExit, match='allow-new-commit'):
        m.main(['resume', '--out', str(out)])
    m.main(['resume', '--out', str(out), '--allow-new-commit'])
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['status'] == 'complete' and manifest['skipped'] == [] and manifest['segments'][-1]['git_commit'] == '0' * 40
    # Nothing had run before the deadline, so no block needs a repeated control.
    assert all(len(cells) == 3 and {c['segment'] for c in cells} == {len(manifest['segments']) - 1} for cells in block_cells(out).values())


def test_a_cell_that_ran_across_a_server_restart_is_never_recorded(fake_server, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(m, 'load_tokenizer', lambda: Tokenizer())
    starts = iter([1., 1., 2.])  # segment start, before the first cell, after it: the server restarted during the cell
    monkeypatch.setattr(m, 'server_process_start', lambda url: next(starts))
    out = tmp_path / 'run'
    with pytest.raises(RuntimeError, match='changed during'):
        m.main(['run', '--out', str(out)] + run_inputs(tmp_path, fake_server))
    manifest = json.loads((out / 'manifest.json').read_text())
    first = manifest['blocks'][0]
    assert manifest['status'] == 'failed' and block_cells(out) == {} and not list((out / first).glob('cell-*'))
    monkeypatch.setattr(m, 'server_process_start', lambda url: 3.)
    m.main(['resume', '--out', str(out)])
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['status'] == 'complete' and [(d['block'], d['files']) for d in manifest['discarded']] == [(first, [])]
    assert [s['server_process_start'] for s in manifest['segments']] == [1., 3.]
