import json
from pathlib import Path

import experiment_tracking as t
from sustained_calibration import KV, RUNNING, TOKENS, WAITING

RESULTS = Path(__file__).resolve().parents[3] / 'docs' / 'results'


class FakeRun:
    def __init__(self, kwargs):
        self.kwargs, self.logged, self.metrics, self.summary, self.finished = kwargs, [], [], {}, False

    def define_metric(self, name, **options):
        self.metrics.append((name, options))

    def log(self, row):
        self.logged.append(row)

    def finish(self):
        self.finished = True


class FakeWandb:
    def __init__(self):
        self.runs = []

    def Settings(self, **options):
        return options

    def Table(self, columns, data):
        return dict(columns=columns, data=data)

    def init(self, **kwargs):
        self.runs.append(FakeRun(kwargs))
        return self.runs[-1]


def test_flat_keeps_numbers_only():
    assert t.flat(dict(a=1, b=dict(c=2.5, d='text', e=None, f=True, g=[1]))) == {'a': 1, 'b/c': 2.5}


def test_telemetry_merges_server_and_gpu_by_second_and_counts_from_the_start():
    server = [dict(at_s=.2, metrics={RUNNING: 1., WAITING: 0., KV[0]: .1, t.PREEMPTIONS: 40., TOKENS: 1000.}),
              dict(at_s=1.1, error='timeout'),
              dict(at_s=2.3, metrics={RUNNING: 5., WAITING: 2., KV[0]: .9, t.PREEMPTIONS: 43., TOKENS: 1600.})]
    device = [dict(at_s=.4, gpu={'utilization.gpu': 97., 'temperature.gpu': 86.}), dict(at_s=1.4, error='x')]
    rows = t.telemetry(server, device)
    assert rows[0] == dict(time_s=0, running=1., waiting=0., kv_cache_usage=.1, preemptions=0., generated_tokens=0.,
                           **{'gpu/utilization_gpu': 97., 'gpu/temperature_gpu': 86.})
    assert rows[1] == dict(time_s=2, running=5., waiting=2., kv_cache_usage=.9, preemptions=3., generated_tokens=600.)


def test_committed_runs_are_described():
    calibration = list(t.calibration_runs(RESULTS / '2026-11-15-sustained-calibration-4080' / 'seed-1042'))
    assert len(calibration) == 1 and len(calibration[0]['history']) == 6 and calibration[0]['job_type'] == 'calibration'
    assert round(calibration[0]['summary']['contended_rate_rps'], 3) == 2.463 and calibration[0]['history'][0]['latency/p95_s'] > 0
    assert calibration[0]['config']['git_commit'] and calibration[0]['config']['hardware']['gpu_count'] == 1
    profile = list(t.profile_runs(RESULTS / '2026-11-15-profiling-4080'))
    assert len(profile) == 10 and len({r['id'] for r in profile}) == 10 and all(r['history'] == [] for r in profile)  # raw telemetry is not committed
    assert all('server/kv_cache/max' in r['summary'] for r in profile)


def test_matched_runs_cover_cells_and_scored_tables(tmp_path):
    cell = dict(cell='cell-0-N1-B0', block='seed-1-light', seed=1, rate_label='light', background_rate_rps=1.2, n=1, budget=0,
                control=False, cell_wall_s=10., generation_tokens=500., streams=dict(target=dict(scheduled=2)), server=dict(kv_cache=dict(max=.4)), gpu={})
    interval = lambda v: dict(value=v, ci95=[v - .1, v + .1])
    action = dict(rate='light', n=1, budget=0, policy=None, jobs=2, questions=2, statuses=dict(no_answer=0), target_gpu_s=3.,
                  **{m: interval(.5) for m in ('accuracy', 'correct_within_tight', 'correct_within_loose',
                                               'correct_within_tight_per_gpu_s', 'correct_within_loose_per_gpu_s')})
    policy = dict(action, n=None, budget=None, policy='kv-ladder')
    (tmp_path / 'seed-1-light').mkdir()
    (tmp_path / 'seed-1-light' / 'cells.json').write_text(json.dumps([cell]))
    (tmp_path / 'seed-1-light' / 'cell-0-N1-B0-server.jsonl').write_text(json.dumps(dict(at_s=0., metrics={KV[0]: .2, TOKENS: 5.})) + '\n')
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(completed=['seed-1-light'], latency_targets_s=[1, 2], questions=2,
                                                            environment=dict(git_commit='abc', git_dirty=False))))
    (tmp_path / 'summary.json').write_text(json.dumps(dict(
        cells=[dict(block='seed-1-light', cell='cell-0-N1-B0', target_correct=2)], actions=[action, policy],
        paired_differences=[dict(rate='light', metric='correct_within_loose', action=[1, 0], minus='kv-ladder', questions=2, value=0., ci95=[-.1, .1])],
        latency_targets_s=dict(tight=1, loose=2), definitions={}, caveat='c', blocks_scored=['seed-1-light'])))
    runs = list(t.matched_runs(tmp_path))
    assert [r['job_type'] for r in runs] == ['matched', 'matched-summary']
    assert runs[0]['summary']['scored/target_correct'] == 2 and runs[0]['history'][0]['kv_cache_usage'] == .2 and runs[0]['config']['seed'] == 1
    columns, rows = runs[1]['tables']['actions']
    assert len(columns) == len(rows[0]) and [r[1] for r in rows] == ['(1,0)', 'kv-ladder'] and rows[0][-3:] == [.5, .4, .6]
    assert runs[1]['tables']['paired_differences'][1][0][2:4] == ['(1,0)', 'kv-ladder']
    assert runs[1]['summary']['light/kv-ladder/accuracy'] == .5


def test_upload_sends_history_tables_and_summary_and_skips_existing_runs():
    run = dict(id='abc', name='n', group='g', job_type='matched', config=dict(seed=1), history=[dict(time_s=0, running=2.)],
               summary={'x': 1}, tables=dict(actions=(['a'], [[1]])))
    fake = FakeWandb()
    assert t.upload(run, fake, 'team', 'project', ['tag']) is True
    sent = fake.runs[0]
    assert (sent.kwargs['id'], sent.kwargs['entity'], sent.kwargs['project'], sent.kwargs['tags']) == ('abc', 'team', 'project', ['tag'])
    # Nothing about the machine or the code is collected by the client.
    assert sent.kwargs['settings']['disable_git'] and sent.kwargs['settings']['x_disable_meta'] and not sent.kwargs['settings']['save_code']
    assert sent.logged == [dict(time_s=0, running=2.), dict(actions=dict(columns=['a'], data=[[1]]))]
    assert ('*', dict(step_metric='time_s', summary='none')) in sent.metrics and sent.summary == {'x': 1} and sent.finished
    assert t.upload(run, fake, 'team', 'project', exists=lambda run_id: True) is False and len(fake.runs) == 1
    assert t.describe('matched', Path('d'), 'c', {}, [], {})['id'] == t.describe('matched', Path('/other/d'), 'c', {}, [], {})['id']
