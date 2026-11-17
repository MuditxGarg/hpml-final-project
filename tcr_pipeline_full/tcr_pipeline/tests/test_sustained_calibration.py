import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

import sustained_calibration as sc

FAKE = Path(__file__).resolve().parents[1] / 'test_fake_vllm_server.py'


def rule(idle_p95=10.0):
    return dict(idle_p95_s=idle_p95, tight_target_s=2 * idle_p95, loose_target_s=4 * idle_p95, min_completion=.99,
                max_backlog_growth=.05, light_max_ratio=.6, queue_free_fraction=.05, queue_sustained_fraction=.5)


def rung(index, rate, p95, waiting_fraction=0., growth=0., completion=1., drops=0, samples=100):
    return dict(rung=index, rate_rps=rate, scheduled=100, completion_rate=completion, statuses=dict(drop=drops),
                latency=dict(unconditional_p95_upper_s=p95),
                queue=dict(samples=samples, scrape_errors=0, waiting_positive_fraction=waiting_fraction,
                           backlog_growth_fraction=growth))


def classified(*rungs):
    return [sc.classify_rung(r, rule()) for r in rungs]


def test_rule_picks_light_and_contended_rungs():
    # Slowdown without a waiting queue counts as contention; the overloaded rung is never chosen.
    rungs = classified(rung(0, 1.0, 14), rung(1, 1.4, 16), rung(2, 2.2, 19), rung(3, 2.8, 28),
                       rung(4, 3.6, 61, waiting_fraction=1., growth=.2))
    assert [r['stable'] for r in rungs] == [True, True, True, True, False]
    assert not rungs[2]['contention_evidence'] and rungs[3]['contention_evidence']
    chosen = sc.select_rates(rungs, rule())
    assert chosen['status'] == 'complete'
    assert (chosen['contended_rung'], chosen['contended_rate_rps']) == (3, 2.8)
    assert (chosen['light_rung'], chosen['light_rate_rps']) == (1, 1.4)  # 2.2 exceeds 0.6 * 2.8


def test_sustained_queue_inside_the_loose_target_is_contended():
    rungs = classified(rung(0, 1.0, 12), rung(1, 2.0, 18, waiting_fraction=.8))
    assert sc.select_rates(rungs, rule())['contended_rung'] == 1


def test_rule_reports_missing_rates_instead_of_guessing():
    uncontended = classified(rung(0, 1.0, 12), rung(1, 1.5, 15))
    chosen = sc.select_rates(uncontended, rule())
    assert chosen['contended_rate_rps'] is None and chosen['light_rate_rps'] is None
    assert chosen['status'].startswith('no contended or light rung')
    no_light = classified(rung(0, 2.0, 15), rung(1, 2.5, 30))
    chosen = sc.select_rates(no_light, rule())
    assert chosen['contended_rung'] == 1 and chosen['light_rate_rps'] is None
    assert chosen['status'].startswith('no light rung')


def test_failures_and_missing_telemetry_are_never_stable():
    cases = classified(rung(0, 1.0, 12, completion=.9), rung(1, 1.0, 12, drops=1), rung(2, 1.0, 12, samples=2),
                       rung(3, 1.0, 12, growth=None), rung(4, 1.0, None, completion=.5))
    assert not any(r['stable'] for r in cases)
    assert 'telemetry insufficient' in cases[2]['unstable_reasons'][0]
    assert not cases[4]['within_loose_target']  # unbounded p95 is outside every target


def test_rung_summary_keeps_failures_in_the_denominator():
    trace = dict(warmup_s=10, duration_s=100, background_rate=.05)
    row = lambda **kw: dict(dict(stream='background', measured=True, status='completed'), **kw)
    rows = [row(at_s=5, measured=False, latency_s=10), row(at_s=20, latency_s=4), row(at_s=30, latency_s=6),
            row(at_s=40, status='timeout'), row(at_s=50, status='drop'), row(at_s=105, latency_s=8)]
    telemetry = [dict(at_s=t, gauges={sc.RUNNING: 2, sc.WAITING: t / 10, sc.TOKENS: 50 * t}) for t in range(0, 130, 10)]
    telemetry.append(dict(at_s=55, error='scrape failed'))
    s = sc.summarize_rung(rows, telemetry, trace)
    assert s['scheduled'] == 5 and s['completion_rate'] == 3 / 5
    assert s['statuses']['timeout'] == 1 and s['statuses']['drop'] == 1
    assert s['throughput_rps'] == 3 / 100  # warm-up arrival finishing in the window counts; the late one does not
    assert s['latency']['p50_s'] == 6 and s['latency']['unconditional_p95_upper_s'] is None
    assert s['queue']['samples'] == 11 and s['queue']['scrape_errors'] == 1
    assert s['queue']['waiting_max'] == 10 and s['queue']['waiting_positive_fraction'] == 1
    assert s['queue']['backlog_slope_rps'] == pytest.approx(.1)
    assert s['queue']['backlog_growth_fraction'] == pytest.approx(2.)
    assert s['queue']['tokens_per_s'] == pytest.approx(50)


@pytest.fixture
def fake_server():
    """Fake vLLM: 1000 tok/s per request, 4000 tok/s in total, 8 concurrent sequences."""
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    env = dict(os.environ, FAKE_VLLM_CAP='4000', FAKE_VLLM_PER_REQ='1000', FAKE_VLLM_MAXSEQ='8')
    proc = subprocess.Popen([sys.executable, str(FAKE), str(port)], env=env)
    url = f'http://127.0.0.1:{port}'
    try:
        for _ in range(100):
            try:
                urllib.request.urlopen(url + '/health', timeout=1).read()
                break
            except OSError:
                time.sleep(.05)
        else:
            raise RuntimeError('fake server did not start')
        yield url
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_ladder_against_fake_server(fake_server, tmp_path):
    # 200-token requests: 0.2 s idle, saturation at 20 req/s. Rungs at about 0.4x, 0.8x and 3x saturation.
    (tmp_path / 'lengths.json').write_text(json.dumps({'0': [5], '200': [200]}))
    (tmp_path / 'prompts.json').write_text(json.dumps([{'thinking': 'think', 'nothinking': 'answer'}]))
    (tmp_path / 'hardware.json').write_text(json.dumps({'device': 'fake CPU HTTP server'}))
    out = tmp_path / 'run'
    argv = ['--url', fake_server, '--model', 'fake', '--lengths', str(tmp_path / 'lengths.json'),
            '--prompts', str(tmp_path / 'prompts.json'), '--hardware-json', str(tmp_path / 'hardware.json'),
            '--output', str(out), '--reference-budget', '200', '--fractions', '.4', '.8', '3', '--anchor-burst', '16',
            '--idle-reps', '4', '--warmup', '.3', '--duration', '2', '--timeout', '30', '--drain', '30',
            '--idle-timeout', '30', '--idle-stable', '.1', '--metrics-interval', '.05',
            '--max-backlog-growth', '.3', '--target-multipliers', '2', '20', '--stop-after-unstable', '1']
    code = sc.main(argv)

    manifest = json.loads((out / 'manifest.json').read_text())
    result = json.loads((out / 'calibration.json').read_text())
    idle = json.loads((out / 'idle.json').read_text())
    anchor = json.loads((out / 'anchor.json').read_text())
    assert manifest['rungs_run'] == [0, 1, 2] and manifest['rungs_skipped'] == []
    assert manifest['hardware_user_supplied'] == {'device': 'fake CPU HTTP server'}
    assert manifest['configuration']['seed'] == 42 and 'git_commit' in manifest['environment']
    assert len(idle['samples']) == 5 and idle['samples'][0]['warmup']
    assert .15 < idle['p95_s'] < .5
    assert 10 < anchor['requests_per_s'] < 25
    assert result['latency_targets_s'] == [2 * idle['p95_s'], 20 * idle['p95_s']]

    light, contended, overloaded = result['rungs']
    assert light['completion_rate'] == 1 and light['stable'], light['unstable_reasons']
    assert light['queue']['samples'] >= sc.MIN_TELEMETRY and light['latency']['p95_s'] is not None
    assert not overloaded['stable']
    assert overloaded['queue']['waiting_max'] > 0 and overloaded['queue']['waiting_positive_fraction'] > .5
    assert overloaded['queue']['backlog_growth_fraction'] > .3
    assert overloaded['latency']['p95_s'] > 2 * light['latency']['p95_s']
    assert overloaded['throughput_rps'] < overloaded['offered_rps']
    # Every scheduled request is on disk, whatever happened to it.
    rows = [json.loads(line) for line in (out / 'rung-02.jsonl').read_text().splitlines()]
    assert sum(r['measured'] for r in rows) == overloaded['scheduled'] == sum(overloaded['statuses'].values())
    assert (out / 'rung-02-metrics.jsonl').exists() and (out / 'rung-02-trace.json').exists()

    # The busy but bounded rung is frozen as contended, the quiet one as light, the overloaded one never.
    assert contended['stable'] and contended['contention_evidence'] and contended['within_loose_target']
    assert light['queue_free'] and light['within_tight_target'] and not light['contention_evidence']
    assert (result['light_rung'], result['contended_rung']) == (0, 1)
    assert result['light_rate_rps'] == light['rate_rps'] and result['contended_rate_rps'] == contended['rate_rps']
    assert code == 0 and result['status'] == 'complete' and manifest['status'] == 'complete'
    with pytest.raises(FileExistsError):
        sc.main(argv)
