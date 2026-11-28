import json

import pytest

import profiling_pass as pp
from test_sustained_calibration import fake_server  # noqa: F401  (pytest fixture)


def test_parse_metrics_sums_labels_and_skips_buckets():
    text = '\n'.join([
        '# TYPE vllm:num_requests_running gauge',
        'vllm:num_requests_running{engine="0",model_name="m"} 3.0',
        'vllm:num_requests_running{engine="1",model_name="m"} 2.0',
        'vllm:num_requests_waiting_by_reason{engine="0",reason="capacity"} 4.0',
        'vllm:num_requests_waiting_by_reason{engine="0",reason="deferred"} 1.0',
        'vllm:request_queue_time_seconds_bucket{le="0.3"} 9.0',
        'vllm:request_queue_time_seconds_sum{engine="0"} 1.5',
        'vllm:request_queue_time_seconds_count{engine="0"} 6.0',
        'vllm:num_preemptions_created{engine="0"} 1.79e+09',
        'python_gc_objects_collected_total 5.0'])
    assert pp.parse_metrics(text) == {
        'vllm:num_requests_running': 5.0, 'vllm:num_requests_waiting_by_reason:capacity': 4.0,
        'vllm:num_requests_waiting_by_reason:deferred': 1.0,
        'vllm:request_queue_time_seconds_sum': 1.5, 'vllm:request_queue_time_seconds_count': 6.0}


def test_decode_by_concurrency_groups_interval_means_by_running_count():
    name = pp.HISTOGRAMS['inter_token_latency_s']
    sample = lambda running, total, count: {pp.RUNNING: running, name + '_sum': total, name + '_count': count}
    samples = [sample(4, 0, 0), sample(6, 1, 100), sample(40, 2, 150), sample(40, 2, 150), sample(44, 6, 250)]
    table = pp.decode_by_concurrency(samples)
    assert [(row['running_from'], row['intervals']) for row in table] == [(0, 1), (16, 1), (32, 1)]  # idle interval skipped
    assert table[0]['inter_token_latency_ms'] == pytest.approx(10)
    assert table[2]['inter_token_latency_ms'] == pytest.approx(40)
    assert table[2]['tokens_per_s_per_sequence'] == pytest.approx(25)


def test_cell_summary_keeps_failed_targets_and_reads_counters():
    trace = dict(warmup_s=1, duration_s=10)
    part = dict(streamed=True, status='completed', dispatch_s=2, first_token_s=2.5, first_tokens=1, last_token_s=4.5,
                tokens=101, max_gap_s=.3)
    rows = [dict(stream='background', measured=True, status='completed', latency_s=3, at_s=2),
            dict(stream='target', measured=True, status='completed', latency_s=5, at_s=2, parts=[part, part]),
            dict(stream='target', measured=True, status='timeout', at_s=3, parts=[dict(part, status='timeout')])]
    server = [dict(at_s=t, metrics={pp.RUNNING: t, pp.WAITING: 0, 'vllm:num_preemptions_total': 2 * t,
                                    pp.TOKENS: 100 * t, 'vllm:request_queue_time_seconds_sum': .5 * t,
                                    'vllm:request_queue_time_seconds_count': t}) for t in range(0, 13)]
    server.append(dict(at_s=5.5, error='scrape failed'))
    cell = pp.summarize_cell(rows, server, [dict(at_s=3, gpu={f: 50. for f in pp.GPU_FIELDS})], trace, [4, 8])
    target = cell['streams']['target']
    assert target['scheduled'] == 2 and target['completion_rate'] == .5 and target['statuses']['timeout'] == 1
    assert target['attainment'] == {'4': 0, '8': .5}
    assert target['streamed_requests']['sent'] == 3 and target['streamed_requests']['completed'] == 2
    assert target['streamed_requests']['time_to_first_token_s']['p50'] == .5
    assert target['streamed_requests']['decode_ms_per_token']['p50'] == pytest.approx(20)
    assert cell['server']['samples'] == 11 and cell['server']['scrape_errors'] == 1
    assert cell['server']['counters']['preemptions'] == 18 and cell['server']['tokens_per_s'] == pytest.approx(100)
    assert cell['server']['histogram_means']['queue_time_s'] == dict(observations=9, mean=.5)
    assert cell['server']['histogram_means']['decode_time_s']['mean'] is None  # absent metric is null, not zero
    assert cell['gpu']['samples'] == 1 and cell['gpu']['utilization.gpu']['max'] == 50


def test_profiling_against_fake_server(fake_server, tmp_path):  # noqa: F811
    # Fake vLLM: 1000 tok/s per request, 4000 tok/s in total, 8 sequences. 200-token background requests.
    files = dict(lengths={'0': [50], '200': [200], '400': [400]}, prompts=[{'thinking': 'think', 'nothinking': 'answer'}],
                 hardware={'device': 'fake CPU HTTP server'},
                 calibration=dict(status='complete', reference_budget=200, latency_targets_s=[.4, 4.],
                                  light_rate_rps=4., contended_rate_rps=12.))
    for name, value in files.items():
        (tmp_path / f'{name}.json').write_text(json.dumps(value))
    out = tmp_path / 'run'
    argv = ['--url', fake_server, '--model', 'fake', '--output', str(out), '--no-gpu',
            '--actions', '1:0', '2:400', '--target-fraction', '.25', '--warmup', '.3', '--duration', '1.2',
            '--timeout', '30', '--drain', '30', '--idle-timeout', '30', '--idle-stable', '.1', '--metrics-interval', '.05']
    for name in ('lengths', 'prompts', 'calibration'):
        argv += ['--' + name, str(tmp_path / f'{name}.json')]
    argv += ['--hardware-json', str(tmp_path / 'hardware.json')]
    assert pp.main(argv) == 0

    manifest = json.loads((out / 'manifest.json').read_text())
    profile = json.loads((out / 'profile.json').read_text())
    assert manifest['status'] == profile['status'] == 'complete' and len(profile['cells']) == 6
    assert manifest['target_rate_rps'] == 3. and manifest['rates_rps'] == dict(light=4., contended=12.)
    for label in ('light', 'contended'):
        cells = [c for c in profile['cells'] if c['rate_label'] == label]
        assert sorted((c['n'], c['budget']) for c in cells) == [(0, 0), (1, 0), (2, 400)]
        assert [[c['n'], c['budget']] for c in cells] == manifest['cell_order'][label]
        # Matched replay: every cell of a rate offers the same background requests.
        assert len({c['streams']['background']['scheduled'] for c in cells}) == 1
        for cell in cells:
            background, target = cell['streams']['background'], cell['streams']['target']
            assert background['completion_rate'] == 1 and background['latency_s']['p95'] > .15
            assert cell['server']['samples'] > 5 and cell['server']['counters']['preemptions'] == 0
            assert cell['server']['histogram_means']['inter_token_latency_s']['mean'] > 0
            assert cell['server']['decode_by_concurrency']
            assert cell['gpu']['samples'] == 0
            streamed = target['streamed_requests']
            if cell['control']:
                assert target['scheduled'] == 0 and streamed['sent'] == 0
                continue
            assert target['scheduled'] > 0 and target['completion_rate'] == 1
            assert streamed['completed'] == streamed['sent'] == cell['n'] * target['scheduled']
            assert 0 < streamed['time_to_first_token_s']['p50'] < 1
            assert .5 < streamed['decode_ms_per_token']['p50'] < 20  # 1 ms per token when alone
    rows = [json.loads(line) for line in (out / 'contended-cell-0-N{}-B{}.jsonl'.format(*manifest['cell_order']['contended'][0])).read_text().splitlines()]
    assert all(not p['streamed'] for r in rows if r['stream'] == 'background' for p in r.get('parts', []))
    with pytest.raises(FileExistsError):
        pp.main(argv)


def test_profiling_refuses_an_incomplete_calibration(tmp_path):
    for name, value in dict(lengths={}, prompts=[], hardware={}, calibration=dict(status='no contended rung')).items():
        (tmp_path / f'{name}.json').write_text(json.dumps(value))
    argv = ['--url', 'http://127.0.0.1:9', '--model', 'fake', '--output', str(tmp_path / 'run'),
            '--hardware-json', str(tmp_path / 'hardware.json')]
    for name in ('lengths', 'prompts', 'calibration'):
        argv += ['--' + name, str(tmp_path / f'{name}.json')]
    with pytest.raises(SystemExit, match='not complete'):
        pp.main(argv)
    assert not (tmp_path / 'run').exists()


def test_nvidia_smi_values_parse_numbers_hex_masks_and_missing_fields():
    assert pp._smi_value(' 102.5') == 102.5 and pp._smi_value('[N/A]') is None and pp._smi_value('0x0000000000000004') == 4


@pytest.fixture
def fresh_gpu_state(monkeypatch):
    monkeypatch.setattr(pp, '_nvml', dict(handle=None, failed=False, smi_fields=None))


def test_gpu_sample_reads_nvml_in_process_and_nulls_unsupported_fields(monkeypatch, fresh_gpu_state):
    import sys
    import types

    class Unsupported(Exception):
        pass

    def unsupported(handle):
        raise Unsupported('Not Supported')
    nv = types.SimpleNamespace(
        nvmlInit=lambda: None, nvmlDeviceGetHandleByIndex=lambda i: 'gpu0', NVML_TEMPERATURE_GPU=0, NVML_CLOCK_SM=1, NVML_CLOCK_MEM=2,
        nvmlDeviceGetUtilizationRates=lambda h: types.SimpleNamespace(gpu=99), nvmlDeviceGetMemoryInfo=lambda h: types.SimpleNamespace(used=3 * 2 ** 20),
        nvmlDeviceGetPowerUsage=lambda h: 123456, nvmlDeviceGetTemperature=lambda h, kind: 80,
        nvmlDeviceGetEnforcedPowerLimit=unsupported, nvmlDeviceGetClockInfo=lambda h, kind: 2100 if kind == 1 else 9000,
        nvmlDeviceGetCurrentClocksThrottleReasons=lambda h: 0x24)  # older binding name only
    monkeypatch.setitem(sys.modules, 'pynvml', nv)
    monkeypatch.setattr(pp.subprocess, 'check_output', lambda *a, **k: pytest.fail('nvidia-smi must not run when NVML loads'))
    sample = pp.gpu_sample()
    assert sample == {'utilization.gpu': 99, 'memory.used': 3., 'power.draw': 123.456, 'temperature.gpu': 80, 'power.limit': None,
                      'clocks.sm': 2100, 'clocks.mem': 9000, 'throttle_reasons': 0x24, 'source': 'nvml'}
    assert pp.throttle_summary([sample, dict(sample, throttle_reasons=1), dict(sample, throttle_reasons=None)]) == dict(
        samples=2, power_cap=.5, hardware_slowdown=0., thermal=.5)
    assert pp.throttle_summary([dict(sample, throttle_reasons=None)]) is None


def test_gpu_sample_falls_back_to_nvidia_smi_and_drops_rejected_state_fields(monkeypatch, fresh_gpu_state):
    import subprocess
    import sys
    monkeypatch.setitem(sys.modules, 'pynvml', None)  # import fails
    calls = []

    def smi(cmd, **kwargs):
        calls.append(cmd[1])
        if 'clocks_throttle_reasons.active' in cmd[1]:
            raise subprocess.CalledProcessError(2, cmd)
        return '50, 1000, 110.5, 70\n'
    monkeypatch.setattr(pp.subprocess, 'check_output', smi)
    assert pp.gpu_sample() == {'utilization.gpu': 50., 'memory.used': 1000., 'power.draw': 110.5, 'temperature.gpu': 70.,
                               'power.limit': None, 'clocks.sm': None, 'clocks.mem': None, 'throttle_reasons': None, 'source': 'nvidia-smi'}
    pp.gpu_sample()
    assert len(calls) == 3 and 'enforced.power.limit' not in calls[2]  # rejected once, then only the base fields
    monkeypatch.setattr(pp.subprocess, 'check_output', lambda *a, **k: '50, 1000, 110.5, 70, [N/A], 2100, 9000, 0x0000000000000004\n')
    monkeypatch.setattr(pp, '_nvml', dict(handle=None, failed=True, smi_fields=None))
    assert pp.gpu_sample()['throttle_reasons'] == 4 and pp.gpu_sample()['clocks.sm'] == 2100.


def test_cell_summary_reports_machine_state_and_skips_missing_values():
    trace = dict(warmup_s=0, duration_s=10)
    base = {'utilization.gpu': 99., 'memory.used': 1., 'power.draw': 100., 'temperature.gpu': 80., 'power.limit': 175.,
            'clocks.sm': 2000., 'clocks.mem': 9000., 'throttle_reasons': 0x4, 'source': 'nvml'}
    device = [dict(at_s=1, gpu=base), dict(at_s=2, gpu=dict(base, **{'power.draw': None, 'clocks.sm': 1000., 'throttle_reasons': 0x1})),
              dict(at_s=3, error='TimeoutExpired')]
    gpu = pp.summarize_cell([], [], device, trace, [4])['gpu']
    assert gpu['samples'] == 2 and gpu['errors'] == 1 and gpu['power.draw'] == dict(mean=100., max=100.)
    assert gpu['clocks.sm']['mean'] == 1500. and gpu['power.limit']['max'] == 175.
    assert gpu['throttle'] == dict(samples=2, power_cap=.5, hardware_slowdown=0., thermal=0.) and gpu['source'] == ['nvml']
