"""Profiling pass: what each action does to a shared server at the frozen light and contended rates.

For each frozen background rate from a sustained calibration, replays one matched trace for a
background-only control and for every action (N, B), and logs:
  server (/metrics, every interval): running and waiting counts, waiting by reason, KV-cache usage,
      preemptions, token counters, and the sum/count of the queue-time, time-to-first-token,
      prefill, decode and inter-token-latency histograms
  GPU (nvidia-smi, every interval): utilisation, memory, power, temperature
  target requests (streamed): time to first token, decode time per token, longest stall
Background requests are not streamed, so they are sent exactly as in the calibration and the
matched replay. Synthetic lengths with ignore_eos; no answer quality. Standard library only; never
launches a server. See docs/profiling-pass.md.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection
import json
import math
from pathlib import Path
import re
import socket
import statistics
import subprocess
import sys
import threading
import time
from urllib.parse import urlparse

from matched_serving import cell_order, digest, generate_trace, http, percentile, requests_for, run_cell, summarize, wait_idle
from sustained_calibration import KV, RUNNING, TOKENS, WAITING, environment, write_json, write_rows

MARK_EVERY = 256  # tokens between progress marks kept for a streamed request
GPU_FIELDS = ('utilization.gpu', 'memory.used', 'power.draw', 'temperature.gpu')
HISTOGRAMS = dict(queue_time_s='vllm:request_queue_time_seconds', time_to_first_token_s='vllm:time_to_first_token_seconds',
                  prefill_time_s='vllm:request_prefill_time_seconds', decode_time_s='vllm:request_decode_time_seconds',
                  inter_token_latency_s='vllm:inter_token_latency_seconds', e2e_latency_s='vllm:e2e_request_latency_seconds',
                  preemptions_per_request='vllm:request_num_preemptions')
COUNTERS = dict(preemptions='vllm:num_preemptions_total', generation_tokens=TOKENS, prompt_tokens='vllm:prompt_tokens_total',
                prefix_cache_queries='vllm:prefix_cache_queries_total', prefix_cache_hits='vllm:prefix_cache_hits_total',
                requests_finished='vllm:request_success_total')


def parse_metrics(text):
    """All vllm gauges, counters and histogram sums/counts, summed over labels (waiting split by reason)."""
    values = {}
    for line in text.splitlines():
        match = re.match(r'(vllm:[A-Za-z_]+)(\{[^}]*\})?\s+(\S+)$', line)
        if not match:
            continue
        name, labels, value = match.groups()
        if name.endswith(('_bucket', '_created')) or name == 'vllm:cache_config_info':
            continue
        if name == 'vllm:num_requests_waiting_by_reason':
            reason = re.search(r'reason="([^"]*)"', labels or '')
            name += ':' + (reason.group(1) if reason else 'unknown')
        try:
            values[name] = values.get(name, 0.) + float(value)
        except ValueError:
            pass
    return values


def gpu_sample():
    line = subprocess.check_output(['nvidia-smi', '--query-gpu=' + ','.join(GPU_FIELDS), '--format=csv,noheader,nounits'],
                                   text=True, stderr=subprocess.DEVNULL, timeout=5).strip().splitlines()[0]
    return {field: float(value) for field, value in zip(GPU_FIELDS, line.split(','))}


class Sampler:
    """Timestamped server and GPU samples on two threads. A failed sample is recorded, never dropped."""

    def __init__(self, url, interval_s, gpu=True):
        self.url, self.interval, self.gpu = url, interval_s, gpu
        self.server, self.device, self._stop, self._threads = [], [], threading.Event(), []

    def _loop(self, read, sink, key):
        while not self._stop.is_set():
            sample = dict(at_s=time.monotonic() - self.t0)
            try:
                sample[key] = read()
            except Exception as exc:
                sample['error'] = f'{type(exc).__name__}: {exc}'
            sink.append(sample)
            if self._stop.wait(self.interval):
                break

    def start(self):
        self.t0 = time.monotonic()
        jobs = [(lambda: parse_metrics(http(self.url + '/metrics', timeout=2)), self.server, 'metrics')]
        if self.gpu:
            jobs.append((gpu_sample, self.device, 'gpu'))
        for job in jobs:
            thread = threading.Thread(target=self._loop, args=job, daemon=True)
            thread.start()
            self._threads.append(thread)
        return self.t0

    def stop(self):
        self._stop.set()
        for thread in self._threads:
            thread.join(6)


def read_stream(url, body, timeout_s, part, t0):
    """One streamed completion. Fills first-token time, progress marks and the longest stall into `part`."""
    target = urlparse(url)
    deadline = time.monotonic() + timeout_s
    conn = HTTPConnection(target.hostname, target.port, timeout=timeout_s)
    try:
        payload = dict(body, stream=True, stream_options=dict(include_usage=True, continuous_usage_stats=True))
        conn.request('POST', '/v1/completions', json.dumps(payload).encode(), {'Content-Type': 'application/json'})
        response = conn.getresponse()
        if response.status != 200:
            raise ValueError(f'HTTP {response.status}')
        tokens, last, marks, stall = 0, None, [], 0.
        for raw in response:
            if not raw.startswith(b'data:'):
                continue
            data = raw[5:].strip()
            if data == b'[DONE]':
                break
            now = time.monotonic()
            if now > deadline:
                raise TimeoutError('stream exceeded its deadline')
            count = (json.loads(data).get('usage') or {}).get('completion_tokens')
            if not count or count == tokens:
                continue
            if tokens == 0:
                part.update(first_token_s=now - t0, first_tokens=count)
            else:
                stall = max(stall, now - last)
            if count // MARK_EVERY > tokens // MARK_EVERY:
                marks.append([round(now - t0, 4), count])
            tokens, last = count, now
        part.update(tokens=tokens, last_token_s=None if last is None else last - t0, max_gap_s=stall, marks=marks)
        return tokens
    finally:
        conn.close()


def make_runner(trace, url, model, t0):
    """Job callback for matched_serving.run_cell: background plain, target streamed, N samples in parallel."""
    def one(body, stream, timeout_s):
        part = dict(dispatch_s=time.monotonic() - t0, max_tokens=body['max_tokens'], streamed=stream)
        try:
            if stream:
                tokens = read_stream(url, body, timeout_s, part, t0)
            else:
                tokens = json.loads(http(url + '/v1/completions', body, timeout_s)).get('usage', {}).get('completion_tokens')
            if tokens != body['max_tokens']:
                raise ValueError(f'token replay mismatch: requested {body["max_tokens"]}, observed {tokens}')
            part['status'] = 'completed'
        except (TimeoutError, socket.timeout) as exc:
            part.update(status='timeout', error=str(exc))
        except Exception as exc:
            part.update(status='error', error=f'{type(exc).__name__}: {exc}')
        part['end_s'] = time.monotonic() - t0
        return part

    def runner(event, n, budget, remaining_s):
        bodies = [dict(body, model=model, seed=trace['seed']) for body in requests_for(trace, event, n, budget)]
        stream = event['stream'] == 'target'
        with ThreadPoolExecutor(max_workers=n) as pool:
            parts = list(pool.map(lambda body: one(body, stream, max(.001, remaining_s)), bodies))
        statuses = {p['status'] for p in parts}
        status = 'completed' if statuses == {'completed'} else 'timeout' if 'timeout' in statuses else 'error'
        return dict(status=status, parts=parts, n=n, budget=budget)
    return runner


def spread(values):
    values = [v for v in values if v is not None]
    return dict(n=len(values), mean=statistics.mean(values) if values else None, p50=percentile(values, .5),
                p95=percentile(values, .95), max=max(values) if values else None)


def decode_by_concurrency(samples, width=16):
    """Mean inter-token latency between consecutive samples, grouped by the running count at that time."""
    name = HISTOGRAMS['inter_token_latency_s']
    bins = {}
    for a, b in zip(samples, samples[1:]):
        tokens = b.get(name + '_count', 0) - a.get(name + '_count', 0)
        if tokens <= 0 or RUNNING not in a or RUNNING not in b:
            continue
        latency = (b[name + '_sum'] - a[name + '_sum']) / tokens
        bins.setdefault(int((a[RUNNING] + b[RUNNING]) / 2 // width) * width, []).append(latency)
    return [dict(running_from=low, running_to=low + width, intervals=len(v), inter_token_latency_ms=1000 * statistics.median(v),
                 tokens_per_s_per_sequence=1 / statistics.median(v)) for low, v in sorted(bins.items())]


def summarize_cell(rows, server, device, trace, targets_s):
    """Everything reported for one cell, over the measurement window only."""
    warm, end = trace['warmup_s'], trace['warmup_s'] + trace['duration_s']
    streams = {}
    by_target = {f'{t:g}': summarize(rows, t) for t in targets_s}
    for stream in ('background', 'target'):
        cohort = [r for r in rows if r['measured'] and r['stream'] == stream and r['status'] != 'not_offered_control']
        done = [r['latency_s'] for r in cohort if r['status'] == 'completed']
        base = by_target[f'{max(targets_s):g}'][stream]
        streams[stream] = dict(scheduled=len(cohort), statuses=base['statuses'],
                               completion_rate=len(done) / len(cohort) if cohort else None,
                               latency_s=spread(done), unconditional_p95_upper_s=base['unconditional_p95_upper_s'],
                               attainment={key: value[stream]['slo_attainment'] for key, value in by_target.items()})
    parts = [p for r in rows if r['measured'] and r['stream'] == 'target' for p in r.get('parts', []) if p.get('streamed')]
    done = [p for p in parts if p['status'] == 'completed']
    streams['target']['streamed_requests'] = dict(
        sent=len(parts), completed=len(done),
        time_to_first_token_s=spread([p['first_token_s'] - p['dispatch_s'] for p in done]),
        decode_ms_per_token=spread([1000 * (p['last_token_s'] - p['first_token_s']) / (p['tokens'] - p['first_tokens'])
                                    for p in done if p['tokens'] > p['first_tokens']]),
        longest_stall_s=spread([p['max_gap_s'] for p in done]),
        note='client-side timing of completed streamed target requests only')
    window = [s for s in server if warm <= s['at_s'] < end]
    samples = [s['metrics'] for s in window if 'metrics' in s]
    series = lambda name: [m[name] for m in samples if name in m]
    stat = lambda values: dict(mean=statistics.mean(values) if values else None, max=max(values) if values else None)
    delta = lambda name: samples[-1][name] - samples[0][name] if len(samples) > 1 and name in samples[0] and name in samples[-1] else None
    span = window and samples and (max(s['at_s'] for s in window if 'metrics' in s) - min(s['at_s'] for s in window if 'metrics' in s))
    waiting = series(WAITING)
    histograms = {}
    for key, name in HISTOGRAMS.items():
        count, total = delta(name + '_count'), delta(name + '_sum')
        histograms[key] = dict(observations=count, mean=total / count if count else None)
    counters = {key: delta(name) for key, name in COUNTERS.items()}
    gpu = [s['gpu'] for s in device if warm <= s['at_s'] < end and 'gpu' in s]
    return dict(
        streams=streams,
        server=dict(samples=len(window), scrape_errors=sum('error' in s for s in window),
                    running=stat(series(RUNNING)), waiting=stat(waiting),
                    waiting_positive_fraction=sum(v > 0 for v in waiting) / len(waiting) if waiting else None,
                    waiting_by_reason={name.rsplit(':', 1)[1]: stat(series(name)) for name in sorted({k for m in samples for k in m})
                                       if name.startswith('vllm:num_requests_waiting_by_reason:')},
                    kv_cache=stat([m[k] for m in samples for k in KV if k in m]),
                    counters=counters, tokens_per_s=counters['generation_tokens'] / span if span and counters['generation_tokens'] is not None else None,
                    histogram_means=histograms, decode_by_concurrency=decode_by_concurrency(samples),
                    note='measurement window only; histogram means cover requests or tokens observed in the window, all streams together'),
        gpu=dict(samples=len(gpu), errors=sum('error' in s for s in device if warm <= s['at_s'] < end),
                 **{field: stat([g[field] for g in gpu]) for field in GPU_FIELDS}))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ('url', 'model', 'lengths', 'prompts', 'hardware-json', 'calibration', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--actions', nargs='+', default=['1:0', '4:0', '1:2048', '4:2048'], help='N:B pairs; the background-only control is always added')
    p.add_argument('--target-fraction', type=float, default=.05, help='target job rate as a fraction of the contended rate')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--warmup', type=float, default=60)
    p.add_argument('--duration', type=float, default=180)
    p.add_argument('--timeout', type=float, default=300)
    p.add_argument('--drain', type=float, default=300)
    p.add_argument('--idle-timeout', type=float, default=900)
    p.add_argument('--idle-stable', type=float, default=1)
    p.add_argument('--max-inflight', type=int, default=512)
    p.add_argument('--metrics-interval', type=float, default=1)
    p.add_argument('--no-gpu', action='store_true', help='do not sample nvidia-smi (fake server, CPU-only tests)')
    args = p.parse_args(argv)
    args.actions = [tuple(map(int, a.split(':'))) for a in args.actions]
    if any(n < 1 or b < 0 for n, b in args.actions) or args.target_fraction <= 0 or args.duration <= 0 or args.warmup < 0:
        p.error('actions need N >= 1 and B >= 0; target-fraction and duration must be positive')
    return args


def main(argv=None):
    args = parse_args(argv)
    url = args.url.rstrip('/')
    calibration = json.loads(Path(args.calibration).read_text())
    if calibration.get('status') != 'complete':
        raise SystemExit(f'calibration is not complete ({calibration.get("status")}); refusing to profile without frozen rates')
    budget, targets = calibration['reference_budget'], calibration['latency_targets_s']
    rates = dict(light=calibration['light_rate_rps'], contended=calibration['contended_rate_rps'])
    target_rate = args.target_fraction * rates['contended']
    lengths = json.loads(Path(args.lengths).read_text())
    prompts = json.loads(Path(args.prompts).read_text())
    budgets = sorted({budget} | {b for _, b in args.actions})
    missing = [b for b in budgets if str(b) not in lengths]
    if missing:
        raise SystemExit(f'budgets {missing} are not in {args.lengths}')
    pools = {str(b): lengths[str(b)] for b in budgets}
    hardware = json.loads(Path(args.hardware_json).read_text())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    idle_wait = lambda: wait_idle(url, args.idle_timeout, args.idle_stable)
    # Persist every trace and the cell order before any cell runs.
    blocks = []
    for index, (label, rate) in enumerate(rates.items()):
        trace = generate_trace(pools, prompts, args.seed + index, rate, target_rate, args.warmup, args.duration,
                               max(n for n, _ in args.actions))
        order = cell_order(args.actions, f'{args.seed}:{label}:order')
        write_json(out / f'{label}-trace.json', trace)
        blocks.append((label, rate, trace, order))
    manifest = dict(
        experiment='profiling pass: matched synthetic-length replay per action at frozen background rates; no answer-quality claim',
        status='running', configuration=dict(vars(args), actions=[list(a) for a in args.actions]),
        hardware_user_supplied=hardware, environment=environment(url), calibration_sha256=digest(calibration),
        calibration_source=str(Path(args.calibration).resolve()), rates_rps=rates, target_rate_rps=target_rate,
        reference_budget=budget, latency_targets_s=targets, lengths_sha256=digest(lengths), prompts_sha256=digest(prompts),
        cell_order={label: [list(a) for a in order] for label, _, _, order in blocks},
        control_definition='(0,0): background only; target events retained but not offered',
        streaming='target requests streamed with per-chunk usage; background requests not streamed',
        started_unix_s=time.time(), cells_completed=[])
    write_json(out / 'manifest.json', manifest)
    result = dict(hardware_user_supplied=hardware, reference_budget=budget, latency_targets_s=targets, rates_rps=rates,
                  target_rate_rps=target_rate, cells=[], status='running',
                  caveat='one machine, one pass per cell; synthetic fixed-length requests, not real answers')
    try:
        for label, rate, trace, order in blocks:
            for index, (n, b) in enumerate(order):
                name = f'{label}-cell-{index}-N{n}-B{b}'
                print(f'CELL_START {name} background={rate:.3f} req/s', flush=True)
                idle_wait()
                sampler = Sampler(url, args.metrics_interval, gpu=not args.no_gpu)
                t0 = sampler.start()
                rows = run_cell(trace, url, args.model, n, b, budget, timeout_s=args.timeout, drain_s=args.drain,
                                max_inflight=args.max_inflight, job_runner=make_runner(trace, url, args.model, t0))
                sampler.stop()
                write_rows(out / f'{name}.jsonl', rows)
                write_rows(out / f'{name}-server.jsonl', sampler.server)
                write_rows(out / f'{name}-gpu.jsonl', sampler.device)
                cell = dict(cell=name, rate_label=label, background_rate_rps=rate, n=n, budget=b, control=n == 0,
                            **summarize_cell(rows, sampler.server, sampler.device, trace, targets))
                result['cells'].append(cell)
                manifest['cells_completed'].append(name)
                write_json(out / 'profile.json', result)
                write_json(out / 'manifest.json', manifest)
                background = cell['streams']['background']
                print(f'CELL_COMPLETE {name} background_p95={background["latency_s"]["p95"]} '
                      f'completion={background["completion_rate"]} waiting_max={cell["server"]["waiting"]["max"]} '
                      f'preemptions={cell["server"]["counters"]["preemptions"]}', flush=True)
                idle_wait()
        result['status'] = manifest['status'] = 'complete'
        write_json(out / 'profile.json', result)
    except Exception as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        manifest['finished_unix_s'] = time.time()
        write_json(out / 'manifest.json', manifest)
    print(out / 'profile.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
