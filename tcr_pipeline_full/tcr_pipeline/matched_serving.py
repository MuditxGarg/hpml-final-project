"""Matched E1 synthetic-length replay. Standard library; never launches a server."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import re
import socket
import threading
import time
import urllib.error
import urllib.request


def cell_order(actions, seed=None):
    ordered = list(dict.fromkeys([(0, 0)] + list(actions)))
    if seed is not None:
        random.Random(seed).shuffle(ordered)
    return ordered


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def generate_trace(lengths, prompts, seed, background_rate, target_rate, warmup, duration, max_n, target_count=None):
    if duration <= 0 or warmup < 0 or max_n < 1 or min(background_rate, target_rate) < 0:
        raise ValueError('duration/max_n must be positive; warmup/rates nonnegative')
    if target_count is not None and (not isinstance(target_count, int) or target_count < 1):
        raise ValueError('target_count must be a positive integer')
    if not prompts or not lengths or any(not v or any(int(x) != x or x < 1 for x in v) for v in lengths.values()):
        raise ValueError('nonempty prompts and positive integer empirical lengths required')
    for prompt in prompts:
        for key in ('thinking', 'nothinking'):
            value = prompt[key]
            if not isinstance(value, (str, list)) or not value:
                raise ValueError('prompts require nonempty thinking/nothinking strings or token ID lists')
            if isinstance(value, list) and any(not isinstance(x, int) or x < 0 for x in value):
                raise ValueError('token IDs must be nonnegative integers')
    events = []
    for stream, rate in [('background', background_rate), ('target', target_rate)]:
        arrival = random.Random(f'{seed}:{stream}:arrival')
        prompt_rng = random.Random(f'{seed}:{stream}:prompt')
        length_rng = {str(b): random.Random(f'{seed}:{stream}:length:{b}') for b in lengths}
        at, times = 0., []
        while rate > 0:
            at += arrival.expovariate(rate)
            if at >= warmup + duration:
                break
            times.append(at)
        if stream == 'target' and target_count is not None:
            fixed_rng = random.Random(f'{seed}:target:conditional-arrival')
            times = [t for t in times if t < warmup] + sorted(warmup + fixed_rng.random()*duration for _ in range(target_count))
        for index, at in enumerate(times):
            events.append(dict(id=f'{stream}-{index}', stream=stream, at_s=at, measured=at >= warmup,
                               prompt_index=prompt_rng.randrange(len(prompts)),
                               lengths={str(b): [length_rng[str(b)].choice(v) for _ in range(max_n)] for b, v in sorted(lengths.items())}))
    return dict(version=1, experiment='E1 synthetic token-count externality; no answer-quality claim', seed=seed,
                background_rate=background_rate, target_rate=target_rate, warmup_s=warmup, duration_s=duration,
                max_n=max_n, target_count=target_count,
                target_arrival_definition='conditional Poisson: fixed measured count with uniform arrival times; warmup Poisson' if target_count is not None else 'Poisson',
                lengths_sha256=digest(lengths), prompts=prompts,
                events=sorted(events, key=lambda e: (e['at_s'], e['id'])))


def requests_for(trace, event, n, budget):
    prompt = trace['prompts'][event['prompt_index']]['nothinking' if budget == 0 else 'thinking']
    lengths = event['lengths'][str(budget)]
    if not 1 <= n <= len(lengths):
        raise ValueError('action N exceeds persisted trace max_n')
    return [dict(prompt=prompt, max_tokens=length, ignore_eos=True, temperature=.6, top_p=.95, stream=False) for length in lengths[:n]]


def http(url, payload=None, timeout=10):
    req = urllib.request.Request(url, data=None if payload is None else json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read().decode()


def server_metrics(url, timeout=10):
    """Sum selected vLLM gauges/counters across labeled replicas."""
    txt = http(url.rstrip('/') + '/metrics', timeout=timeout)
    gauges = {}
    required = ('vllm:num_requests_running', 'vllm:num_requests_waiting')
    optional = ('vllm:gpu_cache_usage_perc', 'vllm:kv_cache_usage_perc',
                'vllm:prompt_tokens_total', 'vllm:generation_tokens_total')
    for name in required + optional:
        values = re.findall(r'^' + re.escape(name) + r'(?:\{[^\n]*\})?\s+([\d.eE+\-]+)', txt, re.M)
        if not values and name in required:
            raise RuntimeError(f'cannot verify server state: missing {name} in /metrics')
        if values:
            gauges[name] = sum(float(v) for v in values)
    return gauges


def wait_idle(url, timeout_s=120, stable_s=1):
    """Require both known vLLM gauges, on every replica, to remain zero."""
    deadline, zero_since = time.monotonic() + timeout_s, None
    while time.monotonic() < deadline:
        gauges = server_metrics(url, timeout=min(10, max(.01, deadline-time.monotonic())))
        now = time.monotonic()
        if all(gauges[name] == 0 for name in ('vllm:num_requests_running', 'vllm:num_requests_waiting')):
            zero_since = now if zero_since is None else zero_since
            if now-zero_since >= stable_s:
                return gauges
        else:
            zero_since = None
        time.sleep(min(.1, max(0, deadline-time.monotonic())))
    raise TimeoutError('server did not become idle; refusing next cell')


def run_cell(trace, url, model, n, budget, reference_budget, timeout_s=120, drain_s=120, max_inflight=128, telemetry=None, metrics_interval_s=1):
    """Open-loop job admission; N parallel HTTP calls per admitted target job.

    Daemon workers cannot extend the cohort censor deadline. Idle verification by
    the caller is mandatory before reuse, since timed-out requests may remain on GPU.
    """
    if min(timeout_s, drain_s, max_inflight, metrics_interval_s) <= 0:
        raise ValueError('timeout, drain and max_inflight must be positive')
    if n < 0 or (n == 0 and budget != 0):
        raise ValueError('control is (N=0,B=0); action N must be positive')
    rows = [dict(e, status='pending') for e in trace['events']]
    start = time.monotonic()
    deadline = start + trace['warmup_s'] + trace['duration_s'] + drain_s
    slots, lock = threading.BoundedSemaphore(max_inflight), threading.Lock()
    closed = threading.Event()
    metrics_stop = threading.Event()

    def sample_metrics():
        while not metrics_stop.is_set():
            sample = dict(at_s=time.monotonic()-start)
            try:
                sample['gauges'] = server_metrics(url, timeout=2)
            except Exception as exc:
                sample['error'] = str(exc)
            sample['sample_duration_s'] = time.monotonic()-start-sample['at_s']
            with lock:
                if not metrics_stop.is_set(): telemetry.append(sample)
            if metrics_stop.wait(metrics_interval_s): break

    monitor = None
    if telemetry is not None:
        monitor = threading.Thread(target=sample_metrics, daemon=True)
        monitor.start()

    def job(row):
        try:
            action_n, action_b = (1, reference_budget) if row['stream'] == 'background' else (n, budget)
            parts, part_lock = [], threading.Lock()
            def part(body):
                dispatched = time.monotonic()
                result = dict(status='inflight', dispatch_lag_s=dispatched-start-row['at_s'])
                with lock:
                    if closed.is_set(): return
                    row['dispatch_lag_s'] = max(row.get('dispatch_lag_s', 0), result['dispatch_lag_s'])
                    part_index = len(row.setdefault('parts', []))
                    row['parts'].append(dict(result))
                try:
                    body.update(model=model, seed=trace['seed'])
                    response = json.loads(http(url.rstrip('/')+'/v1/completions', body,
                        timeout=max(.001, min(timeout_s, deadline-dispatched))))
                    if not response.get('choices'):
                        raise ValueError('response has no completion choices')
                    actual = response.get('usage', {}).get('completion_tokens')
                    if actual != body['max_tokens']:
                        raise ValueError(f'token replay mismatch: requested {body["max_tokens"]}, observed {actual}')
                    result.update(status='completed', completion_tokens=actual)
                except (TimeoutError, socket.timeout) as exc:
                    result.update(status='timeout', error=str(exc))
                except Exception as exc:
                    result.update(status='error', error=str(exc))
                result.update(completed_at_s=time.monotonic()-start, transport_s=time.monotonic()-dispatched)
                with part_lock:
                    parts.append(result)
                with lock:
                    if not closed.is_set(): row['parts'][part_index] = dict(result)
            children = [threading.Thread(target=part, args=(body,), daemon=True) for body in requests_for(trace, row, action_n, action_b)]
            for child in children: child.start()
            for child in children: child.join(max(0, deadline-time.monotonic()))
            with lock:
                if closed.is_set(): return
                if len(parts) == action_n:
                    row['parts'] = list(parts)
                if len(parts) < action_n or time.monotonic() > deadline:
                    row.update(status='censored', observed_s=deadline-start-row['at_s'])
                elif all(p['status']=='completed' for p in parts):
                    row.update(status='completed', latency_s=max(p['completed_at_s'] for p in parts)-row['at_s'])
                else:
                    row['status'] = 'timeout' if any(p['status']=='timeout' for p in parts) else 'error'
        finally:
            slots.release()

    workers = []
    for row in rows:
        if n == 0 and row['stream'] == 'target':
            row['status'] = 'not_offered_control'
            continue
        time.sleep(max(0, start+row['at_s']-time.monotonic()))
        if time.monotonic() >= deadline:
            row.update(status='censored', observed_s=max(0, deadline-start-row['at_s']))
        elif not slots.acquire(blocking=False):
            row.update(status='drop', dispatch_lag_s=time.monotonic()-start-row['at_s'])
        else:
            worker = threading.Thread(target=job, args=(row,), daemon=True)
            worker.start(); workers.append(worker)
    time.sleep(max(0, start+trace['warmup_s']+trace['duration_s']-time.monotonic()))
    for worker in workers: worker.join(max(0, deadline-time.monotonic()))
    with lock:
        closed.set()
        for row in rows:
            if row['status']=='pending':
                row.update(status='censored', observed_s=deadline-start-row['at_s'])
    metrics_stop.set()
    if monitor is not None: monitor.join(.1)
    return rows


def percentile(values, q=.95):
    return sorted(values)[max(0, math.ceil(q*len(values))-1)] if values else None


def summarize(rows, slo_s):
    output = {}
    for stream in ('background', 'target'):
        cohort = [r for r in rows if r['measured'] and r['stream']==stream and r['status'] != 'not_offered_control']
        complete = [r['latency_s'] for r in cohort if r['status']=='completed']
        lower = [r['latency_s'] if r['status']=='completed' else r.get('observed_s', math.inf) if r['status']=='censored' else math.inf for r in cohort]
        upper = [r['latency_s'] if r['status']=='completed' else math.inf for r in cohort]
        lo, hi = percentile(lower), percentile(upper)
        output[stream] = dict(scheduled=len(cohort), not_offered_control=sum(r['measured'] and r['stream']==stream and r['status']=='not_offered_control' for r in rows), statuses={s:sum(r['status']==s for r in cohort) for s in ('completed','drop','timeout','error','censored')},
            slo_attainment=sum(v<=slo_s for v in complete)/len(cohort) if cohort else None,
            completed_p95_s=percentile(complete), unconditional_p95_lower_s=lo if lo is not None and math.isfinite(lo) else None,
            unconditional_p95_upper_s=hi if hi is not None and math.isfinite(hi) else None,
            unconditional_bound_note='null means unbounded/infinite or empty cohort; drops/errors/timeouts are infinite misses; censored latency lies in [observed, infinity)',
            dispatch_lag_p95_s=percentile([r['dispatch_lag_s'] for r in cohort if r.get('dispatch_lag_s') is not None]))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    gen = subs.add_parser('generate')
    for name in ('lengths','prompts','output'): gen.add_argument('--'+name, required=True)
    gen.add_argument('--seed', type=int, default=42)
    gen.add_argument('--background-rate', type=float, required=True)
    gen.add_argument('--target-rate', type=float, required=True)
    gen.add_argument('--warmup', type=float, default=30)
    gen.add_argument('--duration', type=float, default=120)
    gen.add_argument('--target-count',type=int,default=None,help='fixed measured target cohort; conditional Poisson arrivals')
    gen.add_argument('--max-n', type=int, default=4)
    run = subs.add_parser('run')
    for name in ('trace','url','model','output','hardware-json'): run.add_argument('--'+name, required=True)
    run.add_argument('--actions', nargs='+', required=True, help='N:B pairs; background-only (0:0) control is automatically first')
    run.add_argument('--order-seed', type=int, default=None, help='randomize all cells including control; omitted retains control first')
    run.add_argument('--reference-budget', type=int, required=True)
    run.add_argument('--cache-policy', choices=['disabled','reset-prefix','retain'], required=True)
    run.add_argument('--slo', type=float, required=True)
    run.add_argument('--attainment', type=float, default=.95)
    run.add_argument('--timeout', type=float, default=120)
    run.add_argument('--drain', type=float, default=120)
    run.add_argument('--idle-timeout', type=float, default=120)
    run.add_argument('--max-inflight', type=int, default=128)
    run.add_argument('--metrics-interval', type=float, default=1)
    args = parser.parse_args(argv)
    if args.command == 'generate':
        trace = generate_trace(json.loads(Path(args.lengths).read_text()), json.loads(Path(args.prompts).read_text()), args.seed,
                               args.background_rate,args.target_rate,args.warmup,args.duration,args.max_n,args.target_count)
        trace['sources'] = {'lengths':str(Path(args.lengths).resolve()), 'prompts':str(Path(args.prompts).resolve())}
        with open(args.output, 'x') as f: json.dump(trace, f, indent=2)
        print(json.dumps({'trace':args.output, 'events':len(trace['events']), 'sha256':digest(trace)}))
        return
    if args.slo <= 0 or not 0 < args.attainment <= 1:
        parser.error('positive SLO and attainment in (0,1] required')
    trace = json.loads(Path(args.trace).read_text())
    actions = cell_order([tuple(map(int, x.split(':'))) for x in args.actions], args.order_seed)
    for n,b in actions:
        if n < 0 or (n == 0 and b != 0): parser.error('control is 0:0; action N must be positive')
        for event in trace['events']:
            if event['stream'] == 'background': requests_for(trace,event,1,args.reference_budget)
            elif n > 0: requests_for(trace,event,n,b)
    hardware = json.loads(Path(args.hardware_json).read_text())
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    manifest = dict(experiment=trace['experiment'], trace_sha256=digest(trace), configuration=vars(args), hardware_user_supplied=hardware,
                    client_platform=platform.platform(), started_unix_s=time.time(), cache_policy_note='disabled is a user assertion; reset-prefix calls vLLM endpoint; retain is order-dependent',
                    cell_order=actions, control_definition='background_only; target events retained but not offered',
                    telemetry='per-cell timestamped /metrics samples; missing/failed samples recorded as errors', status='running')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (out/'trace.json').write_text(json.dumps(trace,indent=2))
    summaries = []
    try:
        for index,(n,b) in enumerate(actions):
            print(f'CELL_START {index+1}/{len(actions)} N={n} B={b} control={n==0}', flush=True)
            before = wait_idle(args.url,args.idle_timeout)
            if args.cache_policy == 'reset-prefix':
                http(args.url.rstrip('/')+'/reset_prefix_cache', {}, timeout=10)
                wait_idle(args.url,args.idle_timeout)
            telemetry = []
            rows = run_cell(trace,args.url,args.model,n,b,args.reference_budget,args.timeout,args.drain,args.max_inflight,telemetry,args.metrics_interval)
            with (out/f'cell-{index:03d}-N{n}-B{b}-metrics.jsonl').open('w') as f:
                for sample in telemetry: f.write(json.dumps(sample)+'\n')
            with (out/f'cell-{index:03d}-N{n}-B{b}.jsonl').open('w') as f:
                for row in rows: f.write(json.dumps(row)+'\n')
            summary = dict(n=n,budget=b,control=n==0,control_definition='background_only' if n==0 else None,idle_before=before,telemetry_samples=len(telemetry),metrics=summarize(rows,args.slo))
            relevant = [summary['metrics']['background']] if n == 0 else list(summary['metrics'].values())
            summary['observed_slo_feasible'] = all(v['scheduled'] > 0 and v['slo_attainment'] >= args.attainment for v in relevant)
            summaries.append(summary)
            (out/'summary.json').write_text(json.dumps(dict(cells=summaries, any_observed_feasible=any(s['observed_slo_feasible'] for s in summaries if not s['control']), inference='observed cohort only; no confidence/quality claim'),indent=2))
            wait_idle(args.url,args.idle_timeout)
            print(f'CELL_COMPLETE {index+1}/{len(actions)} N={n} B={b} observed_slo_feasible={summary["observed_slo_feasible"]}', flush=True)
        manifest['status']='complete'
    except Exception as exc:
        manifest.update(status='failed', error=str(exc))
        raise
    finally:
        manifest['finished_unix_s']=time.time()
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(str(out/'summary.json'))
