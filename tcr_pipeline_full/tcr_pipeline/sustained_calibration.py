"""Sustained calibration: background-only open-loop traffic over a rate ladder.

Measures, on one machine and against an already running vLLM server:
  (a) idle latency of the reference request        -> latency targets (multiples of idle p95)
  (b) a finite-burst throughput anchor              -> places the ladder (skipped with --rates)
  (c) one sustained rung per rate                   -> completion, queue, latency percentiles
and freezes a light rate and a contended rate from a rule fixed before the run.

Requests replay empirical reference-budget lengths with ignore_eos (synthetic lengths, no answer
quality). Standard library only; never launches a server. See docs/sustained-calibration.md.
"""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time

from matched_serving import digest, generate_trace, http, percentile, run_cell, summarize, wait_idle

RUNNING, WAITING = 'vllm:num_requests_running', 'vllm:num_requests_waiting'
KV = ('vllm:kv_cache_usage_perc', 'vllm:gpu_cache_usage_perc')
TOKENS = 'vllm:generation_tokens_total'
MIN_TELEMETRY = 5  # fewer in-window samples than this cannot verify a rung


def _replace(path, lines):
    """Write to a temporary file next to `path`, then rename it over `path`: a process stopped mid-write leaves the old file or none."""
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.partial')
    with temporary.open('w') as f:
        for line in lines:
            f.write(line)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def write_json(path, value):
    _replace(path, [json.dumps(value, indent=2) + '\n'])


def write_rows(path, rows):
    _replace(path, (json.dumps(row) + '\n' for row in rows))


def slope(points):
    """Least-squares slope of (x, y) points; None if it is not defined."""
    if len(points) < 2:
        return None
    mx = sum(x for x, _ in points) / len(points)
    my = sum(y for _, y in points) / len(points)
    sxx = sum((x - mx) ** 2 for x, _ in points)
    return sum((x - mx) * (y - my) for x, y in points) / sxx if sxx else None


def environment(url):
    """Best-effort record of code version, client and server. Missing items are null, not guessed."""
    here = Path(__file__).resolve().parent

    def sh(*cmd):
        try:
            return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=10, cwd=here).strip()
        except Exception:
            return None

    def get(path):
        try:
            return json.loads(http(url + path, timeout=5))
        except Exception:
            return None

    dirty = sh('git', 'status', '--porcelain')
    return dict(git_commit=sh('git', 'rev-parse', 'HEAD'), git_dirty=None if dirty is None else bool(dirty),
                client_platform=platform.platform(), python=sys.version.split()[0],
                nvidia_smi=sh('nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'),
                server_version=get('/version'), server_models=get('/v1/models'))


def measure_idle(url, model, prompts, pool, budget, reps, seed, timeout_s):
    """Sequential reference requests on an idle server. The first request is a discarded warm-up."""
    rng = random.Random(f'{seed}:idle')
    key = 'nothinking' if budget == 0 else 'thinking'
    samples = []
    for index in range(reps + 1):
        body = dict(model=model, seed=seed, prompt=rng.choice(prompts)[key], max_tokens=rng.choice(pool),
                    ignore_eos=True, temperature=.6, top_p=.95, stream=False)
        started = time.monotonic()
        response = json.loads(http(url + '/v1/completions', body, timeout_s))
        latency = time.monotonic() - started
        actual = response.get('usage', {}).get('completion_tokens')
        if actual != body['max_tokens']:
            raise RuntimeError(f'idle request {index}: requested {body["max_tokens"]} tokens, observed {actual}')
        samples.append(dict(index=index, warmup=index == 0, max_tokens=body['max_tokens'], latency_s=latency))
    latencies = [s['latency_s'] for s in samples if not s['warmup']]
    return dict(definition='sequential single reference requests on an idle server; first request discarded; nearest-rank percentiles',
                reference_budget=budget, reps=reps, p50_s=percentile(latencies, .5), p95_s=percentile(latencies, .95),
                samples=samples)


def measure_anchor(url, model, prompts, pool, budget, burst, seed, timeout_s):
    """Throughput of one simultaneous burst. Only used to place the ladder on this machine."""
    rng = random.Random(f'{seed}:anchor')
    events = [dict(id=f'background-{k}', stream='background', at_s=0, measured=True,
                   prompt_index=rng.randrange(len(prompts)), lengths={str(budget): [rng.choice(pool)]})
              for k in range(burst)]
    trace = dict(seed=seed, warmup_s=0, duration_s=0, prompts=prompts, events=events)
    rows = run_cell(trace, url, model, 0, 0, budget, timeout_s=timeout_s, drain_s=timeout_s, max_inflight=burst)
    failed = [r['status'] for r in rows if r['status'] != 'completed']
    if failed:
        raise RuntimeError(f'anchor burst incomplete ({len(failed)} of {burst} failed); refusing to place the ladder')
    elapsed = max(r['latency_s'] for r in rows)
    return dict(definition='finite-burst throughput proxy; places the ladder only, not a capacity estimate',
                burst=burst, elapsed_s=elapsed, requests_per_s=burst / elapsed), rows


def summarize_rung(rows, telemetry, trace):
    """Per-rung measurements. Every scheduled request in the window stays in the denominator."""
    warm, duration = trace['warmup_s'], trace['duration_s']
    end = warm + duration
    cohort = [r for r in rows if r['measured']]
    base = summarize(rows, math.inf)['background']
    done = [r['latency_s'] for r in cohort if r['status'] == 'completed']
    finished_in_window = sum(r['status'] == 'completed' and warm <= r['at_s'] + r['latency_s'] < end for r in rows)
    window = [s for s in telemetry if warm <= s['at_s'] < end]
    gauges = [(s['at_s'], s['gauges']) for s in window if 'gauges' in s]
    waiting = [g[WAITING] for _, g in gauges]
    running = [g[RUNNING] for _, g in gauges]
    cache = [g[k] for _, g in gauges for k in KV if k in g]
    tokens = [(t, g[TOKENS]) for t, g in gauges if TOKENS in g]
    backlog = slope([(t, g[RUNNING] + g[WAITING]) for t, g in gauges])
    offered = len(cohort) / duration
    mean = lambda values: statistics.mean(values) if values else None
    return dict(
        rate_rps=trace['background_rate'], scheduled=len(cohort), offered_rps=offered,
        statuses=base['statuses'], completion_rate=len(done) / len(cohort) if cohort else None,
        throughput_rps=finished_in_window / duration,
        latency=dict(mean_s=mean(done), p50_s=percentile(done, .5), p90_s=percentile(done, .9),
                     p95_s=percentile(done, .95), p99_s=percentile(done, .99), max_s=max(done) if done else None,
                     unconditional_p95_lower_s=base['unconditional_p95_lower_s'],
                     unconditional_p95_upper_s=base['unconditional_p95_upper_s'],
                     note='percentiles over completed requests; unconditional bounds count every failure as an infinite miss'),
        dispatch_lag_p95_s=base['dispatch_lag_p95_s'],
        queue=dict(samples=len(window), scrape_errors=sum('error' in s for s in window),
                   waiting_mean=mean(waiting), waiting_max=max(waiting) if waiting else None,
                   waiting_positive_fraction=sum(v > 0 for v in waiting) / len(waiting) if waiting else None,
                   running_mean=mean(running), running_max=max(running) if running else None,
                   kv_cache_mean=mean(cache), kv_cache_max=max(cache) if cache else None,
                   backlog_slope_rps=backlog,
                   backlog_growth_fraction=backlog / offered if backlog is not None and offered else None,
                   tokens_per_s=(tokens[-1][1] - tokens[0][1]) / (tokens[-1][0] - tokens[0][0])
                   if len(tokens) > 1 and tokens[-1][0] > tokens[0][0] else None,
                   note='measurement window only; a missing sample is not zero load'))


def make_rule(idle_p95, args):
    tight, loose = (m * idle_p95 for m in (min(args.target_multipliers), max(args.target_multipliers)))
    return dict(
        idle_p95_s=idle_p95, target_multipliers=sorted(args.target_multipliers), tight_target_s=tight, loose_target_s=loose,
        min_completion=args.min_completion, max_backlog_growth=args.max_backlog_growth,
        light_max_ratio=args.light_max_ratio, queue_free_fraction=.05, queue_sustained_fraction=.5,
        stable='completion_rate >= min_completion, no dropped requests, backlog_growth_fraction <= max_backlog_growth, '
               f'and at least {MIN_TELEMETRY} telemetry samples in the window',
        contended_rate='highest stable rung with unconditional p95 <= loose target and contention evidence: '
                       'unconditional p95 > tight target, or waiting > 0 in at least queue_sustained_fraction of samples',
        light_rate='highest stable rung with rate <= light_max_ratio * contended rate, unconditional p95 <= tight target, '
                   'and waiting > 0 in at most queue_free_fraction of samples')


def classify_rung(rung, rule):
    """Attach the pre-specified flags to one rung. Unverifiable is never treated as stable."""
    queue = rung['queue']
    reasons = []
    if not rung['scheduled']:
        reasons.append('no measured requests')
    elif rung['completion_rate'] < rule['min_completion']:
        reasons.append(f'completion rate {rung["completion_rate"]:.3f} below {rule["min_completion"]}')
    if rung['statuses'].get('drop'):
        reasons.append(f'{rung["statuses"]["drop"]} requests dropped at the client in-flight limit')
    growth = queue['backlog_growth_fraction']
    if queue['samples'] - queue['scrape_errors'] < MIN_TELEMETRY or growth is None:
        reasons.append('telemetry insufficient to verify a bounded backlog')
    elif growth > rule['max_backlog_growth']:
        reasons.append(f'backlog grows at {growth:.3f} of the offered rate')
    p95 = rung['latency']['unconditional_p95_upper_s']
    fraction = queue['waiting_positive_fraction']
    rung.update(
        stable=not reasons, unstable_reasons=reasons,
        p95_over_idle=p95 / rule['idle_p95_s'] if p95 is not None else None,
        within_tight_target=p95 is not None and p95 <= rule['tight_target_s'],
        within_loose_target=p95 is not None and p95 <= rule['loose_target_s'],
        queue_free=fraction is not None and fraction <= rule['queue_free_fraction'],
        queue_sustained=fraction is not None and fraction >= rule['queue_sustained_fraction'])
    rung['contention_evidence'] = rung['queue_sustained'] or (p95 is not None and p95 > rule['tight_target_s'])
    return rung


def select_rates(rungs, rule):
    """Apply the frozen-rate rule. Returns None for a rate the ladder does not support."""
    stable = [r for r in rungs if r['stable']]
    contended = max((r for r in stable if r['within_loose_target'] and r['contention_evidence']),
                    key=lambda r: r['rate_rps'], default=None)
    light = None
    if contended is not None:
        light = max((r for r in stable if r['within_tight_target'] and r['queue_free']
                     and r['rate_rps'] <= rule['light_max_ratio'] * contended['rate_rps']),
                    key=lambda r: r['rate_rps'], default=None)
    missing = [name for name, rung in (('contended', contended), ('light', light)) if rung is None]
    return dict(light_rate_rps=light['rate_rps'] if light else None, light_rung=light['rung'] if light else None,
                contended_rate_rps=contended['rate_rps'] if contended else None,
                contended_rung=contended['rung'] if contended else None,
                status='complete' if not missing else 'no ' + ' or '.join(missing) + ' rung satisfies the rule; extend or refine the ladder')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ('url', 'model', 'lengths', 'prompts', 'hardware-json', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--reference-budget', type=int, required=True, help='budget B of the background request; must be a key of --lengths')
    ladder = p.add_mutually_exclusive_group()
    ladder.add_argument('--rates', type=float, nargs='+', help='absolute ladder in requests/s; skips the anchor burst')
    ladder.add_argument('--fractions', type=float, nargs='+', default=[.3, .5, .7, .85, 1., 1.15, 1.3],
                        help='ladder as fractions of the anchor burst throughput (default)')
    p.add_argument('--anchor-burst', type=int, default=64)
    p.add_argument('--idle-reps', type=int, default=24)
    p.add_argument('--target-multipliers', type=float, nargs='+', default=[2., 4.], help='latency targets as multiples of idle p95')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--warmup', type=float, default=30)
    p.add_argument('--duration', type=float, default=180)
    p.add_argument('--timeout', type=float, default=300)
    p.add_argument('--drain', type=float, default=300)
    p.add_argument('--idle-timeout', type=float, default=600)
    p.add_argument('--idle-stable', type=float, default=1)
    p.add_argument('--max-inflight', type=int, default=512)
    p.add_argument('--metrics-interval', type=float, default=1)
    p.add_argument('--min-completion', type=float, default=.99)
    p.add_argument('--max-backlog-growth', type=float, default=.05)
    p.add_argument('--light-max-ratio', type=float, default=.6)
    p.add_argument('--stop-after-unstable', type=int, default=1, help='stop the ascending ladder after this many unstable rungs')
    args = p.parse_args(argv)
    if min(args.rates or args.fractions) <= 0 or min(args.target_multipliers) <= 0 or len(args.target_multipliers) < 2:
        p.error('rates, fractions and target multipliers must be positive; give at least two multipliers')
    if min(args.idle_reps, args.anchor_burst, args.stop_after_unstable) < 1 or args.duration <= 0 or args.warmup < 0:
        p.error('idle-reps, anchor-burst, stop-after-unstable and duration must be positive')
    if not 0 < args.min_completion <= 1 or not 0 < args.light_max_ratio < 1:
        p.error('min-completion must be in (0,1] and light-max-ratio in (0,1)')
    return args


def main(argv=None):
    args = parse_args(argv)
    url, budget = args.url.rstrip('/'), args.reference_budget
    lengths = json.loads(Path(args.lengths).read_text())
    prompts = json.loads(Path(args.prompts).read_text())
    if str(budget) not in lengths:
        raise SystemExit(f'reference budget {budget} is not in {args.lengths}')
    pool = lengths[str(budget)]
    hardware = json.loads(Path(args.hardware_json).read_text())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    idle_wait = lambda: wait_idle(url, args.idle_timeout, args.idle_stable)
    manifest = dict(
        experiment='sustained calibration: background-only open-loop rate ladder; synthetic lengths, no answer-quality claim',
        status='running', configuration=vars(args), hardware_user_supplied=hardware, environment=environment(url),
        sources=dict(lengths=str(Path(args.lengths).resolve()), prompts=str(Path(args.prompts).resolve())),
        lengths_sha256=digest(lengths), prompts_sha256=digest(prompts), started_unix_s=time.time(),
        ladder_order='ascending; each rung starts and ends on an idle server', rungs_run=[], rungs_skipped=[])
    write_json(out / 'manifest.json', manifest)
    selection = None
    try:
        idle_wait()
        idle = measure_idle(url, model=args.model, prompts=prompts, pool=pool, budget=budget, reps=args.idle_reps,
                            seed=args.seed, timeout_s=args.timeout)
        write_json(out / 'idle.json', idle)
        rule = make_rule(idle['p95_s'], args)
        print(f'IDLE p95={idle["p95_s"]:.3f}s targets={rule["tight_target_s"]:.3f}s,{rule["loose_target_s"]:.3f}s', flush=True)
        anchor = None
        if args.rates:
            rates = sorted(set(args.rates))
        else:
            idle_wait()
            anchor, rows = measure_anchor(url, args.model, prompts, pool, budget, args.anchor_burst, args.seed, args.timeout)
            write_json(out / 'anchor.json', anchor)
            write_rows(out / 'anchor.jsonl', rows)
            rates = [f * anchor['requests_per_s'] for f in sorted(set(args.fractions))]
            print(f'ANCHOR {anchor["requests_per_s"]:.3f} req/s from a burst of {args.anchor_burst}', flush=True)
        manifest['ladder_rps'] = rates
        write_json(out / 'manifest.json', manifest)
        result = dict(hardware_user_supplied=hardware, reference_budget=budget, idle_p95_s=idle['p95_s'],
                      latency_targets_s=[m * idle['p95_s'] for m in sorted(args.target_multipliers)],
                      anchor=anchor, rule=rule, rungs=[], status='running',
                      caveat='one machine, one pass per rung, ascending order; synthetic fixed-length requests, not real answers')
        unstable = 0
        for index, rate in enumerate(rates):
            trace = generate_trace({str(budget): pool}, prompts, args.seed + 1 + index, rate, 0, args.warmup, args.duration, 1)
            write_json(out / f'rung-{index:02d}-trace.json', trace)
            print(f'RUNG_START {index + 1}/{len(rates)} rate={rate:.3f} req/s', flush=True)
            idle_wait()
            telemetry = []
            rows = run_cell(trace, url, args.model, 0, 0, budget, timeout_s=args.timeout, drain_s=args.drain,
                            max_inflight=args.max_inflight, telemetry=telemetry, metrics_interval_s=args.metrics_interval)
            write_rows(out / f'rung-{index:02d}.jsonl', rows)
            write_rows(out / f'rung-{index:02d}-metrics.jsonl', telemetry)
            rung = classify_rung(dict(rung=index, trace_sha256=digest(trace), **summarize_rung(rows, telemetry, trace)), rule)
            result['rungs'].append(rung)
            manifest['rungs_run'].append(index)
            write_json(out / 'calibration.json', result)
            write_json(out / 'manifest.json', manifest)
            print(f'RUNG_COMPLETE {index + 1}/{len(rates)} completion={rung["completion_rate"]} '
                  f'p95={rung["latency"]["p95_s"]} waiting_max={rung["queue"]["waiting_max"]} stable={rung["stable"]}', flush=True)
            idle_wait()
            unstable += not rung['stable']
            if unstable >= args.stop_after_unstable:
                manifest['rungs_skipped'] = list(range(index + 1, len(rates)))
                break
        selection = select_rates(result['rungs'], rule)
        result.update(selection, status=selection['status'])
        write_json(out / 'calibration.json', result)
        manifest['status'] = 'complete' if selection['status'] == 'complete' else 'complete_without_frozen_rates'
    except Exception as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        manifest['finished_unix_s'] = time.time()
        write_json(out / 'manifest.json', manifest)
    print(f'FROZEN light={selection["light_rate_rps"]} contended={selection["contended_rate_rps"]} req/s ({selection["status"]})', flush=True)
    print(out / 'calibration.json')
    return 0 if selection['status'] == 'complete' else 2


if __name__ == '__main__':
    sys.exit(main())
