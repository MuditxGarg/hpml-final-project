"""Matched real-answer comparison of fixed actions at the frozen light and contended rates.

  prepare  choose a fixed question cohort (stratified by level) before any run
  run      for each seed and frozen rate, replay one matched trace for a background-only control and
           every action (N, B) in a seeded random order; target questions get real, EOS-aware answers
  score    grade the answers offline and report correct answers within the latency target per
           GPU-second, with question-level bootstrap intervals

Background traffic is the calibrated synthetic stream (reference-budget lengths, ignore_eos), byte
for byte the same in every cell of a block. Only target answers are real and scored. No router is
evaluated. Never launches a server. See docs/matched-real-answers.md.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import signal
import socket
import sys
import time

from answer_scoring import is_correct
from load_policy import load_policies, read_signal
from matched_serving import cell_order, digest, generate_trace, http, percentile, requests_for, run_cell, wait_idle
from profiling_pass import Sampler, summarize_cell
from real_answer_validation import MODEL, REVISION, aggregate, generate_sample
from sustained_calibration import TOKENS, environment, write_json, write_rows

LEVELS = (1, 2, 3, 4, 5)
FAILED = ('timeout', 'error', 'censored', 'drop')


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL, revision=REVISION)


def select_questions(rows, count, seed, proportional=False):
    """Seeded sample from id-sorted pools per difficulty level: equal counts, or counts proportional to the pools."""
    pools = {level: sorted((r for r in rows if int(r['level']) == level), key=lambda r: r['id']) for level in LEVELS}
    if proportional:
        exact = {level: count * len(pool) / len(rows) for level, pool in pools.items()}
        sizes = {level: int(value) for level, value in exact.items()}
        # Largest remainders take the questions left over after rounding down.
        for level in sorted(LEVELS, key=lambda l: (sizes[l] - exact[l], l))[:count - sum(sizes.values())]:
            sizes[level] += 1
    elif count % len(LEVELS):
        raise ValueError(f'question count must be a multiple of {len(LEVELS)}')
    else:
        sizes = dict.fromkeys(LEVELS, count // len(LEVELS))
    rng = random.Random(seed)
    cohort = []
    for level in LEVELS:
        cohort += rng.sample(pools[level], sizes[level])
    return cohort


def prepare(args):
    prompts = json.loads(Path(args.prompts).read_text())
    chunks = sorted(Path(args.labels).glob('chunk_*.json'))
    labels = {r['id']: r for path in chunks for r in json.loads(path.read_text())}
    if {p['id'] for p in prompts} != set(labels):
        raise ValueError('prompt and label question ids differ')
    cohort = select_questions([dict(p, level=labels[p['id']]['level']) for p in prompts], args.questions, args.seed, args.proportional)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'plan.json', dict(
        name='matched real-answer comparison', model=MODEL, revision=REVISION, seed=args.seed, questions=cohort,
        selection=('count per MATH level proportional to the full set' if args.proportional else 'equal count per MATH level')
        + ', seeded sample from id-sorted pools; chosen before any run, never from outcomes',
        vote_rule='plurality of valid normalized boxes; ties by lowest sample index; all N transports must complete',
        prompts_sha256=hashlib.sha256(Path(args.prompts).read_bytes()).hexdigest(),
        label_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in chunks}))
    write_json(out / 'gold.json', {q['id']: labels[q['id']]['gold'] for q in cohort})
    print(out / 'plan.json')


SCHEDULES = ('step', 'square')


def schedule_phases(kind, rates, warmup, duration, phase_s):
    """Piecewise-constant background rate over a cell, from a machine's frozen light and contended rates.

      step    light, contended, light: the measurement window in three equal phases
      square  light and contended alternating in phases of phase_s seconds, starting light; the last phase may be shorter
    The warm-up runs at the light rate and is its own phase."""
    if kind == 'step':
        cuts = [warmup + duration * k / 3 for k in range(4)]
    elif kind == 'square':
        if phase_s <= 0:
            raise ValueError('square phases need a positive length')
        cuts = [warmup]
        while cuts[-1] + phase_s < warmup + duration - 1e-9:
            cuts.append(cuts[-1] + phase_s)
        cuts.append(warmup + duration)
    else:
        raise ValueError(f'unknown schedule {kind!r}; known: {", ".join(SCHEDULES)}')
    phases = [dict(phase='warmup', rate_label='light', rate_rps=rates['light'], start_s=0., end_s=float(warmup))] if warmup > 0 else []
    for index, (start, end) in enumerate(zip(cuts, cuts[1:])):
        label = ('light', 'contended')[index % 2]
        phases.append(dict(phase=f'{index + 1}-{label}', rate_label=label, rate_rps=rates[label], start_s=start, end_s=end))
    return phases


def phase_of(phases, at_s):
    """The phase an arrival time falls in; phases are contiguous from 0, and the end of the last one belongs to it."""
    for phase in phases:
        if at_s < phase['end_s']:
            return phase
    return phases[-1]


def piecewise_background(pool, prompts, budget, seed, phases):
    """Poisson arrivals at a piecewise-constant rate, by time rescaling: a unit-rate stream on the cumulative-rate axis,
    mapped back to time through the inverse of the cumulative rate. Prompts and lengths use the same seeded streams as
    a constant-rate trace, so only the arrival times depend on the schedule."""
    total = sum(p['rate_rps'] * (p['end_s'] - p['start_s']) for p in phases)
    trace = generate_trace({str(budget): pool}, prompts, seed, 1., 0, 0, total, 1)
    bounds, cumulative = [], 0.
    for p in phases:
        bounds.append((cumulative, p))
        cumulative += p['rate_rps'] * (p['end_s'] - p['start_s'])
    for event in trace['events']:
        before, p = next((c, p) for c, p in reversed(bounds) if event['at_s'] >= c)
        event['at_s'] = p['start_s'] + (event['at_s'] - before) / p['rate_rps']
    return trace


def make_trace(questions, background_prompts, pool, budget, seed, rate, warmup, duration, phases=None):
    """Poisson synthetic background plus one evenly spaced target job per question, in seeded order.

    The target schedule depends only on the seed, so it is identical at every background rate and schedule. With `phases`
    (see schedule_phases) the background rate follows the schedule, `rate` is ignored, and every event carries its phase.
    """
    if phases is None:
        trace = generate_trace({str(budget): pool}, background_prompts, seed, rate, 0, warmup, duration, 1)
    else:
        trace = piecewise_background(pool, background_prompts, budget, seed, phases)
        for event in trace['events']:
            event['measured'] = event['at_s'] >= warmup
        trace.update(background_rate=None, warmup_s=warmup, duration_s=duration, schedule=phases,
                     background_arrival_definition='Poisson at a piecewise-constant rate (time rescaling of a unit-rate stream)')
    order = list(questions)
    random.Random(f'{seed}:targets').shuffle(order)
    for index, question in enumerate(order):
        trace['events'].append(dict(id=f'target-{index}', stream='target', question_id=question['id'],
                                    at_s=warmup + (index + .5) * duration / len(order), measured=True))
    trace['events'].sort(key=lambda e: (e['at_s'], e['id']))
    if phases is not None:
        for event in trace['events']:
            phase = phase_of(phases, event['at_s'])
            event.update(phase=phase['phase'], phase_rate=phase['rate_label'])
    trace.update(experiment='matched real answers: synthetic background, real scored target answers',
                 target_count=len(order), target_arrival_definition='one job per question, evenly spaced over the window, seeded order')
    return trace


def make_runner(trace, questions, tokenizer, url, model):
    """Job callback for matched_serving.run_cell: background replayed, target answered for real."""
    by_id = {q['id']: q for q in questions}

    def runner(event, n, budget, remaining_s):
        if event['stream'] == 'background':
            body = dict(requests_for(trace, event, 1, budget)[0], model=model, seed=trace['seed'])
            try:
                tokens = json.loads(http(url + '/v1/completions', body, max(.001, remaining_s))).get('usage', {}).get('completion_tokens')
                if tokens != body['max_tokens']:
                    raise ValueError(f'token replay mismatch: requested {body["max_tokens"]}, observed {tokens}')
                return dict(status='completed', completion_tokens=tokens)
            except (TimeoutError, socket.timeout) as exc:
                return dict(status='timeout', error=str(exc))
        started = time.monotonic()

        def sample(k):
            seed = int(hashlib.sha256(f'{trace["seed"]}:{event["question_id"]}:{k}'.encode()).hexdigest()[:8], 16) % (2 ** 31)
            send = lambda body, timeout: json.loads(http(url + '/v1/completions', dict(body, model=model), timeout))
            return generate_sample(by_id[event['question_id']], budget, seed, tokenizer, send,
                                   max(0, remaining_s - (time.monotonic() - started)))
        with ThreadPoolExecutor(max_workers=n) as pool:
            parts = list(pool.map(sample, range(n)))
        for part in parts:  # keep the answer text and timings, drop the bulky token-id copies
            for stage in part['stages']:
                stage['request']['prompt'] = f'<{len(stage["request"]["prompt"])} prompt tokens>'
                for choice in stage.get('response', {}).get('choices', []):
                    choice.pop('token_ids', None)
                    choice.pop('prompt_token_ids', None)
        return dict(aggregate(parts), parts=parts, n=n, budget=budget,
                    completion_tokens=sum(p['completion_tokens'] for p in parts))
    return runner


def policy_runner(runner, policy, sampler, seed):
    """Let a policy set the action of each target job from the newest server sample of the cell."""
    def choose(event, n, budget, remaining_s):
        if event['stream'] == 'background':
            return runner(event, n, budget, remaining_s)
        signal = read_signal(sampler.server, time.monotonic() - sampler.t0)
        (n, budget), reason = policy.choose(signal, seed, event['question_id'])
        return dict(runner(event, n, budget, remaining_s), policy=policy.name, policy_reason=reason, policy_signal=signal)
    return choose


def context(args):
    """Everything a run needs from its inputs. `run` and `resume` both build it, so a resumed run reads the same protocol."""
    calibration = json.loads(Path(args.calibration).read_text())
    if calibration.get('status') != 'complete':
        raise SystemExit(f'calibration is not complete ({calibration.get("status")}); refusing to run without frozen rates')
    plan = json.loads(Path(args.plan).read_text())
    questions = plan['questions']
    rates = dict(light=calibration['light_rate_rps'], contended=calibration['contended_rate_rps'])
    # The target schedule always follows the calibrated contended rate, so it is the same for any background rate.
    target_rate = args.target_fraction * rates['contended']
    schedules = {}
    if getattr(args, 'schedules', None):
        if args.rates:
            raise SystemExit('--schedules uses the frozen light and contended rates and cannot be combined with --rates')
        if len(set(args.schedules)) != len(args.schedules):
            raise SystemExit('--schedules must be distinct')
        duration = len(questions) / target_rate
        try:
            schedules = {kind: schedule_phases(kind, rates, args.warmup, duration, args.phase_s) for kind in args.schedules}
        except ValueError as exc:
            raise SystemExit(str(exc))
    if args.rates:
        rates = {label: float(value) for label, value in (item.split('=') for item in args.rates)}
        if len(rates) != len(args.rates) or min(rates.values()) <= 0:
            raise SystemExit('--rates needs distinct labels and positive rates')
    lengths = json.loads(Path(args.lengths).read_text())
    chosen = {q['id'] for q in questions}
    policy_specs, policies = load_policies(args.policies) if args.policies else ([], [])
    actions = [tuple(map(int, a.split(':'))) for a in args.actions]
    if not actions and not policies:
        sys.exit('nothing to run: give at least one action or a policy file')
    return dict(url=args.url.rstrip('/'), calibration=calibration, plan=plan, questions=questions, budget=calibration['reference_budget'],
                targets=calibration['latency_targets_s'], rates=rates, target_rate=target_rate, duration=len(questions) / target_rate,
                lengths=lengths, background=[p for p in json.loads(Path(args.prompts).read_text()) if p['id'] not in chosen],
                actions=actions, policy_specs=policy_specs, by_name={p.name: p for p in policies}, schedules=schedules)


def cell_name(position, item, by_name):
    policy = by_name.get(item) if isinstance(item, str) else None
    return f'cell-{position}-P-{policy.name}' if policy else f'cell-{position}-N{item[0]}-B{item[1]}'


def server_process_start(url):
    """Start time of the server process from its /metrics, or None when it does not export one. Raises when the server is unreachable."""
    match = re.search(r'^process_start_time_seconds\s+(\S+)', http(url + '/metrics', timeout=10), re.M)
    return float(match.group(1)) if match else None


def run(args):
    ctx = context(args)
    hardware = json.loads(Path(args.hardware_json).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'plan.json', ctx['plan'])
    # Persist every trace and cell order before any outcome exists.
    blocks = []
    for seed in args.seeds:
        # A block runs at one constant rate, or under one time-varying schedule when --schedules is given.
        labels = list(ctx['schedules'] or ctx['rates'])
        random.Random(f'{seed}:rates').shuffle(labels)
        for label in labels:
            trace = make_trace(ctx['questions'], ctx['background'], ctx['lengths'][str(ctx['budget'])], ctx['budget'], seed, ctx['rates'].get(label),
                               args.warmup, ctx['duration'], ctx['schedules'].get(label))
            order = cell_order(ctx['actions'] + list(ctx['by_name']), f'{seed}:{label}:order')  # a policy cell is listed by its name
            name = f'seed-{seed}-{label}'
            (out / name).mkdir()
            write_json(out / name / 'trace.json', trace)
            write_json(out / name / 'order.json', order)
            blocks.append((name, seed, label, trace, order))
    manifest = dict(
        experiment='matched real-answer comparison of fixed actions' + (' and per-job policies' if ctx['by_name'] else '')
        + '; synthetic background, real target answers' + ('' if ctx['by_name'] else '; no router'),
        status='running', configuration={k: v for k, v in vars(args).items() if k != 'fn'},
        hardware_user_supplied=hardware, environment=environment(ctx['url']), plan_sha256=digest(ctx['plan']),
        calibration_sha256=digest(ctx['calibration']), calibration_source=str(Path(args.calibration).resolve()),
        rates_rps=ctx['rates'], target_rate_rps=ctx['target_rate'], window_s=ctx['duration'], reference_budget=ctx['budget'],
        latency_targets_s=ctx['targets'], questions=len(ctx['questions']), background_prompts=len(ctx['background']),
        lengths_sha256=digest(ctx['lengths']), blocks=[name for name, *_ in blocks], completed=[], skipped=[], started_unix_s=time.time(),
        policies=ctx['policy_specs'], schedules=ctx['schedules'], segments=[], discarded=[], current_cell=None,
        control_definition='(0,0): background only; target jobs retained but not offered',
        segment_definition='one harness process against one server process; a block resumed in a later segment first repeats its control')
    write_json(out / 'manifest.json', manifest)
    execute(args, ctx, out, manifest, blocks, 'start')


def resume(args):
    """Continue an interrupted run at its next unfinished cell; completed cells are never touched or rerun."""
    out = Path(args.out)
    manifest = json.loads((out / 'manifest.json').read_text())
    if manifest['status'] == 'complete':
        sys.exit(f'{out} is already complete')
    # Settings added after the run started take their defaults; the run's own settings win; only the URL and deadline may change.
    settings = {a.dest: a.default for a in build_parser()[1]['run']._actions if a.dest != 'help'}
    settings.update(manifest['configuration'])
    settings['deadline_epoch'] = args.deadline_epoch
    if args.url:
        settings['url'] = args.url
    run_args = argparse.Namespace(**settings)
    ctx = context(run_args)
    # The protocol must be the one the run started with: same inputs, same window, same rates, same policies.
    stored = json.loads((out / 'plan.json').read_text())
    checks = dict(plan_sha256=(digest(ctx['plan']), digest(stored), manifest['plan_sha256']),
                  calibration_sha256=(digest(ctx['calibration']), manifest['calibration_sha256']),
                  lengths_sha256=(digest(ctx['lengths']), manifest['lengths_sha256']), rates_rps=(ctx['rates'], manifest['rates_rps']),
                  window_s=(ctx['duration'], manifest['window_s']), policies=(ctx['policy_specs'], manifest['policies']),
                  schedules=(ctx['schedules'], manifest.get('schedules', {})))
    changed = [key for key, values in checks.items() if any(v != values[0] for v in values[1:])]
    if changed:
        sys.exit(f'inputs changed since the run started ({", ".join(changed)}); refusing to resume')
    commit = environment(ctx['url'])['git_commit']
    if commit != manifest['environment']['git_commit'] and not args.allow_new_commit:
        sys.exit(f'code is at {commit}, the run started at {manifest["environment"]["git_commit"]}; pass --allow-new-commit to resume anyway')
    blocks = []
    for name in manifest['blocks']:
        trace = json.loads((out / name / 'trace.json').read_text())
        order = [item if isinstance(item, str) else tuple(item) for item in json.loads((out / name / 'order.json').read_text())]
        blocks.append((name, trace['seed'], name.split('-', 2)[2], trace, order))
    if 'segments' not in manifest:  # a run started before segments were recorded: its cells all belong to segment 0
        manifest['segments'] = [dict(index=0, kind='start', note='recorded before segments existed')]
    manifest.setdefault('discarded', [])
    manifest.setdefault('current_cell', None)
    manifest['status'], manifest['skipped'] = 'running', []
    execute(run_args, ctx, out, manifest, blocks, 'resume')


def discard_partial(out, name, cells, manifest, segment):
    """Move the files of any cell of a block that is not in its cells.json into discarded/, and record it. Partial cells are never scored."""
    kept = {f'{c["cell"]}{suffix}.jsonl' for c in cells for suffix in ('', '-server', '-gpu')}
    orphans = sorted(p for p in (out / name).glob('cell-*.jsonl') if p.name not in kept)
    current = manifest.get('current_cell')
    interrupted = current and current['block'] == name and current['cell'] not in {c['cell'] for c in cells}
    if not orphans and not interrupted:
        return
    target = out / name / 'discarded' / f'segment-{segment}'
    target.mkdir(parents=True, exist_ok=True)
    for path in orphans:
        path.rename(target / path.name)
    manifest['discarded'].append(dict(block=name, cell=current['cell'] if interrupted else None, files=[p.name for p in orphans],
                                      moved_to=str(target.relative_to(out)), found_in_segment=segment,
                                      reason='cell started but never recorded in cells.json (process stopped mid-cell)'))
    if interrupted:
        manifest['current_cell'] = None


def execute(args, ctx, out, manifest, blocks, kind):
    url, questions, budget, targets, rates, by_name = (ctx[k] for k in ('url', 'questions', 'budget', 'targets', 'rates', 'by_name'))
    tokenizer = load_tokenizer()
    idle_wait = lambda: wait_idle(url, args.idle_timeout, args.idle_stable)
    segment = len(manifest['segments'])
    manifest['segments'].append(dict(index=segment, kind=kind, started_unix_s=time.time(), pid=os.getpid(), server_process_start=None,
                                     git_commit=environment(url)['git_commit']))
    cell_bound = args.warmup + ctx['duration'] + args.timeout + 60  # longest a cell can take before its idle check
    previous = signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(143))  # a job limit or spot shutdown lands in the finally below
    try:
        server_start = manifest['segments'][segment]['server_process_start'] = server_process_start(url)
        write_json(out / 'manifest.json', manifest)
        for index, (name, seed, label, trace, order) in enumerate(blocks):
            if name in manifest['completed']:
                continue
            cells = json.loads((out / name / 'cells.json').read_text()) if (out / name / 'cells.json').exists() else []
            discard_partial(out, name, cells, manifest, segment)
            done = {c['cell'] for c in cells}
            pending = [(cell_name(position, item, by_name), item) for position, item in enumerate(order) if cell_name(position, item, by_name) not in done]
            if pending and cells and not any(item == (0, 0) for _, item in pending):
                # The block continues in a new process and its control already ran in an earlier one: repeat the control first,
                # so this segment has its own machine-state check. A control still pending runs in this segment anyway.
                pending.insert(0, (f'cell-s{segment}-control-N0-B0', (0, 0)))
            for cell, item in pending:
                if args.deadline_epoch and args.deadline_epoch - time.time() < cell_bound:
                    manifest['skipped'] = [n for n, *_ in blocks[index:] if n not in manifest['completed']]
                    manifest['status'] = 'partial_deadline'
                    return
                policy = by_name.get(item) if isinstance(item, str) else None
                n, b = (1, 0) if policy else item  # a policy cell offers every target job; the policy sets its action
                print(f'CELL_START {name} {cell}', flush=True)
                manifest['current_cell'] = dict(block=name, cell=cell, segment=segment, started_unix_s=time.time())
                write_json(out / 'manifest.json', manifest)
                idle_wait()
                if server_process_start(url) != server_start:
                    raise RuntimeError('the server process changed since this segment started; resume the run to start a new segment')
                sampler = Sampler(url, args.metrics_interval, gpu=not args.no_gpu)
                sampler.start()
                runner = make_runner(trace, questions, tokenizer, url, args.model)
                if policy:
                    runner = policy_runner(runner, policy, sampler, seed)
                rows = run_cell(trace, url, args.model, n, b, budget, timeout_s=args.timeout, drain_s=args.timeout,
                                max_inflight=args.max_inflight, job_runner=runner)
                sampler.stop()
                # A cell that ran across a server restart or crash is not a measurement of this protocol: never record it.
                if server_process_start(url) != server_start:
                    raise RuntimeError(f'the server process changed during {name} {cell}; the cell is discarded')
                write_rows(out / name / f'{cell}.jsonl', rows)
                write_rows(out / name / f'{cell}-server.jsonl', sampler.server)
                write_rows(out / name / f'{cell}-gpu.jsonl', sampler.device)
                seen = [s for s in sampler.server if TOKENS in s.get('metrics', {})]
                cells.append(dict(cell=cell, block=name, seed=seed, rate_label=label, background_rate_rps=rates.get(label),
                                  schedule=label if label in ctx['schedules'] else None,
                                  n=None if policy else n, budget=None if policy else b, control=n == 0,
                                  policy=policy.name if policy else None, segment=segment,
                                  cell_wall_s=seen[-1]['at_s'] - seen[0]['at_s'] if len(seen) > 1 else None,
                                  generation_tokens=seen[-1]['metrics'][TOKENS] - seen[0]['metrics'][TOKENS] if len(seen) > 1 else None,
                                  **summarize_cell(rows, sampler.server, sampler.device, trace, targets)))
                write_json(out / name / 'cells.json', cells)
                manifest['current_cell'] = None
                write_json(out / 'manifest.json', manifest)
                target = cells[-1]['streams']['target']
                power = cells[-1]['gpu']['power.draw']['mean']
                control = [c['gpu']['power.draw']['mean'] for c in cells if c['control'] and c.get('segment', 0) == segment]
                drift = power / control[0] - 1 if power is not None and control and control[0] else None
                print(f'CELL_COMPLETE {name} {cell} target={target["statuses"]} '
                      f'background_p95={cells[-1]["streams"]["background"]["latency_s"]["p95"]} power_w={power} '
                      f'power_vs_control={"-" if drift is None else f"{drift:+.1%}"}'
                      + (' POWER_WARNING' if drift is not None and abs(drift) > args.power_tolerance else ''), flush=True)
                idle_wait()
            manifest['completed'].append(name)
            write_json(out / 'manifest.json', manifest)
        manifest['status'] = 'complete'
    except BaseException as exc:
        interrupted = isinstance(exc, (KeyboardInterrupt, SystemExit))
        manifest.update(status='interrupted' if interrupted else 'failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)
        manifest['segments'][segment]['finished_unix_s'] = manifest['finished_unix_s'] = time.time()
        manifest['segments'][segment]['status'] = manifest['status']
        write_json(out / 'manifest.json', manifest)
        print(out / 'manifest.json', flush=True)


DEFAULT_POWER_TOLERANCE = .10  # relative gap in mean GPU power against the block control; set per run with --power-tolerance


def power_check(cells, tolerance, segments=None):
    """Each cell's mean GPU power against the background-only control of its block segment (scored cell dicts).

    The control carries the same background as every other cell of its block and keeps the GPU busy, so on one machine
    state their mean power is close; a larger gap means the machine ran at a different power state (power mode, heat,
    a changed limit) and latency comparisons across those cells are not like for like. Missing power is reported, never assumed.

    A segment without its own control (its cells ran before a stop, and the block's control after it) is compared with the
    block's control of a segment on the same server process, when `segments` (from the manifest) shows one; the row says so."""
    server = {s['index']: s.get('server_process_start') for s in segments or [] if 'index' in s}
    rows = []
    for cell in cells:
        if cell['n'] == 0:
            continue
        block, segment = cell['block'], cell.get('segment', 0)
        controls = [c for c in cells if c['block'] == block and c.get('segment', 0) == segment and c['n'] == 0]
        source = 'control of the same segment'
        if not controls and server.get(segment) is not None:
            controls = [c for c in cells if c['block'] == block and c['n'] == 0 and server.get(c.get('segment', 0)) == server[segment]]
            source = 'control of another segment on the same server process'
        powers = [c['gpu_power_mean_w'] for c in controls if c['gpu_power_mean_w'] is not None]
        reference = sum(powers) / len(powers) if powers and len(powers) == len(controls) else None
        power = cell['gpu_power_mean_w']
        gap = power / reference - 1 if power is not None and reference else None
        rows.append(dict(block=block, segment=segment, cell=cell['cell'], entry=cell['policy'] or f'{cell["n"]}:{cell["budget"]}',
                         power_w=power, control_power_w=reference, relative_gap=gap, reference=source if reference else None,
                         status='unknown' if gap is None else 'flagged' if abs(gap) > tolerance else 'ok'))
    flagged, unknown = [r for r in rows if r['status'] == 'flagged'], [r for r in rows if r['status'] == 'unknown']
    warnings = []
    if flagged:
        warnings.append(f'{len(flagged)} of {len(rows)} cells differ from their block control by more than {tolerance:.0%} in mean GPU power: '
                        + ', '.join(f'{r["block"]} {r["entry"]} {r["relative_gap"]:+.1%}' for r in flagged)
                        + '. Latency comparisons involving these cells mix machine power states.')
    if unknown:
        warnings.append(f'{len(unknown)} of {len(rows)} cells have no mean GPU power for the cell or its control: '
                        + ', '.join(f'{r["block"]} {r["entry"]}' for r in unknown) + '. Their power state is not verified.')
    return dict(tolerance=tolerance, rule='flag a cell whose mean GPU power differs from the control of its block segment by more than the tolerance (relative); '
                'a segment without a control uses the block control of a segment on the same server process',
                cells=rows, flagged=len(flagged), unknown=len(unknown), warnings=warnings)


def bootstrap(by_question, statistic, draws, seed):
    """Percentile interval from resampling questions with replacement (all seeds of a question move together)."""
    ids = sorted(by_question)
    rng = random.Random(seed)
    values = []
    for _ in range(draws):
        value = statistic([by_question[rng.choice(ids)] for _ in ids])
        if value is not None:
            values.append(value)
    values.sort()
    return [values[int(.025 * (len(values) - 1))], values[int(.975 * (len(values) - 1))]] if values else None


def ratio(groups, top, bottom):
    total = sum(r[bottom] for g in groups for r in g)
    return sum(r[top] for g in groups for r in g) / total if total else None


PAIRED_METRICS = (('accuracy', 'correct', 'job'),
                  ('correct_within_loose', 'correct_loose', 'job'), ('correct_within_tight', 'correct_tight', 'job'),
                  ('correct_within_loose_per_gpu_s', 'correct_loose', 'gpu_s'),
                  ('correct_within_tight_per_gpu_s', 'correct_tight', 'gpu_s'))


def paired_differences(rate, groups, draws, name, label):
    """Every pair of entries at one rate, first minus second, on the questions both have; the bootstrap resamples the same questions for both.

    `groups` maps an entry key to {question: [records]}; `name` turns a key into what the output calls the entry and `label` into the seed of its interval."""
    out = []
    keys = sorted(groups)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            shared = sorted(set(groups[a]) & set(groups[b]))
            paired = {q: [(groups[a][q], groups[b][q])] for q in shared}
            for key, top, bottom in PAIRED_METRICS:
                def gap(sample, top=top, bottom=bottom):
                    first = ratio([x for pair in sample for x, _ in pair], top, bottom)
                    second = ratio([y for pair in sample for _, y in pair], top, bottom)
                    return None if first is None or second is None else first - second
                out.append(dict(rate=rate, metric=key, action=name(a), minus=name(b), questions=len(shared),
                                value=gap(list(paired.values())), ci95=bootstrap(paired, gap, draws, f'{rate}:{label(a)}:{label(b)}:{key}')))
    return out


def compare(args):
    """Paired differences between entries of several scored runs that share questions, seeds and latency targets.

    A controller and its baseline mixture are often run separately on the same seeds, so their traces are identical but no single
    `summary.json` pairs them. Entries are named `N:B` or by policy; pairs whose seeds differ are listed but not compared."""
    runs = [Path(r) for r in args.runs]
    summaries = [json.loads((r / 'summary.json').read_text()) for r in runs]
    if len({json.dumps(s['latency_targets_s'], sort_keys=True) for s in summaries}) != 1:
        sys.exit('runs have different latency targets and cannot be compared')
    groups, origin = {}, {}
    for index, run in enumerate(runs):
        for r in (json.loads(line) for line in (run / 'scored.jsonl').read_text().splitlines()):
            key = r.get('policy') or f"{r['n']}:{r['budget']}"
            if origin.setdefault((r['rate'], key), index) != index:
                sys.exit(f'{key} at rate {r["rate"]} appears in more than one run')
            groups.setdefault(r['rate'], {}).setdefault(key, {}).setdefault(r['question'], []).append(r)
    entries, differences, unpaired = [], [], []
    for rate, by_key in sorted(groups.items()):
        seeds = {key: sorted({r['seed'] for g in by_question.values() for r in g}) for key, by_question in by_key.items()}
        for key, by_question in sorted(by_key.items()):
            everything = list(by_question.values())
            entries.append(dict(rate=rate, entry=key, run=runs[origin[(rate, key)]].name, jobs=sum(map(len, everything)), questions=len(by_question), seeds=seeds[key],
                                **{m: ratio(everything, top, bottom) for m, top, bottom in PAIRED_METRICS}))
        for key in sorted(by_key):
            others = sorted(k for k in by_key if seeds[k] != seeds[key])
            if others:
                unpaired.append(dict(rate=rate, entry=key, seeds=seeds[key], not_compared_with=others))
        for same in {tuple(v): [k for k in by_key if seeds[k] == v] for v in seeds.values()}.values():
            differences += paired_differences(rate, {k: by_key[k] for k in same}, args.draws, str, str)
    # Entries from different runs may have run at different machine power states: compare their mean cell power.
    power = {}
    for summary in summaries:
        for c in summary.get('cells', []):
            if c['n'] != 0 and c.get('gpu_power_mean_w') is not None:
                power.setdefault((c['rate'], c.get('policy') or f'{c["n"]}:{c["budget"]}'), []).append(c['gpu_power_mean_w'])
    power = {key: sum(v) / len(v) for key, v in power.items()}
    for entry in entries:
        entry['gpu_power_mean_w'] = power.get((entry['rate'], entry['entry']))
    tolerance = max((s.get('power_check') or {}).get('tolerance', DEFAULT_POWER_TOLERANCE) for s in summaries)
    warnings = []
    for d in differences:
        if d['metric'] != 'accuracy':
            continue
        a, b = power.get((d['rate'], d['action'])), power.get((d['rate'], d['minus']))
        if a is None or b is None:
            warnings.append(f'{d["rate"]}: {d["action"]} vs {d["minus"]}: mean GPU power unknown for at least one entry')
        elif abs(a / b - 1) > tolerance:
            warnings.append(f'{d["rate"]}: {d["action"]} ran at {a:.1f} W and {d["minus"]} at {b:.1f} W, a gap above {tolerance:.0%}; '
                            'their latency metrics mix machine power states')
    warnings += [w for s, r in zip(summaries, runs) for w in (f'{r.name}: {x}' for x in s.get('warnings', []))]
    write_json(Path(args.output), dict(
        runs=[str(r) for r in runs], latency_targets_s=summaries[0]['latency_targets_s'], entries=entries, paired_differences=differences,
        not_compared=unpaired, power_tolerance=tolerance, warnings=warnings,
        definitions=dict(ci95=f'percentile bootstrap, {args.draws} resamples of questions with replacement; seeds of a question are kept together; '
                         'paired differences resample the same questions for both entries and are computed only between entries run on the same seeds',
                         caveat='entries from different runs share questions, seeds and background traces but were run at different times on the same machine')))
    print(args.output)


SHARED = ('experiment', 'configuration', 'hardware_user_supplied', 'plan_sha256', 'calibration_sha256', 'calibration_source', 'rates_rps',
          'target_rate_rps', 'window_s', 'reference_budget', 'latency_targets_s', 'questions', 'background_prompts', 'lengths_sha256', 'control_definition')
MUST_MATCH = ('plan_sha256', 'calibration_sha256', 'rates_rps', 'target_rate_rps', 'window_s', 'latency_targets_s', 'questions', 'lengths_sha256')


def assemble(args):
    """Gather complete blocks of several runs of one protocol into a new directory that `score` reads like a single run.

    Use it when blocks had to be repeated (a block run while the machine was in a different state, say): the kept blocks of the
    original run and the repeated blocks of later runs are copied unchanged, and the manifest records where each block came from.
    Every source must share the plan, calibration, window and latency targets, and the rate of every block it contributes must have
    the same value in every source that ran it (a repeated block run with `--rates` carries only its own rate). Block names must be distinct."""
    out = Path(args.out)
    sources, blocks = {}, []
    for item in args.blocks:
        run, _, block = item.rpartition('/')
        if run not in sources:
            sources[run] = json.loads((Path(run) / 'manifest.json').read_text())
        if block not in sources[run]['completed']:
            sys.exit(f'{block} is not a complete block of {run}')
        if block in [b for _, b in blocks]:
            sys.exit(f'block {block} given twice')
        blocks.append((run, block))
    first = next(iter(sources.values()))
    for run, manifest in sources.items():
        for key in MUST_MATCH:
            if key != 'rates_rps' and manifest.get(key) != first.get(key):
                sys.exit(f'{run} differs from {next(iter(sources))} in {key}; blocks of different protocols cannot be assembled')
    rates, schedules = {}, {}
    for run, block in blocks:
        label = block.split('-', 2)[2]  # blocks are named seed-<seed>-<rate label or schedule>
        if label in (sources[run].get('schedules') or {}):
            # A schedule block needs the same phases, and so the same frozen rates, in every source that ran it.
            value = sources[run]['schedules'][label]
            if schedules.setdefault(label, value) != value or sources[run]['rates_rps'] != first['rates_rps']:
                sys.exit(f'{run} differs from the other runs in the {label} schedule; blocks of different protocols cannot be assembled')
            continue
        value = sources[run]['rates_rps'].get(label)
        if value is None:
            sys.exit(f'{run} has no rate {label} for block {block}')
        if rates.setdefault(label, value) != value:
            sys.exit(f'{run} differs from the other runs in rates_rps for {label}; blocks of different protocols cannot be assembled')
    out.mkdir(parents=True, exist_ok=False)
    for run, block in blocks:
        shutil.copytree(Path(run) / block, out / block)
    shutil.copy(Path(next(iter(sources))) / 'plan.json', out / 'plan.json')
    write_json(out / 'manifest.json', dict(
        {k: first.get(k) for k in SHARED}, rates_rps=dict(first['rates_rps']) if schedules else rates, schedules=schedules, status='complete', completed=[b for _, b in blocks], skipped=[],
        experiment=first['experiment'] + '; assembled from the complete blocks of several runs',
        environment=dict(assembled_at_unix_s=time.time(), **{k: v for k, v in environment('').items() if not k.startswith('server')}),
        assembled_from=[dict(run=run, block=block, git_commit=sources[run]['environment'].get('git_commit'),
                             started_unix_s=sources[run].get('started_unix_s'), hardware=sources[run].get('hardware_user_supplied'))
                        for run, block in blocks]))
    print(out / 'manifest.json')


def score(args, verifier=None):
    if verifier is None:
        import math_verify  # noqa: F401  require the project's symbolic grader; no silent string fallback
        verifier = is_correct
    out = Path(args.out)
    gold = json.loads(Path(args.gold).read_text())
    manifest = json.loads((out / 'manifest.json').read_text())
    tight, loose = min(manifest['latency_targets_s']), max(manifest['latency_targets_s'])
    records, cells = [], []
    for name in manifest['completed']:  # only blocks whose every cell and idle check finished
        for cell in json.loads((out / name / 'cells.json').read_text()):
            rows = [json.loads(line) for line in (out / name / f'{cell["cell"]}.jsonl').read_text().splitlines()]
            jobs = [r for r in rows if r['stream'] == 'target' and r['measured'] and r['status'] != 'not_offered_control']
            background = [r for r in rows if r['stream'] == 'background' and r['measured']]
            for r in jobs:
                done = r['status'] == 'completed'
                correct = done and r.get('pred') is not None and bool(verifier(gold[r['question_id']], r['pred']))
                tokens = r.get('completion_tokens') or 0
                records.append(dict(
                    rate=cell['rate_label'], n=cell['n'], budget=cell['budget'], policy=cell.get('policy'),
                    chosen=f'{r["n"]}:{r["budget"]}' if 'n' in r else None, policy_reason=r.get('policy_reason'),
                    seed=cell['seed'], question=r['question_id'], phase=r.get('phase'), phase_rate=r.get('phase_rate'),
                    status=r['status'], job=1, correct=int(correct), latency_s=r.get('latency_s'),
                    correct_tight=int(correct and r['latency_s'] <= tight), correct_loose=int(correct and r['latency_s'] <= loose),
                    tokens=tokens,
                    # Share of this cell's GPU time attributed to the job by its share of generated tokens.
                    gpu_s=cell['cell_wall_s'] * tokens / cell['generation_tokens'] if cell['generation_tokens'] else 0.))
            good = lambda limit, rows=background: sum(r['status'] == 'completed' and r['latency_s'] <= limit for r in rows)
            phased = {}
            for r in background:
                if r.get('phase_rate'):
                    phased.setdefault(r['phase_rate'], []).append(r)
            mine = [r for r in records if (r['rate'], r['n'], r['budget'], r['policy'], r['seed']) == (
                cell['rate_label'], cell['n'], cell['budget'], cell.get('policy'), cell['seed'])] if jobs else []
            cells.append(dict(
                block=name, cell=cell['cell'], seed=cell['seed'], rate=cell['rate_label'], n=cell['n'], budget=cell['budget'],
                policy=cell.get('policy'),
                target_jobs=len(jobs), target_statuses={s: sum(r['status'] == s for r in jobs) for s in ('completed', 'no_answer') + FAILED},
                target_correct=sum(r['correct'] for r in mine), target_correct_tight=sum(r['correct_tight'] for r in mine),
                target_correct_loose=sum(r['correct_loose'] for r in mine),
                target_latency_p95_s=percentile([r['latency_s'] for r in jobs if r['status'] in ('completed', 'no_answer')]),
                target_tokens=sum(r['tokens'] for r in mine), target_gpu_s=sum(r['gpu_s'] for r in mine),
                background_jobs=len(background), background_within_tight=good(tight), background_within_loose=good(loose),
                background_p95_s=cell['streams']['background']['latency_s']['p95'],
                # Under a schedule: background requests by the load level of the phase they arrived in (p95 over completed requests).
                background_by_phase={k: dict(jobs=len(v), within_tight=good(tight, v), within_loose=good(loose, v),
                                             p95_s=percentile([r['latency_s'] for r in v if r['status'] == 'completed']))
                                     for k, v in sorted(phased.items())} or None,
                cell_wall_s=cell['cell_wall_s'], generation_tokens=cell['generation_tokens'],
                # Everything the server delivered on time, per second of GPU wall time in the cell.
                goodput_tight_per_gpu_s=(sum(r['correct_tight'] for r in mine) + good(tight)) / cell['cell_wall_s'] if cell['cell_wall_s'] else None,
                goodput_loose_per_gpu_s=(sum(r['correct_loose'] for r in mine) + good(loose)) / cell['cell_wall_s'] if cell['cell_wall_s'] else None,
                waiting_max=cell['server']['waiting']['max'], kv_cache_max=cell['server']['kv_cache']['max'],
                preemptions=cell['server']['counters']['preemptions'],
                # Mean GPU power in the cell: a drop against other cells means the machine itself ran slower.
                gpu_power_mean_w=((cell.get('gpu') or {}).get('power.draw') or {}).get('mean'),
                gpu_samples=(cell.get('gpu') or {}).get('samples'),
                gpu_power_limit_w=((cell.get('gpu') or {}).get('power.limit') or {}).get('mean'),
                gpu_sm_clock_mean_mhz=((cell.get('gpu') or {}).get('clocks.sm') or {}).get('mean'),
                gpu_throttle=(cell.get('gpu') or {}).get('throttle'), segment=cell.get('segment', 0)))
    def entries(subset, rate, seed_label):
        """Entries and paired differences of the records of one rate, or of one load level of a schedule. `seed_label` seeds the intervals."""
        groups = {}
        for r in subset:
            # Fixed actions sort first, by (N, B); policies follow by name.
            groups.setdefault((r['policy'] or '', r['n'] or 0, r['budget'] or 0), {}).setdefault(r['question'], []).append(r)
        found = []
        for (policy, n, b), by_question in sorted(groups.items()):
            everything = list(by_question.values())
            entry = dict(rate=rate, n=None if policy else n, budget=None if policy else b, policy=policy or None, jobs=sum(len(g) for g in everything), questions=len(by_question),
                         seeds=sorted({r['seed'] for g in everything for r in g}),
                         statuses={s: sum(r['status'] == s for g in everything for r in g) for s in ('completed', 'no_answer') + FAILED},
                         target_gpu_s=sum(r['gpu_s'] for g in everything for r in g))
            powers = [c['gpu_power_mean_w'] for c in cells if c['rate'] == rate and (c['policy'] or '', c['n'] or 0, c['budget'] or 0) == (policy, n, b)
                      and c['gpu_power_mean_w'] is not None]
            entry['gpu_power_mean_w'] = sum(powers) / len(powers) if powers else None  # mean over the entry's cells that have power samples
            if policy:  # what the policy actually did, and why
                done = [r for g in everything for r in g]
                entry['chosen'] = {c: sum(str(r['chosen']) == c for r in done) for c in sorted({str(r['chosen']) for r in done})}
                entry['reasons'] = {c: sum(str(r['policy_reason']) == c for r in done) for c in sorted({str(r['policy_reason']) for r in done})}
            for key, top, bottom in (('accuracy', 'correct', 'job'), ('correct_within_tight', 'correct_tight', 'job'),
                                     ('correct_within_loose', 'correct_loose', 'job'),
                                     ('correct_within_tight_per_gpu_s', 'correct_tight', 'gpu_s'),
                                     ('correct_within_loose_per_gpu_s', 'correct_loose', 'gpu_s')):
                entry[key] = dict(value=ratio(everything, top, bottom),
                                  ci95=bootstrap(by_question, lambda g, t=top, b_=bottom: ratio(g, t, b_), args.draws,
                                                 f'{seed_label}:{policy}:{n}:{b}:{key}' if policy else f'{seed_label}:{n}:{b}:{key}'))
            found.append(entry)
        name = lambda k: k[0] or list(k[1:])
        label = lambda k: str(k[1:] if not k[0] else k)
        gaps = paired_differences(seed_label, groups, args.draws, name, label)
        for gap in gaps:
            gap['rate'] = rate
        return found, gaps

    actions, differences, phase_actions, phase_differences = [], [], [], []
    # Constant rates sort by rate; schedules, which have no single rate, follow by name.
    for rate in sorted({r['rate'] for r in records}, key=lambda label: (manifest['rates_rps'].get(label, float('inf')), label)):
        subset = [r for r in records if r['rate'] == rate]
        found, gaps = entries(subset, rate, rate)
        actions += found
        differences += gaps
        for level in sorted({r['phase_rate'] for r in subset if r['phase_rate']}, key=lambda label: (manifest['rates_rps'].get(label, float('inf')), label)):
            found, gaps = entries([r for r in subset if r['phase_rate'] == level], rate, f'{rate}:{level}')
            phase_actions += [dict(e, phase_rate=level) for e in found]
            phase_differences += [dict(d, phase_rate=level) for d in gaps]
    tolerance = (manifest.get('configuration') or {}).get('power_tolerance')
    check = power_check(cells, DEFAULT_POWER_TOLERANCE if tolerance is None else tolerance, manifest.get('segments'))
    check['tolerance_source'] = 'default; the run predates the setting' if tolerance is None else 'run configuration'
    write_rows(out / 'scored.jsonl', records)
    write_json(out / 'summary.json', dict(
        status=manifest['status'], blocks_scored=manifest['completed'], blocks_skipped=manifest.get('skipped', []),
        latency_targets_s=dict(tight=tight, loose=loose), cells=cells, actions=actions, paired_differences=differences,
        power_check=check, warnings=check['warnings'],
        phases=dict(actions=phase_actions, paired_differences=phase_differences, schedules=manifest.get('schedules', {}),
                    definition='target jobs of each schedule grouped by the load level of the phase they arrived in; a job may finish in the next phase; '
                               'GPU seconds are attributed over the whole cell, so per-GPU-second values by phase are approximate') if phase_actions else None,
        definitions=dict(
            correct_within_target='target job completed, majority answer verified correct, and job latency (scheduled arrival to last sample) within the target; every scheduled job is in the denominator',
            per_gpu_s='correct-within-target jobs divided by the GPU seconds attributed to target jobs: cell wall time (one GPU) times the job share of all tokens generated in the cell; prompt processing is not attributed separately',
            goodput_per_gpu_s='per cell: (target jobs correct within target + background requests completed within target) / cell wall seconds',
            ci95=f'percentile bootstrap, {args.draws} resamples of questions with replacement; seeds of a question are kept together; paired differences resample the same questions for both actions'),
        caveat='one machine; ' + ('fixed actions and frozen per-job policies' if manifest.get('policies') else 'fixed actions only, no router')
        + '; intervals reflect question sampling and seed-level noise only as far as the seeds run'))
    print(out / 'summary.json')


def build_parser():
    """The command line, and each subcommand's parser by name (resume reads the run defaults from it)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    commands = {}
    p = commands['prepare'] = sub.add_parser('prepare')
    for name in ('prompts', 'labels', 'out'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--questions', type=int, default=50)
    p.add_argument('--seed', type=int, default=20261115)
    p.add_argument('--proportional', action='store_true', help='keep the level mix of the full question set instead of equal counts')
    p.set_defaults(fn=prepare)
    p = commands['run'] = sub.add_parser('run')
    for name in ('plan', 'calibration', 'lengths', 'prompts', 'hardware-json', 'out'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--url', default='http://127.0.0.1:8001')
    p.add_argument('--model', default='m')
    p.add_argument('--actions', nargs='*', default=['1:0', '4:0', '1:2048', '4:2048'],
                   help='fixed actions N:B; may be empty when --policies is given, so a baseline policy can run on its own')
    p.add_argument('--seeds', type=int, nargs='+', default=[20261116, 20261117, 20261118])
    p.add_argument('--target-fraction', type=float, default=.05, help='target job rate as a fraction of the contended rate')
    p.add_argument('--policies', default=None, help='JSON list of per-job policies (load_policy.py); each gets its own cell')
    p.add_argument('--rates', nargs='+', default=None, metavar='LABEL=RPS',
                   help='background rates to run instead of the frozen pair, for example between them')
    p.add_argument('--schedules', nargs='+', default=None, choices=SCHEDULES,
                   help='time-varying background: one block per schedule instead of one per constant rate (frozen light and contended rates)')
    p.add_argument('--phase-s', type=float, default=300, help='phase length of the square schedule, seconds')
    p.add_argument('--warmup', type=float, default=60)
    p.add_argument('--timeout', type=float, default=300, help='per-job observation limit, also the drain limit')
    p.add_argument('--idle-timeout', type=float, default=900)
    p.add_argument('--idle-stable', type=float, default=1)
    p.add_argument('--max-inflight', type=int, default=512)
    p.add_argument('--metrics-interval', type=float, default=1)
    p.add_argument('--deadline-epoch', type=float, default=None, help='do not start a cell that cannot finish before this time; resume later')
    p.add_argument('--no-gpu', action='store_true')
    p.add_argument('--power-tolerance', type=float, default=DEFAULT_POWER_TOLERANCE,
                   help='flag a cell whose mean GPU power differs from its block control by more than this fraction')
    p.set_defaults(fn=run)
    p = commands['resume'] = sub.add_parser('resume', help='continue an interrupted run at its next unfinished cell')
    p.add_argument('--out', required=True, help='the run directory to continue')
    p.add_argument('--url', default=None, help='server URL, when it differs from the one the run started with')
    p.add_argument('--deadline-epoch', type=float, default=None, help='do not start a cell that cannot finish before this time')
    p.add_argument('--allow-new-commit', action='store_true', help='resume with code at a different commit than the run started with (recorded)')
    p.set_defaults(fn=resume)
    p = commands['assemble'] = sub.add_parser('assemble', help='gather complete blocks of several runs of one protocol into a new directory for score')
    p.add_argument('--blocks', nargs='+', required=True, metavar='RUN_DIR/BLOCK')
    p.add_argument('--out', required=True)
    p.set_defaults(fn=assemble)
    p = commands['score'] = sub.add_parser('score')
    p.add_argument('--out', required=True)
    p.add_argument('--gold', required=True)
    p.add_argument('--draws', type=int, default=2000)
    p.set_defaults(fn=score)
    p = commands['compare'] = sub.add_parser('compare', help='paired differences between entries of several scored runs on the same seeds')
    p.add_argument('--runs', nargs='+', required=True, metavar='DIR')
    p.add_argument('--output', required=True, help='JSON file to write')
    p.add_argument('--draws', type=int, default=2000)
    p.set_defaults(fn=compare)
    return parser, commands


def main(argv=None):
    args = build_parser()[0].parse_args(argv)
    args.fn(args)


if __name__ == '__main__':
    sys.exit(main())
