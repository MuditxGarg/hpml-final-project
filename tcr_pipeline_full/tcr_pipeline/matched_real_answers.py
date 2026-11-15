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
from pathlib import Path
import random
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


def make_trace(questions, background_prompts, pool, budget, seed, rate, warmup, duration):
    """Poisson synthetic background plus one evenly spaced target job per question, in seeded order.

    The target schedule depends only on the seed, so it is identical at every background rate.
    """
    trace = generate_trace({str(budget): pool}, background_prompts, seed, rate, 0, warmup, duration, 1)
    order = list(questions)
    random.Random(f'{seed}:targets').shuffle(order)
    for index, question in enumerate(order):
        trace['events'].append(dict(id=f'target-{index}', stream='target', question_id=question['id'],
                                    at_s=warmup + (index + .5) * duration / len(order), measured=True))
    trace['events'].sort(key=lambda e: (e['at_s'], e['id']))
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


def run(args):
    url = args.url.rstrip('/')
    calibration = json.loads(Path(args.calibration).read_text())
    if calibration.get('status') != 'complete':
        raise SystemExit(f'calibration is not complete ({calibration.get("status")}); refusing to run without frozen rates')
    plan = json.loads(Path(args.plan).read_text())
    questions = plan['questions']
    budget, targets = calibration['reference_budget'], calibration['latency_targets_s']
    rates = dict(light=calibration['light_rate_rps'], contended=calibration['contended_rate_rps'])
    # The target schedule always follows the calibrated contended rate, so it is the same for any background rate.
    target_rate = args.target_fraction * rates['contended']
    if args.rates:
        rates = {label: float(value) for label, value in (item.split('=') for item in args.rates)}
        if len(rates) != len(args.rates) or min(rates.values()) <= 0:
            raise SystemExit('--rates needs distinct labels and positive rates')
    duration = len(questions) / target_rate
    lengths = json.loads(Path(args.lengths).read_text())
    chosen = {q['id'] for q in questions}
    background = [p for p in json.loads(Path(args.prompts).read_text()) if p['id'] not in chosen]
    actions = [tuple(map(int, a.split(':'))) for a in args.actions]
    policy_specs, policies = load_policies(args.policies) if args.policies else ([], [])
    by_name = {p.name: p for p in policies}
    hardware = json.loads(Path(args.hardware_json).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'plan.json', plan)
    tokenizer = load_tokenizer()
    idle_wait = lambda: wait_idle(url, args.idle_timeout, args.idle_stable)
    # Persist every trace and cell order before any outcome exists.
    blocks = []
    for seed in args.seeds:
        labels = list(rates)
        random.Random(f'{seed}:rates').shuffle(labels)
        for label in labels:
            trace = make_trace(questions, background, lengths[str(budget)], budget, seed, rates[label], args.warmup, duration)
            order = cell_order(actions + list(by_name), f'{seed}:{label}:order')  # a policy cell is listed by its name
            name = f'seed-{seed}-{label}'
            (out / name).mkdir()
            write_json(out / name / 'trace.json', trace)
            write_json(out / name / 'order.json', order)
            blocks.append((name, seed, label, trace, order))
    block_bound = (len(actions) + len(policies) + 1) * (args.warmup + duration + args.timeout + 60)
    manifest = dict(
        experiment='matched real-answer comparison of fixed actions' + (' and per-job policies' if policies else '')
        + '; synthetic background, real target answers' + ('' if policies else '; no router'),
        status='running', configuration={k: v for k, v in vars(args).items() if k != 'fn'},
        hardware_user_supplied=hardware, environment=environment(url), plan_sha256=digest(plan),
        calibration_sha256=digest(calibration), calibration_source=str(Path(args.calibration).resolve()),
        rates_rps=rates, target_rate_rps=target_rate, window_s=duration, reference_budget=budget, latency_targets_s=targets,
        questions=len(questions), background_prompts=len(background), lengths_sha256=digest(lengths),
        blocks=[name for name, *_ in blocks], completed=[], skipped=[], started_unix_s=time.time(),
        policies=policy_specs,
        control_definition='(0,0): background only; target jobs retained but not offered')
    write_json(out / 'manifest.json', manifest)
    try:
        for index, (name, seed, label, trace, order) in enumerate(blocks):
            if args.deadline_epoch and args.deadline_epoch - time.time() < block_bound:
                manifest['skipped'] = [n for n, *_ in blocks[index:]]
                manifest['status'] = 'partial_deadline'
                break
            cells = []
            for position, item in enumerate(order):
                policy = by_name.get(item) if isinstance(item, str) else None
                n, b = (1, 0) if policy else item  # a policy cell offers every target job; the policy sets its action
                cell = f'cell-{position}-P-{policy.name}' if policy else f'cell-{position}-N{n}-B{b}'
                print(f'CELL_START {name} {cell}', flush=True)
                idle_wait()
                sampler = Sampler(url, args.metrics_interval, gpu=not args.no_gpu)
                sampler.start()
                runner = make_runner(trace, questions, tokenizer, url, args.model)
                if policy:
                    runner = policy_runner(runner, policy, sampler, seed)
                rows = run_cell(trace, url, args.model, n, b, budget, timeout_s=args.timeout, drain_s=args.timeout,
                                max_inflight=args.max_inflight, job_runner=runner)
                sampler.stop()
                write_rows(out / name / f'{cell}.jsonl', rows)
                write_rows(out / name / f'{cell}-server.jsonl', sampler.server)
                write_rows(out / name / f'{cell}-gpu.jsonl', sampler.device)
                seen = [s for s in sampler.server if TOKENS in s.get('metrics', {})]
                cells.append(dict(cell=cell, block=name, seed=seed, rate_label=label, background_rate_rps=rates[label],
                                  n=None if policy else n, budget=None if policy else b, control=n == 0,
                                  policy=policy.name if policy else None,
                                  cell_wall_s=seen[-1]['at_s'] - seen[0]['at_s'] if len(seen) > 1 else None,
                                  generation_tokens=seen[-1]['metrics'][TOKENS] - seen[0]['metrics'][TOKENS] if len(seen) > 1 else None,
                                  **summarize_cell(rows, sampler.server, sampler.device, trace, targets)))
                write_json(out / name / 'cells.json', cells)
                target = cells[-1]['streams']['target']
                print(f'CELL_COMPLETE {name} {cell} target={target["statuses"]} '
                      f'background_p95={cells[-1]["streams"]["background"]["latency_s"]["p95"]}', flush=True)
                idle_wait()
            manifest['completed'].append(name)
            write_json(out / 'manifest.json', manifest)
        else:
            manifest['status'] = 'complete'
    except Exception as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        manifest['finished_unix_s'] = time.time()
        write_json(out / 'manifest.json', manifest)
    print(out / 'manifest.json')


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
                    seed=cell['seed'], question=r['question_id'],
                    status=r['status'], job=1, correct=int(correct), latency_s=r.get('latency_s'),
                    correct_tight=int(correct and r['latency_s'] <= tight), correct_loose=int(correct and r['latency_s'] <= loose),
                    tokens=tokens,
                    # Share of this cell's GPU time attributed to the job by its share of generated tokens.
                    gpu_s=cell['cell_wall_s'] * tokens / cell['generation_tokens'] if cell['generation_tokens'] else 0.))
            good = lambda limit: sum(r['status'] == 'completed' and r['latency_s'] <= limit for r in background)
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
                cell_wall_s=cell['cell_wall_s'], generation_tokens=cell['generation_tokens'],
                # Everything the server delivered on time, per second of GPU wall time in the cell.
                goodput_tight_per_gpu_s=(sum(r['correct_tight'] for r in mine) + good(tight)) / cell['cell_wall_s'] if cell['cell_wall_s'] else None,
                goodput_loose_per_gpu_s=(sum(r['correct_loose'] for r in mine) + good(loose)) / cell['cell_wall_s'] if cell['cell_wall_s'] else None,
                waiting_max=cell['server']['waiting']['max'], kv_cache_max=cell['server']['kv_cache']['max'],
                preemptions=cell['server']['counters']['preemptions']))
    actions, differences = [], []
    for rate in sorted({r['rate'] for r in records}, key=lambda label: (manifest['rates_rps'][label], label)):
        groups = {}
        for r in records:
            if r['rate'] == rate:
                # Fixed actions sort first, by (N, B); policies follow by name.
                groups.setdefault((r['policy'] or '', r['n'] or 0, r['budget'] or 0), {}).setdefault(r['question'], []).append(r)
        for (policy, n, b), by_question in sorted(groups.items()):
            everything = list(by_question.values())
            entry = dict(rate=rate, n=None if policy else n, budget=None if policy else b, policy=policy or None, jobs=sum(len(g) for g in everything), questions=len(by_question),
                         seeds=sorted({r['seed'] for g in everything for r in g}),
                         statuses={s: sum(r['status'] == s for g in everything for r in g) for s in ('completed', 'no_answer') + FAILED},
                         target_gpu_s=sum(r['gpu_s'] for g in everything for r in g))
            if policy:  # what the policy actually did, and why
                done = [r for g in everything for r in g]
                entry['chosen'] = {c: sum(str(r['chosen']) == c for r in done) for c in sorted({str(r['chosen']) for r in done})}
                entry['reasons'] = {c: sum(str(r['policy_reason']) == c for r in done) for c in sorted({str(r['policy_reason']) for r in done})}
            for key, top, bottom in (('accuracy', 'correct', 'job'), ('correct_within_tight', 'correct_tight', 'job'),
                                     ('correct_within_loose', 'correct_loose', 'job'),
                                     ('correct_within_tight_per_gpu_s', 'correct_tight', 'gpu_s'),
                                     ('correct_within_loose_per_gpu_s', 'correct_loose', 'gpu_s')):
                entry[key] = dict(value=ratio(everything, top, bottom),
                                  ci95=bootstrap(by_question, lambda g, t=top, b_=bottom: ratio(g, t, b_), args.draws, f'{rate}:{policy}:{n}:{b}:{key}' if policy else f'{rate}:{n}:{b}:{key}'))
            actions.append(entry)
        keys = sorted(groups)
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                shared = sorted(set(groups[a]) & set(groups[b]))
                paired = {q: [(groups[a][q], groups[b][q])] for q in shared}
                for key, top, bottom in (('correct_within_loose', 'correct_loose', 'job'), ('correct_within_tight', 'correct_tight', 'job'),
                                         ('correct_within_loose_per_gpu_s', 'correct_loose', 'gpu_s'),
                                         ('correct_within_tight_per_gpu_s', 'correct_tight', 'gpu_s')):
                    def gap(sample, top=top, bottom=bottom):
                        first = ratio([x for pair in sample for x, _ in pair], top, bottom)
                        second = ratio([y for pair in sample for _, y in pair], top, bottom)
                        return None if first is None or second is None else first - second
                    name = lambda k: k[0] or list(k[1:])
                    differences.append(dict(rate=rate, metric=key, action=name(a), minus=name(b), questions=len(shared),
                                            value=gap(list(paired.values())),
                                            ci95=bootstrap(paired, gap, args.draws, f'{rate}:{a[1:] if not a[0] else a}:{b[1:] if not b[0] else b}:{key}')))
    write_rows(out / 'scored.jsonl', records)
    write_json(out / 'summary.json', dict(
        status=manifest['status'], blocks_scored=manifest['completed'], blocks_skipped=manifest.get('skipped', []),
        latency_targets_s=dict(tight=tight, loose=loose), cells=cells, actions=actions, paired_differences=differences,
        definitions=dict(
            correct_within_target='target job completed, majority answer verified correct, and job latency (scheduled arrival to last sample) within the target; every scheduled job is in the denominator',
            per_gpu_s='correct-within-target jobs divided by the GPU seconds attributed to target jobs: cell wall time (one GPU) times the job share of all tokens generated in the cell; prompt processing is not attributed separately',
            goodput_per_gpu_s='per cell: (target jobs correct within target + background requests completed within target) / cell wall seconds',
            ci95=f'percentile bootstrap, {args.draws} resamples of questions with replacement; seeds of a question are kept together; paired differences resample the same questions for both actions'),
        caveat='one machine; ' + ('fixed actions and frozen per-job policies' if manifest.get('policies') else 'fixed actions only, no router')
        + '; intervals reflect question sampling and seed-level noise only as far as the seeds run'))
    print(out / 'summary.json')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    for name in ('prompts', 'labels', 'out'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--questions', type=int, default=50)
    p.add_argument('--seed', type=int, default=20261115)
    p.add_argument('--proportional', action='store_true', help='keep the level mix of the full question set instead of equal counts')
    p.set_defaults(fn=prepare)
    p = sub.add_parser('run')
    for name in ('plan', 'calibration', 'lengths', 'prompts', 'hardware-json', 'out'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--url', default='http://127.0.0.1:8001')
    p.add_argument('--model', default='m')
    p.add_argument('--actions', nargs='+', default=['1:0', '4:0', '1:2048', '4:2048'])
    p.add_argument('--seeds', type=int, nargs='+', default=[20261116, 20261117, 20261118])
    p.add_argument('--target-fraction', type=float, default=.05, help='target job rate as a fraction of the contended rate')
    p.add_argument('--policies', default=None, help='JSON list of per-job policies (load_policy.py); each gets its own cell')
    p.add_argument('--rates', nargs='+', default=None, metavar='LABEL=RPS',
                   help='background rates to run instead of the frozen pair, for example between them')
    p.add_argument('--warmup', type=float, default=60)
    p.add_argument('--timeout', type=float, default=300, help='per-job observation limit, also the drain limit')
    p.add_argument('--idle-timeout', type=float, default=900)
    p.add_argument('--idle-stable', type=float, default=1)
    p.add_argument('--max-inflight', type=int, default=512)
    p.add_argument('--metrics-interval', type=float, default=1)
    p.add_argument('--deadline-epoch', type=float, default=None, help='do not start a block that cannot finish before this time')
    p.add_argument('--no-gpu', action='store_true')
    p.set_defaults(fn=run)
    p = sub.add_parser('score')
    p.add_argument('--out', required=True)
    p.add_argument('--gold', required=True)
    p.add_argument('--draws', type=int, default=2000)
    p.set_defaults(fn=score)
    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == '__main__':
    sys.exit(main())
