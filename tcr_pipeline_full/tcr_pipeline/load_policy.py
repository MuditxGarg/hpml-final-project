"""Per-job action policies for the matched real-answer run.

A policy picks the action (N, B) of one target job at the moment the job arrives.

  threshold  load-adaptive: the first rule whose load conditions all hold gives the action;
             otherwise the fallback. Conditions read the newest server sample of the cell.
  mixture    load-blind baseline: a fixed share of jobs per action, assigned by a hash of the
             block seed and the question, so the assignment is the same at every background rate.

Policies are plain JSON, frozen before a run and copied into its manifest. Nothing here is
fitted; thresholds must come from earlier runs on other seeds. See docs/load-adaptive-controller.md.
"""
import hashlib
import json
from pathlib import Path

from sustained_calibration import KV, RUNNING, WAITING

CONDITIONS = dict(kv_below=('kv', lambda value, limit: value < limit),
                  waiting_at_most=('waiting', lambda value, limit: value <= limit),
                  running_below=('running', lambda value, limit: value < limit))


def parse_action(text):
    n, budget = (int(part) for part in str(text).split(':'))
    if n < 1 or budget < 0:
        raise ValueError(f'bad action {text!r}')
    return n, budget


def read_signal(samples, now_s):
    """Newest successful server sample as the load signal, with its age. None when there is none yet."""
    for sample in reversed(samples):
        metrics = sample.get('metrics')
        kv = [metrics[key] for key in KV if key in metrics] if metrics else []  # the gauge was renamed across vLLM versions
        if kv:
            return dict(kv=kv[0], waiting=metrics.get(WAITING, 0.), running=metrics.get(RUNNING, 0.),
                        age_s=now_s - sample['at_s'])
    return None


class Threshold:
    kind = 'threshold'

    def __init__(self, spec):
        self.name, self.fallback = spec['name'], parse_action(spec['fallback'])
        self.max_age_s = float(spec.get('max_signal_age_s', 5))
        self.rules = []
        for rule in spec['rules']:
            unknown = set(rule) - set(CONDITIONS) - {'action'}
            if unknown or len(rule) < 2:
                raise ValueError(f'rule needs an action and known conditions, got {sorted(rule)}')
            self.rules.append((parse_action(rule['action']), {k: float(v) for k, v in rule.items() if k != 'action'}))

    def choose(self, signal, seed, question_id):
        # Without a fresh signal the policy cannot show that an expensive action is affordable.
        if signal is None or signal['age_s'] > self.max_age_s:
            return self.fallback, 'no fresh signal'
        for index, (action, limits) in enumerate(self.rules):
            if all(CONDITIONS[key][1](signal[CONDITIONS[key][0]], limit) for key, limit in limits.items()):
                return action, f'rule {index}'
        return self.fallback, 'fallback'


class Mixture:
    kind = 'mixture'

    def __init__(self, spec):
        self.name = spec['name']
        self.shares = [(parse_action(action), float(share)) for action, share in spec['shares'].items()]
        if not self.shares or min(s for _, s in self.shares) < 0 or abs(sum(s for _, s in self.shares) - 1) > 1e-9:
            raise ValueError('mixture shares must be non-negative and sum to 1')

    def choose(self, signal, seed, question_id):
        draw = int(hashlib.sha256(f'{self.name}:{seed}:{question_id}'.encode()).hexdigest()[:12], 16) / 16 ** 12
        total = 0.
        for action, share in self.shares:
            total += share
            if draw < total:
                return action, 'mixture'
        return self.shares[-1][0], 'mixture'


def load_policies(path):
    specs = json.loads(Path(path).read_text())
    policies = [dict(threshold=Threshold, mixture=Mixture)[spec['kind']](spec) for spec in specs]
    names = [p.name for p in policies]
    if len(set(names)) != len(names) or any(not name or not name.replace('-', '').isalnum() for name in names):
        raise ValueError('policy names must be distinct and use only letters, digits and hyphens')
    return specs, policies


def arrival_table(runs, width=.1):
    """On-time rate of target jobs of each fixed action, by KV-cache usage at the job's arrival.

    `runs` are matched real-answer run directories. On time means completed within the loose
    latency target of that run, whether or not the answer was right; failures count as late.
    """
    bins = {}
    for run in map(Path, runs):
        manifest = json.loads((run / 'manifest.json').read_text())
        loose = max(manifest['latency_targets_s'])
        for block in manifest['completed']:
            for cell in json.loads((run / block / 'cells.json').read_text()):
                if cell['control'] or cell.get('policy'):
                    continue
                samples = [json.loads(line) for line in (run / block / f'{cell["cell"]}-server.jsonl').read_text().splitlines()]
                for line in (run / block / f'{cell["cell"]}.jsonl').read_text().splitlines():
                    row = json.loads(line)
                    if row['stream'] != 'target' or not row.get('measured'):
                        continue
                    signal = read_signal([s for s in samples if s['at_s'] <= row['at_s']], row['at_s'])
                    if signal is None:
                        continue
                    key = (f'{cell["n"]}:{cell["budget"]}', min(int(signal['kv'] / width), int(round(1 / width)) - 1))
                    entry = bins.setdefault(key, dict(jobs=0, on_time=0))
                    entry['jobs'] += 1
                    entry['on_time'] += int(row['status'] in ('completed', 'no_answer') and row['latency_s'] <= loose)
    return [dict(action=action, kv_from=round(index * width, 6), kv_to=round((index + 1) * width, 6), **entry,
                 on_time_rate=entry['on_time'] / entry['jobs']) for (action, index), entry in sorted(bins.items())]


def freeze_threshold(table, action, min_on_time, min_jobs):
    """Upper KV limit for an action: the start of the first bin that is too late or too thin to judge.

    Bins below the lowest well-populated one are taken as affordable: a lighter server is assumed not to be slower.
    """
    rows = sorted((r for r in table if r['action'] == action), key=lambda r: r['kv_from'])
    while rows and rows[0]['jobs'] < min_jobs:
        rows.pop(0)
    limit = rows[0]['kv_from'] if rows else 0.
    for row in rows:
        if row['kv_from'] > limit + 1e-9 or row['jobs'] < min_jobs or row['on_time_rate'] < min_on_time:
            break
        limit = row['kv_to']
    return limit if rows and limit > rows[0]['kv_from'] else 0.


def freeze(argv=None):
    import argparse
    import hashlib as h
    p = argparse.ArgumentParser(description='Freeze a KV-threshold policy from earlier matched runs (other seeds than the run it will be used in).')
    p.add_argument('--runs', nargs='+', required=True, help='matched real-answer run directories')
    p.add_argument('--prefer', nargs='+', required=True, help='actions in order of preference, for example 1:2048 4:0')
    p.add_argument('--fallback', default='1:0')
    p.add_argument('--name', default='kv-ladder')
    p.add_argument('--min-on-time', type=float, default=.9)
    p.add_argument('--min-jobs', type=int, default=20)
    p.add_argument('--bin-width', type=float, default=.1)
    p.add_argument('--output', required=True)
    args = p.parse_args(argv)
    table = arrival_table(args.runs, args.bin_width)
    rules = [dict(action=a, kv_below=freeze_threshold(table, a, args.min_on_time, args.min_jobs)) for a in args.prefer]
    policy = dict(kind='threshold', name=args.name, fallback=args.fallback, rules=[r for r in rules if r['kv_below'] > 0])
    Threshold(policy)  # reject a malformed result before writing anything
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'policies.json').write_text(json.dumps([policy], indent=2) + '\n')
    (out / 'freeze.json').write_text(json.dumps(dict(
        rule='an action is allowed below the start of the first KV-usage bin whose on-time rate is under min_on_time or that holds fewer than min_jobs jobs; bins below the lowest well-populated bin count as affordable',
        arguments=vars(args), thresholds=rules, table=table,
        sources={str(Path(r).resolve()): h.sha256((Path(r) / 'manifest.json').read_bytes()).hexdigest() for r in args.runs},
        seeds_used=sorted({json.loads((Path(r) / b / 'cells.json').read_text())[0]['seed'] for r in args.runs
                           for b in json.loads((Path(r) / 'manifest.json').read_text())['completed']}),
        caveat='thresholds describe these runs on this machine; evaluate the policy on other seeds only'), indent=2) + '\n')
    print(out / 'policies.json')
