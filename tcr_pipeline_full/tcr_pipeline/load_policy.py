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
