"""Upload logged runs to Weights & Biases. Reads finished run directories; never measures anything.

  calibration directory (13)  one tracked run per pass, one step per rung
  profiling directory (14)    one tracked run per cell, per-second server and GPU telemetry
  matched directory (15)      one tracked run per cell with telemetry, plus one run holding the
                              scored tables (actions with intervals, paired differences)

Only numbers and run settings are uploaded: no question text and no model output. Run ids are
derived from the directory and cell, so uploading a directory twice skips what is already there.
See docs/experiment-tracking.md.
"""
import argparse
import hashlib
import json
from pathlib import Path

from sustained_calibration import KV, RUNNING, TOKENS, WAITING

PREEMPTIONS = 'vllm:num_preemptions_total'
ID_VERSION = 2  # the service never reuses the id of a deleted run; raise this to upload everything again under new ids


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()] if Path(path).exists() else []


def flat(value, prefix=''):
    """Numeric leaves of a nested dict as {'a/b': number}; everything else is left out."""
    if isinstance(value, bool) or value is None:
        return {}
    if isinstance(value, (int, float)):
        return {prefix: value}
    if isinstance(value, dict):
        return {k: v for key, item in value.items() for k, v in flat(item, f'{prefix}/{key}' if prefix else str(key)).items()}
    return {}


def telemetry(server, device):
    """Per-second history from raw samples: load gauges, preemptions and tokens since the start, GPU readings."""
    seconds, first = {}, None
    for sample in server:
        metrics = sample.get('metrics')
        if not metrics:
            continue
        first = first or metrics
        kv = [metrics[k] for k in KV if k in metrics]
        seconds.setdefault(int(sample['at_s']), {}).update(
            running=metrics.get(RUNNING), waiting=metrics.get(WAITING), kv_cache_usage=kv[0] if kv else None,
            preemptions=metrics.get(PREEMPTIONS, 0) - first.get(PREEMPTIONS, 0),
            generated_tokens=metrics.get(TOKENS, 0) - first.get(TOKENS, 0))
    for sample in device:
        if sample.get('gpu'):
            seconds.setdefault(int(sample['at_s']), {}).update(
                {'gpu/' + key.replace('.', '_'): value for key, value in sample['gpu'].items()})
    return [dict(time_s=second, **{k: v for k, v in values.items() if v is not None}) for second, values in sorted(seconds.items())]


def describe(kind, source, cell, config, history, summary, tables=None):
    name = f'{source.name}/{cell}' if cell else source.name
    return dict(id=hashlib.sha256(f'{kind}:{name}:{ID_VERSION}'.encode()).hexdigest()[:16], name=name, group=source.name, job_type=kind,
                config=config, history=history, summary=summary, tables=tables or {})


def common(manifest):
    env = manifest.get('environment', {})
    return dict(hardware=manifest.get('hardware_user_supplied'), git_commit=env.get('git_commit'), git_dirty=env.get('git_dirty'),
                experiment=manifest.get('experiment'), arguments=manifest.get('configuration'))


def calibration_runs(source):
    result, manifest = json.loads((source / 'calibration.json').read_text()), json.loads((source / 'manifest.json').read_text())
    history = [dict(rung=r['rung'], rate_rps=r['rate_rps'], stable=int(r['stable']),
                    **flat({k: r[k] for k in ('completion_rate', 'throughput_rps', 'p95_over_idle', 'latency', 'queue') if k in r}))
               for r in result['rungs']]
    summary = flat({k: result.get(k) for k in ('idle_p95_s', 'light_rate_rps', 'contended_rate_rps')})
    yield describe('calibration', source, None, dict(common(manifest), rule=result.get('rule'), status=result['status']),
                   history, dict(summary, status=result['status']))


def profile_runs(source):
    profile, manifest = json.loads((source / 'profile.json').read_text()), json.loads((source / 'manifest.json').read_text())
    for cell in profile['cells']:
        config = dict(common(manifest), rate_label=cell['rate_label'], background_rate_rps=cell['background_rate_rps'],
                      n=cell['n'], budget=cell['budget'], control=cell['control'], latency_targets_s=profile['latency_targets_s'])
        yield describe('profiling', source, cell['cell'], config,
                       telemetry(read_rows(source / f'{cell["cell"]}-server.jsonl'), read_rows(source / f'{cell["cell"]}-gpu.jsonl')),
                       flat({k: cell[k] for k in ('streams', 'server', 'gpu')}))


def matched_runs(source):
    manifest = json.loads((source / 'manifest.json').read_text())
    scored = json.loads((source / 'summary.json').read_text()) if (source / 'summary.json').exists() else None
    by_cell = {(c['block'], c['cell']): c for c in scored['cells']} if scored else {}
    for block in manifest['completed']:
        for cell in json.loads((source / block / 'cells.json').read_text()):
            config = dict(common(manifest), seed=cell['seed'], rate_label=cell['rate_label'],
                          background_rate_rps=cell['background_rate_rps'], n=cell['n'], budget=cell['budget'],
                          policy=cell.get('policy'), control=cell['control'], latency_targets_s=manifest['latency_targets_s'],
                          questions=manifest['questions'])
            summary = flat({k: cell[k] for k in ('streams', 'server', 'gpu', 'cell_wall_s', 'generation_tokens')})
            summary.update(flat(by_cell.get((block, cell['cell']), {}), 'scored'))
            yield describe('matched', source, f'{block}/{cell["cell"]}', config,
                           telemetry(read_rows(source / block / f'{cell["cell"]}-server.jsonl'),
                                     read_rows(source / block / f'{cell["cell"]}-gpu.jsonl')), summary)
    if scored:
        label = lambda a: a.get('policy') or f'({a["n"]},{a["budget"]})'
        name = lambda value: value if isinstance(value, str) else f'({value[0]},{value[1]})'
        metrics = ('accuracy', 'correct_within_tight', 'correct_within_loose', 'correct_within_tight_per_gpu_s', 'correct_within_loose_per_gpu_s')
        columns = ['rate', 'action', 'jobs', 'questions', 'no_answer', 'target_gpu_s'] + [c for m in metrics for c in (m, m + '_low', m + '_high')]
        rows = [[a['rate'], label(a), a['jobs'], a['questions'], a['statuses']['no_answer'], a['target_gpu_s']]
                + [x for m in metrics for x in (a[m]['value'], *(a[m]['ci95'] or (None, None)))] for a in scored['actions']]
        gaps = [[d['rate'], d['metric'], name(d['action']), name(d['minus']), d['questions'], d['value'], *(d['ci95'] or (None, None))]
                for d in scored['paired_differences']]
        summary = {f'{a["rate"]}/{label(a)}/{m}': a[m]['value'] for a in scored['actions'] for m in metrics}
        tables = dict(actions=(columns, rows),
                      paired_differences=(['rate', 'metric', 'action', 'minus', 'questions', 'difference', 'low', 'high'], gaps))
        check = scored.get('power_check')
        if check:  # how many cells ran at a different power state than their block control
            summary.update(power_cells_flagged=check['flagged'], power_cells_unknown=check['unknown'])
        phases = (scored.get('phases') or {}).get('actions', [])
        if phases:  # time-varying schedules: the same metrics by the load level of the phase of arrival
            summary.update({f'{a["rate"]}/{a["phase_rate"]}/{label(a)}/{m}': a[m]['value'] for a in phases for m in metrics})
            tables['phase_actions'] = (['rate', 'phase_rate'] + columns[1:],
                                       [[a['rate'], a['phase_rate'], label(a), a['jobs'], a['questions'], a['statuses']['no_answer'], a['target_gpu_s']]
                                        + [x for m in metrics for x in (a[m]['value'], *(a[m]['ci95'] or (None, None)))] for a in phases])
        yield describe('matched-summary', source, 'summary',
                       dict(common(manifest), latency_targets_s=scored['latency_targets_s'], definitions=scored['definitions'],
                            caveat=scored['caveat'], blocks_scored=scored['blocks_scored'], policies=manifest.get('policies'),
                            schedules=manifest.get('schedules') or None),
                       [], summary, tables)


def upload(run, wandb, entity, project, tags=(), exists=lambda run_id: False):
    """Send one described run. Returns False when a run with the same id is already there."""
    if exists(run['id']):
        return False
    # No code, git remote, host name or system metrics: only what the run directory recorded.
    settings = wandb.Settings(disable_git=True, save_code=False, x_disable_meta=True, x_disable_stats=True, console='off', silent=True)
    live = wandb.init(entity=entity, project=project, id=run['id'], name=run['name'], group=run['group'], job_type=run['job_type'],
                      config=run['config'], tags=list(tags), settings=settings)
    # Plot against time or offered rate, and keep last-step values out of the summary: it holds whole-run results only.
    axis = next((key for key in ('time_s', 'rate_rps') if run['history'] and key in run['history'][0]), None)
    if axis:
        live.define_metric(axis, summary='none')
        live.define_metric('*', step_metric=axis, summary='none')
    for row in run['history']:
        live.log(row)
    for name, (columns, rows) in run['tables'].items():
        live.log({name: wandb.Table(columns=columns, data=rows)})
    live.summary.update(run['summary'])
    live.finish()
    return True


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--entity', required=True)
    p.add_argument('--project', required=True)
    p.add_argument('--calibration', action='append', default=[], metavar='DIR')
    p.add_argument('--profile', action='append', default=[], metavar='DIR')
    p.add_argument('--matched', action='append', default=[], metavar='DIR')
    p.add_argument('--tags', nargs='*', default=[], help='for example the hardware label')
    p.add_argument('--dry-run', action='store_true', help='list what would be uploaded without contacting the service')
    args = p.parse_args(argv)
    runs = [run for reader, sources in ((calibration_runs, args.calibration), (profile_runs, args.profile), (matched_runs, args.matched))
            for source in sources for run in reader(Path(source))]
    if args.dry_run:
        for run in runs:
            print(f'{run["id"]} {run["job_type"]:16} {run["name"]} steps={len(run["history"])} summary={len(run["summary"])}')
        return
    import wandb
    api = wandb.Api()

    def exists(run_id):
        try:
            api.run(f'{args.entity}/{args.project}/{run_id}')
            return True
        except Exception:
            return False
    sent = [upload(run, wandb, args.entity, args.project, args.tags, exists) for run in runs]
    print(f'uploaded {sum(sent)} runs, skipped {len(sent) - sum(sent)} already present')
