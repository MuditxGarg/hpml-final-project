"""Summarize a downloaded bounded replay run without treating missing results as wins."""
import argparse
import json
from pathlib import Path


def table_rows(blocks):
    rows = []
    for block in blocks:
        cells = block['summary']['cells']
        control = next(c for c in cells if c['control'])['metrics']['background']
        for c in cells:
            bg, tg = c['metrics']['background'], c['metrics']['target']
            failures = bg['scheduled']-bg['statuses'].get('completed', 0)
            control_failures = control['scheduled']-control['statuses'].get('completed', 0)
            p95, cp95 = bg['completed_p95_s'], control['completed_p95_s']
            rows.append(dict(block=block['block'], partial_block=block.get('partial', False), seed=block['seed'], load=block['rho'],
                             n=c['n'], budget=c['budget'], control=c['control'],
                             background_scheduled=bg['scheduled'], background_failures=failures,
                             background_slo_attainment=bg['slo_attainment'], completed_p95_s=p95,
                             completed_p95_delta_s=p95-cp95 if p95 is not None and cp95 is not None else None,
                             matched_background_count=bg['scheduled']==control['scheduled'],
                             target_scheduled=tg['scheduled'], target_slo_attainment=tg['slo_attainment'],
                             target_completed_p95_s=tg.get('completed_p95_s'),
                             latency_comparison='completed-only; failures/censoring present' if failures or control_failures
                             else 'all background requests completed'))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    state = json.loads((args.run/'pilot-summary.json').read_text())
    blocks = list(state['completed_blocks'])
    # Preserve usable cells from a block that failed/stopped before completion.
    known = {b['block'] for b in blocks}
    for path in sorted(args.run.glob('seed*-rho*/summary.json')):
        if path.parent.name in known:
            continue
        name = path.parent.name
        seed_text, rho_text = name.split('-rho')
        blocks.append(dict(block=name, seed=int(seed_text[4:]), rho=float(rho_text),
                           summary=json.loads(path.read_text()), partial=True))
    rows = table_rows(blocks)
    for row in rows:
        directory = args.run/row['block']
        manifest = json.loads((directory/'manifest.json').read_text())
        trace = json.loads((directory/'trace.json').read_text())
        action = [row['n'], row['budget']]
        index = manifest['cell_order'].index(action)
        telemetry = directory/f'cell-{index:03d}-N{row["n"]}-B{row["budget"]}-metrics.jsonl'
        samples = [json.loads(line) for line in telemetry.read_text().splitlines()] if telemetry.exists() else []
        measured = [s for s in samples if trace['warmup_s'] <= s['at_s'] < trace['warmup_s']+trace['duration_s']]
        waiting = [s['gauges']['vllm:num_requests_waiting'] for s in measured
                   if 'vllm:num_requests_waiting' in s.get('gauges', {})]
        row['waiting_max_measured'] = max(waiting) if waiting else None
        row['metrics_scrape_errors'] = sum('error' in s for s in measured)
    result = dict(status=state['status'], calibration=state.get('calibration'),
                  completed_blocks=len(state['completed_blocks']), skipped_blocks=state['skipped_blocks'],
                  error=state.get('error'), rows=rows)
    protocol = {key: state.get(key) for key in (
        'protocol_phase', 'corrective_reason', 'request_timeout_s', 'drain_s',
        'reused_source', 'selected_actions', 'omitted_default_actions')}
    result['protocol'] = protocol
    def fmt(x):
        return '—' if x is None else f'{x:.3f}'
    lines = ['# Real T4 matched externality pilot', '', f"Run status: {state['status']}.",
             f"Completed blocks: {len(state['completed_blocks'])}; skipped blocks: {len(state['skipped_blocks'])}.", '',
             'Synthetic token-count replay; this does not measure final answer accuracy or routing gains. '
             'All scheduled cohort requests, including drops/errors/timeouts, count in SLO attainment. '
             'The p95 column is completed-only; it must not be read as an unconditional tail when failures exist.', '',
             f"Calibration: `{json.dumps(state.get('calibration'))}`", '',
             f"Protocol and correction provenance: `{json.dumps(protocol)}`", '',
             f"Recovered partial blocks: {sum(bool(b.get('partial')) for b in blocks)}. Partial cells are retained as incomplete evidence.", '',
             '| seed | load | N | B | BG count | BG failures | BG p95 completed (s) | delta vs control (s) | BG SLO fraction | target count | target SLO fraction | target p95 completed (s) | max waiting |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        lines.append(f"| {r['seed']} | {r['load']} | {r['n']} | {r['budget']} | {r['background_scheduled']} | "
                     f"{r['background_failures']} | {fmt(r['completed_p95_s'])} | {fmt(r['completed_p95_delta_s'])} | "
                     f"{fmt(r['background_slo_attainment'])} | {r['target_scheduled']} | {fmt(r['target_slo_attainment'])} | {fmt(r['target_completed_p95_s'])} | {fmt(r['waiting_max_measured'])} |")
    lines += ['', 'N=0 is a background-only control. Candidate order is randomized, but control runs first. '
              'These short runs and few seeds do not establish statistical significance; empty target cohorts '
              'provide no target-performance evidence. Capacity and SLO are local pilot heuristics, not imported '
              'RTX4080 values or a user SLA. Corrected B=0 cap2048 and single chat-template prefixes differ from '
              'the historical sweep. Compare actions within each paired block before comparing across loads.', '',
              'Raw cohort events, telemetry and manifests remain in the downloaded run directory. '
              'An unavailable or skipped condition is not counted as a passing condition.']
    if any(not r['matched_background_count'] for r in rows):
        lines += ['', '**Background counts differ from control in at least one cell; investigate matching before interpreting differences.**']
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2))
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')
    print(args.output/'report.md')


if __name__ == '__main__':
    main()
