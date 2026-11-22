"""Report generator for the mid-point baseline: tables and plots built only from logged runs.

Reads the output directories of the sustained calibration (13), the profiling pass (14) and the
scored matched real-answer comparison (15), and writes into a new directory:
  README.md       what was measured, what was not, and every table
  figures/*.png   plots (skipped with a note when matplotlib is not installed)
  sources.json    path and SHA-256 of every input file
No number is typed by hand: every cell is formatted from a logged JSON value. Interpretation belongs
in a separate hand-written note next to the generated README. See docs/baseline-report.md.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

# Chart colours: fixed slot per action so an action keeps its colour in every figure.
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
CONTROL = '#8a897f'
# Colour follows the action, whichever actions a figure happens to show.
COLOR = {(1, 0): SERIES[0], (1, 2048): SERIES[1], (4, 0): SERIES[2], (4, 2048): SERIES[3]}


def fmt(value, digits=1):
    return '-' if value is None else f'{value:.{digits}f}'


def table(header, rows):
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    return '\n'.join(lines + ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows])


def action_name(n, budget):
    return 'control' if n == 0 else f'({n},{budget})'


def interval(entry, digits=3):
    low, high = entry['ci95'] or (None, None)
    return f'{fmt(entry["value"], digits)} [{fmt(low, digits)}, {fmt(high, digits)}]'


class Sources:
    def __init__(self):
        self.files = {}

    def load(self, path):
        path = Path(path)
        self.files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())


def calibration_section(directory, sources):
    result = sources.load(Path(directory) / 'calibration.json')
    manifest = sources.load(Path(directory) / 'manifest.json')
    rows = []
    for r in result['rungs']:
        q, lat = r['queue'], r['latency']
        role = 'light' if r['rung'] == result.get('light_rung') else 'contended' if r['rung'] == result.get('contended_rung') else ''
        rows.append([fmt(r['rate_rps'], 3), r['scheduled'], r['statuses']['completed'], fmt(lat['p50_s']), fmt(lat['p95_s']),
                     fmt(r['p95_over_idle'], 2), f'{fmt(q["running_mean"])} / {fmt(q["running_max"], 0)}',
                     f'{fmt(q["waiting_mean"])} / {fmt(q["waiting_max"], 0)}', f'{fmt(q["kv_cache_mean"], 2)} / {fmt(q["kv_cache_max"], 2)}',
                     '-' if q['backlog_growth_fraction'] is None else f'{q["backlog_growth_fraction"]:+.3f}',
                     fmt(q['tokens_per_s'], 0), 'yes' if r['stable'] else 'no', role])
    targets = result['latency_targets_s']
    text = [f'### Calibration seed {manifest["configuration"]["seed"]}', '',
            f'Source `{directory}`, commit `{(manifest["environment"]["git_commit"] or "unknown")[:7]}`'
            f'{" (dirty tree)" if manifest["environment"]["git_dirty"] else ""}, status `{result["status"]}`. '
            f'Idle p95 {fmt(result["idle_p95_s"], 2)} s; latency targets {fmt(targets[0], 2)} s and {fmt(targets[-1], 2)} s; '
            f'frozen light rate {fmt(result.get("light_rate_rps"), 3)} req/s, contended rate {fmt(result.get("contended_rate_rps"), 3)} req/s.', '',
            table(['Rate (req/s)', 'Requests', 'Completed', 'p50 (s)', 'p95 (s)', 'p95 / idle', 'Running mean / max', 'Waiting mean / max',
                   'KV cache mean / max', 'Backlog growth', 'Tokens/s', 'Stable', 'Role'], rows)]
    return '\n'.join(text), dict(seed=manifest['configuration']['seed'], result=result)


def profile_section(directory, sources):
    profile = sources.load(Path(directory) / 'profile.json')
    manifest = sources.load(Path(directory) / 'manifest.json')
    cells = sorted(profile['cells'], key=lambda c: (c['rate_label'] != 'light', c['n'], c['budget']))
    tight, loose = (f'{t:g}' for t in (min(profile['latency_targets_s']), max(profile['latency_targets_s'])))
    latency, server, streamed = [], [], []
    for c in cells:
        name, b, t, s = action_name(c['n'], c['budget']), c['streams']['background'], c['streams']['target'], c['server']
        h = s['histogram_means']
        latency.append([c['rate_label'], name, fmt(b['latency_s']['p50']), fmt(b['latency_s']['p95']), fmt(b['attainment'][tight], 2),
                        fmt(b['attainment'][loose], 2), t['scheduled'], fmt(t['latency_s']['p50']), fmt(t['latency_s']['p95']),
                        fmt(t['attainment'][loose], 2)])
        itl = h['inter_token_latency_s']['mean']
        server.append([c['rate_label'], name, f'{fmt(s["running"]["mean"])} / {fmt(s["running"]["max"], 0)}',
                       f'{fmt(s["waiting"]["mean"])} / {fmt(s["waiting"]["max"], 0)}', f'{fmt(s["kv_cache"]["mean"], 2)} / {fmt(s["kv_cache"]["max"], 2)}',
                       fmt(s['counters']['preemptions'], 0), fmt(h['preemptions_per_request']['mean'], 2), fmt(h['queue_time_s']['mean'], 2),
                       fmt(h['time_to_first_token_s']['mean'], 2), fmt(None if itl is None else 1000 * itl), fmt(s['tokens_per_s'], 0),
                       fmt(c['gpu']['utilization.gpu']['mean'], 0), fmt(c['gpu']['temperature.gpu']['mean'], 0)])
        if c['n']:
            r = t['streamed_requests']
            streamed.append([c['rate_label'], name, f'{r["completed"]} of {r["sent"]}',
                             f'{fmt(r["time_to_first_token_s"]["p50"], 2)} / {fmt(r["time_to_first_token_s"]["p95"], 2)}',
                             f'{fmt(r["decode_ms_per_token"]["p50"])} / {fmt(r["decode_ms_per_token"]["p95"])}',
                             f'{fmt(r["longest_stall_s"]["p95"], 2)} / {fmt(r["longest_stall_s"]["max"], 2)}'])
    text = ['### Profiling pass', '',
            f'Source `{directory}`, commit `{(manifest["environment"]["git_commit"] or "unknown")[:7]}`, status `{profile["status"]}`. '
            f'Background rates: light {fmt(profile["rates_rps"]["light"], 3)} req/s, contended {fmt(profile["rates_rps"]["contended"], 3)} req/s; '
            f'target stream {fmt(profile["target_rate_rps"], 3)} jobs/s.', '',
            table(['Rate', 'Action', 'Background p50 (s)', 'Background p95 (s)', f'Within {tight} s', f'Within {loose} s', 'Target jobs',
                   'Target p50 (s)', 'Target p95 (s)', f'Target within {loose} s'], latency), '',
            table(['Rate', 'Action', 'Running mean / max', 'Waiting mean / max', 'KV cache mean / max', 'Preemptions', 'Preemptions per finished request',
                   'Mean queue wait (s)', 'Mean time to first token (s)', 'Mean inter-token latency (ms)', 'Tokens/s', 'GPU util (%)', 'GPU temp (C)'], server), '',
            table(['Rate', 'Action', 'Streamed requests', 'Time to first token p50 / p95 (s)', 'Decode ms per token p50 / p95',
                   'Longest stall p95 / max (s)'], streamed)]
    return '\n'.join(text), dict(profile=profile, cells=cells, directory=Path(directory))


def matched_section(directory, sources):
    summary = sources.load(Path(directory) / 'summary.json')
    manifest = sources.load(Path(directory) / 'manifest.json')
    actions = [[a['rate'], action_name(a['n'], a['budget']), a['jobs'], len(a['seeds']), a['statuses']['completed'], a['statuses']['no_answer'],
                sum(a['statuses'][s] for s in ('timeout', 'error', 'censored', 'drop')), interval(a['accuracy']),
                interval(a['correct_within_tight']), interval(a['correct_within_loose']), fmt(a['target_gpu_s'], 0),
                interval(a['correct_within_tight_per_gpu_s'], 4), interval(a['correct_within_loose_per_gpu_s'], 4)] for a in summary['actions']]
    cells = [[c['block'], action_name(c['n'], c['budget']), c['target_jobs'], c['target_correct'], c['target_correct_tight'], c['target_correct_loose'],
              fmt(c['target_latency_p95_s']), c['background_jobs'], c['background_within_tight'], c['background_within_loose'], fmt(c['background_p95_s']),
              fmt(c['waiting_max'], 0), fmt(c['kv_cache_max'], 2), fmt(c['preemptions'], 0), fmt(c.get('gpu_power_mean_w')),
              fmt(c['goodput_tight_per_gpu_s'], 3), fmt(c['goodput_loose_per_gpu_s'], 3)] for c in summary['cells']]
    gaps = [[d['rate'], d['metric'], action_name(*d['action']), action_name(*d['minus']), d['questions'],
             '-' if d['value'] is None else f'{d["value"]:+.4f}', '-' if not d['ci95'] else f'[{d["ci95"][0]:+.4f}, {d["ci95"][1]:+.4f}]',
             '-' if not d['ci95'] else 'no' if d['ci95'][0] <= 0 <= d['ci95'][1] else 'yes'] for d in summary['paired_differences']]
    targets = summary['latency_targets_s']
    text = ['### Matched real-answer comparison', '',
            f'Source `{directory}`, commit `{(manifest["environment"]["git_commit"] or "unknown")[:7]}`, status `{summary["status"]}`; '
            f'blocks scored: {len(summary["blocks_scored"])}, skipped: {len(summary["blocks_skipped"])}. '
            f'Latency targets: tight {fmt(targets["tight"], 2)} s, loose {fmt(targets["loose"], 2)} s. '
            'Values are followed by 95% bootstrap intervals over questions.', '',
            table(['Rate', 'Action', 'Jobs', 'Seeds', 'Completed', 'No answer', 'Failed', 'Accuracy', 'Correct within tight', 'Correct within loose',
                   'Target GPU-s', 'Correct within tight per GPU-s', 'Correct within loose per GPU-s'], actions), '',
            'Per cell:', '',
            table(['Block', 'Action', 'Target jobs', 'Correct', 'Correct within tight', 'Correct within loose', 'Target p95 (s)', 'Background requests',
                   'Background within tight', 'Background within loose', 'Background p95 (s)', 'Waiting max', 'KV cache max', 'Preemptions', 'GPU power mean (W)',
                   'Goodput tight per GPU-s', 'Goodput loose per GPU-s'], cells), '',
            'Paired differences between actions at the same rate (first minus second):', '',
            table(['Rate', 'Metric', 'Action', 'Minus', 'Questions', 'Difference', '95% interval', 'Interval excludes zero'], gaps), '',
            'Definitions: ' + ' '.join(f'**{k}**: {v}.' for k, v in summary['definitions'].items())]
    return '\n'.join(text), dict(summary=summary, rates=manifest.get('rates_rps', {}), name=Path(directory).name)


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc='left', color=INK, fontsize=11, pad=10)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(axis='y', color=GRID, linewidth=.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for side, spine in ax.spines.items():
        spine.set_visible(side == 'bottom')
        spine.set_color(GRID)


def make_figures(out, calibrations, profile, matched):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = []

    def save(fig, name, caption):
        fig.patch.set_facecolor(SURFACE)
        fig.tight_layout()
        fig.savefig(out / 'figures' / name, dpi=160, facecolor=SURFACE)
        plt.close(fig)
        figures.append((name, caption))

    if calibrations:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        for index, c in enumerate(calibrations):
            rungs = c['result']['rungs']
            for ax, key, pick in ((axes[0], 'p95', lambda r: r['latency']['p95_s']), (axes[1], 'kv', lambda r: r['queue']['kv_cache_max'])):
                x, y = [r['rate_rps'] for r in rungs], [pick(r) for r in rungs]
                ax.plot(x, y, color=SERIES[index % 4], linewidth=2, label=f'seed {c["seed"]}')
                for r, xv, yv in zip(rungs, x, y):  # hollow marker: rung classed unstable
                    ax.plot(xv, yv, 'o', markersize=7, color=SERIES[index % 4], markerfacecolor=SERIES[index % 4] if r['stable'] else SURFACE, markeredgewidth=2)
        for value, label in zip(calibrations[0]['result']['latency_targets_s'], ('tight target', 'loose target')):
            axes[0].axhline(value, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
            axes[0].annotate(label, (axes[0].get_xlim()[0], value), xytext=(4, 3), textcoords='offset points', color=MUTED, fontsize=8)
        style(axes[0], 'Background p95 latency by offered rate', 'Background rate (req/s)', 'p95 latency (s)')
        style(axes[1], 'Peak KV-cache usage by offered rate', 'Background rate (req/s)', 'Peak KV-cache usage (fraction)')
        axes[1].set_ylim(0, 1.05)
        axes[0].legend(frameon=False, fontsize=8, labelcolor=MUTED, loc='upper left', bbox_to_anchor=(0, .8))
        save(fig, 'calibration-ladder.png', 'Sustained calibration: background p95 and peak KV-cache usage against offered rate. Hollow markers are rungs classed unstable.')

    if profile:
        cells, targets = profile['cells'], profile['profile']['latency_targets_s']
        labels = ['light', 'contended']
        names = sorted({(c['n'], c['budget']) for c in cells})
        colour = {a: CONTROL if a[0] == 0 else SERIES[(i - 1) % 4] for i, a in enumerate(names)}
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        for ax, title, pick, unit in ((axes[0], 'Background p95 latency by action', lambda c: c['streams']['background']['latency_s']['p95'], 'p95 latency (s)'),
                                      (axes[1], 'Preemptions in the window by action', lambda c: c['server']['counters']['preemptions'], 'Preemptions')):
            width = .8 / len(names)
            for i, a in enumerate(names):
                for j, label in enumerate(labels):
                    cell = next((c for c in cells if c['rate_label'] == label and (c['n'], c['budget']) == a), None)
                    if cell is None or pick(cell) is None:
                        continue
                    x = j + (i - (len(names) - 1) / 2) * width
                    ax.bar(x, pick(cell), width=width * .88, color=colour[a], label=action_name(*a) if j == 0 else None)
                    ax.annotate(fmt(pick(cell), 0), (x, pick(cell)), xytext=(0, 2), textcoords='offset points', ha='center', color=MUTED, fontsize=7,
                                bbox=dict(facecolor=SURFACE, edgecolor='none', pad=.6))
            ax.set_xticks(range(len(labels)), [f'{label} rate' for label in labels])
            style(ax, title, '', unit)
        for value in targets:
            axes[0].axhline(value, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=0)
        axes[0].legend(frameon=False, fontsize=8, labelcolor=MUTED, loc='upper left', ncols=2)
        save(fig, 'profile-actions.png', 'Profiling pass: background p95 (dashed lines: latency targets) and preemptions for each action at the two frozen rates.')

        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        for flag, colour_, label in ((False, SERIES[0], 'cells without preemptions'), (True, SERIES[1], 'cells with preemptions')):
            points = [(r['running_from'] + 8, r['inter_token_latency_ms']) for c in cells if ((c['server']['counters']['preemptions'] or 0) > 0) == flag
                      for r in c['server']['decode_by_concurrency'] if r['intervals'] >= 10]
            ax.plot([p[0] for p in points], [p[1] for p in points], 'o', markersize=7, color=colour_, markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        style(ax, 'Per-token decode time against running sequences', 'Running sequences (bin centre)', 'Inter-token latency (ms)')
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=8, labelcolor=MUTED, loc='lower right')
        save(fig, 'profile-decode-by-concurrency.png', 'Profiling pass: per-cell median inter-token latency for each bin of 16 running sequences (bins with at least 10 one-second intervals).')

        worst = max(cells, key=lambda c: c['server']['counters']['preemptions'] or 0)
        raw = profile['directory'] / f'{worst["cell"]}-server.jsonl'
        if raw.exists() and (worst['server']['counters']['preemptions'] or 0) > 0:
            samples = [s for s in (json.loads(line) for line in raw.read_text().splitlines()) if 'metrics' in s]
            t = [s['at_s'] for s in samples]
            series = (('KV-cache usage (fraction)', [s['metrics'].get('vllm:kv_cache_usage_perc') for s in samples]),
                      ('Requests waiting', [s['metrics'].get('vllm:num_requests_waiting') for s in samples]),
                      ('Preemptions since start', [s['metrics'].get('vllm:num_preemptions_total', 0) - samples[0]['metrics'].get('vllm:num_preemptions_total', 0) for s in samples]))
            fig, axes = plt.subplots(3, 1, figsize=(7.5, 6.2), sharex=True)
            for ax, (label, values) in zip(axes, series):
                ax.plot(t, values, color=SERIES[0], linewidth=2)
                style(ax, label, '', '')
                ax.set_ylim(bottom=0)
            axes[0].set_title(f'{action_name(worst["n"], worst["budget"])} at the {worst["rate_label"]} rate: KV-cache usage (fraction)', loc='left', color=INK, fontsize=11, pad=10)
            axes[-1].set_xlabel('Seconds from the start of the cell (warm-up included)', color=MUTED, fontsize=9)
            save(fig, 'profile-collapse-timeline.png', f'Profiling pass, {action_name(worst["n"], worst["budget"])} at the {worst["rate_label"]} rate: KV-cache usage, waiting requests and preemptions over time.')

    for index, run in enumerate(matched):
        actions = run['summary']['actions']
        names = sorted({(a['n'], a['budget']) for a in actions})
        labels = sorted({a['rate'] for a in actions}, key=lambda r: (run['rates'].get(r, {'light': 0, 'contended': 1}.get(r, 2)), r))
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        for ax, key, title, unit in ((axes[0], 'correct_within_loose', 'Correct answers within the loose target', 'Fraction of target jobs'),
                                     (axes[1], 'correct_within_loose_per_gpu_s', 'Correct answers within the loose target per GPU-second', 'Per attributed GPU-second')):
            width = .8 / len(names)
            for i, a in enumerate(names):
                for j, label in enumerate(labels):
                    entry = next((e for e in actions if e['rate'] == label and (e['n'], e['budget']) == a), None)
                    if entry is None or entry[key]['value'] is None:
                        continue
                    x, value, ci = j + (i - (len(names) - 1) / 2) * width, entry[key]['value'], entry[key]['ci95']
                    ax.bar(x, value, width=width * .88, color=COLOR.get(a, CONTROL), label=action_name(*a) if j == 0 else None)
                    if ci:
                        ax.plot([x, x], ci, color=INK, linewidth=1.2)
            ax.set_xticks(range(len(labels)), [f'{label} rate' for label in labels])
            style(ax, title, '', unit)
        axes[0].legend(frameon=False, fontsize=8, labelcolor=MUTED, loc='upper right', ncols=2)
        save(fig, 'matched-actions.png' if not index else f'matched-actions-{index + 1}.png',
             f'Matched real-answer comparison ({run["name"]}): correct answers within the loose latency target, as a fraction of target jobs and per attributed GPU-second. Black lines: 95% bootstrap intervals over questions.')

    # Every background rate of every matched run on one load axis. Rates without a recorded value are left out.
    points = {}
    for run in matched:
        for a in run['summary']['actions']:
            if a['rate'] in run['rates'] and a['correct_within_loose']['value'] is not None:
                points.setdefault((a['n'], a['budget']), []).append((run['rates'][a['rate']], a['correct_within_loose']))
    if len({rate for series in points.values() for rate, _ in series}) >= 3:
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        for a, series in sorted(points.items()):
            series.sort(key=lambda item: item[0])
            xs = [rate for rate, _ in series]
            ax.plot(xs, [v['value'] for _, v in series], color=COLOR.get(a, CONTROL), linewidth=2, marker='o', markersize=5, label=action_name(*a))
            for rate, v in series:
                if v['ci95']:
                    ax.plot([rate, rate], v['ci95'], color=COLOR.get(a, CONTROL), linewidth=1, alpha=.55)
        ax.set_ylim(0, 1)
        ax.margins(x=.12)
        style(ax, 'Correct answers within the loose target against background load', 'Background rate (req/s)', 'Fraction of target jobs')
        ax.legend(frameon=False, fontsize=8, labelcolor=MUTED, loc='lower left')
        save(fig, 'matched-load-curve.png', 'Matched real-answer comparison: correct answers within the loose latency target against the background rate, all matched runs. Vertical lines: 95% bootstrap intervals over questions.')
    return figures


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--calibration', action='append', default=[], help='sustained calibration output directory (repeatable)')
    p.add_argument('--profile', help='profiling pass output directory')
    p.add_argument('--matched', action='append', default=[], metavar='DIR', help='scored matched real-answer output directory; repeat to add runs at other rates')
    p.add_argument('--title', default='Mid-point baseline and profiling')
    p.add_argument('--hardware', required=True, help='one-line hardware label; results from different machines are never pooled')
    p.add_argument('--output', required=True)
    p.add_argument('--no-plots', action='store_true')
    args = p.parse_args(argv)
    if not (args.calibration or args.profile or args.matched):
        p.error('give at least one run directory')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    sources = Sources()
    sections, calibrations, profile, matched = [], [], None, []
    for directory in args.calibration:
        text, data = calibration_section(directory, sources)
        sections.append(text)
        calibrations.append(data)
    if args.profile:
        text, profile = profile_section(args.profile, sources)
        sections.append(text)
    for index, directory in enumerate(args.matched):
        text, run = matched_section(directory, sources)
        sections.append(text.replace('### Matched real-answer comparison', f'### Matched real-answer comparison ({run["name"]})', 1) if index else text)
        matched.append(run)
    figures, note = [], 'Plots were not requested.'
    if not args.no_plots:
        try:
            (out / 'figures').mkdir()
            figures = make_figures(out, calibrations, profile, matched)
        except ImportError:
            note = 'matplotlib is not installed, so no plots were written.'
    measured = []
    if calibrations:
        measured.append('- **Sustained calibration:** background-only synthetic traffic (reference-budget lengths, `ignore_eos`) over a rate ladder; idle latency, per-rung completion, latency, queue and KV-cache usage; frozen light and contended rates.')
    if profile:
        measured.append('- **Profiling pass:** a background-only control and each action at the two frozen rates on one matched synthetic trace per rate; server gauges, counters and histograms, GPU samples, and client timing of streamed target requests.')
    if matched:
        measured.append('- **Matched real-answer comparison:** real, graded answers to a fixed question cohort for each fixed action, next to matched synthetic background traffic at the background rates of each run.')
    lines = [f'# {args.title}', '', f'Hardware: {args.hardware}. Results from other machines must not be pooled with these.', '',
             'Generated by `16_make_baseline_report.py` from the logged run directories listed in `sources.json`. Every number below is formatted from a logged value; none is typed by hand.', '',
             '## What was measured', '', *measured, '',
             '## What was not measured', '',
             '- No router or controller. Every action is fixed for a whole cell, so nothing here is a routing gain.',
             '- Calibration and profiling use replayed lengths and score no answers.',
             '- One machine and one server configuration. One pass per calibration rung and per profiling cell.',
             '- Failures are kept in every denominator and listed by status; nothing was dropped.', '',
             '## Figures', '']
    lines += [f'![{caption}](figures/{name})\n\n{caption}\n' for name, caption in figures] or [note, '']
    lines += ['## Tables', ''] + [s + '\n' for s in sections]
    (out / 'README.md').write_text('\n'.join(lines))
    (out / 'sources.json').write_text(json.dumps(dict(files=sources.files, figures=[name for name, _ in figures]), indent=2) + '\n')
    print(out / 'README.md')
    return 0


if __name__ == '__main__':
    sys.exit(main())
