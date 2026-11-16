import json
from pathlib import Path

import pytest

import baseline_report as br

RESULTS = Path(__file__).resolve().parents[3] / 'docs' / 'results'
CALIBRATION = RESULTS / '2026-11-15-sustained-calibration-4080' / 'seed-1042'
PROFILE = RESULTS / '2026-11-15-profiling-4080'


def matched_run(tmp_path):
    value = lambda v: dict(value=v, ci95=[v - .1, v + .1])
    statuses = dict(completed=9, no_answer=1, timeout=0, error=0, censored=0, drop=0)
    action = lambda rate, n, b, v: dict(rate=rate, n=n, budget=b, jobs=10, questions=5, seeds=[1, 2], statuses=statuses, target_gpu_s=40.,
                                        accuracy=value(.8), correct_within_tight=value(.5), correct_within_loose=value(v),
                                        correct_within_tight_per_gpu_s=value(.2), correct_within_loose_per_gpu_s=value(v / 4))
    cell = dict(block='seed-1-light', n=1, budget=0, target_jobs=5, target_correct=4, target_correct_tight=3, target_correct_loose=4,
                target_latency_p95_s=12.34, background_jobs=40, background_within_tight=38, background_within_loose=40, background_p95_s=18.5,
                waiting_max=0., kv_cache_max=.42, preemptions=0., goodput_tight_per_gpu_s=.21, goodput_loose_per_gpu_s=.23)
    summary = dict(status='complete', blocks_scored=['seed-1-light'], blocks_skipped=[], latency_targets_s=dict(tight=20.9, loose=41.8),
                   cells=[cell, dict(cell, n=None, budget=None, policy='ladder')],
                   actions=[action('light', 1, 0, .7), action('light', 4, 2048, .6), action('contended', 1, 0, .6),
                            dict(action('light', None, None, .65), policy='ladder', chosen={'1:0': 4, '4:2048': 6}, reasons={'rule 0': 6, 'fallback': 4})],
                   paired_differences=[dict(rate='light', metric='correct_within_loose', action=[1, 0], minus=[4, 2048], questions=5, value=.1, ci95=[.02, .2]),
                                       dict(rate='light', metric='correct_within_loose', action=[1, 0], minus='ladder', questions=5, value=.05, ci95=[-.02, .1]),
                                       dict(rate='light', metric='correct_within_tight', action=[1, 0], minus=[4, 2048], questions=5, value=.1, ci95=[-.1, .3])],
                   definitions=dict(correct_within_target='as logged'))
    run = tmp_path / 'matched'
    run.mkdir()
    (run / 'summary.json').write_text(json.dumps(summary))
    (run / 'manifest.json').write_text(json.dumps(dict(environment=dict(git_commit='abcdef1234', git_dirty=False))))
    return run


def test_tables_are_formatted_from_logged_values(tmp_path):
    out = tmp_path / 'report'
    argv = ['--calibration', str(CALIBRATION), '--profile', str(PROFILE), '--matched', str(matched_run(tmp_path)),
            '--hardware', 'test machine', '--output', str(out), '--no-plots']
    assert br.main(argv) == 0
    text = (out / 'README.md').read_text()
    calibration = json.loads((CALIBRATION / 'calibration.json').read_text())
    for rung in calibration['rungs']:  # every rung appears with its logged rate and p95
        assert f'| {rung["rate_rps"]:.3f} | {rung["scheduled"]} | {rung["statuses"]["completed"]} |' in text
        assert f'| {rung["latency"]["p95_s"]:.1f} |' in text
    assert text.count('| contended |') >= 1 and '| light |' in text
    profile = json.loads((PROFILE / 'profile.json').read_text())
    worst = max(profile['cells'], key=lambda c: c['server']['counters']['preemptions'])
    assert f'| {worst["server"]["counters"]["preemptions"]:.0f} |' in text
    assert '| light | (1,0) | - | 10 | 2 | 9 | 1 | 0 | 0.800 [0.700, 0.900] |' in text
    assert '| light | ladder | 1:0 4, 4:2048 6 | 10 | 2 |' in text and '| seed-1-light | ladder | 5 |' in text
    assert '| light | correct_within_loose | (1,0) | ladder | 5 | +0.0500 | [-0.0200, +0.1000] | no |' in text
    assert '| light | correct_within_loose | (1,0) | (4,2048) | 5 | +0.1000 | [+0.0200, +0.2000] | yes |' in text
    assert '| light | correct_within_tight | (1,0) | (4,2048) | 5 | +0.1000 | [-0.1000, +0.3000] | no |' in text
    assert 'test machine' in text and 'What was not measured' in text and 'Plots were not requested.' in text
    sources = json.loads((out / 'sources.json').read_text())
    assert len(sources['files']) == 6 and all(len(digest) == 64 for digest in sources['files'].values())
    with pytest.raises(FileExistsError):
        br.main(argv)


def test_figures_are_written_when_matplotlib_is_available(tmp_path):
    pytest.importorskip('matplotlib')
    out = tmp_path / 'report'
    br.main(['--calibration', str(CALIBRATION), '--profile', str(PROFILE), '--matched', str(matched_run(tmp_path)),
             '--hardware', 'test machine', '--output', str(out)])
    names = json.loads((out / 'sources.json').read_text())['figures']
    # The collapse timeline needs the raw per-second samples, which are not committed.
    assert names == ['calibration-ladder.png', 'profile-actions.png', 'profile-decode-by-concurrency.png', 'matched-actions.png']
    assert all((out / 'figures' / name).stat().st_size > 5000 for name in names)
    assert all(f'figures/{name}' in (out / 'README.md').read_text() for name in names)


def test_missing_values_render_as_a_dash():
    assert br.fmt(None) == '-' and br.fmt(1.25, 2) == '1.25'
    assert br.action_name(0, 0) == 'control' and br.action_name(4, 2048) == '(4,2048)'
    assert br.interval(dict(value=None, ci95=None)) == '- [-, -]'


def test_several_matched_runs_share_one_load_curve(tmp_path):
    first = matched_run(tmp_path)
    second = tmp_path / 'ladder'
    second.mkdir()
    summary = json.loads((first / 'summary.json').read_text())
    summary['actions'] = [dict(a, rate='mid') for a in summary['actions'] if a['rate'] == 'light']
    (second / 'summary.json').write_text(json.dumps(summary))
    for run, rates in ((first, dict(light=1.2, contended=2.5)), (second, dict(mid=1.7))):
        manifest = json.loads((first / 'manifest.json').read_text())
        (run / 'manifest.json').write_text(json.dumps(dict(manifest, rates_rps=rates)))
    out = tmp_path / 'report'
    br.main(['--matched', str(first), '--matched', str(second), '--hardware', 'test machine', '--output', str(out)])
    text = (out / 'README.md').read_text()
    assert '### Matched real-answer comparison\n' in text and '### Matched real-answer comparison (ladder)' in text
    assert {'matched-actions.png', 'matched-actions-2.png', 'matched-load-curve.png'} <= {p.name for p in (out / 'figures').iterdir()}
