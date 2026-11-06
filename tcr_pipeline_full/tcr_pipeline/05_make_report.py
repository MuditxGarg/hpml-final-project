"""Final stage: one readable report from decisions.json + ext_cells.jsonl (+ label accuracy)."""
import json, os, sys
sys.path.insert(0, os.environ.get('TCR_DIR', '.'))
from tcr_common import *

out = RUN / 'report'
out.mkdir(exist_ok=True)
L = []
say = L.append
dec = json.load(open(RUN / 'decisions.json'))
say(f"# Overnight run report (model {CFG['model']})\n")
say('## Values decided from data (frozen)')
say(f"- questions: {dec['n_questions']} (dev {len(dec['dev_ids'])}, test {dec['n_test']}), traces per question: {dec['K']}")
say(f"- N grid: {dec['N_GRID']} (rule: {dec['rule']})")
say(f"- B grid: {dec['B_GRID']}; budgets beaten by a cheaper one at N=1: {dec['dominated_B']}")
if 'mu_max' in dec:
    say(f"- capacity mu_max: {dec['mu_max']:.2f} requests/s; idle reference p95: {dec['idle_p95']:.1f}s; SLOs: {dec['slos']}s")
    say(f"- load regimes (fraction of capacity): {dec['load_fracs']}")
say('\n## Accuracy by (N, B), majority vote, all questions')
say('| N \\ B | ' + ' | '.join(str(b) for b in dec['B_GRID']) + ' |')
say('|---|' + '---|' * len(dec['B_GRID']))
for N, row in dec['acc_vote'].items():
    say(f'| {N} | ' + ' | '.join('%.3f' % row[str(b)] for b in dec['B_GRID']) + ' |')

f = RUN / 'load' / 'ext_cells.jsonl'
if f.exists():
    cells = [json.loads(x) for x in open(f) if x.strip()]
    first = next((c for c in cells if c['bg'].get('n')), None)
    slo_keys = [k for k in first['bg'] if k.startswith('slo_')] if first else []
    say('\n## Externality: effect of one action stream on background traffic')
    say('Delta = background p95 minus the control cell (no targets) at the same load, averaged over reps. '
        'Censored = requests that never finished (counted as SLO misses).\n')
    say('| load | action (N,B) | bg p95 (s) | delta p95 (s) | ' + ' | '.join(k.replace('slo_', 'bg within ') for k in slo_keys) +
        ' | bg censored | target p50 (s) | server waiting |')
    say('|---|---|---|---|' + '---|' * (len(slo_keys) + 3))
    fr = sorted({c['frac'] for c in cells})
    for fc in fr:
        cc = [c for c in cells if c['frac'] == fc]
        ctrl = [c['bg'].get('lat_p95') for c in cc if c['N'] == 0 and c['bg'].get('n')]
        ctrl = sum(ctrl) / len(ctrl) if ctrl else None
        keys = sorted({(c['N'], c['B']) for c in cc})
        for (N, B) in keys:
            xs = [c for c in cc if (c['N'], c['B']) == (N, B)]
            def avg(fn):
                v = [fn(c) for c in xs]
                v = [a for a in v if a is not None]
                return sum(v) / len(v) if v else None
            p95 = avg(lambda c: c['bg'].get('lat_p95'))
            fmt = lambda v, p='%.2f': (p % v) if v is not None else '-'
            d = (p95 - ctrl) if (p95 is not None and ctrl is not None) else None
            slos = [fmt(avg(lambda c, k=k: c['bg'].get(k)), '%.2f') for k in slo_keys]
            say(f"| {fc:.2f} | {'control' if N == 0 else f'({N},{B})'} | {fmt(p95)} | {fmt(d)} | " + ' | '.join(slos) +
                f" | {fmt(avg(lambda c: c['bg'].get('n_cens')), '%.1f')} | {fmt(avg(lambda c: c['tg'].get('lat_p50')))} | "
                f"{fmt(avg(lambda c: c['server']['waiting']))} |")
    wmax = max([(c['server']['waiting'] or 0) for c in cells if c['frac'] > 0] or [0])
    say(f'\nMax mean server queue (waiting requests) in any cell: {wmax:.1f}. '
        + ('A real queue formed.' if wmax >= 1 else 'No queue formed: raise the load fractions before drawing conclusions.'))
(out / 'overnight_report.md').write_text('\n'.join(L))
print('\n'.join(L))
