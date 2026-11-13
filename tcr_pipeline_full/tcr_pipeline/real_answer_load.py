"""Matched real-answer load diagnostic; a running dedicated server is required."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

from matched_serving import run_cell, wait_idle, cell_order, summarize, digest, http
from real_answer_validation import generate_sample, aggregate, write_json, MODEL, REVISION
from answer_scoring import is_correct

WARMUP, DURATION, TIMEOUT, IDLE = 20, 100, 180, 30
SEEDS = (20261116, 20261117)
ACTIONS = ((1, 0), (4, 0), (1, 2048))
SLO = 87


def block_bound():
    return 4*(WARMUP+DURATION+TIMEOUT+2*IDLE)


def block_fits(deadline):
    return deadline-time.time() >= block_bound()+180


def make_trace(plan, background, seed, rate):
    arrival=random.Random(f'{seed}:arrival')
    choose=random.Random(f'{seed}:background')
    events=[]; at=0
    while rate>0:
        at+=arrival.expovariate(rate)
        if at>=WARMUP+DURATION: break
        q=choose.choice(background)
        events.append(dict(id=f'b-{len(events)}',question_id=q['id'],stream='background',at_s=at,measured=at>=WARMUP))
    targets=list(plan['prompts']);random.Random(f'{seed}:targets').shuffle(targets)
    for i,q in enumerate(targets):
        events.append(dict(id=f't-{i}',question_id=q['id'],stream='target',at_s=WARMUP+(i+.5)*DURATION/len(targets),measured=True))
    return dict(seed=seed,background_rate=rate,warmup_s=WARMUP,duration_s=DURATION,
                events=sorted(events,key=lambda e:(e['at_s'],e['id'])),
                target_arrivals='fixed evenly spaced five-question cohort; randomized question order')


def callback(trace, prompts, tokenizer, url, model):
    def execute(event,n,budget,remaining):
        started=time.monotonic()
        def sample(k):
            seed=int(hashlib.sha256(f"{trace['seed']}:{event['id']}:{event['question_id']}:{k}".encode()).hexdigest()[:8],16)%(2**31)
            def send(body,timeout):
                return json.loads(http(url+'/v1/completions',dict(body,model=model),timeout))
            return generate_sample(prompts[event['question_id']],budget,seed,tokenizer,send,
                                   max(0,remaining-(time.monotonic()-started)))
        with ThreadPoolExecutor(max_workers=n) as pool: parts=list(pool.map(sample,range(n)))
        return dict(aggregate(parts),parts=parts,n=n,budget=budget)
    return execute


def score_cell(rows,gold,verifier=is_correct):
    metrics=summarize(rows,SLO)
    cohort=[r for r in rows if r['stream']=='target' and r['measured'] and r['status']!='not_offered_control']
    for r in cohort:
        r['correct']=r['status']=='completed' and bool(verifier(gold[r['question_id']],r['pred']))
        r['correct_within_slo']=r['correct'] and r['latency_s']<=SLO
    metrics['target'].update(correct=sum(r['correct'] for r in cohort),
                             correct_within_slo=sum(r['correct_within_slo'] for r in cohort))
    relevant=[metrics['background']] if not cohort else list(metrics.values())
    metrics['observed_delivery_feasible']=all(r['scheduled'] and r['slo_attainment']>=.95 for r in relevant)
    return metrics


def save_rows(path,rows):
    with Path(path).open('w') as f:
        for row in rows:f.write(json.dumps(row)+'\n')


def execute_cell(trace,action,prompts,tokenizer,args,path):
    before=wait_idle(args.url,IDLE)
    telemetry=[]
    rows=run_cell(trace,args.url,args.model,*action,0,timeout_s=TIMEOUT,drain_s=TIMEOUT,
                  telemetry=telemetry,job_runner=callback(trace,prompts,tokenizer,args.url,args.model))
    save_rows(path.with_suffix('.jsonl'),rows)
    save_rows(path.with_name(path.name+'-metrics.jsonl'),telemetry)
    write_json(path.with_name(path.name+'-idle.json'),before)
    wait_idle(args.url,IDLE)
    return rows


def calibrate(background,prompts,tokenizer,args,out):
    trials=[]
    for i in range(3):
        if args.deadline_epoch-time.time()<TIMEOUT+2*IDLE+180:
            raise TimeoutError('insufficient time for calibration')
        rng=random.Random(f'calibration:{i}')
        trace=dict(seed=20261120+i,warmup_s=0,duration_s=0,
                   events=[dict(id=f'b-{k}',question_id=rng.choice(background)['id'],stream='background',at_s=0,measured=True) for k in range(64)])
        write_json(out/f'calibration-{i}-trace.json',trace)
        rows=execute_cell(trace,(0,0),prompts,tokenizer,args,out/f'calibration-{i}')
        if any(r['status'] not in ('completed','no_answer') for r in rows):
            raise RuntimeError('calibration transport failed; refusing to define load from incomplete burst')
        elapsed=max(r['latency_s'] for r in rows)
        trials.append(dict(count=len(rows),elapsed_s=elapsed,requests_per_s=len(rows)/elapsed,
                           no_answer=sum(r['status']=='no_answer' for r in rows)))
    result=dict(definition='finite-burst throughput proxy, not established steady-state capacity',trials=trials,
                requests_per_s=statistics.median(r['requests_per_s'] for r in trials))
    write_json(out/'calibration.json',result)
    return result


def run(args):
    from transformers import AutoTokenizer
    plan=json.loads(Path(args.plan).read_text());background=json.loads(Path(args.background).read_text())
    assert len(plan['prompts'])==5
    assert not ({p['id'] for p in plan['prompts']} & {p['id'] for p in background})
    prompts={p['id']:p for p in background+plan['prompts']}
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    write_json(out/'plan.json',plan);write_json(out/'background.json',background)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    blocks=[(seed,load) for seed in SEEDS for load in (.5,.9)]
    random.Random(20261118).shuffle(blocks)
    state=dict(status='running',configuration=vars(args),blocks=blocks,completed=[],skipped=[],
               plan_sha256=digest(plan),background_sha256=digest(background),
               definition='real EOS-aware target and background answers; no router',slo_s=SLO,
               server_models=json.loads(http(args.url+'/v1/models')))
    write_json(out/'manifest.json',state)
    try:
        calibration=calibrate(background,prompts,tokenizer,args,out)
        # Persist every comparison trace and cell order before any comparison outcome.
        specifications=[]
        for seed,load in blocks:
            trace=make_trace(plan,background,seed,load*calibration['requests_per_s'])
            order=cell_order(ACTIONS,f'{seed}:{load}:order')
            block=out/f'seed-{seed}-load-{load}';block.mkdir()
            write_json(block/'trace.json',trace);write_json(block/'order.json',order)
            specifications.append((block,trace,order))
        for index,(block,trace,order) in enumerate(specifications):
            if not block_fits(args.deadline_epoch):
                state['skipped']=[p.name for p,_,_ in specifications[index:]]
                state['status']='partial_deadline';break
            for cell,action in enumerate(order):
                print(f'CELL_START {block.name} {action}',flush=True)
                execute_cell(trace,action,prompts,tokenizer,args,block/f'cell-{cell}-N{action[0]}-B{action[1]}')
                print(f'CELL_COMPLETE {block.name} {action}',flush=True)
            state['completed'].append(block.name);write_json(out/'manifest.json',state)
        else:state['status']='complete'
    except Exception as exc:
        state.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        state['finished_unix_s']=time.time();write_json(out/'manifest.json',state)


def summarize_telemetry(samples):
    measured=[s for s in samples if WARMUP<=s['at_s']<WARMUP+DURATION]
    waiting=[s['gauges']['vllm:num_requests_waiting'] for s in measured if 'vllm:num_requests_waiting' in s.get('gauges',{})]
    cache=[s['gauges'][key] for s in measured for key in ('vllm:kv_cache_usage_perc','vllm:gpu_cache_usage_perc') if key in s.get('gauges',{})]
    return dict(samples=len(measured),scrape_errors=sum('error' in s for s in measured),
                waiting_max=max(waiting) if waiting else None,
                waiting_mean=statistics.mean(waiting) if waiting else None,
                waiting_positive_fraction=sum(v>0 for v in waiting)/len(waiting) if waiting else None,
                cache_usage_max=max(cache) if cache else None,
                note='measured arrival window only; missing telemetry is not zero load')


def paired_differences(blocks):
    result=[]
    for block in blocks:
        if not block['paired_complete']:continue
        control=next((c for c in block['cells'] if c['n']==0),None)
        if control is None:continue
        base=control['metrics']['background']
        for cell in block['cells']:
            if cell['n']==0:continue
            bg=cell['metrics']['background'];target=cell['metrics']['target']
            all_complete=all(g['scheduled']>0 and g['statuses']['completed']==g['scheduled'] for g in (bg,base))
            delta=bg['slo_attainment']-base['slo_attainment'] if bg['scheduled'] and base['scheduled'] else None
            result.append(dict(block=block['block'],n=cell['n'],budget=cell['budget'],
                               background_slo_delta=delta,
                               background_p95_delta_s=bg['completed_p95_s']-base['completed_p95_s'] if all_complete else None,
                               target_correct_within_slo=target['correct_within_slo'],target_scheduled=target['scheduled'],
                               p95_note='paired p95 difference omitted if either background cohort contains failures'))
    return result


def score(args):
    import math_verify  # Require the project's symbolic grader, including in offline mode.
    out=Path(args.out);gold=json.loads(Path(args.gold).read_text());reports=[]
    manifest=json.loads((out/'manifest.json').read_text())
    for block in sorted(out.glob('seed-*-load-*')):
        cells=[]
        for path in sorted(block.glob('cell-*-N*-B*.jsonl')):
            if path.name.endswith(('-metrics.jsonl','-scored.jsonl')):continue
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            action=path.stem.split('-')
            n,b=int(action[-2][1:]),int(action[-1][1:])
            metrics=score_cell(rows,gold)
            save_rows(path.with_name(path.stem+'-scored.jsonl'),rows)
            telemetry_path=path.with_name(path.stem+'-metrics.jsonl')
            telemetry=[json.loads(line) for line in telemetry_path.read_text().splitlines()] if telemetry_path.exists() else []
            cells.append(dict(n=n,budget=b,metrics=metrics,telemetry=summarize_telemetry(telemetry)))
        # A cell must have passed post-run idle validation for its block to be complete.
        reports.append(dict(block=block.name,paired_complete=block.name in manifest['completed'],cells=cells))
    write_json(out/'summary.json',dict(blocks=reports,paired_differences=paired_differences(reports),status=manifest['status'],
               caveat='two seeds/five questions; descriptive paired comparisons only; no routing benefit claim'))


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('run')
    for name in ('plan','background','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--url',default='http://127.0.0.1:8001');p.add_argument('--model',default='m')
    p.add_argument('--deadline-epoch',required=True,type=float);p.set_defaults(fn=run)
    p=sub.add_parser('score');p.add_argument('--out',required=True);p.add_argument('--gold',required=True);p.set_defaults(fn=score)
    args=parser.parse_args();args.fn(args)

if __name__=='__main__':main()
