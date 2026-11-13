"""Idle real-answer diagnostic: prepare a fixed cohort, generate, then score offline."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import queue
import threading
import socket
import time

from answer_scoring import last_boxed, norm_key, is_correct
from matched_serving import http, wait_idle, percentile

MODEL = 'Qwen/Qwen3-1.7B'
REVISION = '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
NATURAL = '</think>\n\nThe final answer is $\\boxed{'
FORCED = '\n\nConsidering the limited time, I have to give the final answer now.\n</think>\n\nThe final answer is $\\boxed{'
ACTIONS = [(1, 0), (4, 0), (1, 2048)]


def select_questions(rows, seed):
    rng = random.Random(seed)
    return [rng.choice(sorted([r for r in rows if int(r['level']) == lv], key=lambda r:r['id'])) for lv in range(1, 6)]


def generate_sample(prompts, budget, seed, tokenizer, send, timeout_s):
    start = time.monotonic()
    stages = []
    result = dict(status='pending', pred=None, stages=stages, seed=seed)
    def enc(text): return tokenizer.encode(text, add_special_tokens=False)
    def call(prompt, max_tokens, final=False):
        remaining = timeout_s - (time.monotonic()-start)
        if remaining <= 0: raise TimeoutError('shared action observation deadline')
        body = dict(prompt=prompt, max_tokens=max_tokens, temperature=0.0 if final else .6,
                    top_p=1.0 if final else .95, top_k=-1 if final else 20,
                    seed=seed, ignore_eos=False, stream=False, return_token_ids=True)
        if budget and not final: body['stop']=['</think>']
        t = time.monotonic()
        stage=dict(request=body, status='pending', stage='final' if final else 'thinking' if budget else 'nothinking')
        stages.append(stage)
        mailbox=queue.Queue(maxsize=1)
        def transport():
            try: mailbox.put((True, send(body, remaining)))
            except Exception as exc: mailbox.put((False, exc))
        threading.Thread(target=transport,daemon=True).start()
        try:
            try: success, response=mailbox.get(timeout=max(.001, timeout_s-(time.monotonic()-start)))
            except queue.Empty: raise TimeoutError('absolute action observation deadline')
            if time.monotonic()-start >= timeout_s: raise TimeoutError('absolute action observation deadline')
            if not success: raise response
            stage.update(response=response,status='completed')
        except Exception as exc:
            stage.update(status='timeout' if isinstance(exc,(TimeoutError,socket.timeout)) else 'error',error=str(exc))
            raise
        finally: stage['seconds']=time.monotonic()-t
        choice = response['choices'][0]
        if choice.get('finish_reason') not in ('length', 'stop'):
            raise ValueError('unexpected completion finish reason')
        return choice
    try:
        if budget == 0:
            choice = call(enc(prompts['nothinking']), 2048)
            result.update(pred=last_boxed(choice['text']), cap_hit=choice['finish_reason']=='length')
        else:
            # Preserve the original labeler's explicit opening marker for comparability.
            base = enc(prompts['thinking']+'<think>\n')
            think = call(base, budget)
            ids = think.get('token_ids')
            if not isinstance(ids, list): raise ValueError('server omitted token_ids; cannot reproduce continuation')
            close = enc('</think>')
            if close and ids[-len(close):] == close: ids=ids[:-len(close)]
            natural = think.get('stop_reason') == '</think>'
            result['thinking_stopped_naturally'] = natural
            suffix = NATURAL if natural else FORCED
            choice = call(base+ids+enc(suffix), 48, final=True)
            result.update(pred=last_boxed('\\boxed{'+choice['text']),
                          cap_hit=choice['finish_reason']=='length', thinking_cap_hit=think['finish_reason']=='length')
        result['status']='completed'
    except (TimeoutError, socket.timeout) as exc:
        result.update(status='timeout', error=str(exc))
    except Exception as exc:
        result.update(status='error', error=f'{type(exc).__name__}: {exc}')
    result.update(seconds=time.monotonic()-start,
                  completion_tokens=sum(s.get('response',{}).get('usage',{}).get('completion_tokens',0) for s in stages),
                  token_count_complete=result['status']=='completed')
    return result


def aggregate(parts):
    if any(p['status']!='completed' for p in parts):
        return dict(status='timeout' if any(p['status']=='timeout' for p in parts) else 'error', pred=None)
    keys = [norm_key(p['pred']) for p in parts]
    counts = Counter(k for k in keys if k is not None)
    winner = max(counts, key=counts.get) if counts else None
    pred = next((p['pred'] for p,k in zip(parts, keys) if k==winner), None) if winner else None
    return dict(status='completed' if pred is not None else 'no_answer', transport_completed=True, pred=pred, vote_keys=keys, valid_samples=sum(k is not None for k in keys))


def score_rows(rows, gold, slo, verifier=is_correct):
    result=[]
    for row in rows:
        ok=row['status']=='completed' and row.get('pred') is not None and bool(verifier(gold[row['id']],row['pred']))
        within=row['status']=='completed' and row['latency_s']<=slo
        result.append(dict(row, correct=ok, within_slo=within, correct_within_slo=ok and within))
    return result


def write_json(path, obj):
    path=Path(path)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2)+'\n'); tmp.replace(path)


def prepare(args):
    prompts=json.loads(Path(args.prompts).read_text())
    labels=[r for p in sorted(Path(args.labels).glob('chunk_*.json')) for r in json.loads(p.read_text())]
    by_id={r['id']:r for r in labels}
    if len(by_id)!=len(labels): raise ValueError('duplicate label IDs')
    if {p['id'] for p in prompts}!=set(by_id): raise ValueError('prompt/label ID coverage differs')
    rows=[dict(p,level=by_id[p['id']]['level']) for p in prompts]
    cohort=select_questions(rows,args.seed)
    blocks=[]
    for p in cohort:
        actions=list(ACTIONS);random.Random(f"{args.seed}:{p['id']}:order").shuffle(actions)
        blocks.append(dict(id=p['id'],level=p['level'],actions=actions))
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    plan=dict(name='Real Answer Completion Validation',model=MODEL,revision=REVISION,seed=args.seed,
              slo_s=87,observation_timeout_s=180,blocks=blocks,prompts=cohort,
              vote_rule='plurality of valid normalized boxes; ties by lowest sample index; all N transports must complete',
              scope='five-question idle diagnostic; no routing or population accuracy claim',
              prompt_sha256=hashlib.sha256(Path(args.prompts).read_bytes()).hexdigest(),
              label_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(args.labels).glob('chunk_*.json'))})
    write_json(out/'plan.json',plan)
    write_json(out/'gold.json',{p['id']:by_id[p['id']]['gold'] for p in cohort})
    print(json.dumps(blocks,indent=2))


def run(args):
    from transformers import AutoTokenizer
    plan=json.loads(Path(args.plan).read_text())
    deadline=datetime.fromisoformat(args.deadline_utc.replace('Z','+00:00'))
    if deadline.utcoffset() is None: raise ValueError('deadline must contain UTC timezone')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    write_json(out/'plan.json',plan)
    tokenizer=AutoTokenizer.from_pretrained(plan['model'],revision=plan['revision'])
    versions={p:importlib.metadata.version(p) for p in ('transformers',)}
    write_json(out/'environment.json',dict(versions=versions,server_url=args.url,model=args.model,
               deadline_utc=deadline.isoformat(),slurm_job_id=os.environ.get('SLURM_JOB_ID'),
               plan_sha256=hashlib.sha256(Path(args.plan).read_bytes()).hexdigest(),
               server_models=json.loads(http(args.url+'/v1/models'))))
    prompts={p['id']:p for p in plan['prompts']}
    state=dict(status='running',completed_blocks=[],skipped_blocks=[],started_utc=datetime.now(timezone.utc).isoformat())
    write_json(out/'status.json',state)
    try:
        for index,block in enumerate(plan['blocks']):
            # Admission reserves every action's worst-case observation and two idle checks.
            bound=len(block['actions'])*(plan['observation_timeout_s']+2*args.idle_timeout)+180
            if (deadline-datetime.now(timezone.utc)).total_seconds()<bound:
                state['skipped_blocks']=plan['blocks'][index:];state['status']='partial_deadline';break
            for n,budget in block['actions']:
                before=wait_idle(args.url,args.idle_timeout)
                start=time.monotonic()
                def worker(k):
                    seed=int(hashlib.sha256(f"{plan['seed']}:{block['id']}:{k}".encode()).hexdigest()[:8],16)%(2**31)
                    def send(body,timeout):
                        return json.loads(http(args.url+'/v1/completions',dict(body,model=args.model),timeout))
                    remaining=plan['observation_timeout_s']-(time.monotonic()-start)
                    return generate_sample(prompts[block['id']],budget,seed,tokenizer,send,remaining)
                with ThreadPoolExecutor(max_workers=n) as pool:
                    parts=list(pool.map(worker,range(n)))
                row=dict(id=block['id'],level=block['level'],n=n,budget=budget,parts=parts,**aggregate(parts))
                row['latency_s']=time.monotonic()-start
                row['before_metrics']=before
                # Persist even if the server cannot drain before the next action.
                with (out/'raw.jsonl').open('a') as f:
                    f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
                print(json.dumps({k:row[k] for k in ('id','n','budget','status','pred','latency_s')}),flush=True)
                wait_idle(args.url,args.idle_timeout)
            state['completed_blocks'].append(block['id']);write_json(out/'status.json',state)
        else: state['status']='complete'
    except Exception as exc:
        state.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        state['ended_utc']=datetime.now(timezone.utc).isoformat();write_json(out/'status.json',state)


def score(args):
    # Require the same symbolic backend as project labeling; don't silently fall back to exact strings.
    import math_verify
    run_dir=Path(args.run);plan=json.loads((run_dir/'plan.json').read_text())
    raw=[json.loads(x) for x in (run_dir/'raw.jsonl').read_text().splitlines()]
    rows=score_rows(raw,json.loads(Path(args.gold).read_text()),plan['slo_s'])
    write_json(run_dir/'scored.json',rows)
    summaries=[]
    for n,b in ACTIONS:
        group=[r for r in rows if r['n']==n and r['budget']==b]
        summaries.append(dict(n=n,budget=b,observed=len(group),planned=len(plan['blocks']),
                             completed=sum(r['status']=='completed' for r in group),
                             correct=sum(r['correct'] for r in group),
                             within_slo=sum(r['within_slo'] for r in group),
                             correct_within_slo=sum(r['correct_within_slo'] for r in group),
                             completed_p95_s=percentile([r['latency_s'] for r in group if r['status']=='completed'])))
    write_json(run_dir/'summary.json',dict(actions=summaries,math_verify=importlib.metadata.version('math-verify'),
               scope=plan['scope'],status=json.loads((run_dir/'status.json').read_text()),
               note='Counts use all observed attempts, including failures. Unoffered blocks are not failures; see planned/observed counts. Completed-only p95 excludes failures. No throughput estimate from serial idle trials.'))
    lines=['# Real Answer Completion Validation','',plan['scope'],'',
           '| N | Thinking budget | Observed / planned | Completed | Correct | Within 87s | Correct within 87s |',
           '|---|---|---|---|---|---|---|']
    for s in summaries: lines.append(f"| {s['n']} | {s['budget']} | {s['observed']} / {s['planned']} | {s['completed']} | {s['correct']} | {s['within_slo']} | {s['correct_within_slo']} |")
    lines+=['','Inspect raw.jsonl for finish reasons, cap hits, missing boxes, two-stage timings and failed requests. Hardware differs from previous T4 trials if this runs on the school cluster; do not pool latency results.']
    (run_dir/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--prompts',required=True);a.add_argument('--labels',required=True);a.add_argument('--out',required=True);a.add_argument('--seed',type=int,default=20261115);a.set_defaults(fn=prepare)
    a=sub.add_parser('run');a.add_argument('--plan',required=True);a.add_argument('--out',required=True);a.add_argument('--url',default='http://127.0.0.1:8001');a.add_argument('--model',default='m');a.add_argument('--deadline-utc',required=True);a.add_argument('--idle-timeout',type=float,default=30);a.set_defaults(fn=run)
    a=sub.add_parser('score');a.add_argument('--run',required=True);a.add_argument('--gold',required=True);a.set_defaults(fn=score)
    args=p.parse_args();args.fn(args)

if __name__=='__main__':main()
