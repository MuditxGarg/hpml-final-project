#!/usr/bin/env python3
"""Bounded real-T4 E1 pilot. Run beside matched_serving.py, lengths.json, prompts.json.

LATTE_GPU_DEADLINE must be an epoch timestamp or an ISO UTC timestamp.
Never installs dependencies or starts a server. Always archives partial results.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import threading
import time

import matched_serving as replay

ARCHIVE_MARGIN_S = 60
ACTIONS = [(1, 0), (4, 0), (1, 2048), (4, 2048)]


def choose_calibration(idle_latencies, trials):
    valid = [t for t in trials if t['completed'] >= 3 and t['errors'] == 0 and t['elapsed_s'] > 0]
    if not valid or len(idle_latencies) < 5:
        raise RuntimeError('insufficient error-free local calibration; refusing imported capacity/SLO')
    capacity = max(t['completed']/t['elapsed_s'] for t in valid)
    slo = math.ceil(4 * replay.percentile(idle_latencies))
    return dict(reference_capacity_jobs_s=capacity, slo_s=slo, target_rate=.10*capacity,
                capacity_definition='maximum error-free completed reference requests / total trial wall time including drain',
                slo_definition='ceil(4 * nearest-rank p95 of 5 isolated local B1024 requests); pilot heuristic, not SLA',
                target_fraction=.10, rho_values=[.5,.8])


def can_start_block(now, deadline, cells=5, warmup=10, duration=45, drain=60, idle_timeout=15):
    # Two idle waits per cell; extra five seconds cover telemetry/thread/file overhead.
    return now + cells*(warmup+duration+drain+2*idle_timeout+5) + ARCHIVE_MARGIN_S < deadline


def deadline_value(value):
    try:
        return float(value)
    except ValueError:
        parsed = datetime.fromisoformat(value.replace('Z','+00:00'))
        if parsed.tzinfo is None:
            raise ValueError('ISO LATTE_GPU_DEADLINE must include UTC/timezone')
        return parsed.timestamp()


def log(message):
    print(datetime.now(timezone.utc).isoformat(timespec='seconds'), message, flush=True)


def command_output(command):
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=10, check=False).stdout.strip()
    except Exception as exc:
        return 'unavailable: '+str(exc)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def request_reference(url, model, prompts, lengths, sample, timeout):
    started = time.monotonic()
    rng = random.Random(f't4-calibration:{sample}')
    body = dict(model=model, prompt=prompts[rng.randrange(len(prompts))]['thinking'],
                max_tokens=rng.choice(lengths['1024']), temperature=.6, top_p=.95,
                ignore_eos=True, stream=False, seed=731)
    row = dict(requested_tokens=body['max_tokens'], sample=sample)
    try:
        answer = json.loads(replay.http(url+'/v1/completions',body,timeout=timeout))
        actual = answer.get('usage',{}).get('completion_tokens')
        if not answer.get('choices') or actual != body['max_tokens']:
            raise ValueError(f'invalid replay response: completion_tokens={actual}')
        row.update(status='completed', completion_tokens=actual)
    except Exception as exc:
        row.update(status='error', error=str(exc))
    row['latency_s'] = time.monotonic()-started
    return row


def calibrate(url, model, prompts, lengths, out, deadline):
    log('Local T4 calibration: 5 isolated B1024 requests, then concurrency 4/16/32 for 12s each')
    idle = []
    trials = []
    replay.wait_idle(url, timeout_s=15)
    for index in range(5):
        if deadline-time.time() < ARCHIVE_MARGIN_S+90:
            raise TimeoutError('deadline reserve reached during idle calibration')
        row = request_reference(url, model, prompts, lengths, 'idle-'+str(index), timeout=60)
        idle.append(row)
        write_json(out/'calibration.json', dict(idle=idle,trials=trials))
        if row['status'] != 'completed':
            raise RuntimeError('isolated calibration failed: '+str(row))
        log(f'Idle reference {index+1}/5: {row["latency_s"]:.2f}s, {row["completion_tokens"]} tokens')
    for concurrency in (4,16,32):
        if deadline-time.time() < ARCHIVE_MARGIN_S+105:
            raise TimeoutError('deadline reserve reached before capacity calibration')
        replay.wait_idle(url, timeout_s=15)
        began = time.monotonic()
        stop_at = began+12
        def worker(worker_index):
            rows = []
            seq = 0
            while time.monotonic() < stop_at:
                row = request_reference(url,model,prompts,lengths,f'c{concurrency}-w{worker_index}-{seq}',timeout=60)
                row['finished_at_s'] = time.monotonic()-began
                rows.append(row); seq += 1
                if row['status'] != 'completed': break
            return rows
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            jobs = [pool.submit(worker,i) for i in range(concurrency)]
            rows = [row for job in as_completed(jobs) for row in job.result()]
        elapsed = time.monotonic()-began
        completed = [r for r in rows if r['status']=='completed']
        trial = dict(concurrency=concurrency, dispatch_window_s=12, elapsed_s=elapsed,
                     completed=len(completed), errors=len(rows)-len(completed),
                     completed_tokens=sum(r['completion_tokens'] for r in completed), raw_requests=rows)
        trial['completed_requests_s'] = len(completed)/elapsed
        trial['completed_tokens_s'] = trial['completed_tokens']/elapsed
        trials.append(trial)
        write_json(out/'calibration.json',dict(idle=idle,trials=trials))
        log(f'Calibration C={concurrency}: {trial["completed_requests_s"]:.3f} req/s, {trial["completed_tokens_s"]:.1f} tok/s, errors={trial["errors"]}')
        replay.wait_idle(url, timeout_s=15)
    selected = choose_calibration([r['latency_s'] for r in idle],trials)
    write_json(out/'calibration.json',dict(idle=idle,trials=trials,selected=selected))
    log(f'Local capacity={selected["reference_capacity_jobs_s"]:.3f} req/s; pilot SLO={selected["slo_s"]}s; target={selected["target_rate"]:.3f} jobs/s')
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:8001')
    parser.add_argument('--model',default='m')
    parser.add_argument('--lengths',default='lengths.json')
    parser.add_argument('--prompts',default='prompts.json')
    parser.add_argument('--output',default=None)
    args = parser.parse_args(argv)
    deadline = deadline_value(os.environ['LATTE_GPU_DEADLINE'])
    out = Path(args.output or ('t4-pilot-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))).resolve()
    out.mkdir(parents=True,exist_ok=False)
    state = dict(experiment='pilot real-T4 synthetic token-count externality; not quality/routing proof',
                 deadline_epoch=deadline, started_epoch=time.time(), status='running', completed_blocks=[], skipped_blocks=[],
                 note='capacity/SLO calibrated locally; no RTX4080 rates or latency reused; target fraction .10 rather than old .05')
    write_json(out/'pilot-summary.json',state)
    monitor_stop = threading.Event()
    def monitor():
        with (out/'gpu-metrics.jsonl').open('w') as f:
            while not monitor_stop.is_set():
                row = dict(epoch=time.time(), csv=command_output(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw','--format=csv,noheader,nounits']))
                f.write(json.dumps(row)+'\n'); f.flush()
                if monitor_stop.wait(5): break
    monitor_thread = threading.Thread(target=monitor,daemon=True)
    monitor_thread.start()
    try:
        lengths = json.loads(Path(args.lengths).read_text())
        prompts = json.loads(Path(args.prompts).read_text())
        if not all(str(b) in lengths for b in (0,1024,2048)):
            raise ValueError('lengths.json must include corrected B0 and budgets1024/2048')
        if deadline-time.time() < 360:
            raise TimeoutError('less than six minutes remain; refusing calibration')
        versions = {}
        for name in ('vllm','torch','transformers','numpy'):
            try: versions[name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError: versions[name]='not installed'
        hardware = dict(nvidia_smi=command_output(['nvidia-smi']), versions=versions, dtype='half',
                        prefix_caching_enabled=False, served_model=args.model, server_url=args.url,
                        other_clients='none; dedicated free Colab T4 assumed',
                        length_provenance='parent-supplied corrected B0-cap2048 empirical lengths; budgets1024/2048',
                        lengths_sha256=replay.digest(lengths),prompts_sha256=replay.digest(prompts))
        write_json(out/'hardware.json',hardware)
        if 'T4' not in hardware['nvidia_smi']:
            raise RuntimeError('this pilot is explicitly T4; nvidia-smi did not identify a T4')
        shutil.copy2(args.lengths,out/'lengths.json')
        shutil.copy2(args.prompts,out/'prompts.json')
        calibration = calibrate(args.url.rstrip('/'),args.model,prompts,lengths,out,deadline)
        state['calibration']=calibration
        write_json(out/'pilot-summary.json',state)
        # Interleave loads: each seed uses both loads, with the second seed reversed.
        blocks = [(731,.5),(731,.8),(947,.8),(947,.5)]
        for seed,rho in blocks:
            key = f'seed{seed}-rho{rho:.1f}'
            if not can_start_block(time.time(),deadline):
                state['skipped_blocks'].append(dict(block=key,reason='conservative block bound plus archive margin exceeds deadline'))
                write_json(out/'pilot-summary.json',state)
                log(f'SKIP {key}: insufficient deadline reserve')
                continue
            actions = list(ACTIONS)
            random.Random(f't4-order:{seed}:{rho}').shuffle(actions)
            trace = replay.generate_trace(lengths,prompts,seed,rho*calibration['reference_capacity_jobs_s'],
                calibration['target_rate'],warmup=10,duration=45,max_n=4)
            trace['sources'] = dict(lengths=str(Path(args.lengths).resolve()), prompts=str(Path(args.prompts).resolve()))
            trace_path = out/f'{key}-trace.json'
            write_json(trace_path,trace)
            log(f'START {key}: background={trace["background_rate"]:.3f} jobs/s; order={actions}; {len(trace["events"])} scheduled events')
            state['active_block']=key
            write_json(out/'pilot-summary.json',state)
            replay.main(['run','--trace',str(trace_path),'--url',args.url,'--model',args.model,
                '--output',str(out/key),'--hardware-json',str(out/'hardware.json'),
                '--actions',*[f'{n}:{b}' for n,b in actions],'--reference-budget','1024',
                '--cache-policy','disabled','--slo',str(calibration['slo_s']),'--attainment','.95',
                '--timeout','60','--drain','60','--idle-timeout','15','--max-inflight','128','--metrics-interval','1'])
            state['completed_blocks'].append(dict(block=key,seed=seed,rho=rho,summary=json.loads((out/key/'summary.json').read_text())))
            state.pop('active_block',None)
            write_json(out/'pilot-summary.json',state)
            log(f'COMPLETE {key}; saved {len(state["completed_blocks"])} matched blocks')
        state['status']='complete' if not state['skipped_blocks'] else 'partial_deadline'
    except BaseException as exc:
        state.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        log('PILOT STOPPED: '+state['error'])
        raise
    finally:
        state['finished_epoch']=time.time()
        write_json(out/'pilot-summary.json',state)
        monitor_stop.set(); monitor_thread.join(4)
        archive = shutil.make_archive(str(out),'zip',root_dir=out.parent,base_dir=out.name)
        log(f'ARCHIVE_READY {archive}; status={state["status"]}')
        print('Download in Colab: from google.colab import files; files.download('+repr(archive)+')',flush=True)


if __name__ == '__main__':
    main()
