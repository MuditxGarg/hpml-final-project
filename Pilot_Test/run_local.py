#!/usr/bin/env python3
"""Run the LATTE go/no-go pipeline on a local Linux / WSL2 GPU machine.

    python run_local.py smoke    # ~15 min, proves the plumbing works (numbers meaningless)
    python run_local.py main     # the real run; safe to re-run, finished chunks are skipped

Results go to ./results/{smoke,main}; the verdict is in results/main/report/report.md
"""
import json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN_DIR = os.environ.get('LATTE_RUN_DIR', str(HERE / 'results'))
os.environ['LATTE_DIR'] = str(HERE)
os.environ['LATTE_CONFIG'] = str(HERE / 'config.json')

gpu_line = os.environ.get('LATTE_FAKE_GPU')
if not gpu_line:
    try:
        gpu_line = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,compute_cap',
                                            '--format=csv,noheader,nounits'], text=True).strip().split('\n')[0]
    except Exception:
        gpu_line = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total',
                                            '--format=csv,noheader,nounits'], text=True).strip().split('\n')[0]
parts = [x.strip() for x in gpu_line.split(',')]
GPU_NAME, MEM_GB = parts[0], float(parts[1]) / 1024
CC = float(parts[2]) if len(parts) > 2 else 8.0
DTYPE = 'bfloat16' if CC >= 8.0 else 'half'
if MEM_GB >= 38:
    PROFILE = dict(model='Qwen/Qwen3-8B', n_prompts=300, S_levels=[1, 8, 32, 64, 128, 256], max_num_seqs=256)
elif MEM_GB >= 22:
    PROFILE = dict(model='Qwen/Qwen3-4B', n_prompts=200, S_levels=[1, 8, 32, 64, 128], max_num_seqs=192)
else:
    PROFILE = dict(model='Qwen/Qwen3-1.7B', n_prompts=150, S_levels=[1, 8, 24, 48, 96], max_num_seqs=128)
print(f'GPU: {GPU_NAME} | {MEM_GB:.0f} GB | dtype {DTYPE} | -> model {PROFILE["model"]}, {PROFILE["n_prompts"]} prompts')


def write_config(smoke):
    cfg = dict(model=PROFILE['model'], dtype=DTYPE, run_dir=RUN_DIR, seed=20261030, gpu_mem_util=float(os.environ.get('LATTE_GPU_UTIL', '0.85')),
               max_num_seqs=PROFILE['max_num_seqs'], smoke=smoke)
    if smoke:
        cfg.update(n_prompts=6, k_traces=4, chunk=3, t_max=256, budgets=[0, 64, 128, 256], max_model_len=2048,
                   load=dict(S_levels=[1, 8], actions=[[1, 128], [4, 32]], dur_s=8, warm_s=3, idle_reps=1))
    else:
        cfg.update(n_prompts=PROFILE['n_prompts'], k_traces=16, chunk=20, t_max=2048,
                   budgets=[0, 256, 512, 1024, 2048], max_model_len=5120,
                   load=dict(S_levels=PROFILE['S_levels'],
                             actions=[[1, 2048], [4, 512], [16, 128], [1, 4096], [4, 1024], [16, 256]],
                             dur_s=45, warm_s=10, idle_reps=2))
    json.dump(cfg, open(HERE / 'config.json', 'w'), indent=1)
    return cfg


SHOW = ('[rollouts]', '[probe]', '[load]')


def run_stage(name, smoke, retries=0):
    for attempt in range(retries + 1):
        t0 = time.time()
        p = subprocess.Popen([sys.executable, str(HERE / f'{name}.py')], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, env=dict(os.environ))
        tail, log = [], []
        for line in p.stdout:
            log.append(line)
            tail = (tail + [line])[-25:]
            if line.startswith(SHOW) or 'Traceback' in line or 'Error' in line or name == 'summarize':
                print(line, end='', flush=True)
        rc = p.wait()
        print(f'--- {name}: exit {rc} after {(time.time() - t0) / 60:.1f} min (attempt {attempt + 1})', flush=True)
        d = Path(RUN_DIR) / ('smoke' if smoke else 'main') / 'logs'
        d.mkdir(parents=True, exist_ok=True)
        (d / f'{name}.log').write_text(''.join(log))
        if rc == 0:
            return True
        print('last lines before failure:\n' + ''.join(tail[-12:]), flush=True)
    return False


def run_all(smoke):
    ok = {}
    ok['rollouts'] = run_stage('stage_rollouts', smoke, retries=2)
    ok['probe'] = run_stage('stage_probe', smoke, retries=1)
    ok['summary_partial'] = run_stage('summarize', smoke)
    ok['load'] = run_stage('stage_load', smoke, retries=1)
    ok['summary'] = run_stage('summarize', smoke)
    return ok


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'smoke'
    assert mode in ('smoke', 'main'), 'usage: python run_local.py smoke|main'
    smoke = mode == 'smoke'
    cfg = write_config(smoke)
    t0 = time.time()
    ok = run_all(smoke)
    rep = Path(RUN_DIR) / mode / 'report' / 'report.md'
    print(f'\nstages: {ok} | {(time.time() - t0) / 60:.1f} min')
    if smoke:
        print('SMOKE OK (numbers are meaningless, only the plumbing matters)' if all(ok.values()) and rep.exists()
              else 'SMOKE FAILED - fix before running main')
    else:
        print('\nReport:', rep)
