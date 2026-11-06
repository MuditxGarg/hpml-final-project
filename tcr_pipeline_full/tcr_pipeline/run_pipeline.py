#!/usr/bin/env python3
"""TCR pipeline: label -> probe -> decide values -> live-server calibration -> externality sweep.

    python run_pipeline.py smoke            # ~15-25 min plumbing check (numbers meaningless)
    python run_pipeline.py serve --rollouts <pilot rollouts folder>   # ~1-1.5 h: live-server sweep first, reuses pilot labels
    python run_pipeline.py main [--hours 9] # the real run; resumable, re-run the same command after any crash

Every stage is resumable. A failed stage does not stop later stages that do not depend on it.
Outputs: tcr_results/main/{decisions.json, lengths.json, load/, probe/, rollouts/, report/overnight_report.md}
"""
import json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
mode = sys.argv[1] if len(sys.argv) > 1 else 'smoke'
RUN_DIR = os.environ.get('TCR_RUN_DIR', str(HERE / ('tcr_results_quick' if mode == 'serve' else 'tcr_results')))
os.environ['TCR_DIR'] = str(HERE)
os.environ['TCR_CONFIG'] = str(HERE / 'tcr_config.json')
assert mode in ('smoke', 'main', 'serve'), 'usage: python run_pipeline.py smoke|main|serve [--hours H] [--rollouts DIR]'
hours = float(sys.argv[sys.argv.index('--hours') + 1]) if '--hours' in sys.argv else 9.0
smoke = mode == 'smoke'
quick = mode == 'serve'   # serve-first: reuse the pilot labels, run only decide + live-server sweep (~1-1.5 h)
T0 = time.time()
# Hardening for unattended runs: avoid vLLM's flashinfer sampler (JIT-compiles with nvcc/ninja and fails on this setup)
os.environ.setdefault('VLLM_USE_FLASHINFER_SAMPLER', '0')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
os.environ['TCR_DEADLINE'] = str(T0 + hours * 3600)

try:
    line = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,compute_cap', '--format=csv,noheader,nounits'],
                                   text=True).strip().split('\n')[0]
except Exception:
    line = os.environ.get('TCR_FAKE_GPU', 'FAKE,12288,8.9')
parts = [x.strip() for x in line.split(',')]
GPU, MEM_GB, CC = parts[0], float(parts[1]) / 1024, float(parts[2]) if len(parts) > 2 else 8.0
DTYPE = 'bfloat16' if CC >= 8.0 else 'half'
if MEM_GB >= 38:
    MODEL, SEQS = 'Qwen/Qwen3-8B', 256
elif MEM_GB >= 22:
    MODEL, SEQS = 'Qwen/Qwen3-4B', 192
else:
    MODEL, SEQS = 'Qwen/Qwen3-1.7B', 128
print(f'GPU {GPU} {MEM_GB:.0f} GB -> {MODEL}; mode {mode}; time budget {hours} h', flush=True)

cfg = dict(model=MODEL, dtype=DTYPE, run_dir=RUN_DIR, seed=20261104, gpu_mem_util=float(os.environ.get('TCR_GPU_UTIL', '0.85')),
           max_num_seqs=SEQS, smoke=smoke)
if smoke:
    cfg.update(n_prompts=8, k_traces=4, chunk=4, t_max=256, budgets=[0, 64, 128, 256], max_model_len=2048,
               serve=dict(ref_B=128, actions_B=[64, 256], idle_reps=3, slo_mult=[2, 4], capacity_clients=[2, 4, 8],
                          cap_warm_s=2, cap_dur_s=6, fractions=[0, 0.5, 1.0], target_rate_frac=0.15, reps=1,
                          dur_s=10, idle_dur_s=6, warm_s=3, grace_s=10, max_inflight=100))
elif quick:
    cfg.update(n_prompts=150, k_traces=16, chunk=20, t_max=2048, budgets=[0, 256, 512, 1024, 2048], max_model_len=5120,
               serve=dict(ref_B=1024, actions_B=[1024, 2048], idle_reps=16, slo_mult=[2, 4],
                          capacity_clients=[8, 16, 32, 64, 96, 128], cap_warm_s=15, cap_dur_s=45,
                          fractions=[0, 0.6, 0.9, 1.1], target_rate_frac=0.10, reps=1,
                          dur_s=90, idle_dur_s=45, warm_s=15, grace_s=90, max_inflight=500))
else:
    cfg.update(n_prompts=500, k_traces=16, chunk=20, t_max=2048, budgets=[0, 256, 512, 1024, 2048], max_model_len=5120,
               serve=dict(ref_B=1024, actions_B=[0, 1024, 2048], idle_reps=24, slo_mult=[2, 4],
                          capacity_clients=[8, 16, 32, 64, 96, 128], cap_warm_s=15, cap_dur_s=50,
                          fractions=[0, 0.5, 0.8, 1.0, 1.3], target_rate_frac=0.05, reps=2,
                          dur_s=120, idle_dur_s=60, warm_s=15, grace_s=90, max_inflight=500))
json.dump(cfg, open(HERE / 'tcr_config.json', 'w'), indent=1)
LOGDIR = Path(RUN_DIR) / ('smoke' if smoke else 'main') / 'logs'
LOGDIR.mkdir(parents=True, exist_ok=True)
SHOW = ('[preflight]', '[rollouts]', '[probe]', '[decide]', '[serve]')


def run_stage(name, retries=0):
    for attempt in range(retries + 1):
        t0 = time.time()
        p = subprocess.Popen([sys.executable, str(HERE / f'{name}.py')], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, env=dict(os.environ))
        log, tail = [], []
        for ln in p.stdout:
            log.append(ln)
            tail = (tail + [ln])[-25:]
            if ln.startswith(SHOW) or 'Traceback' in ln or 'Error' in ln or name.endswith('report'):
                print(ln, end='', flush=True)
        rc = p.wait()
        print(f'--- {name}: exit {rc} after {(time.time() - t0) / 60:.1f} min (attempt {attempt + 1})', flush=True)
        (LOGDIR / f'{name}.log').write_text(''.join(log))
        if rc == 0:
            return True
        print('last lines:\n' + ''.join(tail[-12:]), flush=True)
        time.sleep(20)
    return False


ok = {}
if not os.environ.get('TCR_SKIP_PREFLIGHT') and not run_stage('00_preflight'):
    print('PREFLIGHT FAILED: fix the problems listed above, then re-run.', flush=True)
    sys.exit(1)
if quick:
    import shutil
    src = Path(sys.argv[sys.argv.index('--rollouts') + 1]) if '--rollouts' in sys.argv else None
    assert src and any(src.glob('chunk_*.json')), 'pass --rollouts <folder with the pilot chunk_*.json files>'
    dst = Path(RUN_DIR) / 'main' / 'rollouts'
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.glob('chunk_*.json'):
        shutil.copy(f, dst / f.name)
    ok['rollouts'], ok['probe'] = True, True
else:
    ok['rollouts'] = run_stage('01_label_rollouts', retries=2)
    ok['probe'] = run_stage('02_probe_features', retries=1) if ok['rollouts'] else False
ok['decide'] = run_stage('03_decide_values') if ok['rollouts'] else False
ok['serve'] = run_stage('04_serve_measure', retries=2) if ok['decide'] else False
ok['report'] = run_stage('05_make_report') if ok['decide'] else False
print(f'\nstages: {ok} | total {(time.time() - T0) / 60:.1f} min', flush=True)
rep = Path(RUN_DIR) / ('smoke' if smoke else 'main') / 'report' / 'overnight_report.md'
if smoke:
    print('SMOKE OK (numbers are meaningless)' if all(ok.values()) and rep.exists() else 'SMOKE FAILED: fix before running main')
else:
    print('Report:', rep)
