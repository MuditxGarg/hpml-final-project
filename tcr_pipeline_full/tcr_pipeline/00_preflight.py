"""Stage 0: fail fast, before any long job, if the machine is not ready."""
import os, shutil, socket, sys
sys.path.insert(0, os.environ.get('TCR_DIR', '.'))
from tcr_common import *

problems = []
free_gb = shutil.disk_usage(str(RUN)).free / 1e9
print(f'[preflight] free disk: {free_gb:.0f} GB', flush=True)
if free_gb < 15:
    problems.append(f'only {free_gb:.0f} GB free disk (need ~15 GB)')
try:
    import torch, vllm
    print(f'[preflight] torch {torch.__version__}, vllm {vllm.__version__}, cuda {torch.cuda.is_available()}', flush=True)
    if not torch.cuda.is_available():
        problems.append('torch cannot see the GPU')
    else:
        free, tot = torch.cuda.mem_get_info()
        print(f'[preflight] GPU memory free {free / 1e9:.1f} / {tot / 1e9:.1f} GB', flush=True)
        if free < 0.8 * CFG['gpu_mem_util'] * tot:
            problems.append(f'GPU has only {free / 1e9:.1f} GB free; close other GPU programs (or an old vLLM server) first')
except Exception as e:
    problems.append(f'cannot import torch/vllm: {e}')
try:
    s = socket.socket()
    s.bind(('127.0.0.1', 8001))
    s.close()
except OSError:
    problems.append('port 8001 is busy (an old vLLM server is still running: `pkill -f "vllm serve"`)')
try:
    load_problems()
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(CFG['model'])
    print('[preflight] dataset and tokenizer available', flush=True)
except Exception as e:
    problems.append(f'cannot load MATH-500 or the tokenizer (internet/HF cache?): {e}')
if problems:
    print('[preflight] PROBLEMS:\n  - ' + '\n  - '.join(problems), flush=True)
    sys.exit(1)
print('[preflight] OK', flush=True)
