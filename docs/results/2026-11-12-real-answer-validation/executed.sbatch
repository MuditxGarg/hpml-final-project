#!/bin/bash
#SBATCH --job-name=latte-real-answers
#SBATCH --no-requeue
#SBATCH --account=edu
#SBATCH --partition=short
#SBATCH --qos=free_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:05:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=latte-real-answers-%j.out
#SBATCH --error=latte-real-answers-%j.err
set -euo pipefail
remote_root="${SLURM_SUBMIT_DIR}"
local_root="/tmp/latte-real-${USER}-${SLURM_JOB_ID}"
result_root="${remote_root}/results/job-${SLURM_JOB_ID}"
mkdir -p "${local_root}" "${result_root}"
start_epoch=$(date +%s)
server_pid=''
function save_results() {
    status=$?
    trap - EXIT TERM INT
    if [ -n "${server_pid}" ]; then kill "${server_pid}" 2>/dev/null || true; fi
    if [ -d "${local_root}/run" ]; then cp -r "${local_root}/run/." "${result_root}/"; fi
    for name in server.log install.log hardware.txt environment.txt server_command.txt; do
        if [ -f "${local_root}/${name}" ]; then cp "${local_root}/${name}" "${result_root}/"; fi
    done
    printf 'JOB_EXIT %s status=%s\n' "$(date -Is)" "${status}"
    exit "${status}"
}
trap save_results EXIT
trap 'exit 143' TERM INT
cp -r "${remote_root}/code" "${remote_root}/input" "${local_root}/"
cd "${local_root}"
module load anaconda/2023.09
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HOME="${local_root}/hf-cache"
export PIP_NO_CACHE_DIR=1
nvidia-smi > hardware.txt
python -m venv .venv
.venv/bin/python -m pip install --no-index --find-links "${remote_root}/wheelhouse" uv > install.log 2>&1
export UV_CACHE_DIR="${local_root}/uv-cache"
export UV_OFFLINE=1 HF_HUB_OFFLINE=1
.venv/bin/uv pip install --python .venv/bin/python --no-index \
    --find-links "${remote_root}/wheelhouse" \
    -r "${remote_root}/requirements-offline.lock" >> install.log 2>&1
.venv/bin/uv pip check --python .venv/bin/python >> install.log 2>&1
model_snapshot="${HF_HOME}/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
mkdir -p "${model_snapshot}"
cp -r "${remote_root}/model/." "${model_snapshot}/"
.venv/bin/python -m pip freeze > environment.txt
export PATH="${local_root}/.venv/bin:${PATH}"
server_args=(vllm serve Qwen/Qwen3-1.7B --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --tokenizer-revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --dtype half --max-model-len 5120 --max-num-seqs 64 --gpu-memory-utilization .85 --served-model-name m --host 127.0.0.1 --port 8001 --no-enable-prefix-caching)
printf '%q ' "${server_args[@]}" > server_command.txt
"${server_args[@]}" > server.log 2>&1 &
server_pid=$!
export LATTE_SERVER_PID="${server_pid}"
export LATTE_SESSION_DEADLINE_EPOCH=$((start_epoch+3780))
python - <<'PY'
import os,time,urllib.request
limit=min(time.time()+900,int(os.environ['LATTE_SESSION_DEADLINE_EPOCH'])-900)
while time.time()<limit:
    try: os.kill(int(os.environ['LATTE_SERVER_PID']),0)
    except ProcessLookupError: raise SystemExit('Model server exited; see server.log')
    try:
        with urllib.request.urlopen('http://127.0.0.1:8001/health',timeout=2) as r:
            if r.status==200: break
    except Exception: time.sleep(2)
else: raise SystemExit('Server not ready within setup budget')
PY
# Short unscored warmup, followed by an idle check in the runner.
python - <<'PY'
import json,urllib.request
body=json.dumps(dict(model='m',prompt='Compute 1+1:',max_tokens=8,temperature=0)).encode()
with urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8001/v1/completions',data=body,headers={'Content-Type':'application/json'}),timeout=60) as r: print(r.read().decode())
PY
deadline=$(python -c 'import os,datetime; print(datetime.datetime.fromtimestamp(int(os.environ["LATTE_SESSION_DEADLINE_EPOCH"]),datetime.timezone.utc).isoformat())')
python code/real_answer_validation.py run --plan input/plan.json --out run --deadline-utc "${deadline}"
python code/real_answer_validation.py score --run run --gold input/gold.json
