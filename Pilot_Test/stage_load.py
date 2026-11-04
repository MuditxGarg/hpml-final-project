"""Stage 3 (test 1): load sweep against a live vLLM server.

Action a=(N,T): a request asking for n=N completions of exactly T tokens each (ignore_eos), so the
token count is controlled. We measure
  * idle latency: sequential requests on an otherwise empty server
  * saturated cost: closed-loop clients (S concurrent sequences) -> generated tokens/s from the
    server's own counter. GPU-seconds per request at saturation = N*T / max tokens/s.
Compare equal-token actions, e.g. (1,2048) vs (4,512) vs (16,128).
"""
import http.client, json, os, random, re, shutil, socket, subprocess, sys, threading, time
from urllib.parse import urlparse
import requests

sys.path.insert(0, os.environ.get('LATTE_DIR', '/content/latte'))
from common import *

LC = CFG['load']
PORT = 8001
EXT = os.environ.get('LATTE_EXTERNAL_SERVER')
URL = EXT or f'http://127.0.0.1:{PORT}'
_u = urlparse(URL)
HOST, HPORT = _u.hostname, _u.port
out = RUN / 'load'
out.mkdir(exist_ok=True)
GEN = 'vllm:generation_tokens_total'


def start_server():
    exe = shutil.which('vllm')
    cmd = ([exe, 'serve', CFG['model']] if exe else
           [sys.executable, '-m', 'vllm.entrypoints.openai.api_server', '--model', CFG['model']])
    cmd += ['--dtype', CFG['dtype'], '--max-model-len', str(CFG['max_model_len']),
            '--gpu-memory-utilization', str(CFG['gpu_mem_util']), '--enable-prefix-caching',
            '--max-num-seqs', str(CFG['max_num_seqs']), '--port', str(PORT),
            '--served-model-name', 'm', '--seed', str(CFG['seed'])]
    print('[load] starting server:', ' '.join(cmd), flush=True)
    return subprocess.Popen(cmd, stdout=open(out / 'server.log', 'w'), stderr=subprocess.STDOUT)


def wait_ready(proc, timeout=1200):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc is not None and proc.poll() is not None:
            tail = open(out / 'server.log').read()[-3000:]
            raise RuntimeError('vLLM server exited early:\n' + tail)
        try:
            if requests.get(URL + '/health', timeout=3).status_code == 200:
                print(f'[load] server ready after {time.time() - t0:.0f}s', flush=True)
                return
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError('server did not become ready')


def metrics():
    try:
        txt = requests.get(URL + '/metrics', timeout=10).text
    except Exception:
        return {}
    d = {}
    for line in txt.splitlines():
        if line.startswith('#'):
            continue
        m = re.match(r'^(vllm:[a-zA-Z_]+)(\{[^}]*\})?\s+([0-9.eE+\-naif]+)$', line)
        if m:
            try:
                d[m.group(1)] = d.get(m.group(1), 0.0) + float(m.group(3))
            except ValueError:
                pass
    return d


def kv_usage(m):
    for k in ('vllm:kv_cache_usage_perc', 'vllm:gpu_cache_usage_perc'):
        if k in m:
            return m[k]
    return None


def body(prompt, N, T):
    return json.dumps(dict(model='m', prompt=prompt, max_tokens=T, n=N, temperature=0.6, top_p=0.95,
                           top_k=20, ignore_eos=True)).encode()


class Worker(threading.Thread):
    def __init__(self, N, T, prompts, stop):
        super().__init__(daemon=True)
        self.N, self.T, self.prompts, self.stop_evt = N, T, prompts, stop
        self.conn, self.done, self.errors = None, [], 0

    def run(self):
        while not self.stop_evt.is_set():
            b = body(random.choice(self.prompts), self.N, self.T)
            t0 = time.perf_counter()
            try:
                self.conn = http.client.HTTPConnection(HOST, HPORT, timeout=3600)
                self.conn.request('POST', '/v1/completions', b, {'Content-Type': 'application/json'})
                resp = self.conn.getresponse()
                resp.read()
                if resp.status != 200:
                    self.errors += 1
                    time.sleep(1)
                else:
                    self.done.append((t0, time.perf_counter()))
            except Exception:
                if self.stop_evt.is_set():
                    break
                self.errors += 1
                time.sleep(0.5)
            finally:
                try:
                    self.conn.close()
                except Exception:
                    pass

    def abort(self):
        try:
            self.conn.sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass


def one_request(prompt, N, T):
    t0 = time.perf_counter()
    r = requests.post(URL + '/v1/completions', data=body(prompt, N, T),
                      headers={'Content-Type': 'application/json'}, timeout=3600)
    r.raise_for_status()
    return time.perf_counter() - t0


def drain(max_wait=150):
    t0 = time.time()
    while time.time() - t0 < max_wait:
        m = metrics()
        if m and m.get('vllm:num_requests_running', 0) == 0 and m.get('vllm:num_requests_waiting', 0) == 0:
            return
        if not m:
            time.sleep(10)
            return
        time.sleep(2)


def run_cell(prompts, N, T, S):
    C = max(1, S // N)
    stop = threading.Event()
    ws = [Worker(N, T, prompts, stop) for _ in range(C)]
    for w in ws:
        w.start()
    time.sleep(LC['warm_s'])
    m0, t0, samples = metrics(), time.perf_counter(), []
    while time.perf_counter() - t0 < LC['dur_s']:
        time.sleep(2.0)
        samples.append(metrics())
    m1, t1 = metrics(), time.perf_counter()
    stop.set()
    for w in ws:
        w.abort()
    for w in ws:
        w.join(timeout=20)
    done = [(a, b) for w in ws for (a, b) in w.done if t0 <= b <= t1]
    dur = t1 - t0
    if GEN in m0 and GEN in m1:
        tps = (m1[GEN] - m0[GEN]) / dur
    else:
        tps = len(done) * N * T / dur
    run_m = [s.get('vllm:num_requests_running') for s in samples if 'vllm:num_requests_running' in s]
    wait_m = [s.get('vllm:num_requests_waiting') for s in samples if 'vllm:num_requests_waiting' in s]
    kv_m = [kv_usage(s) for s in samples if kv_usage(s) is not None]
    mean = lambda x: float(sum(x) / len(x)) if x else None
    res = dict(N=N, T=T, S=S, C=C, tps=tps, rps=tps / (N * T), n_done=len(done),
               lat_mean=mean([b - a for a, b in done]), running=mean(run_m), waiting=mean(wait_m),
               kv=mean(kv_m), errors=sum(w.errors for w in ws), dur=dur)
    drain()
    return res


def main():
    proc = None
    if not EXT:
        proc = start_server()
    try:
        wait_ready(proc)
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(CFG['model'])
        probs = load_problems()
        prompts = [chat_prompt(tok, p['problem'], True) + '<think>\n' for p in probs[:60]]
        one_request(prompts[0], 1, 16)  # warm-up / CUDA graphs

        idle_f = out / 'idle.json'
        if not idle_f.exists():
            idle = []
            for N, T in LC['actions']:
                for rep in range(LC['idle_reps']):
                    lat = one_request(random.choice(prompts), N, T)
                    idle.append(dict(N=N, T=T, lat=lat, rep=rep))
                    print(f'[load] idle N={N} T={T}: {lat:.1f}s', flush=True)
            atomic_write_json(idle_f, idle)

        cells_f = out / 'cells.jsonl'
        have = set()
        if cells_f.exists():
            for line in open(cells_f):
                if line.strip():
                    r = json.loads(line)
                    have.add((r['N'], r['T'], r['C']))
        for N, T in LC['actions']:
            for S in LC['S_levels']:
                C = max(1, S // N)
                if (N, T, C) in have:
                    continue
                res = run_cell(prompts, N, T, S)
                have.add((N, T, C))
                with open(cells_f, 'a') as f:
                    f.write(json.dumps(res) + '\n')
                print(f"[load] N={N} T={T} S={S} C={C}: {res['tps']:.0f} tok/s, running={res['running']}, "
                      f"waiting={res['waiting']}, kv={res['kv']}", flush=True)
        print('[load] finished', flush=True)
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except Exception:
                proc.kill()


if __name__ == '__main__':
    main()
