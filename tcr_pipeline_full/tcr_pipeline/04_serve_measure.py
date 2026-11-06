"""Stage 4: live-server measurements, with every value derived from data.

 (a) idle latency of the reference request           -> latency targets (SLOs)
 (b) closed-loop saturation throughput                -> capacity mu_max (requests/s)
 (c) load regimes = fractions of capacity             -> background arrival rates
 (d) externality sweep: background Poisson traffic at each regime, with and without an extra stream
     of "target" requests using action (N,B). Compare background latency / SLO attainment against the
     control cell (no targets). Requests that do not finish are kept as censored (counted as SLO misses),
     never silently capped.

Requests carry the empirical token count for their budget (ignore_eos), N samples = N parallel requests
sharing one prompt (prefix caching), latency = slowest of the N. Resumable; stops starting new cells after
TCR_DEADLINE (epoch seconds).
"""
import http.client, json, os, random, re, shutil, socket, subprocess, sys, threading, time
from urllib.parse import urlparse
import numpy as np
import requests

sys.path.insert(0, os.environ.get('TCR_DIR', '.'))
from tcr_common import *

SC = CFG['serve']
PORT = 8001
EXT = os.environ.get('TCR_EXTERNAL_SERVER')
URL = EXT or f'http://127.0.0.1:{PORT}'
_u = urlparse(URL)
HOST, HPORT = _u.hostname, _u.port
DEADLINE = float(os.environ.get('TCR_DEADLINE', '0')) or None
out = RUN / 'load'
out.mkdir(exist_ok=True)
DEC = json.load(open(RUN / 'decisions.json'))
LEN = {int(k): v for k, v in json.load(open(RUN / 'lengths.json')).items()}
GEN = 'vllm:generation_tokens_total'
rnd = random.Random(CFG['seed'] + 1)


def start_server():
    exe = shutil.which('vllm')
    cmd = ([exe, 'serve', CFG['model']] if exe else
           [sys.executable, '-m', 'vllm.entrypoints.openai.api_server', '--model', CFG['model']])
    cmd += ['--dtype', CFG['dtype'], '--max-model-len', str(CFG['max_model_len']),
            '--gpu-memory-utilization', str(CFG['gpu_mem_util']), '--enable-prefix-caching',
            '--max-num-seqs', str(CFG['max_num_seqs']), '--port', str(PORT),
            '--served-model-name', 'm', '--seed', str(CFG['seed'])]
    print('[serve] starting server', flush=True)
    return subprocess.Popen(cmd, stdout=open(out / 'server.log', 'w'), stderr=subprocess.STDOUT)


def wait_ready(proc, timeout=1200):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError('vLLM exited early:\n' + open(out / 'server.log').read()[-3000:])
        try:
            if requests.get(URL + '/health', timeout=3).status_code == 200:
                print(f'[serve] server ready after {time.time() - t0:.0f}s', flush=True)
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


def make_prompts():
    probs = load_problems()
    if os.environ.get('TCR_FAKE_TOK'):
        return [p['problem'] for p in probs], [p['problem'] for p in probs]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CFG['model'])
    think = [chat_prompt(tok, p['problem'], True) + '<think>\n' for p in probs]
    nothink = [chat_prompt(tok, p['problem'], False) for p in probs]
    return think, nothink


PT, PN = make_prompts()


def draw_len(B):
    return max(1, int(rnd.choice(LEN[B])))


class Req(threading.Thread):
    """One single-completion request with a controlled token count; abortable."""
    def __init__(self, prompt, L):
        super().__init__(daemon=True)
        self.prompt, self.L = prompt, L
        self.t0 = time.perf_counter()
        self.lat, self.ok, self.conn = None, None, None

    def run(self):
        b = json.dumps(dict(model='m', prompt=self.prompt, max_tokens=self.L, temperature=0.6, top_p=0.95,
                            top_k=20, ignore_eos=True)).encode()
        try:
            self.conn = http.client.HTTPConnection(HOST, HPORT, timeout=3600)
            self.conn.request('POST', '/v1/completions', b, {'Content-Type': 'application/json'})
            r = self.conn.getresponse()
            r.read()
            self.ok = r.status == 200
        except Exception:
            self.ok = False
        finally:
            self.lat = time.perf_counter() - self.t0
            try:
                self.conn.close()
            except Exception:
                pass

    def abort(self):
        try:
            self.conn.sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass


class Job:
    """A logical request = N parallel samples for budget B. Latency = slowest sample."""
    def __init__(self, N, B, arr):
        i = rnd.randrange(len(PT))
        prompt = (PN if B == 0 else PT)[i]
        self.N, self.B, self.arr = N, B, arr
        self.parts = [Req(prompt, draw_len(B)) for _ in range(N)]
        for p in self.parts:
            p.start()

    def done(self):
        return all(p.lat is not None for p in self.parts)

    def result(self, now):
        if self.done():
            return dict(ok=all(p.ok for p in self.parts), lat=max(p.lat for p in self.parts), cens=False)
        return dict(ok=False, lat=now - self.parts[0].t0, cens=True)

    def abort(self):
        for p in self.parts:
            p.abort()


def poisson(rate, N, B, stop_at, jobs, flag):
    """Open-loop arrivals at `rate` req/s until stop_at."""
    def run():
        t = time.perf_counter()
        while not flag.is_set():
            t += rnd.expovariate(rate)
            if t >= stop_at:
                return
            time.sleep(max(0.0, t - time.perf_counter()))
            if sum(1 for j in jobs if not j.done()) < SC['max_inflight']:
                jobs.append(Job(N, B, time.perf_counter()))
    th = threading.Thread(target=run, daemon=True)
    th.start()
    return th


def drain(max_wait=180):
    t0 = time.time()
    while time.time() - t0 < max_wait:
        m = metrics()
        if not m or (m.get('vllm:num_requests_running', 0) == 0 and m.get('vllm:num_requests_waiting', 0) == 0):
            return
        time.sleep(2)


def summarize(jobs, t_lo, t_hi, slos):
    rs = [j.result(time.perf_counter()) for j in jobs if t_lo <= j.arr < t_hi]
    n = len(rs)
    if n == 0:
        return dict(n=0)
    lat = np.array([r['lat'] for r in rs])
    ok = np.array([r['ok'] for r in rs])
    cens = np.array([r['cens'] for r in rs])
    d = dict(n=n, n_cens=int(cens.sum()), n_fail=int((~ok & ~cens).sum()), lat_mean=float(lat.mean()),
             lat_p50=float(np.percentile(lat, 50)), lat_p95=float(np.percentile(lat, 95)))
    for s in slos:
        d[f'slo_{s:.0f}s'] = float(((lat <= s) & ok).mean())
    return d


def run_cell(rate_bg, target, slos, dur, warm, grace):
    flag = threading.Event()
    t_start = time.perf_counter()
    stop_at = t_start + warm + dur
    bg, tg, samples = [], [], []
    ths = []
    if rate_bg > 0:
        ths.append(poisson(rate_bg, 1, SC['ref_B'], stop_at, bg, flag))
    if target:
        ths.append(poisson(SC['target_rate_frac'] * DEC['mu_max'], target[0], target[1], stop_at, tg, flag))
    time.sleep(warm)
    m0 = metrics()
    while time.perf_counter() < stop_at:
        time.sleep(2.0)
        samples.append(metrics())
    m1 = metrics()
    t_end = time.perf_counter()
    t_grace = t_end + grace
    while time.perf_counter() < t_grace and not all(j.done() for j in bg + tg):
        time.sleep(1.0)
    flag.set()
    res = dict(bg=summarize(bg, t_start + warm, stop_at, slos), tg=summarize(tg, t_start + warm, stop_at, slos))
    for j in bg + tg:
        if not j.done():
            j.abort()
    mean = lambda x: float(np.mean(x)) if x else None
    res['server'] = dict(running=mean([s['vllm:num_requests_running'] for s in samples if 'vllm:num_requests_running' in s]),
                         waiting=mean([s['vllm:num_requests_waiting'] for s in samples if 'vllm:num_requests_waiting' in s]),
                         kv=mean([kv_usage(s) for s in samples if kv_usage(s) is not None]),
                         tok_per_s=((m1[GEN] - m0[GEN]) / (t_end - t_start - warm)) if GEN in m0 and GEN in m1 else None)
    res['dur'] = t_end - t_start - warm
    drain()
    return res


def calibrate():
    f = out / 'calibration.json'
    if f.exists():
        return json.load(open(f))
    ref = SC['ref_B']
    mean_len = float(np.mean(LEN[ref]))
    Job(1, 0, 0).parts[0].join()  # warm-up request
    # (a) idle latency of the reference request
    lats = []
    for _ in range(SC['idle_reps']):
        j = Job(1, ref, time.perf_counter())
        j.parts[0].join()
        lats.append(j.parts[0].lat)
    idle_p95 = float(np.percentile(lats, 95))
    slos = [round(m * idle_p95, 1) for m in SC['slo_mult']]
    print(f'[serve] idle reference latency: median {np.median(lats):.1f}s p95 {idle_p95:.1f}s -> SLOs {slos}s', flush=True)
    # (b) closed-loop saturation throughput
    best, curve = 0.0, []
    for C in SC['capacity_clients']:
        stop = threading.Event()
        jobs = []

        def client():
            while not stop.is_set():
                j = Job(1, ref, time.perf_counter())
                j.parts[0].join()
                jobs.append(j)
        ths = [threading.Thread(target=client, daemon=True) for _ in range(C)]
        for t in ths:
            t.start()
        time.sleep(SC['cap_warm_s'])
        m0, t0 = metrics(), time.perf_counter()
        time.sleep(SC['cap_dur_s'])
        m1, t1 = metrics(), time.perf_counter()
        stop.set()
        if GEN in m0 and GEN in m1:
            tps = (m1[GEN] - m0[GEN]) / (t1 - t0)
        else:
            tps = sum(j.parts[0].L for j in jobs if j.parts[0].t0 >= t0) / (t1 - t0)
        for j in jobs:
            pass
        curve.append(dict(C=C, tps=tps, rps=tps / mean_len))
        print(f'[serve] capacity probe C={C}: {tps:.0f} tok/s = {tps / mean_len:.2f} req/s', flush=True)
        drain()
        if best and tps < 1.05 * best:
            best = max(best, tps)
            break
        best = max(best, tps)
    cal = dict(idle_lat=lats, idle_p95=idle_p95, slos=slos, mu_max=best / mean_len, tps_max=best, curve=curve,
               mean_len_ref=mean_len)
    json.dump(cal, open(f, 'w'), indent=1)
    print(f'[serve] capacity mu_max = {cal["mu_max"]:.2f} req/s ({best:.0f} tok/s)', flush=True)
    return cal


def main():
    proc = None if EXT else start_server()
    try:
        wait_ready(proc)
        cal = calibrate()
        DEC.update(mu_max=cal['mu_max'], slos=cal['slos'], idle_p95=cal['idle_p95'], load_fracs=SC['fractions'])
        json.dump(DEC, open(RUN / 'decisions.json', 'w'), indent=1)
        slos = cal['slos']
        Ns = DEC['N_GRID']
        n_sel = sorted({Ns[0], Ns[len(Ns) // 2], Ns[-1]})
        b_sel = [b for b in SC['actions_B'] if b in DEC['B_GRID']]
        actions = [(n, b) for n in n_sel for b in b_sel]
        cells_f = out / 'ext_cells.jsonl'
        have = set()
        if cells_f.exists():
            for line in open(cells_f):
                if line.strip():
                    r = json.loads(line)
                    have.add((r['rep'], r['frac'], r['N'], r['B']))
        for rep in range(SC['reps']):
            for frac in SC['fractions']:
                rate = frac * cal['mu_max']
                todo = ([(0, 0)] if frac > 0 else []) + actions   # (0,0) = control cell (no targets)
                for (N, B) in todo:
                    if (rep, frac, N, B) in have:
                        continue
                    if DEADLINE and time.time() > DEADLINE:
                        print('[serve] time budget reached, stopping new cells', flush=True)
                        return
                    dur = SC['dur_s'] if frac > 0 else SC['idle_dur_s']
                    res = run_cell(rate, (N, B) if N else None, slos, dur, SC['warm_s'], SC['grace_s'])
                    res.update(rep=rep, frac=frac, rate_bg=rate, N=N, B=B, slos=slos)
                    with open(cells_f, 'a') as f:
                        f.write(json.dumps(res) + '\n')
                    have.add((rep, frac, N, B))
                    b = res['bg']
                    print(f"[serve] rep{rep} load={frac:.2f} a=({N},{B}) bg_p95={b.get('lat_p95')} "
                          f"cens={b.get('n_cens')} waiting={res['server']['waiting']}", flush=True)
        print('[serve] finished', flush=True)
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except Exception:
                proc.kill()


if __name__ == '__main__':
    main()
