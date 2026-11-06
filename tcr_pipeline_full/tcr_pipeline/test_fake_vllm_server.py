"""Fake vLLM-compatible server (processor-sharing) to test the load-generator logic without a GPU."""
import json, threading, time, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
SPEED = 20.0   # time compression
CAP, PER_REQ, MAXSEQ = 5000.0 * SPEED, 90.0 * SPEED, 128
lock = threading.Lock()
running, waiting, gen_total = [], [], [0.0]


class R:
    def __init__(self, L):
        self.rem, self.ev = float(L), threading.Event()


def loop():
    last = time.time()
    while True:
        time.sleep(0.02)
        now = time.time(); dt = now - last; last = now
        with lock:
            while waiting and len(running) < MAXSEQ:
                running.append(waiting.pop(0))
            if running:
                rate = min(PER_REQ, CAP / len(running))
                for r in list(running):
                    s = rate * dt
                    r.rem -= s; gen_total[0] += s
                    if r.rem <= 0:
                        running.remove(r); r.ev.set()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200); self.end_headers(); self.wfile.write(b'ok'); return
        with lock:
            txt = (f'vllm:num_requests_running {len(running)}\nvllm:num_requests_waiting {len(waiting)}\n'
                   f'vllm:kv_cache_usage_perc {len(running) / MAXSEQ}\nvllm:generation_tokens_total {gen_total[0]}\n')
        self.send_response(200); self.end_headers(); self.wfile.write(txt.encode())
    def do_POST(self):
        n = int(self.headers['Content-Length']); b = json.loads(self.rfile.read(n))
        r = R(b['max_tokens'])
        with lock: waiting.append(r)
        r.ev.wait()
        self.send_response(200); self.end_headers(); self.wfile.write(b'{}')


threading.Thread(target=loop, daemon=True).start()
ThreadingHTTPServer(('127.0.0.1', int(sys.argv[1])), H).serve_forever()
