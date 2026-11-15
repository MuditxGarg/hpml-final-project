"""Fake vLLM-compatible server (processor-sharing) to test the load-generator logic without a GPU."""
import json, os, threading, time, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
SPEED = 20.0   # time compression
# Total tokens/s, per-request tokens/s and concurrent sequences; override from the environment for tests.
CAP = float(os.environ.get('FAKE_VLLM_CAP', 5000.0 * SPEED))
PER_REQ = float(os.environ.get('FAKE_VLLM_PER_REQ', 90.0 * SPEED))
MAXSEQ = int(os.environ.get('FAKE_VLLM_MAXSEQ', 128))
lock = threading.Lock()
running, waiting, gen_total = [], [], [0.0]
queue_time, token_time = [0.0, 0], [0.0, 0.0]   # [sum, count] for the queue-time and inter-token histograms


class R:
    def __init__(self, L):
        self.rem, self.ev, self.t = float(L), threading.Event(), time.time()


def admit(r):
    queue_time[0] += time.time() - r.t; queue_time[1] += 1
    running.append(r)


def loop():
    last = time.time()
    while True:
        time.sleep(0.02)
        now = time.time(); dt = now - last; last = now
        with lock:
            while waiting and len(running) < MAXSEQ:
                admit(waiting.pop(0))
            if running:
                rate = min(PER_REQ, CAP / len(running))
                for r in list(running):
                    s = rate * dt
                    r.rem -= s; gen_total[0] += s
                    token_time[0] += dt; token_time[1] += s
                    if r.rem <= 0:
                        running.remove(r); r.ev.set()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200); self.end_headers(); self.wfile.write(b'ok'); return
        with lock:
            txt = (f'vllm:num_requests_running {len(running)}\nvllm:num_requests_waiting {len(waiting)}\n'
                   f'vllm:kv_cache_usage_perc {len(running) / MAXSEQ}\nvllm:generation_tokens_total {gen_total[0]}\n'
                   f'vllm:num_preemptions_total 0\n'
                   f'vllm:request_queue_time_seconds_sum {queue_time[0]}\nvllm:request_queue_time_seconds_count {queue_time[1]}\n'
                   f'vllm:inter_token_latency_seconds_sum {token_time[0]}\nvllm:inter_token_latency_seconds_count {token_time[1]}\n')
        self.send_response(200); self.end_headers(); self.wfile.write(txt.encode())
    def do_POST(self):
        n = int(self.headers['Content-Length']); b = json.loads(self.rfile.read(n))
        r = R(b['max_tokens'])
        with lock:   # admit at once when a sequence slot is free; queue only when the server is full
            if len(running) < MAXSEQ and not waiting: admit(r)
            else: waiting.append(r)
        try:
            if b.get('stream'):   # server-sent events with a running token count, as vLLM does with continuous usage
                self.send_response(200); self.send_header('Content-Type', 'text/event-stream'); self.end_headers()
                sent = 0
                while True:
                    done = r.ev.wait(0.02)
                    count = b['max_tokens'] if done else int(b['max_tokens'] - max(r.rem, 0))
                    if count > sent:
                        self.wfile.write(b'data: ' + json.dumps({'choices': [{'text': ''}], 'usage': {'completion_tokens': count}}).encode() + b'\n\n')
                        self.wfile.flush(); sent = count
                    if done:
                        self.wfile.write(b'data: [DONE]\n\n'); return
            r.ev.wait()
            self.send_response(200); self.end_headers()
            self.wfile.write(json.dumps({'choices': [{'text': ''}], 'usage': {'completion_tokens': b['max_tokens']}}).encode())
        except (BrokenPipeError, ConnectionResetError):
            pass   # client gave up (timeout or censor deadline)


threading.Thread(target=loop, daemon=True).start()
class Server(ThreadingHTTPServer):
    request_queue_size = 256   # accept bursts without SYN retries


Server(('127.0.0.1', int(sys.argv[1])), H).serve_forever()
