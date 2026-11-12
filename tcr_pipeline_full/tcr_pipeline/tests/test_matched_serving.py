import importlib.util
import json
from pathlib import Path
import threading
import time
import tempfile
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PATH = Path(__file__).resolve().parents[1] / 'matched_serving.py'
spec = importlib.util.spec_from_file_location('matched_serving', PATH)
m = importlib.util.module_from_spec(spec)
if PATH.exists():
    spec.loader.exec_module(m)


class ReplayTests(unittest.TestCase):
    def test_trace_determinism_and_stream_independence(self):
        self.assertTrue(hasattr(m, 'generate_trace'), 'trace generator is missing')
        kw = dict(lengths={'0': [2, 3], '8': [4, 5]}, prompts=[{'thinking': [1, 2], 'nothinking': [1]}], seed=4, background_rate=8, target_rate=9, warmup=0.2, duration=1, max_n=3)
        a = m.generate_trace(**kw)
        self.assertEqual(a, m.generate_trace(**kw))
        kw['target_rate'] = 1
        b = m.generate_trace(**kw)
        self.assertEqual([e for e in a['events'] if e['stream']=='background'], [e for e in b['events'] if e['stream']=='background'])
        event = a['events'][0]
        self.assertEqual(m.requests_for(a, event, 1, 8)[0], m.requests_for(a, event, 3, 8)[0])

    def test_cohort_denominator_and_latency(self):
        self.assertTrue(hasattr(m, 'summarize'), 'cohort summarizer is missing')
        rows = [dict(stream='target', measured=True, status='completed', latency_s=2, dispatch_lag_s=.4), dict(stream='target', measured=True, status='drop'), dict(stream='target', measured=True, status='censored', observed_s=3), dict(stream='target', measured=False, status='completed', latency_s=99)]
        result = m.summarize(rows, 2.5)['target']
        self.assertEqual(result['scheduled'], 3)
        self.assertAlmostEqual(result['slo_attainment'], 1/3)
        self.assertEqual(result['completed_p95_s'], 2)
        self.assertIsNone(result['unconditional_p95_upper_s'])
        self.assertEqual(result['unconditional_p95_lower_s'], None)  # drop is an infinite miss

    def test_http_cohort_completes_after_arrival_window(self):
        self.assertTrue(hasattr(m, 'run_cell'), 'HTTP replay is missing')
        bodies = []
        active = [0]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_GET(self):
                data = f'vllm:num_requests_running{{model_name="fake"}} {active[0]}\nvllm:num_requests_waiting{{model_name="fake"}} 0\n'.encode()
                self.send_response(200); self.end_headers(); self.wfile.write(data)
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                bodies.append(body)
                active[0] += 1
                time.sleep(.05)
                active[0] -= 1
                self.send_response(200); self.end_headers()
                self.wfile.write(json.dumps({'choices':[{'text':'ok'}], 'usage':{'completion_tokens':body['max_tokens']}}).encode())
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            trace = m.generate_trace({'0':[2], '8':[3]}, [{'thinking':[1,2], 'nothinking':[1]}], 1, 0, 100, 0, .01, 2)
            trace['events'] = [dict(id='target-0', stream='target', at_s=0, measured=True, prompt_index=0, lengths={'0':[2,2], '8':[3,3]})]
            url = 'http://127.0.0.1:' + str(server.server_port)
            m.wait_idle(url, timeout_s=1, stable_s=.01)
            rows = m.run_cell(trace, url, 'fake', 2, 8, 8, timeout_s=1, drain_s=1, max_inflight=2)
            self.assertEqual(rows[0]['status'], 'completed')
            self.assertGreater(rows[0]['latency_s'], .01)
            self.assertEqual(len(bodies), 2)
            self.assertTrue(all(b['ignore_eos'] and b['max_tokens']==3 for b in bodies))
            self.assertEqual(m.summarize(rows, 1)['target']['slo_attainment'], 1)
            trace['events'].append(dict(trace['events'][0], id='background-0', stream='background'))
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root/'trace.json').write_text(json.dumps(trace))
                (root/'hardware.json').write_text(json.dumps({'device':'fake CPU HTTP server'}))
                args = ['run','--trace',str(root/'trace.json'),'--url',url,'--model','fake','--output',str(root/'run'), '--hardware-json',str(root/'hardware.json'),'--actions','2:8','--reference-budget','8','--cache-policy','disabled','--slo','1','--timeout','1','--drain','1','--metrics-interval','.005']
                m.main(args)
                manifest = json.loads((root/'run/manifest.json').read_text())
                self.assertEqual(manifest['status'],'complete')
                cells = json.loads((root/'run/summary.json').read_text())['cells']
                self.assertEqual(len(cells),2)
                self.assertTrue(cells[0]['control'])
                self.assertTrue(cells[0]['observed_slo_feasible']) # background alone is the control cohort
                self.assertEqual(cells[0]['metrics']['target']['scheduled'], 0)
                self.assertEqual(cells[1]['metrics']['target']['scheduled'], 1)
                self.assertEqual(manifest['control_definition'], 'background_only; target events retained but not offered')
                telemetry = [json.loads(line) for line in (root/'run/cell-001-N2-B8-metrics.jsonl').read_text().splitlines()]
                self.assertGreater(len(telemetry),1)
                self.assertTrue(any(row.get('gauges', {}).get('vllm:num_requests_running',0) > 0 for row in telemetry))
                with self.assertRaises(FileExistsError): m.main(args)

        finally:
            server.shutdown(); server.server_close()

    def test_control_offers_same_background_and_no_target(self):
        bodies = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                bodies.append(body)
                self.send_response(200); self.end_headers()
                self.wfile.write(json.dumps({'choices':[{'text':'ok'}], 'usage':{'completion_tokens':body['max_tokens']}}).encode())
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            trace = m.generate_trace({'8':[3]}, [{'thinking':'background','nothinking':'background'}, {'thinking':'target','nothinking':'target'}], 1,1,1,0,.03,2)
            trace['events'] = [dict(id=stream+'-0', stream=stream, at_s=0, measured=True, prompt_index=i, lengths={'8':[3,3]}) for i,stream in enumerate(('background','target'))]
            url = 'http://127.0.0.1:' + str(server.server_port)
            control = m.run_cell(trace,url,'fake',0,0,8,timeout_s=1,drain_s=1)
            control_bodies = list(bodies)
            self.assertEqual(len(control_bodies), 1)
            self.assertEqual(control_bodies[0]['prompt'], 'background')
            self.assertEqual(control[1]['status'], 'not_offered_control')
            metrics = m.summarize(control,1)
            self.assertEqual(metrics['target']['scheduled'],0)
            self.assertEqual(metrics['target']['not_offered_control'],1)
            self.assertEqual(metrics['background']['scheduled'],1)
            bodies.clear()
            action = m.run_cell(trace,url,'fake',2,8,8,timeout_s=1,drain_s=1)
            self.assertEqual([b for b in bodies if b['prompt']=='background'],control_bodies)
            self.assertEqual(len([b for b in bodies if b['prompt']=='target']),2)
            self.assertEqual(control[0]['at_s'],action[0]['at_s'])
        finally:
            server.shutdown(); server.server_close()

    def test_cli_roundtrip_and_stale_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'lengths.json').write_text(json.dumps({'0':[2]}))
            (root/'prompts.json').write_text(json.dumps([{'thinking':'a', 'nothinking':'b'}]))
            args = ['generate', '--lengths', str(root/'lengths.json'), '--prompts', str(root/'prompts.json'), '--output', str(root/'trace.json'), '--background-rate', '1', '--target-rate', '1', '--warmup', '0', '--duration', '1']
            m.main(args)
            self.assertEqual(json.loads((root/'trace.json').read_text())['seed'], 42)
            with self.assertRaises(FileExistsError): m.main(args)

    def test_drop_and_censor_preserve_dispatch(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                time.sleep(.2)
                try:
                    self.send_response(200); self.end_headers()
                    self.wfile.write(b'{"choices":[{}],"usage":{"completion_tokens":2}}')
                except (BrokenPipeError, ConnectionResetError): pass
        server = ThreadingHTTPServer(('127.0.0.1',0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            trace = m.generate_trace({'0':[2]}, [{'thinking':'a','nothinking':'b'}], 1,0,1,0,.01,1)
            trace['events'] = [dict(id=f'target-{i}',stream='target',at_s=0,measured=True,prompt_index=0,lengths={'0':[2]}) for i in range(2)]
            rows = m.run_cell(trace,'http://127.0.0.1:'+str(server.server_port),'fake',1,0,0,timeout_s=1,drain_s=.04,max_inflight=1)
            self.assertEqual(rows[1]['status'],'drop')
            self.assertIn(rows[0]['status'], ('censored','timeout'))
            self.assertIsNotNone(rows[0].get('dispatch_lag_s'))
            self.assertEqual(m.summarize(rows,1)['target']['scheduled'],2)
            self.assertEqual(m.summarize(rows,1)['target']['slo_attainment'],0)
        finally:
            server.shutdown(); server.server_close()

if __name__ == '__main__': unittest.main()


def test_randomized_order_preserves_one_control_and_all_actions():
    import matched_serving as r
    assert hasattr(r, 'cell_order')
    original = [(4,0),(1,2048),(4,2048)]
    orders = [r.cell_order(original, seed) for seed in range(12)]
    assert all(sorted(x) == sorted([(0,0)]+original) for x in orders)
    assert any(x[0] != (0,0) for x in orders)
    assert r.cell_order(original, 11) == r.cell_order(original, 11)
    assert r.cell_order(original, None) == [(0,0)]+original
    assert original == [(4,0),(1,2048),(4,2048)]
