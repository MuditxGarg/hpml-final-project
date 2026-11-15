import importlib.util
from pathlib import Path
import time

P=Path(__file__).resolve().parents[1]/'real_answer_load.py'
spec=importlib.util.spec_from_file_location('real_answer_load',P)
m=importlib.util.module_from_spec(spec)
if P.exists():spec.loader.exec_module(m)


def test_matching_and_seeded_target_cohort():
    prompts=[dict(id='b',thinking='p',nothinking='n')]
    plan=dict(prompts=[dict(id=str(i),level=i,thinking='p',nothinking='n') for i in range(5)])
    a=m.make_trace(plan,prompts,4,2)
    b=m.make_trace(plan,prompts,4,4)
    targets=lambda t:[e for e in t['events'] if e['stream']=='target']
    assert targets(a)==targets(b)
    assert len(targets(a))==5 and len({e['question_id'] for e in targets(a)})==5
    assert a==m.make_trace(plan,prompts,4,2)
    assert {e['question_id'] for e in a['events'] if e['stream']=='background'}=={'b'}
    assert m.block_bound()==1440
    assert not m.block_fits(time.time()+100)


def test_load_score_keeps_no_answers_and_late_correct():
    rows=[dict(stream='target',measured=True,id='t',question_id='q',status='completed',pred='1',latency_s=90),dict(stream='target',measured=True,id='u',question_id='q',status='no_answer',pred=None,latency_s=2),dict(stream='background',measured=True,id='b',question_id='q',status='completed',pred='1',latency_s=1)]
    report=m.score_cell(rows,{'q':'1'},verifier=lambda g,p:g==p)
    assert report['target']['scheduled']==2
    assert report['target']['correct']==1 and report['target']['correct_within_slo']==0
    assert report['target']['statuses']['no_answer']==1
    assert report['background']['slo_attainment']==1


def test_rescoring_ignores_previous_scored_files(tmp_path,monkeypatch):
    import json,sys,types
    monkeypatch.setitem(sys.modules,'math_verify',types.SimpleNamespace())
    monkeypatch.setattr(m,'is_correct',lambda g,p:g==p)
    block=tmp_path/'seed-1-load-0.5';block.mkdir()
    (tmp_path/'manifest.json').write_text(json.dumps(dict(status='complete',completed=[block.name])))
    gold=tmp_path/'gold.json';gold.write_text('{}')
    rows=[dict(id='b',question_id='b',stream='background',measured=True,status='completed',latency_s=1,pred='1')]
    (block/'cell-0-N0-B0.jsonl').write_text(json.dumps(rows[0])+'\n')
    args=types.SimpleNamespace(out=str(tmp_path),gold=str(gold))
    m.score(args);m.score(args)
    report=json.loads((tmp_path/'summary.json').read_text())
    assert len(report['blocks'][0]['cells'])==1


def test_real_callback_http_under_open_loop_preserves_background(monkeypatch):
    import json,threading
    from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
    from test_real_answer_validation import Tokenizer
    seen=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            seen.append(body)
            text='1}' if body['max_tokens']==48 else 'work' if 'stop' in body else r'\boxed{1}'
            result=dict(choices=[dict(text=text,token_ids=[ord(c) for c in text],finish_reason='length' if text=='work' else 'stop')],usage=dict(completion_tokens=len(text)))
            self.send_response(200);self.end_headers();self.wfile.write(json.dumps(result).encode())
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    trace=dict(seed=8,warmup_s=0,duration_s=.02,events=[dict(id='b',question_id='b',stream='background',at_s=0,measured=True),dict(id='t',question_id='t',stream='target',at_s=.001,measured=True)])
    prompts={'b':dict(thinking='bg',nothinking='bg'),'t':dict(thinking='target',nothinking='target')}
    url=f'http://127.0.0.1:{server.server_port}'
    backgrounds=[]
    try:
        for action in [(0,0),(4,0),(1,2048)]:
            seen.clear()
            rows=m.run_cell(trace,url,'m',*action,0,timeout_s=2,drain_s=2,job_runner=m.callback(trace,prompts,Tokenizer(),url,'m'))
            backgrounds.append([r for r in seen if r['prompt']==Tokenizer().encode('bg')])
            if action[0]:
                assert rows[1]['pred']=='1' and len(rows[1]['parts'])==action[0]
                assert rows[1]['latency_s']>=max(p['seconds'] for p in rows[1]['parts'])
                if action[1]:assert len(rows[1]['parts'][0]['stages'])==2
            else:assert rows[1]['status']=='not_offered_control'
        assert backgrounds[0]==backgrounds[1]==backgrounds[2]
    finally:server.shutdown();server.server_close()


def test_report_pairs_only_complete_blocks_and_exposes_contention():
    telemetry=[dict(at_s=1,gauges={'vllm:num_requests_waiting':99}),dict(at_s=21,gauges={'vllm:num_requests_waiting':2,'vllm:kv_cache_usage_perc':.4}),dict(at_s=22,error='scrape failed'),dict(at_s=121,gauges={'vllm:num_requests_waiting':99})]
    q=m.summarize_telemetry(telemetry)
    assert q['waiting_max']==2 and q['scrape_errors']==1 and q['samples']==2
    control=dict(n=0,budget=0,metrics=dict(background=dict(scheduled=10,statuses={'completed':10},completed_p95_s=2,slo_attainment=1)))
    action=dict(n=1,budget=2048,metrics=dict(background=dict(scheduled=10,statuses={'completed':9,'timeout':1},completed_p95_s=3,slo_attainment=.9),target=dict(correct_within_slo=3,scheduled=5)))
    blocks=[dict(block='seed-1-load-0.9',paired_complete=True,cells=[control,action]),dict(block='seed-2-load-0.9',paired_complete=False,cells=[control,action])]
    pairs=m.paired_differences(blocks)
    assert len(pairs)==1 and pairs[0]['background_p95_delta_s'] is None
    assert abs(pairs[0]['background_slo_delta']+.1)<1e-9
    assert pairs[0]['target_correct_within_slo']==3


def test_run_cli_serializes_configuration_and_skips_unaffordable_blocks(tmp_path,monkeypatch):
    import json,sys,types
    monkeypatch.setitem(sys.modules,'transformers',types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a,**k:object())))
    monkeypatch.setattr(m,'http',lambda *a,**k:'{"data":[]}')
    monkeypatch.setattr(m,'calibrate',lambda *a,**k:dict(requests_per_s=1))
    plan=tmp_path/'plan.json';plan.write_text(json.dumps(dict(prompts=[dict(id=str(i),thinking='p',nothinking='n') for i in range(5)])))
    bg=tmp_path/'background.json';bg.write_text(json.dumps([dict(id='bg',thinking='p',nothinking='n')]))
    out=tmp_path/'run'
    monkeypatch.setattr(sys,'argv',['real_answer_load.py','run','--plan',str(plan),'--background',str(bg),'--out',str(out),'--deadline-epoch',str(time.time())])
    m.main()
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['status']=='partial_deadline' and len(manifest['skipped'])==4
    assert 'fn' not in manifest['configuration']
    assert manifest['configuration']['command']=='run'
