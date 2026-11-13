import importlib.util
from pathlib import Path
import socket

P = Path(__file__).resolve().parents[1] / 'real_answer_validation.py'
spec = importlib.util.spec_from_file_location('real_answer_validation', P)
m = importlib.util.module_from_spec(spec)
if P.exists(): spec.loader.exec_module(m)

class Tokenizer:
    def encode(self, text, add_special_tokens=False): return [ord(x) for x in text]
    def decode(self, ids, **kw): return ''.join(chr(x) for x in ids)


def test_deep_budget_continuation_preserves_tokens_and_common_deadline():
    assert hasattr(m, 'generate_sample'), 'real answer generation missing'
    seen = []
    def send(body, timeout):
        seen.append((body, timeout))
        if len(seen) == 1:
            return {'choices':[{'text':'work', 'token_ids':[88,89], 'finish_reason':'length', 'stop_reason':None}], 'usage':{'completion_tokens':2}}
        return {'choices':[{'text':'42}', 'token_ids':[52,50,125], 'finish_reason':'stop'}], 'usage':{'completion_tokens':3}}
    result = m.generate_sample({'thinking':'prompt', 'nothinking':'short'}, 2048, 7, Tokenizer(), send, 10)
    assert result['status']=='completed' and result['pred']=='42'
    assert seen[0][0]['ignore_eos'] is False and seen[0][0]['max_tokens']==2048
    assert seen[1][0]['prompt']==Tokenizer().encode('prompt<think>\n')+[88,89]+Tokenizer().encode(m.FORCED)
    assert seen[1][0]['max_tokens']==48 and seen[1][1] <= seen[0][1]
    assert result['completion_tokens']==5


def test_natural_stop_does_not_duplicate_think_close():
    assert hasattr(m, 'generate_sample')
    seen=[]
    def send(body, timeout):
        seen.append(body)
        if len(seen)==1:
            return {'choices':[{'text':'work','token_ids':[88,99],'finish_reason':'stop','stop_reason':'</think>'}], 'usage':{'completion_tokens':2}}
        return {'choices':[{'text':'1}','token_ids':[49,125],'finish_reason':'stop'}], 'usage':{'completion_tokens':2}}
    tok=Tokenizer()
    # Model token IDs may retain stop text. Use a literal stop suffix to exercise removal.
    def natural(body, timeout):
        out=send(body,timeout)
        if len(seen)==1: out['choices'][0]['token_ids']=[88]+tok.encode('</think>')
        return out
    r=m.generate_sample({'thinking':'p','nothinking':'n'},2048,1,tok,natural,10)
    assert seen[1]['prompt']==tok.encode('p<think>\n')+[88]+tok.encode(m.NATURAL)
    assert r['pred']=='1'


def test_shallow_cap_and_missing_tokens_are_not_silently_accepted():
    assert hasattr(m, 'generate_sample')
    seen=[]
    def send(body,timeout):
        seen.append(body)
        return {'choices':[{'text':r'\boxed{3}','finish_reason':'length'}],'usage':{'completion_tokens':2048}}
    shallow=m.generate_sample({'thinking':'p','nothinking':'n'},0,1,Tokenizer(),send,10)
    assert shallow['pred']=='3' and shallow['cap_hit']
    assert seen[0]['max_tokens']==2048 and not seen[0]['ignore_eos']
    deep=m.generate_sample({'thinking':'p','nothinking':'n'},2048,1,Tokenizer(),send,10)
    assert deep['status']=='error' and 'token_ids' in deep['error']


def test_failed_sample_cannot_be_a_successful_vote_and_late_correct_is_miss():
    assert hasattr(m, 'aggregate')
    parts=[{'status':'completed','pred':'2'},{'status':'completed','pred':'1'},{'status':'completed','pred':'1'},{'status':'timeout','pred':None}]
    assert m.aggregate(parts)['pred'] is None
    parts[-1]={'status':'completed','pred':None}
    assert m.aggregate(parts)['pred']=='1'
    assert m.aggregate(parts[:2])['pred']=='2' # sample-index tie rule
    rows=[dict(status='completed',pred='1',latency_s=90,n=1,budget=0,id='a'),dict(status='timeout',pred=None,latency_s=180,n=1,budget=0,id='b')]
    s=m.score_rows(rows,{'a':'1','b':'1'},87,lambda g,p:g==p)
    assert s[0]['correct'] and not s[0]['correct_within_slo']
    assert not s[1]['correct'] and not s[1]['within_slo']


def test_timeout_retains_completed_stage_and_sampling_is_order_independent():
    assert hasattr(m, 'generate_sample')
    calls=[]
    def send(body,timeout):
        calls.append(body)
        if len(calls)==1:return {'choices':[{'text':'work','token_ids':[88],'finish_reason':'length'}],'usage':{'completion_tokens':1}}
        raise socket.timeout('observation ended')
    r=m.generate_sample({'thinking':'p','nothinking':'n'},2048,1,Tokenizer(),send,10)
    assert r['status']=='timeout' and len(r['stages'])==2 and r['stages'][1]['status']=='timeout'
    rows=[{'id':str(i),'level':i//2+1} for i in range(10)]
    assert m.select_questions(rows,3)==m.select_questions(list(reversed(rows)),3)
    assert sorted(r['level'] for r in m.select_questions(rows,3))==[1,2,3,4,5]


def test_wall_deadline_rejects_late_transport_and_retains_request():
    import time
    def send(body,timeout):
        time.sleep(.2)
        return {'choices':[{'text':r'\boxed{1}','finish_reason':'stop'}]}
    start=time.monotonic()
    r=m.generate_sample({'thinking':'p','nothinking':'n'},0,1,Tokenizer(),send,.02)
    assert time.monotonic()-start < .15
    assert r['status']=='timeout'
    assert r['stages'][0]['request']['max_tokens']==2048
    assert r['stages'][0]['status']=='timeout'


def test_transport_success_without_box_is_not_answer_delivery():
    r=m.aggregate([{'status':'completed','pred':None}])
    assert r['status']=='no_answer' and r['transport_completed']
    row=dict(r,id='a',latency_s=1)
    assert not m.score_rows([row],{'a':'1'},87,lambda g,p:g==p)[0]['within_slo']


def test_runner_http_records_all_actions_and_deadline_skips_whole_block(tmp_path,monkeypatch):
    import json,sys,threading,types
    from datetime import datetime,timedelta,timezone
    from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            data={'data':[{'id':'m'}]}
            self.send_response(200);self.end_headers();self.wfile.write(json.dumps(data).encode())
        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            text='1}' if body['max_tokens']==48 else 'work' if body['max_tokens']==2048 and 'stop' in body else r'\boxed{1}'
            data={'choices':[{'text':text,'token_ids':[ord(c) for c in text], 'finish_reason':'length' if text=='work' else 'stop'}],'usage':{'completion_tokens':len(text)}}
            self.send_response(200);self.end_headers();self.wfile.write(json.dumps(data).encode())
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    monkeypatch.setitem(sys.modules,'transformers',types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a,**k:Tokenizer())))
    monkeypatch.setattr(m.importlib.metadata,'version',lambda p:'test')
    monkeypatch.setattr(m,'wait_idle',lambda *a,**k:{'running':0})
    plan=dict(model='fake',revision='fake',seed=1,slo_s=87,observation_timeout_s=2,
              prompts=[dict(id='q',thinking='p',nothinking='n')],blocks=[dict(id='q',level=1,actions=m.ACTIONS)])
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan))
    args=types.SimpleNamespace(plan=str(path),url=f'http://127.0.0.1:{server.server_port}',model='m',idle_timeout=1,out=str(tmp_path/'run'),deadline_utc=(datetime.now(timezone.utc)+timedelta(seconds=500)).isoformat())
    try:
        m.run(args)
        rows=[json.loads(l) for l in (tmp_path/'run/raw.jsonl').read_text().splitlines()]
        assert len(rows)==3 and all(r['pred']=='1' for r in rows)
        assert len(rows[1]['parts'])==4 and len(rows[2]['parts'][0]['stages'])==2
        assert json.loads((tmp_path/'run/status.json').read_text())['status']=='complete'
        args.out=str(tmp_path/'skipped');args.deadline_utc=datetime.now(timezone.utc).isoformat()
        m.run(args)
        assert not (tmp_path/'skipped/raw.jsonl').exists()
        assert len(json.loads((tmp_path/'skipped/status.json').read_text())['skipped_blocks'])==1
    finally:server.shutdown();server.server_close()


def test_empty_deadline_guarded_run_has_zero_observed_not_zero_accuracy(tmp_path,monkeypatch):
    import json,sys,types
    monkeypatch.setitem(sys.modules,'math_verify',types.SimpleNamespace())
    monkeypatch.setattr(m.importlib.metadata,'version',lambda p:'test')
    (tmp_path/'plan.json').write_text(json.dumps(dict(slo_s=87,scope='diagnostic',blocks=[{'id':'q'}])))
    (tmp_path/'status.json').write_text(json.dumps(dict(status='partial_deadline',skipped_blocks=[{'id':'q'}])))
    (tmp_path/'gold.json').write_text(json.dumps({'q':'1'}))
    m.score(types.SimpleNamespace(run=str(tmp_path),gold=str(tmp_path/'gold.json')))
    summary=json.loads((tmp_path/'summary.json').read_text())
    assert all(a['observed']==0 and a['planned']==1 and a['completed_p95_s'] is None for a in summary['actions'])
    assert summary['status']['status']=='partial_deadline'
