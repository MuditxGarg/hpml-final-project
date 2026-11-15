import importlib
import json


def test_replay_lengths_match_cap_and_original_trace_order(tmp_path):
    root = tmp_path/'labels'
    (root/'rollouts').mkdir(parents=True)
    (root/'rollouts_nothink_cap').mkdir()
    original=[dict(id='z',traces=[dict(L=1500,b={'1024':{'ft':20},'2048':{'ft':30}}),dict(L=2500,b={'1024':{'ft':40},'2048':{'ft':50}})])]
    cap=[dict(id='z',samples=[dict(ft=3000),dict(ft=100)])]
    (root/'rollouts/chunk_0.json').write_text(json.dumps(original))
    (root/'rollouts_nothink_cap/chunk_0.json').write_text(json.dumps(cap))
    m=importlib.import_module('12_prepare_replay_lengths')
    lengths,manifest=m.prepare_lengths(root,cap=2048)
    assert lengths == {'0':[2048,100],'1024':[1044,1064],'2048':[1530,2098]}
    assert manifest['questions'] == 1
    assert manifest['samples_per_budget'] == {'0':2,'1024':2,'2048':2}
    assert len(manifest['files_sha256']) == 2
