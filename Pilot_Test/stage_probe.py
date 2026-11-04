"""Stage 2 (test 3 features): last-token hidden states at the end of the prompt (prefill only)."""
import os, sys, time
sys.path.insert(0, os.environ.get('LATTE_DIR', '/content/latte'))
from common import *
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

out_dir = RUN / 'probe'
out_dir.mkdir(exist_ok=True)
if (out_dir / 'hidden.npz').exists():
    print('[probe] already done')
    sys.exit(0)

tok = AutoTokenizer.from_pretrained(CFG['model'])
dtype = torch.bfloat16 if CFG['dtype'] == 'bfloat16' else torch.float16
model = AutoModelForCausalLM.from_pretrained(CFG['model'], torch_dtype=dtype).to('cuda').eval()
nL = model.config.num_hidden_layers
sel = sorted(set(list(range(0, nL + 1, 4)) + [nL]))
probs = load_problems()

X, ids, plen, ms = [], [], [], []
for i, p in enumerate(probs):
    text = chat_prompt(tok, p['problem'], True)
    enc = tok(text, return_tensors='pt', add_special_tokens=False).to('cuda')
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        hs = model(**enc, output_hidden_states=True).hidden_states
    torch.cuda.synchronize()
    ms.append((time.perf_counter() - t0) * 1000)
    X.append(torch.stack([hs[l][0, -1] for l in sel]).float().cpu().numpy())
    ids.append(p['id'])
    plen.append(int(enc['input_ids'].shape[1]))
    if i % 50 == 0:
        print(f'[probe] {i}/{len(probs)}', flush=True)
np.savez_compressed(out_dir / 'hidden.npz', X=np.stack(X).astype(np.float16), layers=np.array(sel),
                    ids=np.array(ids), plen=np.array(plen), prefill_ms=np.array(ms))
print(f'[probe] saved {len(X)} prompts, {len(sel)} layers, median prefill {np.median(ms[1:] or ms):.1f} ms', flush=True)
