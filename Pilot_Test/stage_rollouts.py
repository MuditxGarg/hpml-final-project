"""Stage 1 (test 2): one-pass grid labeling.

For each prompt generate K thinking traces once (up to t_max tokens). Cut every trace at each
thinking budget B, force an answer, and record correctness + token counts. Resumable per chunk.
"""
import os, sys, time
sys.path.insert(0, os.environ.get('LATTE_DIR', '/content/latte'))
from common import *
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer



def main():
    K, TMAX, BUD, CH = CFG['k_traces'], CFG['t_max'], CFG['budgets'], CFG['chunk']
    assert max(BUD) <= TMAX, 'largest budget must be <= t_max'

    tok = AutoTokenizer.from_pretrained(CFG['model'])
    probs = load_problems()
    out_dir = RUN / 'rollouts'
    out_dir.mkdir(exist_ok=True)
    chunks = [probs[i:i + CH] for i in range(0, len(probs), CH)]
    todo = [ci for ci in range(len(chunks)) if not (out_dir / f'chunk_{ci:03d}.json').exists()]
    print(f'[rollouts] {len(probs)} prompts, {len(chunks)} chunks, {len(todo)} to do', flush=True)
    if not todo:
        sys.exit(0)

    llm = LLM(model=CFG['model'], dtype=CFG['dtype'], max_model_len=CFG['max_model_len'],
              gpu_memory_utilization=CFG['gpu_mem_util'], enable_prefix_caching=True,
              max_num_seqs=CFG['max_num_seqs'], seed=CFG['seed'])


    def enc(s):
        return list(tok(s, add_special_tokens=False).input_ids)


    PRE = 'The final answer is $\\boxed{'
    END_NAT = enc('</think>\n\n' + PRE)
    END_FORCED = enc('\n\nConsidering the limited time, I have to give the final answer now.\n</think>\n\n' + PRE)
    spA = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=TMAX, n=K, stop=['</think>'])
    spN = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=1024, n=K)
    spF = SamplingParams(temperature=0.0, max_tokens=48)

    T_START = time.time()
    sample = None
    for ci in todo:
        t0 = time.time()
        ch = chunks[ci]
        base = [enc(chat_prompt(tok, p['problem'], True) + '<think>\n') for p in ch]
        nothink = [enc(chat_prompt(tok, p['problem'], False)) for p in ch]
        outA = llm.generate([{'prompt_token_ids': b} for b in base], spA, use_tqdm=False)
        outN = llm.generate([{'prompt_token_ids': b} for b in nothink], spN, use_tqdm=False)
        think = [[list(o.token_ids) for o in ro.outputs] for ro in outA]
        fin = [[o.finish_reason == 'stop' for o in ro.outputs] for ro in outA]
        gen_tokens = sum(len(t) for pt in think for t in pt)

        items, where = [], {}

        def add(key, ids):
            if key not in where:
                where[key] = len(items)
                items.append(ids)

        for pi in range(len(ch)):
            for k in range(K):
                L = len(think[pi][k])
                if fin[pi][k]:
                    add((pi, k, 'nat'), base[pi] + think[pi][k] + END_NAT)
                for b in BUD:
                    if b == 0 or (fin[pi][k] and L <= b):
                        continue
                    add((pi, k, b), base[pi] + think[pi][k][:b] + END_FORCED)
        outF = llm.generate([{'prompt_token_ids': x} for x in items], spF, use_tqdm=False) if items else []

        recs = []
        for pi, p in enumerate(ch):
            traces = []
            for k in range(K):
                L = len(think[pi][k])
                d = {}
                for b in BUD:
                    if b == 0:
                        o = outN[pi].outputs[k]
                    elif fin[pi][k] and L <= b:
                        o = outF[where[(pi, k, 'nat')]].outputs[0]
                    else:
                        o = outF[where[(pi, k, b)]].outputs[0]
                    pred = last_boxed(o.text if b == 0 else '\\boxed{' + o.text)
                    if pi == 0 and k == 0 and b == BUD[-1]:
                        sample = o.text
                    d[str(b)] = dict(pred=pred, key=norm_key(pred),
                                     ok=bool(pred is not None and is_correct(p['answer'], pred)),
                                     ft=len(o.token_ids))
                traces.append(dict(L=L, fin=bool(fin[pi][k]), b=d))
            recs.append(dict(id=p['id'], level=p['level'], subject=p['subject'], gold=p['answer'],
                             prompt_tokens=len(base[pi]), traces=traces))
        atomic_write_json(out_dir / f'chunk_{ci:03d}.json', recs)
        if ci == todo[0]:
            print('[rollouts] sample forced-answer output (sanity check, should look like a boxed answer):', repr((sample or '')[:160]), flush=True)
        dt = time.time() - t0
        print(f'[rollouts] chunk {ci} done in {dt:.0f}s | {gen_tokens / dt:.0f} thinking tok/s | '
              f'elapsed {(time.time() - T_START) / 60:.1f} min', flush=True)
    print('[rollouts] finished', flush=True)



if __name__ == '__main__':
    main()
