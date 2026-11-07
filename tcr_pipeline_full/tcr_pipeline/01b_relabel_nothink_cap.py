"""Re-label non-thinking mode (B=0) with a much larger token cap (default 4096).

Why: in the main run 22% of B=0 samples hit the 1024-token cap and have no answer, so B=0 accuracy is a lower
bound. One run with a 4096 cap lets 01c_compare_nothink_cap.py report accuracy and cost at any cap up to 4096.
Same questions and chunking as stage 01. Resumable per chunk. Needs the GPU free (no vLLM server running).
"""
import os, sys, time
os.environ.setdefault('VLLM_USE_FLASHINFER_SAMPLER', '0')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
sys.path.insert(0, os.environ.get('TCR_DIR', '.'))
from tcr_common import *
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

CAP = int(os.environ.get('TCR_NOTHINK_CAP', '4096'))


def main():
    K, CH = CFG['k_traces'], CFG['chunk']
    assert CAP + 400 <= CFG['max_model_len'], 'cap + prompt must fit max_model_len'
    tok = AutoTokenizer.from_pretrained(CFG['model'])
    probs = load_problems()
    out_dir = RUN / 'rollouts_nothink_cap'
    out_dir.mkdir(exist_ok=True)
    chunks = [probs[i:i + CH] for i in range(0, len(probs), CH)]
    todo = [ci for ci in range(len(chunks)) if not (out_dir / f'chunk_{ci:03d}.json').exists()]
    print(f'[nothink] {len(probs)} prompts, {len(chunks)} chunks, {len(todo)} to do, cap {CAP}', flush=True)
    if not todo:
        return
    llm = LLM(model=CFG['model'], dtype=CFG['dtype'], max_model_len=CFG['max_model_len'],
              gpu_memory_utilization=CFG['gpu_mem_util'], enable_prefix_caching=True,
              max_num_seqs=CFG['max_num_seqs'], seed=CFG['seed'] + 7)
    sp = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=CAP, n=K)
    t_start = time.time()
    for ci in todo:
        t0 = time.time()
        ch = chunks[ci]
        prompts = [list(tok(chat_prompt(tok, p['problem'], False), add_special_tokens=False).input_ids) for p in ch]
        outs = llm.generate([{'prompt_token_ids': b} for b in prompts], sp, use_tqdm=False)
        recs, ntok = [], 0
        for p, ro in zip(ch, outs):
            samples = []
            for o in ro.outputs:
                pred = last_boxed(o.text)
                ntok += len(o.token_ids)
                samples.append(dict(pred=pred, key=norm_key(pred),
                                    ok=bool(pred is not None and is_correct(p['answer'], pred)),
                                    ft=len(o.token_ids), fin=(o.finish_reason == 'stop')))
            recs.append(dict(id=p['id'], gold=p['answer'], samples=samples))
        atomic_write_json(out_dir / f'chunk_{ci:03d}.json', recs)
        dt = time.time() - t0
        print(f'[nothink] chunk {ci} done in {dt:.0f}s | {ntok / dt:.0f} tok/s | elapsed {(time.time() - t_start) / 60:.1f} min', flush=True)
    print('[nothink] finished', flush=True)


if __name__ == '__main__':
    main()
