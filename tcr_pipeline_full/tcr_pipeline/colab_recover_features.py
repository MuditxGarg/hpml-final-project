"""Run in a free T4 Colab cell after installing datasets, transformers and torch.

Regenerates prompt features, not labels. T4 uses FP16; this is a new feature
extraction run, not a bitwise reproduction of the original RTX4080 BF16 run.
Outputs under /content/latte_features (override LATTE_FEATURE_OUTPUT).
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import zipfile

import numpy as np
import torch
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    out = Path(os.environ.get('LATTE_FEATURE_OUTPUT', '/content/latte_features'))
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'hidden.npz').exists():
        raise RuntimeError('Output already exists; choose a new LATTE_FEATURE_OUTPUT')
    model_name = 'Qwen/Qwen3-1.7B'
    if not torch.cuda.is_available():
        raise RuntimeError('A GPU runtime is required')
    template = ('Solve the following math problem carefully. Think through the steps internally. '
                'After finishing your thinking, give ONLY the final answer formatted as '
                '\\boxed{ANSWER}, with no additional explanation in the final response.\n\n{problem}')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Explicit eager attention avoids optional-kernel installs on a short-lived T4.
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16,
                                                attn_implementation='eager').to('cuda').eval()
    dataset = load_dataset('HuggingFaceH4/MATH-500', split='test')
    rows = sorted(dataset, key=lambda r: str(r['unique_id']))
    assert len(rows) == 500 and len({r['unique_id'] for r in rows}) == 500
    layers = sorted(set(range(0, model.config.num_hidden_layers+1, 4)) | {model.config.num_hidden_layers})
    features, ids, plen, elapsed, replay_prompts = [], [], [], [], []
    start = time.monotonic()
    for i, row in enumerate(rows):
        if time.monotonic()-start > 1800:
            raise TimeoutError('Feature extraction exceeded 30 minutes')
        message = [{'role':'user', 'content':template.replace('{problem}', row['problem'])}]
        thinking = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        nothinking = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = tokenizer(thinking, add_special_tokens=False, return_tensors='pt').to('cuda')
        torch.cuda.synchronize()
        t0 = time.monotonic()
        with torch.inference_mode():
            states = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
        torch.cuda.synchronize()
        elapsed.append(1000*(time.monotonic()-t0))
        features.append(torch.stack([states[layer][0,-1] for layer in layers]).float().cpu().numpy())
        ids.append(str(row['unique_id']))
        plen.append(enc['input_ids'].shape[1])
        replay_prompts.append(dict(id=ids[-1], thinking=thinking, nothinking=nothinking))
        del states, enc
        if i % 50 == 0:
            print(f'FEATURES {i}/500 elapsed={time.monotonic()-start:.1f}s', flush=True)
    x = np.stack(features).astype(np.float16)
    if not np.isfinite(x).all():
        raise ValueError('Non-finite hidden states')
    np.savez_compressed(out/'hidden.npz', X=x, ids=np.array(ids), layers=np.array(layers),
                        plen=np.array(plen), prefill_ms=np.array(elapsed))
    (out/'prompts.json').write_text(json.dumps(replay_prompts))
    manifest = dict(model=model_name, model_revision=getattr(model.config, '_commit_hash', None),
                    dataset='HuggingFaceH4/MATH-500', dataset_fingerprint=dataset._fingerprint,
                    questions=len(ids), layers=layers, shape=list(x.shape), dtype='float16',
                    gpu=torch.cuda.get_device_name(0), python=platform.python_version(), torch=torch.__version__,
                    transformers=transformers.__version__, prompt_template=template,
                    prompt_mode='enable_thinking=True', median_prefill_ms=float(np.median(elapsed[1:])),
                    extraction_seconds=time.monotonic()-start,
                    provenance='Regenerated on free T4; differs in precision/backend from original RTX4080 features',
                    hidden_sha256=hashlib.sha256((out/'hidden.npz').read_bytes()).hexdigest())
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    archive = out.parent/'latte_features_t4.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in ['hidden.npz','prompts.json','manifest.json']:
            z.write(out/name, name)
    print(json.dumps(manifest, indent=2))
    print(f'READY {archive}', flush=True)


if __name__ == '__main__':
    main()
