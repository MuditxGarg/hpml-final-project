"""Shared helpers for the LATTE go/no-go pipeline."""
import json, os, re, random
from functools import lru_cache
from pathlib import Path

CFG_PATH = os.environ.get('LATTE_CONFIG', '/content/latte/config.json')
CFG = json.load(open(CFG_PATH))
RUN = Path(CFG['run_dir']) / ('smoke' if CFG.get('smoke') else 'main')
RUN.mkdir(parents=True, exist_ok=True)

USER_TMPL = ('Solve the following math problem carefully. Think through the steps internally. '
             'After finishing your thinking, give ONLY the final answer formatted as '
             '\\boxed{ANSWER}, with no additional explanation in the final response.\n\n{problem}')


def load_problems():
    """Deterministic subset of MATH-500 (or a local json for tests)."""
    local = os.environ.get('LATTE_PROBLEMS_JSON')
    if local:
        rows = json.load(open(local))
    else:
        from datasets import load_dataset
        ds = load_dataset('HuggingFaceH4/MATH-500', split='test')
        rows = [dict(id=str(r['unique_id']), problem=r['problem'], answer=r['answer'],
                     level=int(r['level']), subject=r['subject']) for r in ds]
    idx = random.Random(CFG['seed']).sample(range(len(rows)), min(CFG['n_prompts'], len(rows)))
    return [rows[i] for i in idx]


def chat_prompt(tok, problem, thinking=True):
    msgs = [{'role': 'user', 'content': USER_TMPL.replace('{problem}', problem)}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=thinking)


def last_boxed(text):
    i = text.rfind('\\boxed')
    if i < 0:
        return None
    j = text.find('{', i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == '{':
            depth += 1
        elif text[k] == '}':
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
    return None


def norm_key(s):
    """Canonical string used to group identical answers when voting."""
    if s is None:
        return None
    t = s.strip()
    t = re.sub(r'\\text\{([^}]*)\}', r'\1', t)
    for a in ['\\left', '\\right', '\\!', '\\,', '\\;', '\\ ', '$', '^\\circ', '^{\\circ}', '\\%', '%']:
        t = t.replace(a, '')
    t = t.replace('dfrac', 'frac').replace('tfrac', 'frac')
    t = re.sub(r'\s+', '', t).rstrip('.')
    return t or None


@lru_cache(maxsize=200000)
def is_correct(gold, pred):
    try:
        from math_verify import parse, verify
        g = parse('\\boxed{%s}' % gold)
        p = parse('\\boxed{%s}' % pred)
        if verify(g, p):
            return True
    except Exception:
        pass
    return norm_key(gold) == norm_key(pred)


def atomic_write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(obj, f)
    os.replace(tmp, path)
