"""Existing project answer extraction and verification, without run-directory side effects."""
import re
from functools import lru_cache

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


