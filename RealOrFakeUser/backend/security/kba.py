from __future__ import annotations
import json, os
from typing import Dict, List
from passlib.hash import bcrypt

KBA_DIR = os.path.join(os.path.dirname(__file__), 'kba_store')
os.makedirs(KBA_DIR, exist_ok=True)

# Normalize answers consistently
_normalize = lambda s: ' '.join(s.strip().lower().split())

def set_kba(username: str, questions: List[str], answers: List[str]):
    assert len(questions)==3 and len(answers)==3
    data = {
        'questions': questions,
        'hashes': [bcrypt.hash(_normalize(a)) for a in answers]
    }
    with open(os.path.join(KBA_DIR, f'{username}.json'), 'w') as f:
        json.dump(data, f)

def get_kba(username: str):
    p = os.path.join(KBA_DIR, f'{username}.json')
    if not os.path.exists(p): return None
    with open(p) as f: return json.load(f)

def verify_kba(username: str, answers: List[str]) -> bool:
    rec = get_kba(username)
    if not rec: return False
    answers_n = [_normalize(a) for a in answers]
    ok = 0
    for a, h in zip(answers_n, rec['hashes']):
        if bcrypt.verify(a, h): ok += 1
    return ok >= 2  # pass if 2 of 3 correct
