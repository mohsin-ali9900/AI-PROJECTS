from __future__ import annotations
import os, json, joblib
from typing import Dict, Any

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# Each user has two files: {user}.joblib (Pipeline) and {user}.meta.json (threshold, metrics)

def paths(user: str):
    return (
        os.path.join(MODELS_DIR, f'{user}.joblib'),
        os.path.join(MODELS_DIR, f'{user}.meta.json'),
    )

def save(user: str, pipeline, meta: Dict[str, Any]):
    mpath, jpath = paths(user)
    joblib.dump(pipeline, mpath)
    with open(jpath, 'w') as f:
        json.dump(meta, f)

def load(user: str):
    mpath, jpath = paths(user)
    if not (os.path.exists(mpath) and os.path.exists(jpath)):
        return None, None
    return joblib.load(mpath), json.load(open(jpath))
