from __future__ import annotations
import os, csv, math
from typing import List, Dict, Any

import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import orjson

# --- our utilities (paths match your layout) ---
from .utils.features import events_to_features
from .model_store import load as load_model, save as save_model
from .security.kba import get_kba, verify_kba, set_kba
from .train_util import train_user  # <-- single-user trainer for auto-train

app = FastAPI(title="Keystroke Auth")

# Allow the browser to call us from Live Server (127.0.0.1:5500) or localhost:5500
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500", "http://localhost:5500",
        # add more Live Server ports here if needed
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lightweight in-memory token store for KBA (dev only)
PENDING: Dict[str, Dict[str, Any]] = {}

class EnrollReq(BaseModel):
    username: str
    events: List[Dict[str, Any]]

class AuthReq(EnrollReq):
    pass

class KBAReq(BaseModel):
    token: str
    a1: str
    a2: str
    a3: str

class SetKBAReq(BaseModel):
    username: str
    questions: List[str]
    answers: List[str]

# Save raw events + append features for offline training
RAW_DIR = os.path.join(os.path.dirname(__file__), 'data', 'raw_events')
FEAT_CSV = os.path.join(os.path.dirname(__file__), 'data', 'features', 'features.csv')
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(os.path.dirname(FEAT_CSV), exist_ok=True)

# Keep this EXACT order in CSV; train_util.py uses the same
FEATURE_ORDER = [
    'H_mean','H_std','H_med','H_mad',
    'UD_mean','UD_std','UD_med','UD_mad',
    'DD_mean','DD_std','DD_med','DD_mad',
    'char_count','cadence_ppm','pause_frac_gt800'
]
if not os.path.exists(FEAT_CSV):
    with open(FEAT_CSV,'w',newline='') as f:
        w = csv.writer(f)
        w.writerow(['username'] + FEATURE_ORDER)

# ------------ helpers --------------

def _norm_user(u: str) -> str:
    """Lowercase + collapse whitespace so enroll/train/login all match."""
    return " ".join(u.strip().lower().split())

def _count_samples(u: str) -> int:
    """Count how many CSV rows exist for this (normalized) user."""
    u = _norm_user(u)
    if not os.path.exists(FEAT_CSV):
        return 0
    n = 0
    with open(FEAT_CSV, newline='') as f:
        r = csv.reader(f)
        _ = next(r, None)  # header
        for row in r:
            if not row: 
                continue
            if row[0].strip().lower() == u:
                n += 1
    return n

# ------------- endpoints ------------

@app.get('/health')
async def health():
    return {'ok': True}

@app.post('/kba/set')
async def kba_set(body: SetKBAReq):
    u = _norm_user(body.username)
    if len(body.questions) != 3 or len(body.answers) != 3:
        return {'status':'ERROR','msg':'Provide exactly 3 questions and 3 answers'}
    set_kba(u, body.questions, body.answers)
    return {'status':'OK','msg':'KBA saved'}

@app.get('/model/meta/{username}')
async def model_meta(username: str):
    u = _norm_user(username)
    pipe, meta = load_model(u)
    if pipe is None:
        return {'status':'NOT_FOUND','msg':'No model for this user'}
    return {'status':'OK','meta': meta}

@app.post('/enroll/sample')
async def enroll(req: EnrollReq, background_tasks: BackgroundTasks):
    u = _norm_user(req.username)

    # 1) save raw for analysis
    with open(os.path.join(RAW_DIR, f"{u}.jsonl"), 'ab') as f:
        f.write(orjson.dumps({'events': req.events}) + b"\n")

    # 2) extract features
    fx = events_to_features(req.events)
    if fx is None:
        return {'status':'ERROR', 'msg':'Not enough keystrokes'}
    X, order = fx  # X is a 1-D vector; order is the names for X

    # 3) VALIDATE finite features
    feats_dict = {k: float(v) for k, v in zip(order, X)}
    bad = [k for k, v in feats_dict.items() if not math.isfinite(v)]
    if bad:
        raise HTTPException(status_code=400, detail=f"Invalid feature values: {bad}")

    # 4) ensure CSV row follows FEATURE_ORDER exactly
    missing_for_csv = [k for k in FEATURE_ORDER if k not in feats_dict]
    if missing_for_csv:
        raise HTTPException(status_code=400, detail=f"Missing features for CSV: {missing_for_csv}")
    csv_row = [u] + [feats_dict[k] for k in FEATURE_ORDER]

    # 5) append a row to features.csv
    with open(FEAT_CSV,'a',newline='') as f:
        csv.writer(f).writerow(csv_row)

    # 6) auto-train: on 10th sample (or every +5 thereafter)
    cnt = _count_samples(u)
    if cnt == 10 or (cnt > 10 and cnt % 5 == 0):
        background_tasks.add_task(train_user, u)

    # 7) return features
    return {'status':'ENROLLED_SAMPLE', 'features': {k: feats_dict[k] for k in FEATURE_ORDER}}

@app.post('/auth/attempt')
async def auth(req: AuthReq):
    u = _norm_user(req.username)

    # Extract features from raw events
    fx = events_to_features(req.events)
    if fx is None:
        return {'status':'ERROR', 'msg':'Not enough keystrokes'}
    X, extracted_order = fx

    # Try to load model
    pipe, meta = load_model(u)
    if pipe is None:
        # If enough samples exist, train synchronously so this attempt gets scored now
        cnt = _count_samples(u)
        if cnt >= 10:
            res = train_user(u)  # synchronous
            pipe, meta = load_model(u)
        else:
            return {
                'status':'ENROLLMENT_REQUIRED',
                'msg': f'No model yet; collect {max(0,10-cnt)} more samples and try again.'
            }

    # Align to the model’s expected feature order (from meta)
    model_order = meta.get('features', FEATURE_ORDER)
    feats_dict = {k: float(v) for k, v in zip(extracted_order, X)}
    missing = [k for k in model_order if k not in feats_dict]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features for scoring: {missing}")

    # build row in correct order + validate finite
    row, bad = [], []
    for k in model_order:
        v = feats_dict[k]
        if not math.isfinite(v):
            bad.append(k)
        row.append(v)
    if bad:
        raise HTTPException(status_code=400, detail=f"Invalid feature values for scoring: {bad}")

    # Predict probability for "this is the user"
    x = np.array([row], dtype=float)
    if hasattr(pipe, "predict_proba"):
        prob = float(pipe.predict_proba(x)[0][1])
    elif hasattr(pipe, "decision_function"):
        prob = float(1 / (1 + np.exp(-pipe.decision_function(x)[0])))
    else:
        prob = float(pipe.predict(x)[0])

    allow = prob >= float(meta.get('threshold', 0.5))
    if allow:
        return {'status':'ALLOW', 'prob': prob}

    # Model not confident → try KBA
    kba = get_kba(u)
    if not kba:
        return {'status':'DENY', 'msg':'KBA not set and model confidence too low.'}

    token = os.urandom(16).hex()
    # Store the ALIGNED vector so partial_fit later uses the right order
    PENDING[token] = {'username': u, 'X': row}
    return {'status':'KBA_REQUIRED', 'prob': prob, 'token': token, 'questions': kba['questions']}

@app.post('/auth/kba')
async def kba_verify(body: KBAReq, background_tasks: BackgroundTasks):
    item = PENDING.pop(body.token, None)
    if not item:
        return {'status':'ERROR', 'msg':'Invalid or expired token'}

    u, row = item['username'], item['X']
    if not verify_kba(u, [body.a1, body.a2, body.a3]):
        return {'status':'DENY', 'msg':'KBA failed'}

    # 1) Append this "stressed" sample to the dataset (write in FEATURE_ORDER)
    #    We have 'row' in model_order; rebuild dict using meta['features'] then emit FEATURE_ORDER.
    pipe, meta = load_model(u)
    model_order = meta.get('features', FEATURE_ORDER) if meta else FEATURE_ORDER
    feats_from_kba = dict(zip(model_order, row))

    # Validate finite and presence
    bad = [k for k in FEATURE_ORDER if (k not in feats_from_kba or not math.isfinite(feats_from_kba[k]))]
    if bad:
        # If anything is off, skip CSV append but still allow (since KBA passed)
        bad_msg = f"Skipped CSV append due to invalid KBA features: {bad}"
    else:
        with open(FEAT_CSV,'a',newline='') as f:
            csv.writer(f).writerow([u] + [feats_from_kba[k] for k in FEATURE_ORDER])
        bad_msg = None

    # 2) Online adapt immediately for instant benefit (keep scaler fixed)
    prob_after = 1.0
    if pipe is not None:
        Z = pipe.named_steps['scaler'].transform([row])
        pipe.named_steps['clf'].partial_fit(Z, [1], classes=[0,1])
        save_model(u, pipe, meta)
        if hasattr(pipe, "predict_proba"):
            prob_after = float(pipe.predict_proba([row])[0][1])

    # 3) Fire a background full retrain so the disk model stays in sync
    background_tasks.add_task(train_user, u)

    resp = {'status':'ALLOW', 'msg':'KBA ok; model updated', 'prob_after': prob_after}
    if bad_msg:
        resp['note'] = bad_msg
    return resp
