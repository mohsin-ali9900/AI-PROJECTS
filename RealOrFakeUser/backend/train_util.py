# backend/train_util.py
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.impute import SimpleImputer

# Use relative import so this works with: python -m backend.train_util
from .model_store import save

FEATURES = [
    'H_mean','H_std','H_med','H_mad',
    'UD_mean','UD_std','UD_med','UD_mad',
    'DD_mean','DD_std','DD_med','DD_mad',
    'char_count','cadence_ppm','pause_frac_gt800'
]

# Expect a single CSV at backend/data/features/features.csv with columns: username + FEATURES
DATA_CSV = os.path.join('backend','data','features','features.csv')
MODELS_DIR = os.path.join('backend', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


# ---------------------------
# helpers
# ---------------------------

def _norm_user(u: str) -> str:
    """Normalize usernames to avoid casing/whitespace mismatches."""
    return " ".join(u.strip().lower().split())

def _load_df() -> pd.DataFrame:
    """Load features.csv and normalize usernames; raise if missing."""
    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(f"No features.csv found at {DATA_CSV}")
    df = pd.read_csv(DATA_CSV)
    # normalize usernames early
    if 'username' not in df.columns:
        raise ValueError("features.csv must have a 'username' column")
    df['username'] = df['username'].astype(str).str.strip().str.lower()
    # keep only declared FEATURES (in correct order)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"features.csv missing columns: {missing}")
    return df

def _build_pipeline() -> Pipeline:
    """Standard pipeline: Imputer → Scaler → SGD (logistic)."""
    return Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('scaler', StandardScaler()),
        ('clf', SGDClassifier(
            loss='log_loss', penalty='l2', alpha=1e-4,
            max_iter=2000, tol=1e-4, class_weight='balanced',
            random_state=42
        ))
    ])

def _fit_and_score(pipe: Pipeline, X: np.ndarray, y: np.ndarray) -> Tuple[float,float,float]:
    """
    Cross-validate, return (roc_auc, average_precision, best_threshold).
    Threshold is chosen by scanning quantiles to approach EER.
    """
    # choose n_splits safely based on the smallest class count
    _, counts = np.unique(y, return_counts=True)
    n_splits = int(max(2, min(5, counts.min())))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=False)

    scores, labels = [], []
    for tr, va in skf.split(X, y):
        pipe.fit(X[tr], y[tr])
        scores.append(pipe.predict_proba(X[va])[:, 1])
        labels.append(y[va])
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)

    roc = float(roc_auc_score(labels, scores))
    ap  = float(average_precision_score(labels, scores))

    # EER-ish threshold selection
    qs = np.quantile(scores, np.linspace(0.05, 0.95, 91))
    best_t, best_gap = 0.5, 9e9
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    for t in qs:
        tpr = (pos >= t).mean() if pos.size else 0.0
        fpr = (neg >= t).mean() if neg.size else 1.0
        gap = abs(fpr - (1 - tpr))
        if gap < best_gap:
            best_gap, best_t = gap, float(t)

    return roc, ap, float(best_t)


# ---------------------------
# public API
# ---------------------------

def train_user(user: str) -> Dict:
    """
    Train (or retrain) a single user's model from features.csv.
    Returns a dict: {'status': 'OK'|'ERROR', ...}
    """
    user = _norm_user(user)
    df = _load_df()

    pos_df = df[df.username == user]
    if len(pos_df) < 10:
        return {'status': 'ERROR', 'msg': f'{user} needs >=10 samples, has {len(pos_df)}'}

    neg_df = df[df.username != user]
    if len(neg_df) == 0:
        return {'status': 'ERROR', 'msg': 'No negatives available'}

    pos = pos_df[FEATURES].to_numpy(dtype=float)
    neg = neg_df.sample(min(len(pos) * 4, len(neg_df)), random_state=42)[FEATURES].to_numpy(dtype=float)

    X = np.vstack([pos, neg])
    y = np.hstack([np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int)])

    pipe = _build_pipeline()
    roc, ap, thr = _fit_and_score(pipe, X, y)

    # final fit on all data
    pipe.fit(X, y)
    meta = {
        'threshold': thr,
        'metrics': {'cv_roc_auc': roc, 'cv_avg_precision': ap},
        'features': FEATURES
    }
    save(user, pipe, meta)
    print(f"[OK] {user}: ROC-AUC={roc:.3f} AP={ap:.3f} thr={thr:.3f} pos={len(pos)}")
    return {'status': 'OK', 'meta': meta, 'pos': int(len(pos))}


def train_all() -> None:
    """
    Train (or retrain) all users found in features.csv.
    Skips users with <10 positive samples.
    """
    df = _load_df()
    users = sorted(df['username'].unique())

    for user in users:
        pos_df = df[df.username == user]
        pos = pos_df[FEATURES].to_numpy(dtype=float)
        if len(pos) < 10:
            print(f"[WARN] {user}: need >=10 positive samples, has {len(pos)}; skipping")
            continue

        neg_df = df[df.username != user]
        neg = neg_df.sample(min(len(pos) * 4, len(neg_df)), random_state=42)[FEATURES].to_numpy(dtype=float)

        X = np.vstack([pos, neg])
        y = np.hstack([np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int)])

        pipe = _build_pipeline()
        roc, ap, thr = _fit_and_score(pipe, X, y)

        # final fit on all data
        pipe.fit(X, y)
        meta = {
            'threshold': thr,
            'metrics': {'cv_roc_auc': roc, 'cv_avg_precision': ap},
            'features': FEATURES
        }
        save(user, pipe, meta)
        print(f"[OK] {user}: ROC-AUC={roc:.3f} AP={ap:.3f} thr={thr:.3f} pos={len(pos)}")


if __name__ == '__main__':
    train_all()
