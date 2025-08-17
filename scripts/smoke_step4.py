# Option B style so you can run `python scripts\smoke_step4.py`
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add project root

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

from backend.model_store import save, load

# --- Make a dummy pipeline ---
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SGDClassifier(loss="log_loss", random_state=42, max_iter=1000, tol=1e-3))
])

# Fake features: 15-dim (matches Step 3 output size)
rng = np.random.default_rng(42)
X = rng.normal(size=(40, 15))       # 40 samples, 15 features
y = np.array([0]*20 + [1]*20)       # two classes

pipe.fit(X, y)

# --- Save per-user model ---
user = "jawad"
meta = {"threshold": 0.5, "note": "dummy model for smoke test"}
save(user, pipe, meta)

# --- Load and test ---
pipe2, meta2 = load(user)
print("loaded?", pipe2 is not None, meta2)

# Make a prediction
proba = pipe2.predict_proba(X[:3])[:,1]
print("proba[:3] =", proba)
