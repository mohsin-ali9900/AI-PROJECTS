from __future__ import annotations
import numpy as np
from typing import List, Dict, Any

# events = [{type:'down'|'up'|'paste', code?:'KeyA', t:float_ms}, ...]

def events_to_features(events: List[Dict[str, Any]]):
    # filter & sort
    E = [e for e in events if e.get('type') in ('down','up','paste')]
    E.sort(key=lambda x: x['t'])

    # pair downs->ups per code
    stacks = {}
    pairs = []  # (code, t_down, t_up)
    for e in E:
        if e['type']=='down':
            stacks.setdefault(e.get('code','UNK'), []).append(e['t'])
        elif e['type']=='up':
            code = e.get('code','UNK')
            if code in stacks and stacks[code]:
                t0 = stacks[code].pop(0)
                if e['t'] >= t0:
                    pairs.append((code, t0, e['t']))
    if not pairs:
        return None  # not enough data

    # sequence ordered by press time
    seq = sorted(pairs, key=lambda x: x[1])
    t_down = np.array([s[1] for s in seq], dtype=float)
    t_up   = np.array([s[2] for s in seq], dtype=float)

    dwell = t_up - t_down                      # H
    dd    = np.diff(t_down, prepend=np.nan)    # DD: current_down - prev_down
    rp    = np.concatenate([(t_down[1:] - t_up[:-1]), [np.nan]])  # RP to next press
    ud    = rp.copy()                          # same def: next_down - this_up

    # cleans
    def clean(x):
        x = x[np.isfinite(x)]
        x = x[(x >= 0) & (x <= 5000.0)]
        return x
    H  = clean(dwell)
    DD = clean(dd[1:])        # skip first NaN
    UD = clean(ud[:-1])       # drop last NaN

    # aggregates per attempt
    def stats(x):
        if x.size == 0:
            return dict(mean=np.nan, std=np.nan, med=np.nan, mad=np.nan)
        med = np.median(x)
        return dict(mean=float(np.mean(x)), std=float(np.std(x, ddof=1)) if x.size>1 else 0.0,
                    med=float(med), mad=float(np.median(np.abs(x - med))))

    Hs, UDs, DDs = stats(H), stats(UD), stats(DD)

    # cadence & pauses based on DD (press->press cadence)
    cadence_ppm = float(60000.0 / np.mean(DD)) if DD.size else np.nan
    pause_frac = float(np.mean(DD > 800.0)) if DD.size else np.nan

    feat = {
        'H_mean':Hs['mean'],'H_std':Hs['std'],'H_med':Hs['med'],'H_mad':Hs['mad'],
        'UD_mean':UDs['mean'],'UD_std':UDs['std'],'UD_med':UDs['med'],'UD_mad':UDs['mad'],
        'DD_mean':DDs['mean'],'DD_std':DDs['std'],'DD_med':DDs['med'],'DD_mad':DDs['mad'],
        'char_count': int(len(seq)),
        'cadence_ppm': cadence_ppm,
        'pause_frac_gt800': pause_frac,
    }
    feature_order = list(feat.keys())
    X = np.array([feat[k] for k in feature_order], dtype=float)
    return X, feature_order
