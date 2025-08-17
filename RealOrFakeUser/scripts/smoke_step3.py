import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add project root

from backend.utils.features import events_to_features
from backend.security.kba import set_kba, get_kba, verify_kba


# --- Feature extraction test ---
events = [
    {'type':'down','code':'KeyA','t':0.0},
    {'type':'up','code':'KeyA','t':120.0},
    {'type':'down','code':'KeyB','t':180.0},
    {'type':'up','code':'KeyB','t':270.0},
    {'type':'down','code':'KeyC','t':320.0},
    {'type':'up','code':'KeyC','t':380.0},
]
fx = events_to_features(events)
print('features:', fx[0].shape, fx[1]) if fx else print('no features')

# --- KBA test ---
set_kba('jawad', ['Fav color?', 'City?', 'Food?'], ['blue', 'karachi', 'pizza'])
print('has kba file:', get_kba('jawad') is not None)
print('verify good:', verify_kba('jawad', ['blue', 'x', 'pizza']))
print('verify bad :', verify_kba('jawad', ['x', 'y', 'z']))
