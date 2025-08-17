# scripts/set_kba.py

# make sure we can import from project root when running this file
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from backend.security.kba import set_kba  # <- correct path

# Edit these per user (use real answers)
user = 'jawad'
questions = [
    "What is your mother's maiden name?",
    "What is the name of your first school?",
    "What city were you born in?"
]
answers = ['<ans1>', '<ans2>', '<ans3>']  # <-- replace with real answers

set_kba(user, questions, answers)
print('KBA set for', user)
