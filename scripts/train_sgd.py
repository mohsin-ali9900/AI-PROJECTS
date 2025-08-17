# scripts/train_sgd.py
import sys, os, argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add project root

from backend.train_util import train_all  # Step 5 file you already created

# Optional: if you added train_user() earlier, we’ll use it when --user is given
try:
    from backend.train_util import train_user as _train_user
except Exception:
    _train_user = None

def main():
    p = argparse.ArgumentParser(description="Train per-user SGD models")
    p.add_argument("-u", "--user", help="Train a single username (optional)")
    args = p.parse_args()

    if args.user:
        if _train_user is None:
            print("[WARN] train_user() not available; running full train_all() instead.")
            return train_all()
        res = _train_user(args.user.strip().lower())
        print(res)
        return

    # Bulk train all users found in features.csv
    train_all()

if __name__ == "__main__":
    main()
