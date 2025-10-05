import json, os, time
from datetime import datetime

def now_iso():
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

def save_model_card(path, **kwargs):
    meta = dict(created_utc=now_iso(), **kwargs)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
