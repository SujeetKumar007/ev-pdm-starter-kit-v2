import argparse, os, json
import numpy as np
import pandas as pd
from joblib import load

def load_artifacts(path):
    clf = load(os.path.join(path, "clf_7d.joblib"))
    reg = load(os.path.join(path, "rul_reg.joblib"))
    iforest = load(os.path.join(path, "iforest.joblib"))
    cols = json.load(open(os.path.join(path, "feature_columns.json"), "r"))
    return clf, reg, iforest, cols

def score(df, clf, reg, iforest, cols):
    X = df[cols].astype("float32").values
    p = clf.predict_proba(X)[:,1]
    rul = np.clip(reg.predict(X), 0, None)
    anom = -iforest.score_samples(X)  # higher = more anomalous
    out = df[["timestamp","evse_id"]].copy()
    out["p_fault_7d"] = p
    out["rul_hours"] = rul
    out["anomaly_score"] = anom
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--tail_hours", type=int, default=24)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    clf, reg, iforest, cols = load_artifacts(args.artifacts)
    df = pd.read_csv(args.features, parse_dates=["timestamp"])
    # Only score last N hours for speed
    cutoff = df["timestamp"].max() - pd.Timedelta(hours=args.tail_hours)
    df_tail = df[df["timestamp"]>=cutoff].copy()
    scored = score(df_tail, clf, reg, iforest, cols)

    # Aggregate per EVSE (latest row per EVSE)
    latest = scored.sort_values("timestamp").groupby("evse_id").tail(1)
    latest["risk_bucket"] = pd.cut(latest["p_fault_7d"],
                                   bins=[-0.01,0.2,0.4,0.6,0.8,1.01],
                                   labels=["Very Low","Low","Medium","High","Critical"])
    latest = latest.sort_values("p_fault_7d", ascending=False)
    print(latest.to_string(index=False))

if __name__ == "__main__":
    main()
