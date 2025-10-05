import argparse, os, json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, mean_absolute_error
from joblib import dump
from datetime import datetime

EXCLUDE = {"timestamp","evse_id","failure","fault_type","next_failure_ts","y_fault_7d","rul_hours"}

def build(df):
    cols = [c for c in df.columns if c not in EXCLUDE]
    X = df[cols].astype("float32").values
    y = df["y_fault_7d"].astype("int8").values
    r = df["rul_hours"].astype("float32").values
    return cols, X, y, r

def split(df, test_frac=0.2):
    cut = df["timestamp"].quantile(1-test_frac)
    return df[df["timestamp"]<=cut], df[df["timestamp"]>cut]

def event_leadtime(df, proba, thr=0.5):
    df = df.copy()
    df["p"] = proba
    res = []
    for ev, g in df.groupby("evse_id"):
        g = g.sort_values("timestamp")
        fails = g[g["failure"]==1]["timestamp"].unique()
        for ft in fails:
            pre = g[g["timestamp"]<=ft]
            cross = pre[pre["p"]>=thr]
            if len(cross)>0:
                lt = (ft - cross.iloc[0]["timestamp"]).total_seconds()/3600.0
                if lt>=0:
                    res.append(lt)
    if not res: return {"detections": 0}
    return {"detections": len(res), "leadtime_mean_h": float(np.mean(res))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--artifacts", required=True)
    args = ap.parse_args()

    os.makedirs(args.artifacts, exist_ok=True)
    df = pd.read_csv(args.features, parse_dates=["timestamp"])
    tr, te = split(df, 0.2)
    cols, Xtr, ytr, rtr = build(tr)
    _, Xte, yte, rte = build(te)

    clf = RandomForestClassifier(n_estimators=200, max_depth=16, n_jobs=-1, random_state=42)
    clf.fit(Xtr, ytr)
    pte = clf.predict_proba(Xte)[:,1]
    auc = roc_auc_score(yte, pte)
    ap = average_precision_score(yte, pte)
    yhat = (pte>=0.5).astype(int)
    p,r,f1,_ = precision_recall_fscore_support(yte, yhat, average="binary", zero_division=0)
    lead = event_leadtime(te, pte, 0.5)

    reg = GradientBoostingRegressor(random_state=42)
    reg.fit(Xtr, rtr)
    mae = mean_absolute_error(rte, reg.predict(Xte))

    normal = tr[tr["y_fault_7d"]==0]
    _, Xn, _, _ = build(normal)
    iforest = IsolationForest(n_estimators=200, contamination=0.02, random_state=42)
    iforest.fit(Xn)

    from joblib import dump
    dump(clf, os.path.join(args.artifacts, "clf_7d.joblib"))
    dump(reg, os.path.join(args.artifacts, "rul_reg.joblib"))
    dump(iforest, os.path.join(args.artifacts, "iforest.joblib"))
    with open(os.path.join(args.artifacts, "feature_columns.json"), "w") as f:
        json.dump(cols, f, indent=2)

    card = {
        "classifier_metrics": {
            "test_auc": float(auc), "test_ap": float(ap),
            "test_precision@0.5": float(p), "test_recall@0.5": float(r), "test_f1@0.5": float(f1),
            **lead
        },
        "rul_mae_hours_test": float(mae),
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "timestamp": datetime.utcnow().isoformat()+"Z",
        "note": "Trained with RandomForest/GBR simple trainer for portability."
    }
    with open(os.path.join(args.artifacts, "model_card.json"), "w") as f:
        json.dump(card, f, indent=2)
    print(json.dumps(card, indent=2))

if __name__ == "__main__":
    main()
