import argparse, os, json
import numpy as np
import pandas as pd
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from joblib import dump
from datetime import datetime
from ..utils import save_model_card

USE_COLS_EXCLUDE = {
    "timestamp","evse_id","failure","fault_type","next_failure_ts","y_fault_7d","rul_hours"
}

def build_Xy(df):
    cols = [c for c in df.columns if c not in USE_COLS_EXCLUDE]
    X = df[cols].astype("float32").values
    y = df["y_fault_7d"].astype("int8").values
    r = df["rul_hours"].astype("float32").values
    return cols, X, y, r

def time_split(df, test_frac=0.2):
    # Simple time-based split
    cutoff = df["timestamp"].quantile(1-test_frac)
    train = df[df["timestamp"] <= cutoff]
    test = df[df["timestamp"] > cutoff]
    return train, test

def eval_classifier(model, X, y, prefix=""):
    prob = model.predict_proba(X)[:,1]
    auc = roc_auc_score(y, prob)
    ap = average_precision_score(y, prob)
    # Choose default threshold 0.5 for reporting
    yhat = (prob>=0.5).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
    return {
        f"{prefix}auc": float(auc),
        f"{prefix}ap": float(ap),
        f"{prefix}precision@0.5": float(p),
        f"{prefix}recall@0.5": float(r),
        f"{prefix}f1@0.5": float(f1),
    }

def event_leadtime(df, proba, threshold=0.5):
    # Compute lead time (hours) for first detection before each failure per EVSE
    res = []
    df = df.copy()
    df["proba"] = proba
    for ev, g in df.groupby("evse_id"):
        g = g.sort_values("timestamp")
        fails = g[g["failure"]==1]["timestamp"].unique()
        for ft in fails:
            pre = g[g["timestamp"]<=ft]
            cross = pre[pre["proba"]>=threshold]
            if len(cross)>0:
                first_detect = cross.iloc[0]["timestamp"]
                lead = (ft - first_detect).total_seconds()/3600.0
                if lead>=0:
                    res.append(lead)
    if len(res)==0:
        return {"detections": 0}
    return {
        "detections": len(res),
        "leadtime_mean_h": float(np.mean(res)),
        "leadtime_p50_h": float(np.percentile(res,50)),
        "leadtime_p90_h": float(np.percentile(res,90)),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--artifacts", required=True)
    args = ap.parse_args()

    os.makedirs(args.artifacts, exist_ok=True)
    df = pd.read_csv(args.features, parse_dates=["timestamp"])

    # Split
    tr, te = time_split(df, test_frac=0.2)
    cols, Xtr, ytr, rtr = build_Xy(tr)
    _, Xte, yte, rte = build_Xy(te)

    # Classifier
    clf = HistGradientBoostingClassifier(max_depth=8, max_iter=300, learning_rate=0.05,
                                         l2_regularization=0.01, random_state=42)
    clf.fit(Xtr, ytr)
    cls_tr = eval_classifier(clf, Xtr, ytr, prefix="train_")
    cls_te = eval_classifier(clf, Xte, yte, prefix="test_")

    # Event-level lead time on test
    proba_te = clf.predict_proba(Xte)[:,1]
    lead = event_leadtime(te, proba_te, threshold=0.5)

    # RUL regressor
    reg = HistGradientBoostingRegressor(max_depth=8, max_iter=300, learning_rate=0.05,
                                        l2_regularization=0.0, random_state=42)
    reg.fit(Xtr, rtr)
    pred_rul = np.clip(reg.predict(Xte), 0, None)
    mae_rul = mean_absolute_error(rte, pred_rul)

    # Anomaly (fit on train only where y=0 to learn "normal")
    normal = tr[tr["y_fault_7d"]==0]
    _, Xn, _, _ = build_Xy(normal)
    iforest = IsolationForest(n_estimators=200, contamination=0.02, random_state=42)
    iforest.fit(Xn)

    # Save models + metadata
    dump(clf, os.path.join(args.artifacts, "clf_7d.joblib"))
    dump(reg, os.path.join(args.artifacts, "rul_reg.joblib"))
    dump(iforest, os.path.join(args.artifacts, "iforest.joblib"))
    with open(os.path.join(args.artifacts, "feature_columns.json"), "w") as f:
        json.dump(cols, f, indent=2)

    card = {
        "classifier_metrics": {**cls_tr, **cls_te, **lead},
        "rul_mae_hours_test": float(mae_rul),
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "timestamp": datetime.utcnow().isoformat()+"Z"
    }
    save_model_card(os.path.join(args.artifacts, "model_card.json"), **card)
    print(json.dumps(card, indent=2))

if __name__ == "__main__":
    main()
