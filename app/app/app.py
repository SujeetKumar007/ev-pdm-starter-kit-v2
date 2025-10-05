import streamlit as st
import pandas as pd
import numpy as np
import json, os
from joblib import load
import gdown, zipfile   # ✅ NEW

st.set_page_config(page_title="EV Fast Charger PDM", layout="wide")

# -------------------- Google Drive download setup --------------------
os.makedirs("data", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

# File IDs from Google Drive
FEATURES_ID = "122RJjkJ3ol1iXv3N39Zq5vX63PCC4wSp"
ARTIFACTS_ID = "13Ks3K-OyzUhin9aF7YNsbpNRTQ35nXZD"

# Download features.csv if not present
features_path_default = "data/features.csv"
if not os.path.exists(features_path_default):
    url = f"https://drive.google.com/uc?id={FEATURES_ID}"
    gdown.download(url, features_path_default, quiet=False)

# Download artifacts.zip if not extracted
artifacts_zip = "artifacts.zip"
if not os.path.exists("artifacts/clf_7d.joblib"):  # check if already extracted
    if not os.path.exists(artifacts_zip):
        url = f"https://drive.google.com/uc?id={ARTIFACTS_ID}"
        gdown.download(url, artifacts_zip, quiet=False)
    with zipfile.ZipFile(artifacts_zip, "r") as zip_ref:
        zip_ref.extractall("artifacts")
# ---------------------------------------------------------------------

@st.cache_resource
def load_models(artifacts):
    clf = load(os.path.join(artifacts, "clf_7d.joblib"))
    reg = load(os.path.join(artifacts, "rul_reg.joblib"))
    iforest = load(os.path.join(artifacts, "iforest.joblib"))
    cols = json.load(open(os.path.join(artifacts, "feature_columns.json"), "r"))
    return clf, reg, iforest, cols

def score(df, clf, reg, iforest, cols):
    X = df[cols].astype("float32").values
    p = clf.predict_proba(X)[:,1]
    rul = np.clip(reg.predict(X), 0, None)
    anom = -iforest.score_samples(X)
    out = df[["timestamp","evse_id"]].copy()
    out["p_fault_7d"] = p
    out["rul_hours"] = rul
    out["anomaly_score"] = anom
    return out

st.title("⚡ EV Fast Charger Predictive Maintenance")

# ✅ Default paths already set to downloaded files
features_path = st.text_input("Features CSV", features_path_default)
artifacts = st.text_input("Artifacts folder", "artifacts")

if st.button("Load and Score"):
    clf, reg, iforest, cols = load_models(artifacts)
    df = pd.read_csv(features_path, parse_dates=["timestamp"])
    last_ts = df["timestamp"].max()
    cutoff = last_ts - pd.Timedelta(hours=24)
    df_tail = df[df["timestamp"]>=cutoff].copy()
    scored = score(df_tail, clf, reg, iforest, cols)

    latest = scored.sort_values("timestamp").groupby("evse_id").tail(1)
    latest["risk_bucket"] = pd.cut(latest["p_fault_7d"],
                                   bins=[-0.01,0.2,0.4,0.6,0.8,1.01],
                                   labels=["Very Low","Low","Medium","High","Critical"])
    st.subheader("Top Risk EVSE (last 24h)")
    st.dataframe(latest.sort_values("p_fault_7d", ascending=False))

    st.subheader("Drilldown")
    evsel = st.selectbox("Choose EVSE", sorted(latest["evse_id"].unique().tolist()))
    hist = scored[scored["evse_id"]==evsel].sort_values("timestamp")
    st.line_chart(hist.set_index("timestamp")[["p_fault_7d"]])
    st.line_chart(hist.set_index("timestamp")[["rul_hours"]])
    st.line_chart(hist.set_index("timestamp")[["anomaly_score"]])




# import streamlit as st
# import pandas as pd
# import numpy as np
# import json, os
# from joblib import load

# st.set_page_config(page_title="EV Fast Charger PDM", layout="wide")

# @st.cache_resource
# def load_models(artifacts):
#     clf = load(os.path.join(artifacts, "clf_7d.joblib"))
#     reg = load(os.path.join(artifacts, "rul_reg.joblib"))
#     iforest = load(os.path.join(artifacts, "iforest.joblib"))
#     cols = json.load(open(os.path.join(artifacts, "feature_columns.json"), "r"))
#     return clf, reg, iforest, cols

# def score(df, clf, reg, iforest, cols):
#     X = df[cols].astype("float32").values
#     p = clf.predict_proba(X)[:,1]
#     rul = np.clip(reg.predict(X), 0, None)
#     anom = -iforest.score_samples(X)
#     out = df[["timestamp","evse_id"]].copy()
#     out["p_fault_7d"] = p
#     out["rul_hours"] = rul
#     out["anomaly_score"] = anom
#     return out

# st.title("⚡ EV Fast Charger Predictive Maintenance")

# features_path = st.text_input("Features CSV", "data/features.csv")
# artifacts = st.text_input("Artifacts folder", "artifacts")

# if st.button("Load and Score"):
#     clf, reg, iforest, cols = load_models(artifacts)
#     df = pd.read_csv(features_path, parse_dates=["timestamp"])
#     last_ts = df["timestamp"].max()
#     cutoff = last_ts - pd.Timedelta(hours=24)
#     df_tail = df[df["timestamp"]>=cutoff].copy()
#     scored = score(df_tail, clf, reg, iforest, cols)

#     latest = scored.sort_values("timestamp").groupby("evse_id").tail(1)
#     latest["risk_bucket"] = pd.cut(latest["p_fault_7d"],
#                                    bins=[-0.01,0.2,0.4,0.6,0.8,1.01],
#                                    labels=["Very Low","Low","Medium","High","Critical"])
#     st.subheader("Top Risk EVSE (last 24h)")
#     st.dataframe(latest.sort_values("p_fault_7d", ascending=False))

#     st.subheader("Drilldown")
#     evsel = st.selectbox("Choose EVSE", sorted(latest["evse_id"].unique().tolist()))
#     hist = scored[scored["evse_id"]==evsel].sort_values("timestamp")
#     st.line_chart(hist.set_index("timestamp")[["p_fault_7d"]])
#     st.line_chart(hist.set_index("timestamp")[["rul_hours"]])
#     st.line_chart(hist.set_index("timestamp")[["anomaly_score"]])
