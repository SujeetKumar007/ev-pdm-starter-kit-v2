import argparse, os
import numpy as np
import pandas as pd

ROLL_COLS = [
    "session_power_kw","output_voltage_v","output_current_a",
    "cabinet_temp_c","connector_temp_c","coolant_temp_c",
    "coolant_flow_lpm","fan_rpm","grid_thd_pct","grid_imbalance_pct",
    "dc_bus_ripple_v","energy_kwh_24h","uptime_pct_24h"
]

def add_rolling_features(df, window=12):
    # window is # of rows per EVSE (5-min * 12 = 1 hour)
    out = []
    for evse, g in df.groupby("evse_id", sort=False):
        g = g.sort_values("timestamp").copy()
        for c in ROLL_COLS:
            r = g[c].rolling(window, min_periods=4)
            g[f"{c}_mean_{window}"] = r.mean()
            g[f"{c}_std_{window}"] = r.std()
            g[f"{c}_min_{window}"] = r.min()
            g[f"{c}_max_{window}"] = r.max()
            g[f"{c}_delta"] = g[c].diff()
        out.append(g)
    return pd.concat(out, axis=0)

def add_labels(df, horizon_days=7):
    df = df.sort_values(["evse_id","timestamp"]).copy()
    # Compute next failure timestamp per row to build y_fault_7d and rul_hours
    df["next_failure_ts"] = pd.NaT
    for evse, g in df.groupby("evse_id"):
        g = g.copy()
        fail_ts = g.loc[g["failure"]==1, "timestamp"]
        # For each row, find next failure
        next_fail = pd.Series(index=g.index, dtype="datetime64[ns]")
        if len(fail_ts)>0:
            nxt = fail_ts.reset_index(drop=True)
            j = 0
            nf = nxt.iloc[j]
            for i, ts in enumerate(g["timestamp"]):
                while j < len(nxt) and ts > nxt.iloc[j]:
                    j += 1
                    if j < len(nxt):
                        nf = nxt.iloc[j]
                if j < len(nxt):
                    next_fail.iloc[i] = nxt.iloc[j]
        df.loc[g.index, "next_failure_ts"] = next_fail.values

    # y_fault_7d
    horizon = pd.Timedelta(days=horizon_days)
    df["y_fault_7d"] = ((df["next_failure_ts"].notna()) & ((df["next_failure_ts"] - df["timestamp"]) <= horizon)).astype(int)
    # RUL hours (cap at 30 days)
    cap = pd.Timedelta(days=30)
    delta = (df["next_failure_ts"] - df["timestamp"])
    df["rul_hours"] = np.where(df["next_failure_ts"].notna(),
                               np.maximum(0, delta.dt.total_seconds()/3600.0),
                               cap.total_seconds()/3600.0)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label_horizon_days", type=int, default=7)
    args = ap.parse_args()

    df = pd.read_csv(args.inp, parse_dates=["timestamp"])
    # Basic cleaning
    df.sort_values(["evse_id","timestamp"], inplace=True)
    # Rolling features (1-hour window by default since freq=5min -> 12 rows)
    feats = add_rolling_features(df, window=12)
    feats = add_labels(feats, horizon_days=args.label_horizon_days)

    # Drop early NaNs from rolling
    feats = feats.groupby("evse_id").apply(lambda x: x.iloc[12:]).reset_index(drop=True)

    feats.to_csv(args.out, index=False)
    print(f"Wrote features: {args.out} with shape {feats.shape}")

if __name__ == "__main__":
    main()
