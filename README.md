# EV Fast Charger Predictive Maintenance — Industrial Starter Kit

This repo gives you a production-style baseline to **simulate data, engineer features, train models, and visualize risk** for EV fast-charging stations on a MacBook (Apple Silicon OK).

## What you get

- **Synthetic data generator** modeling realistic charger use + degradation
- **Feature pipeline**: rolling stats, deltas, health indicators
- **Three models**:
  1) Early-warning **classifier** (failure in next 7 days)
  2) **RUL regressor** (remaining useful life in hours; capped if censored)
  3) **Anomaly detection** (Isolation Forest) on rolling z-scores
- **Streamlit dashboard** to explore risks per EVSE
- **Artifacts** saved with model cards + config tracking

> ⚠️ This is a strong baseline for an *industrial-style prototype*. Swap the simulator with your real feeds (OCPP, SCADA, BMS, site sensors) to move toward production.

## Quickstart (Mac)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1) Generate synthetic data (edit configs/sim.yaml if you want)
python -m src.data.generate_synthetic_evse --out data/sim.csv --days 60 --evse 20 --freq_min 5

# 2) Build features + labels
python -m src.features.build_features --in data/sim.csv --out data/features.csv --label_horizon_days 7

# 3) Train models
python -m src.models.train --features data/features.csv --artifacts artifacts

# 4) Try inference on latest day or stream the CSV
python -m src.serve.predict --features data/features.csv --artifacts artifacts --tail_hours 24

# 5) Launch dashboard
streamlit run app/app.py
```

## Data Columns (raw sim)

- `timestamp`, `evse_id`
- Operational: `session_active`, `session_power_kw`, `output_voltage_v`, `output_current_a`, `energy_kwh_24h`, `uptime_pct_24h`
- Thermal/flow: `ambient_temp_c`, `cabinet_temp_c`, `connector_temp_c`, `coolant_temp_c`, `coolant_flow_lpm`, `fan_rpm`
- Power quality: `grid_thd_pct`, `grid_imbalance_pct`, `dc_bus_voltage_v`, `dc_bus_ripple_v`
- Wear/counters: `contactor_cycles`, `plug_insertions`
- Events/labels: `failure`, `fault_type`

## Labels (features.csv)

- `y_fault_7d` (1 if failure occurs within next 7 days)
- `rul_hours` (remaining useful life; censored at cap if none forthcoming)
- Rolling features: mean/std/min/max for critical metrics + deltas

## Notes

- Use `HistGradientBoosting` to keep CPU-friendly and fast on MacBook.
- Models are saved with `joblib` along with `model_card.json` metadata.
- Threshold defaults to 0.5; tune for your use case (precision vs recall).

---

**License**: MIT
