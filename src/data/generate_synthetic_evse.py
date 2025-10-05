import argparse, os, math, random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yaml

def diurnal_multiplier(ts):
    # Peak around 6-9pm, low at night
    hour = ts.hour + ts.minute/60.0
    return 0.5 + 0.5 * (np.sin((hour-13)*np.pi/12) + 1)/2 + 0.25 * (np.sin((hour-20)*np.pi/6)+1)/2

def weekend_boost(ts):
    return 1.15 if ts.weekday() >= 5 else 1.0

def jitter(x, pct=0.1):
    return x * (1 + np.random.uniform(-pct, pct))

def simulate(config):
    rng = np.random.default_rng(config.get("seed", 42))
    start = pd.Timestamp(config.get("start", "2025-01-01 00:00:00"))
    end = start + pd.Timedelta(days=config.get("days", 60))
    freq = f"{config.get('freq_min', 5)}min"
    ts_index = pd.date_range(start, end, freq=freq, inclusive="left")

    evse_n = config.get("evse", 20)
    deg = config.get("degradation", {})
    thr = config.get("thresholds", {})
    usage = config.get("usage", {})
    pq = config.get("power_quality", {})

    rows = []
    for ev in range(evse_n):
        ev_id = f"EVSE_{ev:03d}"
        # Per-EVSE random knobs
        wear_per_insert = jitter(deg.get("connector_wear_per_insertion", 0.00008), 0.3)
        flow_drop_day = jitter(deg.get("coolant_flow_drop_per_day", 0.002), 0.4)
        fan_drop_day = jitter(deg.get("fan_rpm_drop_per_day", 3.0), 0.5)
        ripple_rise_mwh = jitter(deg.get("dc_ripple_rise_per_mwh", 0.4), 0.4)
        therm_margin_drop = jitter(deg.get("thermal_margin_drop_per_day", 0.015), 0.4)

        # State
        connector_wear = rng.uniform(0.05, 0.2)   # 0..1
        coolant_flow = rng.uniform(13.5, 16.0)    # LPM
        fan_rpm = rng.uniform(2100, 2600)
        dc_ripple = rng.uniform(2.0, 3.0)         # V
        therm_margin = rng.uniform(12.0, 18.0)    # degC margin to threshold

        energy_cum_mwh = 0.0
        contactor_cycles = 0
        plug_insertions = int(rng.integers(500, 2500))

        # Random dominant fault mode per EVSE (still allow others)
        fault_mode = rng.choice([
            "connector_overheat","coolant_pump_degradation","fan_degradation",
            "dc_cap_aging","grid_power_quality"
        ], p=[0.28,0.22,0.2,0.2,0.1])

        failure_flag = False
        failure_time = None
        fault_type = None

        # Daily profile seeds
        base_sessions_per_hour = usage.get("base_sessions_per_hour", 0.6) * rng.uniform(0.8,1.25)
        peak_mult = usage.get("peak_multiplier", 2.1) * rng.uniform(0.8,1.2)
        weekend_mult = usage.get("weekend_multiplier", 1.1) * rng.uniform(0.9,1.2)

        energy_24h = 0.0
        uptime_24h = 1.0
        ocpp_errors_24h = 0

        last_day = None
        for ts in ts_index:
            if last_day is None or ts.date() != last_day:
                # New day: degrade some states
                coolant_flow = max(4.0, coolant_flow - flow_drop_day)
                fan_rpm = max(700, fan_rpm - fan_drop_day)
                therm_margin = max(2.0, therm_margin - therm_margin_drop)
                last_day = ts.date()
                # Daily counters
                energy_24h = 0.0
                ocpp_errors_24h = 0
                uptime_24h = 1.0

            # Usage intensity
            use_lambda = base_sessions_per_hour * diurnal_multiplier(ts) * weekend_boost(ts) * weekend_mult
            session_start = rng.random() < use_lambda * (1/12.0)  # per 5-min tick
            session_active = session_start or (rng.random() < 0.85 and 'active' in locals() and active)
            active = session_active

            power_kw = 0.0
            out_v = 800 + rng.normal(0,4)
            out_a = 0.0
            if session_active:
                target_kw = rng.uniform(80, 200) * (1.1 if fault_mode=="grid_power_quality" and rng.random()<0.1 else 1.0)
                # Throttle if thermal margins are low
                throttle = max(0.6, min(1.0, (therm_margin/12.0)))
                power_kw = target_kw * throttle * rng.uniform(0.9,1.05)
                out_v = 700 + rng.normal(0,6)
                out_a = (power_kw*1000) / max(out_v, 1)

                # Counters
                if session_start:
                    contactor_cycles += 1
                    plug_insertions += 1
                    connector_wear += wear_per_insert

            # Thermal dynamics
            ambient = 18 + 12*np.sin((ts.hour-6)/24*2*np.pi) + rng.normal(0,1.2)
            cabinet_temp = ambient + max(0, power_kw/40) + rng.normal(0,0.8)
            # Cooling
            cabinet_temp -= (coolant_flow-10)/8 if coolant_flow>10 else 0
            cabinet_temp -= (fan_rpm-2000)/800 if fan_rpm>2000 else 0

            connector_temp = ambient + (power_kw/25) + 45*connector_wear + rng.normal(0,1.2)

            # Coolant temp follows cabinet
            coolant_temp = ambient + max(0, power_kw/60) + rng.normal(0,0.6)

            # DC bus behavior
            dc_bus_v = 900 + rng.normal(0, 5)
            dc_ripple += (power_kw/2000.0) * (ripple_rise_mwh/1000.0)
            dc_ripple = min(dc_ripple + rng.normal(0,0.03), 12.0)

            # Power quality
            thd = pq.get("thd_base_pct", 3.5) + rng.normal(0,0.3)
            imb = pq.get("imb_base_pct", 1.2) + rng.normal(0,0.15)
            if fault_mode=="grid_power_quality" and rng.random()<0.06:
                thd += rng.uniform(2.0, pq.get("thd_spike_pct", 9.0))
                imb += rng.uniform(1.0, pq.get("imb_spike_pct", 4.0))
                ocpp_errors_24h += 1

            # Update energy
            energy_24h += power_kw * (5/60.0) / 1000.0  # MWh
            energy_cum_mwh += power_kw * (5/60.0) / 1000.0

            # Uptime degrade if temps too high / PQ bad
            penalty = 0.0
            if cabinet_temp > 75 or connector_temp > 85 or thd>8.0 or imb>3.5:
                penalty = 0.2 if session_active else 0.05
                uptime_24h = max(0.75, uptime_24h - penalty*(5/60.0))

            # Natural fault progression
            if fault_mode=="coolant_pump_degradation" and rng.random()<0.02:
                coolant_flow -= rng.uniform(0.05, 0.15)
            if fault_mode=="fan_degradation" and rng.random()<0.03:
                fan_rpm -= rng.uniform(10, 50)

            # Trip to failure if thresholds breached persistently
            if not failure_flag:
                breach = (
                    connector_temp > 90 or
                    cabinet_temp > 80 or
                    coolant_flow < 5.0 or
                    fan_rpm < 900 or
                    dc_ripple > 9.0
                )
                if breach and rng.random()<0.08:
                    failure_flag = True
                    failure_time = ts
                    # Determine dominant fault
                    if connector_temp>90: fault_type = "connector_overheat"
                    elif coolant_flow<5.0: fault_type = "coolant_pump_degradation"
                    elif fan_rpm<900: fault_type = "fan_degradation"
                    elif dc_ripple>9.0: fault_type = "dc_cap_aging"
                    else: fault_type = "cabinet_overheat"

            failure = 1 if failure_flag else 0

            rows.append(dict(
                timestamp=ts, evse_id=ev_id,
                session_active=int(session_active), session_power_kw=round(power_kw,2),
                output_voltage_v=round(out_v,1), output_current_a=round(out_a,1),
                energy_kwh_24h=round(energy_24h*1000,2), uptime_pct_24h=round(uptime_24h*100,2),
                ambient_temp_c=round(ambient,2), cabinet_temp_c=round(cabinet_temp,2),
                connector_temp_c=round(connector_temp,2), coolant_temp_c=round(coolant_temp,2),
                coolant_flow_lpm=round(coolant_flow,2), fan_rpm=round(fan_rpm,0),
                grid_thd_pct=round(thd,2), grid_imbalance_pct=round(imb,2),
                dc_bus_voltage_v=round(dc_bus_v,1), dc_bus_ripple_v=round(dc_ripple,2),
                contactor_cycles=contactor_cycles, plug_insertions=plug_insertions,
                ocpp_error_count_24h=ocpp_errors_24h, failure=failure,
                fault_type=fault_type if failure else ""
            ))

            # After failure, simulate downtime + partial recovery, then continue
            if failure_flag and rng.random()<0.04:
                # 1-6 hours outage
                outage_ticks = int(rng.integers(12, 72))
                for i in range(outage_ticks):
                    ts2 = ts + pd.Timedelta(minutes=(i+1)*5)
                    if ts2 >= end: break
                    rows.append(dict(
                        timestamp=ts2, evse_id=ev_id,
                        session_active=0, session_power_kw=0.0,
                        output_voltage_v=0.0, output_current_a=0.0,
                        energy_kwh_24h=round(energy_24h*1000,2), uptime_pct_24h=round(max(0.0, uptime_24h-0.4),2),
                        ambient_temp_c=round(ambient,2), cabinet_temp_c=round(max(ambient-1, 15),2),
                        connector_temp_c=round(ambient,2), coolant_temp_c=round(ambient-1,2),
                        coolant_flow_lpm=round(max(0.0, coolant_flow-1.0),2), fan_rpm=round(max(0, fan_rpm-200),0),
                        grid_thd_pct=round(thd,2), grid_imbalance_pct=round(imb,2),
                        dc_bus_voltage_v=0.0, dc_bus_ripple_v=round(dc_ripple+0.2,2),
                        contactor_cycles=contactor_cycles, plug_insertions=plug_insertions,
                        ocpp_error_count_24h=ocpp_errors_24h+1, failure=1, fault_type=fault_type
                    ))
                failure_flag = False  # return to service
                # Reset some states post-maintenance, but keep aging accumulated
                coolant_flow = max(6.5, coolant_flow + 2.0)
                fan_rpm = max(fan_rpm, 1600)
                dc_ripple = max(2.5, dc_ripple - 1.0)
                therm_margin = max(6.0, therm_margin + 2.0)

    df = pd.DataFrame(rows).sort_values(["evse_id","timestamp"]).reset_index(drop=True)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sim.yaml")
    ap.add_argument("--out", default="data/sim.csv")
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--evse", type=int, default=None)
    ap.add_argument("--freq_min", type=int, default=None)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.days is not None: config["days"] = args.days
    if args.evse is not None: config["evse"] = args.evse
    if args.freq_min is not None: config["freq_min"] = args.freq_min

    df = simulate(config)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows to {args.out}")

if __name__ == "__main__":
    main()
