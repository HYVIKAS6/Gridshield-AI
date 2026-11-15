"""
Cyber-Physical Anomaly Detection Engine for GridShield AI.
Implements single-phase/three-phase electrical physical consistency checks,
Rate-of-Change-of-Frequency (RoCoF), voltage stability corridors,
and sensor telemetry integrity diagnostics.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


# Operational Grid Safety Thresholds
NOMINAL_VOLTAGE_V = 230.0  # RMS single-phase nominal
VOLTAGE_LOWER_PU = 0.90    # 0.90 p.u. (207 V) - emergency lower bound
VOLTAGE_UPPER_PU = 1.10    # 1.10 p.u. (253 V) - emergency upper bound
NOMINAL_FREQ_HZ = 50.0     # Nominal grid frequency (50 Hz or 60 Hz)
FREQ_LOWER_HZ = 49.5       # Operational statutory limit
FREQ_UPPER_HZ = 50.5       # Operational statutory limit
MAX_ROCOF_HZ_PER_S = 0.5   # Maximum allowable RoCoF (Hz/s)
MAX_PHYSICAL_POWER_RESIDUAL = 0.15  # 15% tolerance on P = V * I / 1000 due to reactive pf


def check_physical_power_consistency(
    voltage: Union[float, np.ndarray, pd.Series],
    current: Union[float, np.ndarray, pd.Series],
    load_kw: Union[float, np.ndarray, pd.Series],
    power_factor: float = 0.95,
    tolerance: float = MAX_PHYSICAL_POWER_RESIDUAL,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Checks electrical power consistency:
    Expected Active Power P (kW) = (V (V) * I (A) * cos(phi)) / 1000.
    Returns:
        (is_violation, residual_ratio)
        where residual_ratio = |P_measured - P_expected| / (P_expected + 1e-4)
    """
    v = np.asarray(voltage, dtype=float)
    i = np.asarray(current, dtype=float)
    p_meas = np.asarray(load_kw, dtype=float)

    p_expected = (v * i * power_factor) / 1000.0
    # Avoid zero division
    denom = np.maximum(np.abs(p_expected), 0.05)
    residual_ratio = np.abs(p_meas - p_expected) / denom
    is_violation = (residual_ratio > tolerance).astype(int)

    return is_violation, residual_ratio


def compute_rocof(
    frequency: Union[pd.Series, np.ndarray],
    time_delta_seconds: float = 1.0,
) -> np.ndarray:
    """
    Computes Rate-of-Change-of-Frequency (RoCoF = |df/dt| in Hz/s).
    """
    f = np.asarray(frequency, dtype=float)
    if len(f) <= 1:
        return np.zeros_like(f)

    # Gradient or first-difference
    df = np.diff(f, prepend=f[0])
    dt = max(float(time_delta_seconds), 1e-3)
    rocof = np.abs(df / dt)
    return rocof


def evaluate_grid_physics(
    df: pd.DataFrame,
    base_voltage: float = NOMINAL_VOLTAGE_V,
    nominal_freq: float = NOMINAL_FREQ_HZ,
    power_factor: float = 0.95,
) -> pd.DataFrame:
    """
    Performs comprehensive cyber-physical rule validation on grid telemetry.
    Appends audit diagnostic columns:
      - p_expected_kw: Theoretical power from V & I
      - power_residual_ratio: Discrepancy ratio
      - physics_power_violation: 1 if physical law violated
      - voltage_pu: Voltage in per-unit
      - voltage_corridor_violation: 1 if outside [0.90, 1.10] p.u.
      - rocof_hz_s: Rate of change of frequency
      - freq_corridor_violation: 1 if outside statutory band
      - rocof_violation: 1 if |df/dt| > 0.5 Hz/s
      - sensor_flatline_violation: 1 if sensor repeats identical value >= 5 steps
      - cyber_physical_anomaly_score: Aggregate physical risk score [0, 1]
    """
    res = df.copy()

    # 1. Power Physical Consistency Check (P = V * I * pf / 1000)
    p_exp = (res["voltage"] * res["current"] * power_factor) / 1000.0
    res["p_expected_kw"] = p_exp
    p_viol, p_res = check_physical_power_consistency(
        res["voltage"], res["current"], res["load"], power_factor=power_factor
    )
    res["power_residual_ratio"] = np.round(p_res, 4)
    res["physics_power_violation"] = p_viol

    # 2. Voltage Corridor Check
    v_pu = res["voltage"] / base_voltage
    res["voltage_pu"] = np.round(v_pu, 4)
    res["voltage_corridor_violation"] = (
        (v_pu < VOLTAGE_LOWER_PU) | (v_pu > VOLTAGE_UPPER_PU)
    ).astype(int)

    # 3. Frequency & RoCoF
    res["rocof_hz_s"] = np.round(compute_rocof(res["frequency"], 1.0), 4)
    res["freq_corridor_violation"] = (
        (res["frequency"] < FREQ_LOWER_HZ) | (res["frequency"] > FREQ_UPPER_HZ)
    ).astype(int)
    res["rocof_violation"] = (res["rocof_hz_s"] > MAX_ROCOF_HZ_PER_S).astype(int)

    # 4. Sensor Flatline / Telemetry Dropout Check
    # Check if voltage or frequency is zero or exactly unchanged for >= 4 consecutive samples
    v_diff = res["voltage"].diff().abs()
    f_diff = res["frequency"].diff().abs()
    dropout_zero = (res["voltage"] <= 1.0) | (res["load"] < 0)
    flatline = ((v_diff == 0) & (f_diff == 0)).astype(int)
    res["sensor_flatline_violation"] = (dropout_zero | flatline).astype(int)

    # 5. Composite Cyber-Physical Anomaly Score (weighted sum normalized to 0-1)
    weights = {
        "physics_power_violation": 0.35,
        "voltage_corridor_violation": 0.25,
        "freq_corridor_violation": 0.20,
        "rocof_violation": 0.10,
        "sensor_flatline_violation": 0.10,
    }
    score = np.zeros(len(res))
    for col, w in weights.items():
        score += res[col].values * w

    res["cyber_physical_anomaly_score"] = np.clip(np.round(score, 3), 0.0, 1.0)
    res["cp_flagged_anomaly"] = (res["cyber_physical_anomaly_score"] >= 0.35).astype(int)

    return res


def get_cyber_physical_summary(df_audited: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes summary telemetry and cyber-physical health statistics.
    """
    total = len(df_audited)
    if total == 0:
        return {}

    p_viols = int(df_audited["physics_power_violation"].sum())
    v_viols = int(df_audited["voltage_corridor_violation"].sum())
    f_viols = int(df_audited["freq_corridor_violation"].sum())
    rocof_viols = int(df_audited["rocof_violation"].sum())
    flatline_viols = int(df_audited["sensor_flatline_violation"].sum())
    total_cp_alerts = int(df_audited["cp_flagged_anomaly"].sum())

    health_pct = max(0.0, 100.0 - (total_cp_alerts / total * 100.0))

    return {
        "total_samples": total,
        "grid_health_index": round(health_pct, 1),
        "total_cyber_physical_alerts": total_cp_alerts,
        "power_inconsistency_count": p_viols,
        "voltage_corridor_breaches": v_viols,
        "frequency_deviations": f_viols,
        "rocof_critical_events": rocof_viols,
        "sensor_flatlines": flatline_viols,
        "avg_power_residual": round(float(df_audited["power_residual_ratio"].mean()), 4),
        "max_rocof_hz_s": round(float(df_audited["rocof_hz_s"].max()), 4),
    }
