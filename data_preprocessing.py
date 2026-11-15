"""
Data Preprocessing, Synthetic Telemetry Generation, and Leak-Free Feature Engineering
for GridShield AI Smart Grid Cyberattack Simulator.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


ATTACK_METADATA: Dict[str, Dict[str, Any]] = {
    "voltage_spike": {
        "name": "Voltage Spike / Surge",
        "description": "Instantaneous voltage spike or transient overvoltage on substation bus.",
        "affected_sensors": ["voltage"],
        "default_magnitude": 12.0,
        "param_description": "Magnitude added to nominal voltage in Volts (e.g. +6V to +18V).",
    },
    "frequency_shift": {
        "name": "Frequency Sag / Shift",
        "description": "Sudden frequency deviation simulating generator trip or spoofed PMU frequency telemetry.",
        "affected_sensors": ["frequency"],
        "default_magnitude": 0.8,
        "param_description": "Absolute frequency offset in Hz applied in positive or negative direction.",
    },
    "load_drift": {
        "name": "Load Inconsistency / Drift",
        "description": "Uncoordinated step scaling on active power reading breaking V-I physical relationship.",
        "affected_sensors": ["load"],
        "default_magnitude": 1.4,
        "param_description": "Multiplicative factor applied to active load (kW) (e.g. 1.25x - 1.6x).",
    },
    "stealth_fdia": {
        "name": "Stealth Coordinated FDIA",
        "description": "Coordinated false data injection across current and load preserving P = V*I/1000 with noise.",
        "affected_sensors": ["current", "load", "voltage"],
        "default_magnitude": 1.25,
        "param_description": "Scaling multiplier for current and load with perturbed voltage.",
    },
    "telemetry_dropout": {
        "name": "Telemetry Dropout / Loss",
        "description": "Simulates sensor packet loss or communication blackout. Sensor flatlines to 0 or last held value.",
        "affected_sensors": ["voltage", "current", "load", "frequency", "all"],
        "default_magnitude": 0.0,
        "param_description": "Target dropped value (e.g., 0.0 for zero-out or -1.0 for last held reading flatline).",
    },
    "ramping_fdi": {
        "name": "Ramping / Slow-Drift FDI",
        "description": "Gradual linear or progressive drift from baseline to peak magnitude over attack duration to evade threshold alarms.",
        "affected_sensors": ["voltage", "frequency", "load", "current"],
        "default_magnitude": 15.0,
        "param_description": "Final peak perturbation delta or multiplier reached at the end of the ramping sequence.",
    },
}


def generate_synthetic_grid_data(
    n_samples: int = 1000,
    attack_prob: float = 0.1,
    random_seed: Optional[int] = 42,
    base_voltage: float = 230.0,
    nominal_frequency: float = 50.0,
) -> pd.DataFrame:
    """
    Generates synthetic smart grid telemetry (Voltage, Current, Load, Frequency)
    with realistic physical dependencies: Load (kW) = Voltage (V) * Current (A) / 1000.
    
    If attack_prob > 0, initial attacks are introduced via inject_attacks_into_df.
    Dynamic and reproducible via explicit random_seed.
    """
    rng = np.random.default_rng(random_seed)
    timestamps = pd.date_range(start="2024-01-01", periods=n_samples, freq="h")

    # Normal baseline telemetry
    voltage = base_voltage + rng.normal(0, 0.5, n_samples)
    current = 10.0 + 2.0 * np.sin(np.linspace(0, 4 * np.pi, n_samples)) + rng.normal(0, 0.2, n_samples)
    current = np.clip(current, 0.1, None)  # Current should remain positive in standard load
    load = (voltage * current) / 1000.0  # Physically consistent active power (kW)
    frequency = nominal_frequency + rng.normal(0, 0.02, n_samples)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "voltage": voltage,
        "current": current,
        "load": load,
        "frequency": frequency,
        "target": 0,  # 0 for Normal
        "attack_type": "none",
    })

    if attack_prob > 0 and n_samples > 0:
        n_attacks = int(round(n_samples * attack_prob))
        if n_attacks > 0:
            attack_indices = rng.choice(n_samples, size=n_attacks, replace=False)
            df = inject_attack_at_indices(df, attack_indices, rng=rng)

    return df


def inject_attack_at_indices(
    df: pd.DataFrame,
    indices: Union[List[int], np.ndarray],
    attack_type: Optional[str] = None,
    magnitude: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    target_sensor: Optional[str] = None,
    dropout_mode: str = "zero",
    ramp_shape: str = "linear",
) -> pd.DataFrame:
    """
    Modifies telemetry in df at specific indices to simulate cyberattack scenarios.
    Ensures telemetry is actually manipulated and flagged in target and attack_type.
    
    Supported attack types:
    - 'voltage_spike': Instantaneous surge added to voltage.
    - 'frequency_shift': Sudden frequency step (sag or swell).
    - 'load_drift': Multiplicative drift on active power reading.
    - 'stealth_fdia': Coordinated current and load manipulation preserving P = V*I/1000.
    - 'telemetry_dropout': Simulates missing sensor readings or communication interruption.
                          target_sensor: 'voltage', 'current', 'load', 'frequency', or 'all'.
                          dropout_mode: 'zero' (drops to 0.0) or 'flatline' (holds previous value / constant).
    - 'ramping_fdi' (or 'slow_drift'): Progressively ramps sensor tampering over the sequence
                          of indices to evade instantaneous delta checks.
    """
    if rng is None:
        rng = np.random.default_rng()

    df = df.copy()
    valid_attack_types = [
        "voltage_spike",
        "frequency_shift",
        "load_drift",
        "stealth_fdia",
        "telemetry_dropout",
        "ramping_fdi",
        "slow_drift",
    ]

    # Normalize slow_drift to ramping_fdi
    canonical_type = None
    if attack_type is not None:
        canonical_type = "ramping_fdi" if attack_type == "slow_drift" else attack_type

    # Filter indices to valid dataframe indices and preserve chronological order for ramping
    valid_indices = [idx for idx in indices if idx in df.index]
    if not valid_indices:
        return df

    # If attack_type is fixed or to be sampled globally
    fixed_type = canonical_type if canonical_type in valid_attack_types else None
    
    num_pts = len(valid_indices)

    # For ramping attacks, compute ramp progression across the sequence of indices
    if fixed_type == "ramping_fdi":
        sensor = target_sensor if target_sensor in ["voltage", "frequency", "load", "current"] else "voltage"
        peak_mag = magnitude if magnitude is not None else (
            float(rng.uniform(10.0, 25.0)) if sensor == "voltage"
            else float(rng.uniform(0.5, 1.5)) if sensor == "frequency"
            else float(rng.uniform(1.3, 1.8))
        )
        if num_pts == 1:
            ramp_factors = np.array([1.0])
        else:
            steps = np.linspace(1.0 / num_pts, 1.0, num_pts)
            if ramp_shape == "progressive":
                ramp_factors = steps ** 2
            else:  # linear
                ramp_factors = steps

        for step_i, idx in enumerate(valid_indices):
            factor = ramp_factors[step_i]
            if sensor == "voltage":
                df.loc[idx, "voltage"] += peak_mag * factor
            elif sensor == "frequency":
                df.loc[idx, "frequency"] += peak_mag * factor
            elif sensor == "current":
                df.loc[idx, "current"] *= (1.0 + (peak_mag - 1.0) * factor if peak_mag > 1.0 else 1.0 + peak_mag * factor)
                df.loc[idx, "load"] = (df.loc[idx, "voltage"] * df.loc[idx, "current"]) / 1000.0
            elif sensor == "load":
                df.loc[idx, "load"] *= (1.0 + (peak_mag - 1.0) * factor if peak_mag > 1.0 else 1.0 + peak_mag * factor)

            df.loc[idx, "attack_type"] = "ramping_fdi"
            df.loc[idx, "target"] = 1
        return df

    for idx in valid_indices:
        selected_type = fixed_type if fixed_type is not None else rng.choice([
            "voltage_spike", "frequency_shift", "load_drift", "stealth_fdia", "telemetry_dropout", "ramping_fdi"
        ])

        if selected_type == "voltage_spike":
            delta_v = magnitude if magnitude is not None else float(rng.uniform(6.0, 18.0))
            df.loc[idx, "voltage"] += delta_v
            df.loc[idx, "attack_type"] = "voltage_spike"

        elif selected_type == "frequency_shift":
            shift = magnitude if magnitude is not None else float(rng.uniform(0.4, 1.2))
            direction = rng.choice([-1.0, 1.0])
            df.loc[idx, "frequency"] += direction * shift
            df.loc[idx, "attack_type"] = "frequency_shift"

        elif selected_type == "load_drift":
            factor = magnitude if magnitude is not None else float(rng.uniform(1.25, 1.6))
            df.loc[idx, "load"] *= factor
            df.loc[idx, "attack_type"] = "load_drift"

        elif selected_type == "stealth_fdia":
            scaling = magnitude if magnitude is not None else float(rng.uniform(1.15, 1.35))
            df.loc[idx, "current"] *= scaling
            df.loc[idx, "load"] = (df.loc[idx, "voltage"] * df.loc[idx, "current"]) / 1000.0
            df.loc[idx, "voltage"] += float(rng.normal(0, 0.8))
            df.loc[idx, "attack_type"] = "stealth_fdia"

        elif selected_type == "telemetry_dropout":
            sensor = target_sensor if target_sensor in ["voltage", "current", "load", "frequency", "all"] else "all"
            target_cols = ["voltage", "current", "load", "frequency"] if sensor == "all" else [sensor]
            
            for col in target_cols:
                if dropout_mode == "flatline":
                    # Flatline: use magnitude if specified, or previous valid observation if available
                    if magnitude is not None and magnitude != 0.0:
                        df.loc[idx, col] = magnitude
                    else:
                        prev_loc = df.index.get_loc(idx) - 1
                        if prev_loc >= 0:
                            df.loc[idx, col] = float(df.iloc[prev_loc][col])
                        else:
                            df.loc[idx, col] = 0.0
                else:
                    # Default 'zero' drop mode
                    val = magnitude if magnitude is not None else 0.0
                    df.loc[idx, col] = val

            df.loc[idx, "attack_type"] = "telemetry_dropout"

        elif selected_type in ["ramping_fdi", "slow_drift"]:
            sensor = target_sensor if target_sensor in ["voltage", "frequency", "load", "current"] else "voltage"
            drift_val = magnitude if magnitude is not None else 10.0
            if sensor == "voltage":
                df.loc[idx, "voltage"] += drift_val
            elif sensor == "frequency":
                df.loc[idx, "frequency"] += drift_val
            elif sensor == "current":
                df.loc[idx, "current"] *= drift_val
            elif sensor == "load":
                df.loc[idx, "load"] *= drift_val
            df.loc[idx, "attack_type"] = "ramping_fdi"

        df.loc[idx, "target"] = 1

    return df


def inject_dynamic_attack_scenario(
    df: pd.DataFrame,
    attack_type: str = "voltage_spike",
    start_idx: Optional[int] = None,
    duration: int = 15,
    magnitude: Optional[float] = None,
    random_seed: Optional[int] = None,
    target_sensor: Optional[str] = None,
    dropout_mode: str = "zero",
    ramp_shape: str = "linear",
) -> Tuple[pd.DataFrame, List[int]]:
    """
    Interactive attack injection for real-time operator simulation.
    Applies the chosen attack type over a continuous or targeted duration.
    Returns the updated DataFrame and the list of modified indices.
    """
    rng = np.random.default_rng(random_seed)
    n = len(df)
    if n == 0:
        return df.copy(), []

    duration = max(1, min(duration, n))
    if start_idx is None or start_idx < 0 or start_idx >= n:
        # Default to the most recent window (or safe offset)
        start_idx = max(0, n - duration)

    end_idx = min(n, start_idx + duration)
    indices = list(range(start_idx, end_idx))

    updated_df = inject_attack_at_indices(
        df,
        indices=indices,
        attack_type=attack_type,
        magnitude=magnitude,
        rng=rng,
        target_sensor=target_sensor,
        dropout_mode=dropout_mode,
        ramp_shape=ramp_shape,
    )
    return updated_df, indices


class LeakFreeGridPipeline:
    """
    Leak-free feature engineering and scaling pipeline.
    Ensures that transformations, rolling statistics, and scalers are strictly
    fitted on training data without leaking test set future information.
    """

    FEATURE_COLS = ["voltage", "current", "load", "frequency", "voltage_roll_mean", "load_lag1"]

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.train_history_voltage: List[float] = []
        self.train_history_load: List[float] = []

    def fit(self, train_df: pd.DataFrame) -> "LeakFreeGridPipeline":
        """
        Fits the scaler on feature-engineered training data.
        """
        train_feat_df = self._compute_features(train_df, is_train=True)
        X_train = train_feat_df[self.FEATURE_COLS].values
        self.scaler.fit(X_train)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforms any incoming partition using the fitted scaler and historical context.
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted on training data before calling transform.")
        feat_df = self._compute_features(df, is_train=False)
        X = feat_df[self.FEATURE_COLS].values
        return self.scaler.transform(X)

    def fit_transform(self, train_df: pd.DataFrame) -> np.ndarray:
        return self.fit(train_df).transform(train_df)

    def _compute_features(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        Calculates rolling and lag features without backward leakage.
        For test sets, context from training boundaries is honored.
        """
        res = df.copy()

        # Handle any residual NaNs or dropouts cleanly before computing features
        for col in ["voltage", "current", "load", "frequency"]:
            if col in res.columns:
                res[col] = pd.to_numeric(res[col], errors="coerce").ffill().bfill().fillna(0.0)

        if is_train:
            # Fit context from train tail
            res["voltage_roll_mean"] = res["voltage"].rolling(window=self.window_size, min_periods=1).mean()
            res["load_lag1"] = res["load"].shift(1)
            # Impute the very first step with the initial observation to avoid leakage
            res["load_lag1"] = res["load_lag1"].fillna(res["load"].iloc[0] if len(res) > 0 else 0.0)

            # Save boundary context for consecutive test sets
            if len(res) >= self.window_size:
                self.train_history_voltage = res["voltage"].iloc[-self.window_size:].tolist()
            else:
                self.train_history_voltage = res["voltage"].tolist()
            self.train_history_load = [res["load"].iloc[-1]] if len(res) > 0 else [0.0]

        else:
            # In test mode, prepend boundary history from train partition to compute seamless rolling stats
            if self.train_history_voltage:
                history_v = pd.Series(self.train_history_voltage)
                combined_v = pd.concat([history_v, res["voltage"]], ignore_index=True)
                roll_v = combined_v.rolling(window=self.window_size, min_periods=1).mean()
                res["voltage_roll_mean"] = roll_v.iloc[len(history_v):].values
            else:
                res["voltage_roll_mean"] = res["voltage"].rolling(window=self.window_size, min_periods=1).mean()

            if self.train_history_load:
                last_load = self.train_history_load[-1]
                lag_s = res["load"].shift(1)
                lag_s.iloc[0] = last_load
                res["load_lag1"] = lag_s
            else:
                res["load_lag1"] = res["load"].shift(1).fillna(res["load"].iloc[0] if len(res) > 0 else 0.0)

        return res


def chronological_train_test_split(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs strict chronological time-series splitting.
    Preserves sequential order and avoids temporal lookahead leakage.
    """
    if df.empty:
        raise ValueError("Cannot split an empty DataFrame.")

    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be strictly between 0 and 1.")

    split_idx = int(len(df) * (1.0 - test_ratio))
    split_idx = max(1, min(split_idx, len(df) - 1))

    # Sort by timestamp if available to strictly guarantee chronological progression
    if "timestamp" in df.columns:
        df_sorted = df.sort_values(by="timestamp").reset_index(drop=True)
    else:
        df_sorted = df.reset_index(drop=True)

    train_df = df_sorted.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df_sorted.iloc[split_idx:].copy().reset_index(drop=True)

    return train_df, test_df


def validate_grid_dataframe(df: pd.DataFrame) -> Tuple[bool, str, pd.DataFrame]:
    """
    Validates uploaded or generated telemetry DataFrame for required columns,
    missing values, and basic physical realism.
    Returns (is_valid, message, cleaned_df).
    """
    if df is None or df.empty:
        return False, "Provided dataset is empty.", pd.DataFrame()

    df_clean = df.copy()

    # Power System Intrusion Dataset mapping
    if "marker" in df_clean.columns and "voltage" not in df_clean.columns:
        if "R1-PA1:VH" in df_clean.columns and "R1-PA4:IH" in df_clean.columns and "R1:F" in df_clean.columns:
            df_clean["voltage"] = pd.to_numeric(df_clean["R1-PA1:VH"], errors="coerce")
            df_clean["current"] = pd.to_numeric(df_clean["R1-PA4:IH"], errors="coerce")
            df_clean["load"] = (df_clean["voltage"] * df_clean["current"]) / 1000.0
            df_clean["frequency"] = pd.to_numeric(df_clean["R1:F"], errors="coerce")
            df_clean["target"] = df_clean["marker"].astype(str).map(lambda x: 1 if "Attack" in x else 0)
        else:
            return False, "Power System Intrusion dataset missing required PMU columns (R1-PA1:VH, R1-PA4:IH, R1:F).", pd.DataFrame()

    # SmartGrid CyberPhysical Attack Dataset (DOI: 10.21227/symr-bz19) Adapter
    # Typically contains PMU/Relay columns: V1, I1, Freq, P, Label
    col_lower_map = {c.lower(): c for c in df_clean.columns}
    if "voltage" not in df_clean.columns:
        v_cand = [c for c in df_clean.columns if c.lower() in ["v1", "voltage_v", "volt", "v_mag", "vm_pu", "v_pmu"]]
        i_cand = [c for c in df_clean.columns if c.lower() in ["i1", "current_a", "curr", "i_mag", "im_pu", "i_pmu"]]
        f_cand = [c for c in df_clean.columns if c.lower() in ["freq", "frequency_hz", "frequency", "f_hz"]]
        p_cand = [c for c in df_clean.columns if c.lower() in ["p", "power", "load", "active_power", "load_kw", "p_mw"]]

        if v_cand and i_cand and f_cand:
            df_clean["voltage"] = pd.to_numeric(df_clean[v_cand[0]], errors="coerce")
            df_clean["current"] = pd.to_numeric(df_clean[i_cand[0]], errors="coerce")
            df_clean["frequency"] = pd.to_numeric(df_clean[f_cand[0]], errors="coerce")
            if p_cand:
                df_clean["load"] = pd.to_numeric(df_clean[p_cand[0]], errors="coerce")
            else:
                df_clean["load"] = (df_clean["voltage"] * df_clean["current"]) / 1000.0

            # Label mapping if present
            lbl_cand = [c for c in df_clean.columns if c.lower() in ["label", "attack", "class", "is_attack", "status"]]
            if lbl_cand and "target" not in df_clean.columns:
                df_clean["target"] = pd.to_numeric(df_clean[lbl_cand[0]], errors="coerce").fillna(0).astype(int)

    # Data4Cyber (Zenodo: 20004612) Adapter
    # Typically contains OT sensor telemetry and cyber flags
    if "sensor_v" in col_lower_map and "voltage" not in df_clean.columns:
        df_clean["voltage"] = pd.to_numeric(df_clean[col_lower_map["sensor_v"]], errors="coerce")
        if "sensor_i" in col_lower_map:
            df_clean["current"] = pd.to_numeric(df_clean[col_lower_map["sensor_i"]], errors="coerce")
        if "sensor_p" in col_lower_map:
            df_clean["load"] = pd.to_numeric(df_clean[col_lower_map["sensor_p"]], errors="coerce")
        else:
            df_clean["load"] = (df_clean["voltage"] * df_clean["current"]) / 1000.0
        if "sensor_f" in col_lower_map:
            df_clean["frequency"] = pd.to_numeric(df_clean[col_lower_map["sensor_f"]], errors="coerce")

    required_cols = ["voltage", "current", "load", "frequency"]
    missing = [c for c in required_cols if c not in df_clean.columns]
    if missing:
        return False, f"Missing required telemetry columns: {missing}", pd.DataFrame()

    # Enforce numeric conversion
    for c in required_cols:
        df_clean[c] = pd.to_numeric(df_clean[c], errors="coerce")

    # Drop any rows with NaN in primary sensors
    initial_len = len(df_clean)
    df_clean = df_clean.dropna(subset=required_cols).reset_index(drop=True)
    if df_clean.empty:
        return False, "All rows contained invalid non-numeric or missing sensor data.", pd.DataFrame()

    # Target column defaults to 0 if missing
    if "target" not in df_clean.columns:
        df_clean["target"] = 0
    else:
        df_clean["target"] = pd.to_numeric(df_clean["target"], errors="coerce").fillna(0).astype(int)

    if "timestamp" not in df_clean.columns:
        df_clean["timestamp"] = pd.date_range(start="2024-01-01", periods=len(df_clean), freq="h")
    else:
        df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"], errors="coerce")
        if df_clean["timestamp"].isna().any():
            df_clean["timestamp"] = pd.date_range(start="2024-01-01", periods=len(df_clean), freq="h")

    if "attack_type" not in df_clean.columns:
        df_clean["attack_type"] = df_clean["target"].map(lambda t: "attack" if t == 1 else "none")

    dropped_count = initial_len - len(df_clean)
    msg = "Dataset validated successfully."
    if dropped_count > 0:
        msg += f" (Dropped {dropped_count} rows with NaN values)"

    return True, msg, df_clean


# Legacy compatibility layer for existing imports
def preprocess_data(df: pd.DataFrame):
    """
    Backwards-compatible wrapper. Fits pipeline on the provided dataframe.
    Note: Direct use is deprecated in favor of LeakFreeGridPipeline to avoid leakage.
    """
    pipeline = LeakFreeGridPipeline()
    X_scaled = pipeline.fit_transform(df)
    y = df["target"].values
    features = LeakFreeGridPipeline.FEATURE_COLS
    return X_scaled, y, features, pipeline.scaler


def prepare_lstm_data(X_scaled, y, window_size=10):
    """
    Reshapes data for sequential models (samples, time_steps, features).
    """
    X_seq, y_seq = [], []
    y_arr = y.values if hasattr(y, "values") else np.array(y)
    for i in range(window_size, len(X_scaled)):
        X_seq.append(X_scaled[i - window_size:i])
        y_seq.append(y_arr[i])
    return np.array(X_seq), np.array(y_seq)
