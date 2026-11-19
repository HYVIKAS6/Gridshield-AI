"""
Download and Dataset Acquisition Utilities for GridShield AI.
Authentic research dataset ingestion from Mississippi State University (MSU) /
Oak Ridge National Laboratory (ORNL) Power System Attack Dataset and IEEE DataPort mirrors.
"""

import io
import os
import urllib.request
import logging
from typing import Optional, Tuple
import pandas as pd

logger = logging.getLogger(__name__)

# MSU / ORNL Industrial Control Systems Power System Benchmark
# Original Authors: Tommy Morris, Uttam Adhikari, Shengyi Pan (Mississippi State University & ORNL)
MSU_BENCHMARK_URL = "https://raw.githubusercontent.com/VictoKu1/IndustrialControlSystemCyberAttackDetectingCourse/master/Class/binaryAllNaturalPlusNormalVsAttacks/data1.csv"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
DATA_SAMPLE_DIR = os.path.join(PROJECT_ROOT, "data", "sample")


def download_msu_dataset(
    target_raw_file: Optional[str] = None,
    target_processed_file: Optional[str] = None,
    max_natural: int = 1500,
    max_attack: int = 1500,
) -> Tuple[bool, str, pd.DataFrame]:
    """
    Downloads and prepares the Mississippi State University / ORNL Power System Attack Dataset.
    Extracts PMU telemetry (voltage, current, frequency, active power) and binary attack labels.
    """
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

    if target_raw_file is None:
        target_raw_file = os.path.join(DATA_RAW_DIR, "msu_power_attack_raw_sample.csv")
    if target_processed_file is None:
        target_processed_file = os.path.join(DATA_PROCESSED_DIR, "msu_pmu_benchmark.csv")

    try:
        req = urllib.request.Request(MSU_BENCHMARK_URL, headers={"User-Agent": "GridShield-AI/2.0"})
        with urllib.request.urlopen(req, timeout=25) as resp:
            header = resp.readline().decode("utf-8")
            natural_rows = []
            attack_rows = []
            count = 0
            while True:
                line = resp.readline().decode("utf-8")
                if not line or count > 30000:
                    break
                count += 1
                if ",Natural" in line and len(natural_rows) < max_natural:
                    natural_rows.append(line)
                elif ",Attack" in line and len(attack_rows) < max_attack:
                    attack_rows.append(line)
                if len(natural_rows) >= max_natural and len(attack_rows) >= max_attack:
                    break

        all_lines = [header] + natural_rows + attack_rows
        with open(target_raw_file, "w", encoding="utf-8") as f:
            f.writelines(all_lines)

        df_raw = pd.read_csv(target_raw_file)
        df_proc = pd.DataFrame()
        df_proc["voltage"] = pd.to_numeric(df_raw["R1-PA1:VH"], errors="coerce")
        df_proc["current"] = pd.to_numeric(df_raw["R1-PA4:IH"], errors="coerce")
        df_proc["load"] = (df_proc["voltage"] * df_proc["current"]) / 1000.0
        df_proc["frequency"] = pd.to_numeric(df_raw["R1:F"], errors="coerce")
        df_proc["target"] = df_raw["marker"].astype(str).map(lambda x: 1 if "Attack" in x else 0)
        df_proc["attack_type"] = df_raw["marker"].astype(str).map(lambda x: "cyber_attack" if "Attack" in x else "natural_normal")
        df_proc["timestamp"] = pd.date_range(start="2024-01-01", periods=len(df_proc), freq="10s")
        df_proc = df_proc.dropna().reset_index(drop=True)

        df_proc.to_csv(target_processed_file, index=False)
        return True, f"Successfully acquired MSU dataset ({len(df_proc)} records).", df_proc

    except Exception as exc:
        logger.error(f"Failed to download MSU dataset: {exc}")
        return False, f"Download failed: {str(exc)}", pd.DataFrame()


def load_dataset(dataset_name: str = "msu_benchmark") -> Tuple[pd.DataFrame, str]:
    """
    Standard dataset loader for GridShield AI supporting:
    - 'synthetic_sample': Offline academic synthetic telemetry
    - 'msu_benchmark': Mississippi State University / ORNL Power System Attack Dataset
    - 'ieee_dataport_mock': Cleaned academic smart grid schema
    """
    if dataset_name == "synthetic_sample":
        sample_path = os.path.join(DATA_SAMPLE_DIR, "smartgrid_telemetry_sample.csv")
        if os.path.exists(sample_path):
            df = pd.read_csv(sample_path)
            return df, "Synthetic Smart Grid Telemetry (Simulated baseline + 6 attack vectors)"
    elif dataset_name == "msu_benchmark":
        proc_path = os.path.join(DATA_PROCESSED_DIR, "msu_pmu_benchmark.csv")
        if os.path.exists(proc_path):
            df = pd.read_csv(proc_path)
            return df, "Mississippi State University & ORNL Power System Attack Dataset (PMU Relay Logs)"
        else:
            success, msg, df = download_msu_dataset(target_processed_file=proc_path)
            if success:
                return df, "Mississippi State University & ORNL Power System Attack Dataset (PMU Relay Logs)"

    # Fallback to synthetic sample if available
    fallback_path = os.path.join(DATA_SAMPLE_DIR, "smartgrid_telemetry_sample.csv")
    if os.path.exists(fallback_path):
        return pd.read_csv(fallback_path), "Offline Academic Telemetry Sample"
    
    raise FileNotFoundError(f"Requested dataset '{dataset_name}' not available.")


if __name__ == "__main__":
    success, msg, df = download_msu_dataset()
    print("Download script result:", success, msg)
    if not df.empty:
        print("Data shape:", df.shape)
        print("Target breakdown:\n", df["target"].value_counts())
