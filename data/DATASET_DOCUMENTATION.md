# GridShield AI — Research Datasets Documentation

## 1. Overview of Datasets

GridShield AI supports both authentic empirical smart grid cyber-physical research datasets and synthetic high-fidelity simulation datasets.

---

## 2. Mississippi State University (MSU) & Oak Ridge National Laboratory (ORNL) Dataset

- **Original Title:** Industrial Control System (ICS) Cyber Attack Dataset / Power System Attack Dataset
- **Authors:** Dr. Tommy Morris, Dr. Uttam Adhikari, Dr. Shengyi Pan
- **Institutions:** Mississippi State University (Center for Computer Security Research) & Oak Ridge National Laboratory
- **Domain:** Electric Power Transmission System, Phasor Measurement Units (PMUs), Protective Relays
- **Testbed Setup:**
  - 2-line, 3-bus and 4-bus power transmission system testbed
  - Microprocessor-based protective relays (SEL-421, SEL-451)
  - Synchrophasor measurements collected via IEEE C37.118 protocol
  - Snort Intrusion Detection System packet inspection logs
- **Classes:**
  - **Natural/Normal:** System load changes, line maintenance, generator trip, transmission fault with relay trip
  - **Attack:** False Data Injection (voltage/current spoofing), Remote Command Injection (unauthorized breaker trip), Relay Setting Modification
- **Key Features Ingested by GridShield AI:**
  - `voltage`: Phase A Voltage Phase Angle & Magnitude (`R1-PA1:VH`, `R1-PM1:V`)
  - `current`: Phase A Current Phase Angle & Magnitude (`R1-PA4:IH`, `R1-PM4:I`)
  - `load`: Calculated Active Power $P = V \cdot I / 1000$ (kW)
  - `frequency`: Grid Frequency ($R1:F$ in Hz, nominal 60.0 Hz)
  - `target`: Binary ground truth (0 = Normal/Natural, 1 = Cyber Attack)
- **Files Stored Locally:**
  - `data/raw/msu_power_attack_raw_sample.csv` (Sample raw records from scenario `data1.csv`)
  - `data/processed/msu_pmu_benchmark.csv` (2,600 verified observations, balanced classes)
- **Reproducible Download:**
  - Can be refreshed or extended via `python data/download_datasets.py`

---

## 3. Synthetic Smart Grid Telemetry (Domain Simulation)

- **Purpose:** Used for real-time classroom cyberattack injection, stress-testing, and offline demonstration.
- **Physics Modeling:**
  - Single-phase AC nominal voltage ($230\text{ V} \pm 0.5\text{ V}$)
  - Diurnal sinusoidal current loading ($10\text{ A} \pm 2\text{ A}$)
  - Statutory frequency ($50.0\text{ Hz} \pm 0.02\text{ Hz}$)
  - Physical Ohm-law active power constraint: $P = (V \cdot I) / 1000$
- **6 Cyberattack Simulation Vectors:**
  1. `voltage_spike`: Transient bus overvoltage ($+12\text{ V}$)
  2. `frequency_shift`: Sudden frequency excursion ($\pm 0.8\text{ Hz}$)
  3. `load_drift`: Active power scaling breaking physical $P = V \cdot I$ relation ($1.4\times$)
  4. `stealth_fdia`: Coordinated multi-sensor false data injection
  5. `telemetry_dropout`: Sensor flatline or packet blackout ($0.0\text{ V}$)
  6. `ramping_fdi`: Gradual slow-drift attack evading static thresholds
- **File Location:**
  - `data/sample/smartgrid_telemetry_sample.csv` (1,200 chronological records)

---

## 4. Academic Verification & Inspection

To open and inspect the datasets in Python or Excel:

```python
import pandas as pd

# 1. Load MSU Research Benchmark
df_msu = pd.read_csv("data/processed/msu_pmu_benchmark.csv")
print("MSU Research Shape:", df_msu.shape)
print("MSU Class Breakdown:\n", df_msu["target"].value_counts())

# 2. Load Synthetic Simulation Benchmark
df_syn = pd.read_csv("data/sample/smartgrid_telemetry_sample.csv")
print("Synthetic Shape:", df_syn.shape)
print("Attack Vectors:\n", df_syn["attack_type"].value_counts())
```
