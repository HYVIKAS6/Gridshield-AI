# GridShield AI — Dataset Documentation & Provenance 📁

**Document Version:** 2.0  
**Status:** IMPLEMENTED AND TESTED  

---

## 1. Supported Telemetry Feeds

GridShield AI supports three distinct data ingestion modalities:

### 1. Offline Academic Sample Dataset (Included)
- **File Location:** `data/sample/smartgrid_telemetry_sample.csv`
- **Total Records:** 1,200 hourly telemetry intervals.
- **Attributes:** Timestamp, Voltage ($V$), Current ($A$), Active Power Load ($kW$), Frequency ($Hz$), Attack Target Flag ($0/1$), Attack Type.
- **Provenance & Integrity:** Built directly for zero-internet college laboratory demonstrations. Verified physically consistent under $P = \frac{V \cdot I \cdot 0.95}{1000}$.

### 2. Authentic Research Datasets (Ingestion Adapters)
- **SmartGrid CyberPhysical Attack Dataset:**
  - **DOI:** `10.21227/symr-bz19`
  - **Description:** Industrial laboratory PMU and protective relay telemetry capturing coordinated FDI attacks and line trips.
  - **Column Normalization:** Ingests `V1`, `I1`, `Freq`, `P`, and status `Label`.
- **Data4Cyber Dataset:**
  - **Zenodo Repository:** `20004612`
  - **Description:** SCADA operational technology network events and power measurement logs.
  - **Column Normalization:** Ingests `sensor_v`, `sensor_i`, `sensor_p`, `sensor_f`.
- **Power System Intrusion Dataset:**
  - **Description:** Multi-relay microgrid PMU recordings.
  - **Column Normalization:** Automatically maps `R1-PA1:VH` (Phase A Voltage), `R1-PA4:IH` (Phase A Current), and `R1:F` (Frequency).

### 3. Synthetic Dynamic Telemetry Generator
- **Generator Function:** `generate_synthetic_grid_data(n_samples, attack_prob, random_seed)`
- **Behavior:** Produces diurnal current and active power curves with Gaussian sensor noise, allowing customized attack frequency testing ($0 - 30\%$).

---

## 2. Telemetry Schema & Units

| Feature Column | Physical Measurement | Engineering Unit | Nominal Operational Range |
| :--- | :--- | :---: | :---: |
| `timestamp` | Sampling Timestamp | Datetime ISO-8601 | Hourly / 15-min intervals |
| `voltage` | RMS Single-Phase Bus Voltage | Volts ($V$) | $230.0\text{ V} \pm 5\%$ ($218.5 - 241.5\text{ V}$) |
| `current` | Feeder Line Current | Amperes ($A$) | $8.0 - 15.0\text{ A}$ |
| `load` | Active Power Consumption | Kilowatts ($kW$) | $P = \frac{V \cdot I \cdot \cos\theta}{1000}$ |
| `frequency` | AC Power System Frequency | Hertz ($Hz$) | $50.0\text{ Hz} \pm 0.2\text{ Hz}$ ($49.8 - 50.2\text{ Hz}$) |
| `target` | Ground Truth Attack Label | Binary ($0/1$) | $0 = \text{Normal}, 1 = \text{Cyberattack}$ |
| `attack_type` | Injected Vector Classification | Categorical String | `voltage_spike`, `load_drift`, etc. |
