"""
Multi-Bus Smart Grid Simulation Engine using Pandapower for GridShield AI.
Simulates IEEE 9-bus and IEEE 14-bus test systems, computes AC Newton-Raphson
power-flow, simulates bus-level False Data Injection and load manipulation attacks,
and returns bus voltage profiles and line loadings.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Safe Pandapower import
try:
    import pandapower as pp
    import pandapower.networks as pn
    import pandapower.topology as top
    HAS_PANDAPOWER = True
except ImportError:
    pp = None
    pn = None
    top = None
    HAS_PANDAPOWER = False


def get_available_networks() -> List[str]:
    """Returns list of supported standard IEEE bus test networks."""
    return ["IEEE 14-Bus System", "IEEE 9-Bus System", "CIGRE Medium Voltage"]


def load_network_case(network_name: str = "IEEE 14-Bus System"):
    """
    Loads a standard IEEE power network case from pandapower.
    """
    if not HAS_PANDAPOWER:
        raise ImportError("pandapower library is required for grid simulation.")

    if network_name == "IEEE 9-Bus System":
        return pn.case9()
    elif network_name == "CIGRE Medium Voltage":
        return pn.create_cigre_network_mv()
    else:
        return pn.case14()


def run_power_flow_simulation(
    net,
    attack_bus_id: Optional[int] = None,
    attack_type: str = "none",
    magnitude: float = 1.0,
) -> Dict[str, Any]:
    """
    Runs an AC power flow simulation on the network.
    Optionally applies a bus-level cyberattack (e.g. Load FDI, Voltage Spoofing, or Line Trip).
    Returns comprehensive bus and line operational telemetry.
    """
    if not HAS_PANDAPOWER:
        return {"success": False, "error": "pandapower not available"}

    import copy
    net_sim = copy.deepcopy(net)

    attack_applied = False
    attack_description = "Normal baseline dispatch"

    if attack_bus_id is not None and attack_bus_id in net_sim.bus.index:
        if attack_type == "load_fdi":
            # Manipulate active power of loads attached to this bus
            matching_loads = net_sim.load[net_sim.load.bus == attack_bus_id].index
            if len(matching_loads) > 0:
                net_sim.load.loc[matching_loads, "p_mw"] *= float(magnitude)
                attack_applied = True
                attack_description = f"Load FDI injected at Bus {attack_bus_id}: load scaled by {magnitude}x"
        elif attack_type == "voltage_spoofing":
            # Spoof generator target voltage or bus shunt
            matching_gens = net_sim.gen[net_sim.gen.bus == attack_bus_id].index
            if len(matching_gens) > 0:
                net_sim.gen.loc[matching_gens, "vm_pu"] = float(magnitude)
                attack_applied = True
                attack_description = f"Voltage setpoint manipulation at Bus {attack_bus_id} set to {magnitude} p.u."
        elif attack_type == "line_trip":
            # Open first line connected to this bus
            connected_lines = net_sim.line[(net_sim.line.from_bus == attack_bus_id) | (net_sim.line.to_bus == attack_bus_id)].index
            if len(connected_lines) > 0:
                target_line = connected_lines[0]
                net_sim.line.loc[target_line, "in_service"] = False
                attack_applied = True
                attack_description = f"Line {target_line} tripped from Bus {attack_bus_id}"

    # Run AC Power Flow (numba=False guarantees stable cross-platform execution)
    try:
        pp.runpp(net_sim, algorithm="nr", numba=False, max_iteration=30)
        converged = True
    except Exception as e:
        converged = False
        return {
            "success": False,
            "converged": False,
            "error": f"Power flow diverged due to extreme perturbation: {str(e)}",
            "attack_description": attack_description,
        }

    # Extract bus results
    bus_df = net_sim.res_bus.copy()
    bus_df["bus_id"] = bus_df.index
    bus_df["vm_pu"] = np.round(bus_df["vm_pu"], 4)
    bus_df["va_degree"] = np.round(bus_df["va_degree"], 2)
    bus_df["is_attacked"] = (bus_df["bus_id"] == attack_bus_id) if attack_applied else False
    bus_df["voltage_status"] = np.where(
        (bus_df["vm_pu"] < 0.95) | (bus_df["vm_pu"] > 1.05),
        "VIOLATION",
        "NORMAL",
    )

    # Extract line results
    line_df = net_sim.res_line.copy()
    line_df["line_id"] = line_df.index
    line_df["from_bus"] = net_sim.line["from_bus"].values
    line_df["to_bus"] = net_sim.line["to_bus"].values
    line_df["loading_percent"] = np.round(line_df["loading_percent"], 2)
    line_df["is_overloaded"] = line_df["loading_percent"] > 100.0

    # Summary metrics
    overloaded_lines = int(line_df["is_overloaded"].sum())
    voltage_violations = int((bus_df["voltage_status"] == "VIOLATION").sum())
    total_active_loss_mw = round(float(line_df["p_from_mw"].sum() + line_df["p_to_mw"].sum()), 4)

    return {
        "success": True,
        "converged": converged,
        "attack_applied": attack_applied,
        "attack_description": attack_description,
        "n_buses": len(bus_df),
        "n_lines": len(line_df),
        "overloaded_lines": overloaded_lines,
        "voltage_violations": voltage_violations,
        "total_active_loss_mw": total_active_loss_mw,
        "bus_df": bus_df,
        "line_df": line_df,
        "network_object": net_sim,
    }


def generate_bus_timeseries(
    n_steps: int = 100,
    network_name: str = "IEEE 14-Bus System",
    attack_step: int = 40,
    attack_bus_id: int = 3,
    attack_magnitude: float = 2.5,
) -> pd.DataFrame:
    """
    Generates time-series telemetry from multi-bus simulation, introducing
    a cyberattack perturbation at attack_step.
    """
    if not HAS_PANDAPOWER:
        # Graceful fallback mock
        t = pd.date_range("2024-01-01", periods=n_steps, freq="15min")
        return pd.DataFrame({"timestamp": t, "voltage": 230.0, "current": 10.0, "load": 2.3, "frequency": 50.0, "target": 0})

    base_net = load_network_case(network_name)
    rows = []
    timestamps = pd.date_range("2024-01-01", periods=n_steps, freq="15min")

    for i in range(n_steps):
        is_attack = (i >= attack_step and i < attack_step + 15)
        mag = attack_magnitude if is_attack else 1.0
        att_type = "load_fdi" if is_attack else "none"

        sim = run_power_flow_simulation(
            base_net,
            attack_bus_id=attack_bus_id if is_attack else None,
            attack_type=att_type,
            magnitude=mag,
        )

        if sim["success"]:
            b_df = sim["bus_df"]
            # Monitor bus 3 (or selected bus)
            mon_bus = b_df.loc[attack_bus_id] if attack_bus_id in b_df.index else b_df.iloc[0]
            v_kv = float(mon_bus["vm_pu"]) * 230.0  # single phase equivalent or base
            v_pu = float(mon_bus["vm_pu"])
            l_mw = float(mon_bus["p_mw"]) if "p_mw" in mon_bus else 2.5
        else:
            v_kv = 230.0
            v_pu = 1.0
            l_mw = 2.5

        # Base nominal frequency with subtle operational drift
        freq = 50.0 + (0.015 * np.sin(i / 5.0)) - (0.4 if is_attack else 0.0)
        curr = max(0.5, (l_mw * 1000.0) / (v_kv * 0.95 + 1e-3))

        rows.append({
            "timestamp": timestamps[i],
            "voltage": round(v_kv, 2),
            "current": round(curr, 2),
            "load": round(l_mw, 3),
            "frequency": round(freq, 3),
            "target": 1 if is_attack else 0,
            "attack_type": "load_fdi" if is_attack else "none",
            "monitored_bus_id": attack_bus_id,
            "bus_vm_pu": round(v_pu, 4),
        })

    return pd.DataFrame(rows)
