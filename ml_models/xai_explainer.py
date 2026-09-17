import numpy as np
import pandas as pd

class XAIModelExplainer:
    """Explainable AI (XAI) module providing feature attribution and risk justifications."""
    def generate_local_attribution(self, sample_row: pd.Series) -> dict:
        # Heuristic-based SHAP-style local attribution for mission-critical transparency
        vibration = sample_row["engine_vibration_hz"]
        oil = sample_row["oil_pressure_psi"]
        temp = sample_row["exhaust_gas_temp_c"]
        
        attributions = {
            "engine_vibration_hz": round(max(0.0, (vibration - 4.5) * 15.2), 2),
            "oil_pressure_psi": round(max(0.0, (70.0 - oil) * 1.8), 2),
            "exhaust_gas_temp_c": round(max(0.0, (temp - 600) * 0.12), 2),
            "flight_operating_hours": round(sample_row["flight_operating_hours"] * 0.001, 2)
        }
        return attributions