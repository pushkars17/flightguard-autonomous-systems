class TelemetryMonitoringAgent:
    """Agent 1: Monitors real-time IoT sensor telemetry and flags anomalies."""
    def inspect_telemetry(self, row: dict) -> dict:
        vib = row["engine_vibration_hz"]
        oil = row["oil_pressure_psi"]
        temp = row["exhaust_gas_temp_c"]
        
        status = "Nominal Operations"
        severity = "LOW"
        
        if vib > 7.5 or oil < 38.0 or temp > 820.0:
            status = "Critical Threshold Breach Detected"
            severity = "HIGH"
        elif vib > 6.0 or oil < 50.0:
            status = "Warning: Elevated Wear Metrics"
            severity = "MEDIUM"
            
        return {
            "sensor_status": status,
            "anomaly_severity": severity
        }