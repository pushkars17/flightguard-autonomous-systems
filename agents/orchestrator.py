from agents.telemetry_agent import TelemetryMonitoringAgent
from agents.xai_audit_agent import XAIAuditAgent
from ml_models.stacking_ensemble_engine import FlightGuardEnsembleEngine
from ml_models.xai_explainer import XAIModelExplainer
from ml_models.rul_estimator import RemainingUsefulLifeEstimator
import pandas as pd
import numpy as np

class FlightGuardOrchestrator:
    """Master Multi-Agent Swarm Orchestrator for FlightGuard Autonomous Systems"""
    def __init__(self):
        self.telemetry_agent = TelemetryMonitoringAgent()
        self.xai_agent = XAIAuditAgent()
        self.ensemble = FlightGuardEnsembleEngine()
        self.explainer = XAIModelExplainer()
        self.rul_estimator = RemainingUsefulLifeEstimator()
        
    def execute_swarm_pipeline(self):
        df, X_test, preds = self.ensemble.train_and_evaluate()
        
        # Compute Remaining Useful Life (RUL) for test records using regression engine
        test_df = df.loc[X_test.index]
        predicted_rul = self.rul_estimator.fit_and_predict_rul(test_df, preds)
        
        audit_records = []
        
        for idx in range(min(50, len(X_test))):
            row_idx = X_test.index[idx]
            row_data = df.loc[row_idx]
            risk_prob = float(preds[idx])
            rul_val = float(predicted_rul[idx])
            
            telemetry_res = self.telemetry_agent.inspect_telemetry(row_data.to_dict())
            attributions = self.explainer.generate_local_attribution(row_data)
            justification = self.xai_agent.compile_audit_justification(attributions, risk_prob)
            
            # Map RUL into actionable maintenance window directives
            if rul_val < 150:
                maintenance_directive = "Immediate Action (< 48 Hours)"
            elif rul_val < 500:
                maintenance_directive = "Schedule Within 2 Weeks"
            elif rul_val < 1500:
                maintenance_directive = "Routine Next Check"
            else:
                maintenance_directive = "Nominal / Healthy"
            
            audit_records.append({
                "Tail Number": row_data["aircraft_tail_number"],
                "Operating Hours": row_data["flight_operating_hours"],
                "Sensor Status": telemetry_res["sensor_status"],
                "Risk Score": round(risk_prob, 3),
                "Estimated RUL (Hrs)": round(rul_val, 1),
                "Maintenance Directive": maintenance_directive,
                "XAI Justification": justification
            })
            
        return pd.DataFrame(audit_records)