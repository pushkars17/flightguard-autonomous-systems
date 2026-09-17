import unittest
import os
import pandas as pd
from ml_models.stacking_ensemble_engine import FlightGuardEnsembleEngine
from agents.orchestrator import FlightGuardOrchestrator

class TestFlightGuardSystem(unittest.TestCase):
    def test_dataset_exists(self):
        self.assertTrue(os.path.exists("data/aircraft_telemetry_dataset.csv"))

    def test_ensemble_model(self):
        engine = FlightGuardEnsembleEngine()
        df, X_test, preds = engine.train_and_evaluate()
        self.assertTrue(len(preds) > 0)
        self.assertTrue(0.0 <= preds[0] <= 1.0)

    def test_orchestrator(self):
        orch = FlightGuardOrchestrator()
        audit_df = orch.execute_swarm_pipeline()
        self.assertIn("Tail Number", audit_df.columns)
        self.assertIn("Risk Score", audit_df.columns)
        self.assertIn("Estimated RUL (Hrs)", audit_df.columns)
        self.assertIn("Maintenance Directive", audit_df.columns)
        self.assertIn("XAI Justification", audit_df.columns)
        self.assertTrue(len(audit_df) > 0)

if __name__ == "__main__":
    unittest.main()