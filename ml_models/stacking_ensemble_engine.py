import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

class FlightGuardEnsembleEngine:
    """Stacking Ensemble Classifier for Aviation Predictive Maintenance"""
    def __init__(self):
        base_estimators = [
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42))
        ]
        self.model = StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression()
        )
        self.feature_names = [
            "flight_operating_hours", "engine_vibration_hz", 
            "oil_pressure_psi", "exhaust_gas_temp_c", 
            "hydraulic_fluid_level_pct", "ambient_temperature_c"
        ]

    def train_and_evaluate(self, csv_path="data/aircraft_telemetry_dataset.csv"):
        df = pd.read_csv(csv_path)
        X = df[self.feature_names]
        y = df["maintenance_urgency_flag"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        
        preds_proba = self.model.predict_proba(X_test)[:, 1]
        return df, X_test, preds_proba