import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

class RemainingUsefulLifeEstimator:
    """Estimates Remaining Useful Life (RUL) in flight hours based on wear indicators & anomaly scores."""
    
    def __init__(self):
        self.model = Ridge(alpha=1.0)
        self.is_trained = False

    def fit_and_predict_rul(self, df: pd.DataFrame, risk_probs: np.ndarray) -> np.ndarray:
        """
        Calculates RUL based on operating hours, engine vibration, exhaust temperature, 
        and stacking ensemble risk probability.
        """
        # Synthetic degradation baseline calculation for training/estimation
        # Max operational limit is typically around 10,000 to 12,000 hours before major overhaul
        max_limit = 10000.0
        
        # Feature vector for RUL degradation modeling
        X = np.column_stack([
            df["flight_operating_hours"].values,
            df["engine_vibration_hz"].values,
            df["exhaust_gas_temp_c"].values,
            risk_probs
        ])
        
        # Ground truth RUL approximation formula for demonstration/training
        # Higher vibration, higher temp, and higher risk probability exponentially reduce RUL
        wear_factor = (df["engine_vibration_hz"].values / 5.0) + (df["exhaust_gas_temp_c"].values / 500.0) + (risk_probs * 5.0)
        y_rul = np.maximum(50.0, max_limit - df["flight_operating_hours"].values - (wear_factor * 250))
        
        # Fit model and predict
        self.model.fit(X, y_rul)
        self.is_trained = True
        
        predicted_rul = self.model.predict(X)
        return np.clip(predicted_rul, 10.0, max_limit)