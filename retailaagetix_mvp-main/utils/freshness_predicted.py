import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

class FreshnessPredictor:
    """
    Hybrid model combining heuristic and ML-based freshness degradation prediction.
    """

    def __init__(self):
        self.rf_model = None
        self.is_trained = False

    # ---------------- Heuristic decay logic ---------------- #
    def heuristic_decay(self, shelf_life_days, avg_temp, humidity, freshness_score):
        """
        Estimate remaining freshness using decay functions inspired by food storage physics.
        """
        # temperature impact (for each degree > 5°C, reduce life by 2%)
        temp_factor = 0.98 ** max(0, avg_temp - 5)
        # humidity effect (ideal 65%, deviation reduces life)
        humidity_factor = 1 - abs(humidity - 65) / 200
        # visual freshness influence
        freshness_factor = freshness_score

        adjusted_shelf_life = shelf_life_days * temp_factor * humidity_factor * freshness_factor
        return round(adjusted_shelf_life, 2)

    # ---------------- Generate mock training data ---------------- #
    def generate_mock_data(self, n=1000):
        """
        Generate realistic mock data for perishable items.
        """
        np.random.seed(42)
        categories = np.random.choice(["Fruit", "Vegetable", "Poultry"], size=n)
        shelf_life = np.select(
            [categories == "Fruit", categories == "Vegetable", categories == "Poultry"],
            [np.random.randint(5, 8, n), np.random.randint(3, 7, n), np.random.randint(2, 5, n)]
        )
        avg_temp = np.random.normal(6.5, 2.0, n)
        humidity = np.random.normal(65, 10, n)
        freshness_score = np.random.uniform(0.7, 1.0, n)
        days_since_receipt = np.random.randint(0, 6, n)

        # Base degradation pattern
        remaining_days = []
        for sl, t, h, f, d in zip(shelf_life, avg_temp, humidity, freshness_score, days_since_receipt):
            heuristic = self.heuristic_decay(sl, t, h, f)
            remaining = max(0, heuristic - d + np.random.normal(0, 0.5))
            remaining_days.append(round(remaining, 2))

        data = pd.DataFrame({
            "category": categories,
            "shelf_life_days": shelf_life,
            "avg_temp": avg_temp,
            "humidity": humidity,
            "freshness_score": freshness_score,
            "days_since_receipt": days_since_receipt,
            "remaining_days": remaining_days
        })

        return data

    # ---------------- Train the Random Forest ---------------- #
    def train(self, data):
        """
        Train Random Forest on historical (or mock) data.
        """
        feature_cols = ["shelf_life_days", "avg_temp", "humidity", "freshness_score", "days_since_receipt"]
        X = data[feature_cols]
        y = data["remaining_days"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        self.rf_model = model
        self.is_trained = True

        print(f"✅ Model trained: MAE={mae:.2f}, R²={r2:.2f}")
        return mae, r2

    # ---------------- Predict freshness ---------------- #
    def predict(self, shelf_life_days, avg_temp, humidity, freshness_score, days_since_receipt):
        """
        Predict remaining freshness days and expiry risk score.
        """
        if not self.is_trained:
            raise RuntimeError("⚠️ Model not trained. Run train() first.")

        heuristic_est = self.heuristic_decay(shelf_life_days, avg_temp, humidity, freshness_score)
        features = np.array([[shelf_life_days, avg_temp, humidity, freshness_score, days_since_receipt]])
        ml_prediction = self.rf_model.predict(features)[0]

        # Weighted blend (60% ML + 40% heuristic)
        remaining_days = round(0.6 * ml_prediction + 0.4 * heuristic_est, 2)
        degradation_rate = round((shelf_life_days - remaining_days) / shelf_life_days, 3)
        expiry_risk_score = round(min(1.0, 1 - remaining_days / shelf_life_days), 2)

        return {
            "predicted_remaining_days": remaining_days,
            "degradation_rate": degradation_rate,
            "expiry_risk_score": expiry_risk_score,
            "model_confidence": round(1 - degradation_rate * 0.5, 2)
        }