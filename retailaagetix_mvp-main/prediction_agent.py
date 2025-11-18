



####################################################################################################
# prediction_agent.py (enhanced with spoilage + footfall impact)
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load
from pathlib import Path

MODEL_DIR = Path("./models")
MODEL_DIR.mkdir(exist_ok=True)

class PredictionAgent:
    def __init__(self):
        self.models = {}  # item -> model instance

    # ────────────────────────────────────────────────
    # def _prepare_features(self, history, weather_series=None):
    #     """Prepare regression features from sales history."""
    #     df = pd.DataFrame(history)
    #     df["day_index"] = np.arange(len(df))
    #     if weather_series is not None:
    #         df["temp"] = weather_series
    #     else:
    #         df["temp"] = 25.0
    #     X = df[["day_index", "temp"]].values
    #     y = df["sold_qty"].values
    #     return X, y

    def _prepare_features(
        self, 
        history, 
        weather_series=None, 
        footfall_series=None, 
        spoilage_series=None, 
        festival_series=None
    ):
        """Prepare regression features with additional contextual inputs."""
        df = pd.DataFrame(history)
        df["day_index"] = np.arange(len(df))
        df["temp"] = weather_series or [25.0] * len(df)
        df["footfall_factor"] = footfall_series or [1.0] * len(df)
        df["spoilage_risk"] = spoilage_series or [0.0] * len(df)
        df["festival_boost"] = festival_series or [0] * len(df)

        feature_cols = ["day_index", "temp", "footfall_factor", "spoilage_risk", "festival_boost"]
        X = df[feature_cols].values
        y = df["sold_qty"].values
        return X, y

    # ────────────────────────────────────────────────
    # def train_item_model(self, item, history, temperature):
    #     """Train a RandomForest model (fallback to mean if data < 5 days)."""
    #     X, y = self._prepare_features(history, weather_series=[temperature]*len(history))
    #     if len(y) < 5:
    #         model = {"type": "mean", "mean": float(np.mean(y) if len(y) > 0 else 0)}
    #         self.models[item] = model
    #         return model

    #     model = RandomForestRegressor(n_estimators=50, random_state=42)
    #     model.fit(X, y)
    #     self.models[item] = model
    #     dump(model, MODEL_DIR / f"{item}_rf.joblib")
    #     return model

    def train_item_model(self, item, history, temperature, footfall_features=None):
        """Train a RandomForest model using sales + daily footfall and event data."""
        df_sales = pd.DataFrame(history)
        if df_sales.empty:
            return None

        df_sales["date"] = pd.to_datetime(df_sales["date"])
        
        # Merge with daily footfall if available
        if footfall_features:
            ff_df = pd.DataFrame(footfall_features)
            ff_df["date"] = pd.to_datetime(ff_df["date"])
            merged = pd.merge(df_sales, ff_df, on="date", how="left")
        else:
            merged = df_sales.copy()
            merged["footfall_factor"] = 1.0
            merged["festival_boost"] = 0
            merged["temp_proxy"] = 1.0

        merged.fillna({"footfall_factor": 1.0, "festival_boost": 0, "temp_proxy": 1.0}, inplace=True)
        merged["day_index"] = np.arange(len(merged))

        # Prepare features
        X = merged[["day_index", "temp_proxy", "footfall_factor", "festival_boost"]].values
        y = merged["sold_qty"].values

        if len(y) < 5:
            model = {"type": "mean", "mean": float(np.mean(y))}
            self.models[item] = model
            return model

        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        self.models[item] = model
        dump(model, MODEL_DIR / f"{item}_rf.joblib")
        return model

    # ────────────────────────────────────────────────
    # def predict_item(self, item, steps_ahead=3, forecast=None,
    #                  festival_boost=False, spoilage_risk=0.0,
    #                  footfall_factor=1.0):
    #     """
    #     Predict short-horizon demand using:
    #     - RandomForest or mean model
    #     - Boost for festivals
    #     - Adjustment for footfall (traffic)
    #     - Adjustment for spoilage risk (reduces sellable demand)
    #     """
    #     m = self.models.get(item)
    #     if isinstance(m, dict) and m.get("type") == "mean":
    #         mean_val = float(m["mean"])
    #         demand = mean_val
    #     elif m is None:
    #         demand = 0.0
    #     else:
    #         future_X = []
    #         if forecast:
    #             temps = [(d["temp_max"] + d["temp_min"]) / 2 for d in forecast[:steps_ahead]]
    #         else:
    #             temps = [25.0] * steps_ahead
    #         for i, t in enumerate(temps):
    #             future_X.append([1000 + i, t])
    #         preds = m.predict(np.array(future_X))
    #         demand = float(max(0, preds[0]))

    #     # ── 1️⃣ Footfall adjustment (positive impact)
    #     demand *= footfall_factor

    #     # ── 2️⃣ Spoilage adjustment (negative impact)
    #     demand *= max(0.0, (1 - spoilage_risk * 0.4))

    #     # ── 3️⃣ Festival boost
    #     if festival_boost:
    #         demand *= 1.3

    #     return {
    #         "future_demand": int(round(demand)),
    #         "method": "rf" if not isinstance(m, dict) else "mean",
    #         "footfall_factor": round(footfall_factor, 2),
    #         "spoilage_risk": round(spoilage_risk, 3),
    #         "festival_boost": festival_boost
    #     }

    def predict_item(self, item, steps_ahead=3, forecast=None,
                 expected_footfall_factor=1.0,
                 expected_festival=False,
                 spoilage_risk=0.0):
        """
        Predict short-horizon demand using trained RF model that already
        incorporates footfall, festival, and weather features.
        """
        m = self.models.get(item)
        if m is None:
            return {
                "future_demand": 0,
                "method": "none",
                "footfall_factor": expected_footfall_factor,
                "spoilage_risk": spoilage_risk,
                "festival_boost": expected_festival
            }

        if isinstance(m, dict) and m.get("type") == "mean":
            demand = float(m["mean"])
        else:
            # Prepare upcoming data for inference
            if forecast:
                temps = [(d["temp_max"] + d["temp_min"]) / 2 for d in forecast[:steps_ahead]]
            else:
                temps = [25.0] * steps_ahead

            # 🧩 Include all features used during training
            X_future = []
            for i, t in enumerate(temps):
                X_future.append([
                    1000 + i,                  # day_index proxy for future
                    1.0 if t > 20 else 0.0,    # temp_proxy simplification
                    expected_footfall_factor,  # use expected normalized footfall
                    1 if expected_festival else 0
                ])

            preds = m.predict(np.array(X_future))
            demand = float(max(0, preds[0]))

        # 🧩 Only adjust for spoilage (still post-model)
        # because spoilage isn't part of sales → it's an external reduction.
        demand *= max(0.0, (1 - spoilage_risk * 0.4))

        return {
            "future_demand": int(round(demand)),
            "method": "rf" if not isinstance(m, dict) else "mean",
            "footfall_factor": round(expected_footfall_factor, 2),
            "spoilage_risk": round(spoilage_risk, 3),
            "festival_boost": expected_festival
        }

    # ────────────────────────────────────────────────
    def run(self, analysis_output):
        """
        Combine AnalysisAgent output + historical training
        for per-item demand prediction.
        """
        analysis = analysis_output["analysis"]
        forecast = analysis_output.get("forecast_used", [])
        events = analysis_output.get("calendar_events", [])
        footfall_summary = analysis_output.get("footfall_summary", {})

        avg_footfall = footfall_summary.get("avg_footfall", 1.0)
        latest_footfall = footfall_summary.get("latest_footfall", 1.0)
        footfall_factor = (latest_footfall / avg_footfall) if avg_footfall > 0 else 1.0

        today = pd.Timestamp.today().date()
        festival_soon = any(
            0 <= (pd.to_datetime(ev["date"]).date() - today).days <= 3
            for ev in events
        )

        predictions = {}
        for item, metrics in analysis.items():
            spoilage_risk = metrics.get("spoilage_risk", 0.0)
            pred = self.predict_item(
                item,
                steps_ahead=3,
                forecast=forecast,
                expected_festival=festival_soon,
                spoilage_risk=spoilage_risk,
                expected_footfall_factor=footfall_factor
            )

            predictions[item] = pred

        return {"predictions": predictions}


if __name__ == "__main__":
    from data_ingestion_agent import DataIngestionAgent
    from analysis_agent import AnalysisAgent

    ing = DataIngestionAgent().run()
    analysis_out = AnalysisAgent().run(ing)

    pa = PredictionAgent()
    # Train models for each SKU
    for item, hist in ing["sales_history"].items():
        # pa.train_item_model(item, hist, ing["weather"]["temperature"])
        pa.train_item_model(
            item=item,
            history=hist,
            temperature=ingestion_output["weather"]["temperature"],
            footfall_features=analysis_out.get("footfall_daily_features")  # 🆕
        )

    # Run demand predictions
    print(json.dumps(pa.run(analysis_out), indent=2))
####################################################################################################
