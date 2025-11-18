
########################################################################################
# analysis_agent.py
import json
import numpy as np
import pandas as pd
from utils.ollama_client import call_ollama
from utils.freshness_predicted import FreshnessPredictor

class AnalysisAgent:



    def __init__(self):
        self.freshness_model = FreshnessPredictor()
        data = self.freshness_model.generate_mock_data(1000)
        self.freshness_model.train(data)

    def compute_sales_trend(self, sales_list):
        # basic slope (linear regression on daily index)
        qtys = [d["sold_qty"] for d in sales_list]
        if len(qtys) < 3:
            return 0.0
        x = np.arange(len(qtys))
        coef = np.polyfit(x, qtys, 1)[0]
        return float(coef)

    # def estimate_spoilage_risk(self, temperature, days_since_arrival=2):
    #     # Simple heuristic: higher temp => higher spoilage
    #     base_risk = 0.05 * days_since_arrival
    #     temp_risk = max(0, (temperature - 25) * 0.02)
    #     risk = min(1.0, base_risk + temp_risk)
    #     return float(round(risk, 3))

    def estimate_spoilage_risk(self, temperature, days_since_arrival, 
                           shelf_life_days, humidity, freshness_score):
        """
        New hybrid-based spoilage risk estimator.
        Returns same type as before: single float between 0 and 1.
        """
        if not hasattr(self, "freshness_model") or not self.freshness_model.is_trained:
            raise RuntimeError("⚠️ Freshness model not initialized or trained.")

        # Use hybrid model for prediction
        result = self.freshness_model.predict(
            shelf_life_days=shelf_life_days,
            avg_temp=temperature,
            humidity=humidity,
            freshness_score=freshness_score,
            days_since_receipt=days_since_arrival
        )

        # Extract and normalize to match old return type
        print('result',result)
        spoilage_risk = result["expiry_risk_score"]
        remaining_days=result["predicted_remaining_days"]
        spoilage_risk_per_day = min(spoilage_risk * 0.03, 0.02)
        return float(round(spoilage_risk_per_day, 3))

    @staticmethod
    def map_expiry_to_daily_spoilage(expiry_risk_score: float,
                            predicted_remaining_days: float,
                            shelf_life_days: float,
                            freshness_score: float = 1.0,
                            avg_temp: float = None,
                            humidity: float = None,
                            cap: float = 0.05,
                            min_rate: float = 0.0005):
        """
        Convert expiry_risk_score (0..1) and predicted_remaining_days into a per-day spoilage probability.
        Uses a remaining-days normalization with light damping + optional environment adjustments.
        Returns float in [min_rate, cap].
        """
        # safety / sane defaults
        expiry = float(np.clip(expiry_risk_score, 0.0, 1.0))
        rem_days = float(predicted_remaining_days) if predicted_remaining_days is not None else float(max(shelf_life_days, 1))
        rem_days = max(rem_days, 0.5)  # avoid div by zero and extremely small divisors

        # Basic per-day estimate: distribute the remaining expiry risk over remaining days
        per_day = expiry / rem_days

        # Slight damping so the per-day value isn't extreme when rem_days is tiny:
        damping = 0.8 if rem_days < max(1, shelf_life_days * 0.3) else 1.0
        per_day *= damping

        # Optional tiny environmental adjustments (non-linear, small).
        if avg_temp is not None:
            # small increase per degree above 5C
            per_day *= np.exp(0.02 * max(0.0, avg_temp - 5.0))
        if humidity is not None:
            per_day *= (1 - abs(humidity - 65.0) / 500.0)  # small effect

        # Combine with baseline rate 1/shelf_life to keep physical consistency
        base_rate = (1.0 / max(shelf_life_days, 1.0)) * 0.10
        alpha = 0.6  # how much weight we give to model-derived rate vs baseline
        combined = alpha * per_day + (1 - alpha) * base_rate

        # clamp between sensible bounds
        combined = float(np.clip(combined, min_rate, cap))
        return combined


    # def estimate_spoilage_risk(self, temperature, days_since_arrival, 
    #                        shelf_life_days, humidity, freshness_score):
    #     """
    #     Composite physical + model-based spoilage risk estimator.
    #     Returns a per-day spoilage probability (0..1) for use in daily shrinkage simulations.
    #     """

    #     if not hasattr(self, "freshness_model") or not self.freshness_model.is_trained:
    #         raise RuntimeError("⚠️ Freshness model not initialized or trained.")

    #     # --- Step 1: Get model-based freshness prediction ---
    #     result = self.freshness_model.predict(
    #         shelf_life_days=shelf_life_days,
    #         avg_temp=temperature,
    #         humidity=humidity,
    #         freshness_score=freshness_score,
    #         days_since_receipt=days_since_arrival
    #     )

    #     expiry_risk = float(result["expiry_risk_score"])              # model risk (0–1)
    #     predicted_remaining_days = float(result["predicted_remaining_days"])
    #     freshness_score = float(freshness_score)

    #     # --- Step 2: Compute base physical rate (baseline spoilage) ---
    #     base_rate = 1.0 / max(shelf_life_days, 1.0)  # e.g., 7 days → 0.142

    #     # --- Step 3: Calculate environmental adjustments ---
    #     # Temperature effect: every °C above 5 increases rate ~3%
    #     temp_adj = 0.03 * max(0.0, temperature - 5.0)

    #     # Humidity effect: deviation from 65% ideal reduces or increases spoilage
    #     humidity_adj = abs(humidity - 65.0) / 500.0  # small impact (±0.1 max)

    #     # Freshness factor: lower freshness means higher spoilage
    #     freshness_adj = (1.0 - freshness_score) * 0.5

    #     # --- Step 4: Combine model + physics ---
    #     # Model risk and physical baseline combined with tunable weights
    #     w_model = 0.7   # 70% weight to ML signal
    #     w_phys = 0.3    # 30% weight to physical baseline

    #     multiplier = 1 + expiry_risk * 0.8 + temp_adj + humidity_adj + freshness_adj
    #     spoilage_risk_per_day = (w_model * expiry_risk / max(predicted_remaining_days, 1.0)) \
    #                             + (w_phys * base_rate * multiplier)

    #     # --- Step 5: Clamp to realistic bounds ---
    #     spoilage_risk_per_day = float(np.clip(spoilage_risk_per_day, 0.001, 0.05))  # 0.1–5% per day typical

    #     spoilage_risk_per_day =self.map_expiry_to_daily_spoilage(
    #         expiry_risk_score=expiry_risk,
    #         predicted_remaining_days=predicted_remaining_days,
    #         shelf_life_days=shelf_life_days,
    #         freshness_score=freshness_score,
    #         avg_temp=temperature,
    #         humidity=humidity,
    #         cap=0.05
    #     )

    #     return spoilage_risk_per_day



    def interpret_user_notes(self, notes: str):
        # Use LLM to extract intents: promotions, festivals, special supply info
        prompt = (
            "You are a retail analyst assistant. Extract a short structured summary from the user's note. "
            "Return a JSON object with keys: 'intent' (one of ['festival','promotion','supplier_issue','other']), "
            "'confidence' (0-1), and a short 'notes' string.\n\n"
            f"User note: '''{notes}'''\n\nRespond with ONLY valid JSON."
        )
        resp = call_ollama(prompt, max_tokens=256)
        text = resp.get("text", "")
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = {"intent": "other", "confidence": 0.6, "notes": notes[:200]}

        # --- enforce required keys ---
        parsed.setdefault("intent", "other")
        parsed.setdefault("confidence", 0.5)
        parsed.setdefault("notes", "")

        return parsed

    

    def analyze(self, ingestion_output):
        stock = ingestion_output["stock"]
        sales_history = ingestion_output["sales_history"]
        weather = ingestion_output["weather"]
        forecast = ingestion_output.get("weather_forecast", [])
        events = ingestion_output.get("calendar_events", [])
        notes = ingestion_output["user_notes"]

        # --- Footfall data from ingestion ---
        ff_current = ingestion_output.get("footfall_current") or {}
        ff_history = ingestion_output.get("footfall_history") or {}

        latest_index = None
        avg_baseline = None
        if isinstance(ff_current, dict) and ff_current.get("summary"):
            cs = ff_current["summary"]
            latest_index = float(cs.get("average_index")) if cs.get("average_index") is not None else None
            avg_baseline = float(cs.get("avg_baseline")) if cs.get("avg_baseline") is not None else None

        historical_avg_count = None
        if isinstance(ff_history, dict) and ff_history.get("summary"):
            hs = ff_history["summary"]
            historical_avg_count = float(hs.get("average_footfall")) if hs.get("average_footfall") is not None else None

        historical_index = None
        if historical_avg_count is not None and avg_baseline:
            historical_index = historical_avg_count / avg_baseline if avg_baseline > 0 else None

        if historical_index is None:
            try:
                hr_records = ff_history.get("records", []) if isinstance(ff_history, dict) else []
                hr_df = pd.DataFrame(hr_records)
                if not hr_df.empty and "footfall_index" in hr_df.columns:
                    historical_index = float(hr_df["footfall_index"].mean())
            except Exception:
                historical_index = None

        if historical_index is None:
            historical_index = 1.0

        if latest_index is None:
            try:
                cr_records = ff_current.get("records", []) if isinstance(ff_current, dict) else []
                cr_df = pd.DataFrame(cr_records)
                if not cr_df.empty and "footfall_index" in cr_df.columns:
                    latest_index = float(cr_df["footfall_index"].mean())
            except Exception:
                latest_index = None

        if latest_index is None:
            latest_index = 1.0

        # --- Compute normalized demand factor ---
        raw_factor = latest_index / historical_index if historical_index and historical_index > 0 else 1.0
        demand_factor = float(max(0.5, min(raw_factor, 2.0)))  # capped multiplier

        footfall_summary = {
            "avg_footfall": round(historical_index, 3),
            "latest_footfall": round(latest_index, 3),
            "footfall_factor": round(demand_factor, 3)
        }

        # 🆕 Normalize daily historical footfall for training
        historical_records = ff_history.get("records", [])
        footfall_df = pd.DataFrame(historical_records)
        if not footfall_df.empty and "avg_footfall_estimated" in footfall_df.columns:
            footfall_df["date"] = pd.to_datetime(footfall_df["date"], errors="coerce")
            avg_base = footfall_df["avg_footfall_estimated"].mean()
            footfall_df["footfall_factor"] = footfall_df["avg_footfall_estimated"] / avg_base
            footfall_df["festival_boost"] = footfall_df["event_flag"].apply(
                lambda x: 1 if str(x).lower() == "festival" else 0
            )
            footfall_df["temp_proxy"] = footfall_df["weather_condition"].map(
                {"Sunny": 1, "Rainy": 0}
            ).fillna(1)
            footfall_daily_features = footfall_df[
                ["date", "footfall_factor", "festival_boost", "temp_proxy"]
            ].to_dict("records")
        else:
            footfall_daily_features = []

        # --- existing per-item analysis (unchanged) ---
        analysis = {}
        for item, st_list in stock.items():
            hist = sales_history.get(item, [])
            trend = self.compute_sales_trend(hist)
            avg_daily = float(np.mean([d["sold_qty"] for d in hist])) if hist else 0.0
            avg_price=float(np.mean([d["price"] for d in hist])) if hist else 0.0
            latest_stock = st_list[-1]
            humidity = latest_stock.get("humidity", 0)
            freshness = latest_stock.get("freshness", 1.0)
            f_temprature = latest_stock.get("f_temprature", 25.0)
            shelf_life_days = latest_stock.get("shelf_life_days", 5)
            days=2
            if forecast:
                avg_future_temp = np.mean([(d["temp_max"] + d["temp_min"]) / 2 for d in forecast])
            else:
                avg_future_temp = weather.get("temperature", 25.0)

            spoilage_risk = self.estimate_spoilage_risk(f_temprature,days,shelf_life_days,humidity,freshness)
            # avg_price = np.mean([st['price_per_unit_usd'] for st in st_list]) if st_list else 0.0
            analysis[item] = {
                "stock_qty": st_list[-1]["available_qty"],
                "avg_daily_sales": avg_daily,
                "trend_slope": trend,
                "spoilage_risk": spoilage_risk,
                "avg_price": avg_price,
                "footfall_factor": round(demand_factor, 3),
            }

        # --- handle calendar/notes (unchanged) ---
        today = pd.Timestamp.today().date()
        upcoming_festival = None
        for ev in events:
            ev_date = pd.to_datetime(ev["date"]).date()
            if 0 <= (ev_date - today).days <= 7:
                upcoming_festival = ev
                break

        notes_summary = self.interpret_user_notes(" ".join(notes))
        if "notes" not in notes_summary or not isinstance(notes_summary["notes"], str):
            notes_summary["notes"] = ""

        if upcoming_festival:
            notes_summary["intent"] = "festival"
            notes_summary["notes"] += f" | Upcoming festival: {upcoming_festival['event']} on {upcoming_festival['date']}"

        # 🆕 Include structured daily footfall in the returned payload
        return {
            "analysis": analysis,
            "footfall_summary": footfall_summary,
            "footfall_daily_features": footfall_daily_features,  # <-- NEW
            "notes_summary": notes_summary,
            "weather": weather,
            "forecast_used": forecast,
            "calendar_events": events
        }



    def run(self, ingestion_output):
        return self.analyze(ingestion_output)


if __name__ == "__main__":
    from data_ingestion_agent import DataIngestionAgent
    ing = DataIngestionAgent().run()
    print(json.dumps(AnalysisAgent().run(ing), indent=2))
##############################################################################################
