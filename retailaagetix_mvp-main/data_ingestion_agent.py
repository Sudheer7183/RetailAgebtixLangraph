
#############################################################################


import requests
import random
import datetime
import os
import pandas as pd

class DataIngestionAgent:
    def __init__(self, use_spreadsheet=False, spreadsheet_api_url="http://localhost:8002"):
        self.weather_api = "https://api.open-meteo.com/v1/forecast"
        self.calendar_api = "https://calendarific.com/api/v2/holidays"
        self.calendar_api_key = "pAJTC7q4vTfoPA39qhcggJvqZRI90iCN"
        self.use_spreadsheet = use_spreadsheet
        self.spreadsheet_api_url = spreadsheet_api_url
    
    def get_weather(self, lat=13.0843, lon=80.2705):
        """Get current weather data"""
        try:
            params = {
                "latitude": lat,
                "longitude": lon,
                "current_weather": True
            }
            resp = requests.get(self.weather_api, params=params, timeout=10)
            data = resp.json()
            return {
                "temperature": data["current_weather"]["temperature"],
                "windspeed": data["current_weather"]["windspeed"],
                "weathercode": data["current_weather"]["weathercode"]
            }
        except Exception as e:
            print("Weather API error:", e)
            return {"temperature": random.randint(20, 35), "windspeed": 5, "weathercode": 0}
    
    def get_weather_forecast(self, lat=13.0843, lon=80.2705, days=5):
        """Get weather forecast"""
        try:
            params = {
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_max,temperature_2m_min",
                "forecast_days": days,
                "timezone": "auto"
            }
            resp = requests.get(self.weather_api, params=params, timeout=10)
            data = resp.json()
            forecast = []
            for i, d in enumerate(data["daily"]["time"]):
                forecast.append({
                    "date": d,
                    "temp_max": data["daily"]["temperature_2m_max"][i],
                    "temp_min": data["daily"]["temperature_2m_min"][i]
                })
            return forecast
        except Exception as e:
            print("Weather forecast API error:", e)
            return [{"date": str(datetime.date.today() + datetime.timedelta(days=i)), 
                    "temp_max": random.randint(25, 35), 
                    "temp_min": random.randint(20, 30)} for i in range(days)]
    
    def get_festival_events(self, country="IN", year=None):
        """Get festival/calendar events"""
        if not year:
            year = datetime.datetime.now().year
        if not self.calendar_api_key:
            return [
                {"date": "2025-09-10", "event": "Ganesh Chaturthi", "impact": "high demand"},
                {"date": "2025-09-15", "event": "Onam", "impact": "spike in vegetables"}
            ]
        try:
            params = {
                "api_key": self.calendar_api_key,
                "country": country,
                "year": year
            }
            resp = requests.get(self.calendar_api, params=params, timeout=10)
            data = resp.json()
            events = []
            for holiday in data["response"]["holidays"]:
                events.append({
                    "date": holiday["date"]["iso"],
                    "event": holiday["name"],
                    "impact": "demand spike"
                })
            return events
        except Exception as e:
            print("Calendar API error:", e)
            return []
    # ────────────────────────────────────────────────
    # 🔹 SALES HISTORY MAPPING
    # ────────────────────────────────────────────────
    
    def get_sales_history_from_spreadsheet(self):
        """Fetch and strictly normalize sales history for analysis agent."""
        try:
            resp = requests.get(f"{self.spreadsheet_api_url}/api/spreadsheet/sales-history", timeout=15)
            resp.raise_for_status()
            data = resp.json().get("data", [])
            if not data:
                print("⚠️ No sales data received from spreadsheet.")
                return {}

            df = pd.DataFrame(data)
            df.columns = df.columns.str.lower().str.strip()

            # ✅ Product column mapping
            if "sku_name" not in df.columns:
                print("⚠️ 'sku_name' column not found in sales data.")
                return {}

            # ✅ Essential columns
            if "units_sold" not in df.columns:
                print("⚠️ 'units_sold' column not found in sales data.")
                return {}

            mapped = {}
            for _, row in df.iterrows():
                product = str(row.get("sku_name", "")).strip().lower()
                if not product:
                    continue

                # Optional date handling
                date = str(row.get("date") or row.get("sales_date") or "")
                if not date:
                    continue

                try:
                    sold_qty = float(row.get("units_sold", 0))
                except (ValueError, TypeError):
                    sold_qty = 0.0

                price = float(
                    row.get("price_per_unit_usd") or
                    row.get("price_usd") or
                    row.get("unit_price") or 0
                )

                revenue = float(row.get("revenue_usd") or sold_qty * price)

                mapped.setdefault(product, []).append({
                    "date": date,
                    "sold_qty": sold_qty,
                    "price": round(price, 2),
                    "revenue": round(revenue, 2)
                })

            return mapped

        except Exception as e:
            print(f"❌ Sales spreadsheet ingestion error: {e}")
            return {}

    # ────────────────────────────────
    # 🔹 Historical inventory DATA (strict)
    # ────────────────────────────────
    
    def get_historical_inventory(self):
        """Fetch and strictly normalize historical inventory data for analysis agent"""
        try:
            resp=requests.get(f"{self.spreadsheet_api_url}/api/spreadsheet/inventory",timeout=15)
            resp.raise_for_status()
            data= resp.json().get("data",[])
            if not data:
                print("No historical inventory data recived from spread sheet")
                return{}
            df=pd.DataFrame(data)
            df.columns=df.columns.str.lower().str.strip()

            if "sku_name" not in df.columns:
                print("⚠️ 'sku_name' column not found in sales data.")
                return {}
            
            mapped={}

            for _,row in df.iterrows():
                product = str(row.get("sku_name", "")).strip().lower()
                if not product:
                    continue

                date = str(row.get("date") or row.get("sales_date") or "")
                if not date:
                    continue

                try:
                    sold_qty = float(row.get("sold_units", 0))
                except (ValueError, TypeError):
                    sold_qty = 0.0

                try:
                    opening_units = float(row.get("opening_units"))
                except(ValueError,TypeError):
                    opening_units=0.0

                try:
                    spoilage_units=float(row.get("spoilage_units"))
                except(ValueError,TypeError):
                    spoilage_units=0.0

                mapped.setdefault(product,[]).append({
                    "date":date,
                    "sold_qty":sold_qty,
                    "opening_units":opening_units,
                    "spoilage_units":spoilage_units,
                })

            return mapped

        except Exception as e:
            print(f"❌ Sales spreadsheet ingestion error: {e}")
            return {}



    # ────────────────────────────────
    # 🔹 STOCK DATA (strict)
    # ────────────────────────────────
    def get_stock_from_spreadsheet(self):
        """Fetch and strictly normalize stock data for analysis agent."""
        try:
            resp = requests.get(f"{self.spreadsheet_api_url}/api/spreadsheet/current-stock", timeout=15)
            resp.raise_for_status()
            data = resp.json().get("data", [])
            if not data:
                print("⚠️ No stock data received from spreadsheet.")
                return {}

            df = pd.DataFrame(data)
            df.columns = df.columns.str.lower().str.strip()

            # ✅ Product column
            if "sku_name" not in df.columns:
                print("⚠️ 'sku_name' column not found in stock data.")
                return {}

            mapped = {}
            today = datetime.date.today()

            for _, row in df.iterrows():
                product = str(row.get("sku_name", "")).strip().lower()
                if not product:
                    continue

                # ✅ Available quantity mapping
                try:
                    total_qty=float(row.get("total_weight_lbs",0))
                    ripping_qty=float(row.get("ripening_aging_softening_lbs",0))
                    fresh_lbs=float(row.get("fresh_lbs",0))
                    # available_qty = float(row.get("total_weight_lbs", 0))
                    available_qty=ripping_qty+fresh_lbs
                except (ValueError, TypeError):
                    available_qty = 0.0

                if available_qty <= 0:
                    continue

                # ✅ Condition mapping (fresh / spoiled / overripe)
                spoiled = float(row.get("spoiled_overripe_lbs", 0))
                overripe = float(row.get("ripening_aging_softening_lbs", 0))
                fresh = float(row.get("fresh_lbs",0))
                rancid=float(row.get("rancid_lbs",0))
                category=str(row.get("category","uncategorized"))

                condition_values = {
                    "spoiled": spoiled,
                    "overripe": overripe,
                    "fresh": fresh,
                    "rancid": rancid
                }

                condition = max(condition_values, key=condition_values.get)

                # ✅ Average price mapping
                avg_price = (
                    # float(row.get("avg_price", 0)) or
                    float(row.get("avg_price", 0)) or 0
                )

                print('agerage price from the ingestion that is being passed',avg_price)

                humidity=float(row.get("humidity_percent",0))
                freshness=float(row.get("freshness_score",0))
                f_temprature=float(row.get("fahrenheit_temp",0))
                shelf_life_days=float(row.get("shelf_life_days",0))
                mapped.setdefault(product, []).append({
                    "available_qty": round(available_qty, 2),
                    "condition": condition,
                    "avg_price": round(avg_price, 2),
                    "date": str(today),
                    "humidity":humidity,
                    "freshness":freshness,
                    "f_temprature":f_temprature,
                    "shelf_life_days":shelf_life_days,
                    "category":category
                })

            return mapped

        except Exception as e:
            print(f"❌ Stock spreadsheet ingestion error: {e}")
            return {}

    # ────────────────────────────────
    # 🔹 POS DATA (strict)
    # ────────────────────────────────

    def get_pos_data(self):
        try:
            resp=requests.get(f"{self.spreadsheet_api_url}/api/spreadsheet/pos-data",timeout=15)
            resp.raise_for_status()
            data= resp.json().get("data", [])

            if not data:
                print("⚠️ No pos data received from spreadsheet.")
                return {}

            df = pd.DataFrame(data)
            df.columns = df.columns.str.lower().str.strip()

            # ✅ Product column mapping
            if "sku_name" not in df.columns:
                print("⚠️ 'sku_name' column not found in sales data.")
                return {}

            mapped = {}
            today = datetime.date.today()

            for _, row in df.iterrows():
                product = str(row.get("sku_name", "")).strip().lower()
                if not product:
                    continue

                # ✅ Available quantity mapping

                try:
                    regular_price=float(row.get("regular_price_usd"))

                except (ValueError,TypeError):
                    regular_price=0

                if regular_price<=0:
                    continue

                mapped.setdefault(product,[]).append({
                    "regular_price":round(regular_price,2)
                })
            print('Mapped pos data',mapped)
            return mapped

        except Exception as e:
            print(f"❌ Stock spreadsheet ingestion error: {e}")
            return {}



    # ────────────────────────────────
    # 🔹 Foot fall DATA (strict)
    # ────────────────────────────────  

    def get_current_foot_fall_index(self):
        """Fetch the latest (hourly) footfall data and compute an average footfall index."""
        try:
            resp = requests.get(f"{self.spreadsheet_api_url}/api/spreadsheet/foot-fall-index", timeout=15)
            resp.raise_for_status()
            data = resp.json().get("data", [])

            if not data:
                print("⚠️ No current footfall index data received.")
                return {}

            df = pd.DataFrame(data)
            df.columns = df.columns.str.lower().str.strip()

            # Validate essential columns
            if "footfall_index" not in df.columns:
                print("⚠️ 'footfall_index' column missing in current footfall data.")
                return {}

            # Compute average index (0.0 - 1.0 range)
            avg_index = float(df["footfall_index"].mean())
            total_footfall = float(df["footfall_count"].sum()) if "footfall_count" in df.columns else None
            avg_baseline = float(df["historical_baseline"].mean()) if "historical_baseline" in df.columns else None

            summary = {
                "average_index": round(avg_index, 3),
                "total_footfall": total_footfall,
                "avg_baseline": avg_baseline,
                "status": "below_avg" if avg_index < 1 else "normal" if avg_index == 1 else "above_avg"
            }

            return {"records": df.to_dict("records"), "summary": summary}

        except Exception as e:
            print(f"❌ Current footfall API error: {e}")
            return {}

    def get_historical_foot_fall(self):
        """Fetch historical daily footfall data for model training."""
        try:
            resp = requests.get(f"{self.spreadsheet_api_url}/api/spreadsheet/historical-foot-fall", timeout=15)
            resp.raise_for_status()
            data = resp.json().get("data", [])

            if not data:
                print("⚠️ No historical footfall data received.")
                return {}

            df = pd.DataFrame(data)
            df.columns = df.columns.str.lower().str.strip()

            # Ensure required columns exist
            required_cols = ["date", "avg_footfall_estimated"]
            for col in required_cols:
                if col not in df.columns:
                    print(f"⚠️ Missing required column '{col}' in historical footfall data.")
                    return {}

            # Parse dates and sort
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date")

            # Optional daily average or fill missing
            df["avg_footfall_estimated"] = pd.to_numeric(df["avg_footfall_estimated"], errors="coerce").fillna(method="ffill")

            # Compute summary stats
            avg_footfall = df["avg_footfall_estimated"].mean()
            max_footfall = df["avg_footfall_estimated"].max()
            min_footfall = df["avg_footfall_estimated"].min()

            summary = {
                "average_footfall": round(avg_footfall, 2),
                "max_footfall": round(max_footfall, 2),
                "min_footfall": round(min_footfall, 2),
                "total_days": len(df)
            }

            return {"records": df.to_dict("records"), "summary": summary}

        except Exception as e:
            print(f"❌ Historical footfall API error: {e}")
            return {}


    
    # ────────────────────────────────
    # 🔹 Bakery SALES HISTORY
    # ──────────────────────────────── 

    def get_bakery_sales_history(self):
        """Fetch bakery sales history from bakery API."""
        try:
            resp = requests.get(f"{self.spreadsheet_api_url}/api/spreadsheet/backer-sales", timeout=15)
            resp.raise_for_status()
            data = resp.json().get("data", [])
            if not data:
                print("⚠️ No bakery sales data received.")
                return {}

            df = pd.DataFrame(data)
            df.columns = df.columns.str.lower().str.strip()

            mapped = {}
            for _, row in df.iterrows():
                sku = str(row.get("sku_name", "")).strip().lower()
                if not sku:
                    continue

                date = str(row.get("date") or "")
                if not date:
                    continue

                sold_qty = float(row.get("units_sold", 0))
                price = float(row.get("unit_price_usd", row.get("price_per_unit_usd", 0)))
                revenue = float(row.get("revenue_usd", sold_qty * price))

                mapped.setdefault(sku, []).append({
                    "date": date,
                    "sold_qty": sold_qty,
                    "price": round(price, 2),
                    "revenue": round(revenue, 2)
                })
            return mapped

        except Exception as e:
            print(f"❌ Bakery sales ingestion error: {e}")
            return {}
    # ────────────────────────────────
    # 🔹 Bakery INVENTORY (historical)
    # ────────────────────────────────
    def get_bakery_inventory(self):
        """Fetch bakery historical inventory data."""
        try:
            resp = requests.get(f"{self.spreadsheet_api_url}/api/spreadsheet/bakcery-inventory", timeout=15)
            resp.raise_for_status()
            data = resp.json().get("data", [])
            if not data:
                print("⚠️ No bakery inventory data received.")
                return {}

            df = pd.DataFrame(data)
            df.columns = df.columns.str.lower().str.strip()

            mapped = {}
            for _, row in df.iterrows():
                sku = str(row.get("sku_name", "")).strip().lower()
                if not sku:
                    continue

                date = str(row.get("date") or "")
                if not date:
                    continue

                sold_qty = float(row.get("sold_units", 0))
                opening_units = float(row.get("opening_units", 0))
                spoilage_units = float(row.get("spoilage_units", 0))

                mapped.setdefault(sku, []).append({
                    "date": date,
                    "sold_qty": sold_qty,
                    "opening_units": opening_units,
                    "spoilage_units": spoilage_units
                })
            return mapped
        except Exception as e:
            print(f"❌ Bakery inventory ingestion error: {e}")
            return {}

    # ────────────────────────────────
    # 🔹 Bakery CURRENT STOCK
    # ────────────────────────────────
    def get_bakery_current_stock(self):
        """Fetch current stock data for bakery."""
        try:
            resp = requests.get(f"{self.spreadsheet_api_url}/api/spreadsheet/backery-current-stock", timeout=15)
            resp.raise_for_status()
            data = resp.json().get("data", [])
            if not data:
                print("⚠️ No bakery stock data received.")
                return {}

            df = pd.DataFrame(data)
            df.columns = df.columns.str.lower().str.strip()

            today = datetime.date.today()
            mapped = {}
            for _, row in df.iterrows():
                sku = str(row.get("sku_name", "")).strip().lower()
                if not sku:
                    continue

                available_qty = float(row.get("total_units", 0))
                avg_price = float(row.get("avg_price", 0))
                category = str(row.get("category", "bakery"))
                shelf_life = float(row.get("expiry_days", row.get("shelf_life_days", 3)))

                mapped.setdefault(sku, []).append({
                    "available_qty": round(available_qty, 2),
                    "avg_price": round(avg_price, 2),
                    "date": str(today),
                    "condition": "fresh",
                    "category": category,
                    "shelf_life_days": shelf_life
                })
            return mapped
        except Exception as e:
            print(f"❌ Bakery current stock ingestion error: {e}")
            return {}

     # ────────────────────────────────
    # 🔹 Bakery POS DATA
    # ────────────────────────────────
    def get_bakery_pos_data(self):
        """Fetch bakery POS pricing data."""
        try:
            resp = requests.get(f"{self.spreadsheet_api_url}/api/spreadsheet/backer-pos-data", timeout=15)
            resp.raise_for_status()
            data = resp.json().get("data", [])
            if not data:
                print("⚠️ No bakery POS data received.")
                return {}

            df = pd.DataFrame(data)
            df.columns = df.columns.str.lower().str.strip()

            mapped = {}
            for _, row in df.iterrows():
                sku = str(row.get("sku_name", "")).strip().lower()
                if not sku:
                    continue
                regular_price = float(row.get("regular_price_usd", 0))
                shelf_life = float(row.get("expiry_days", 0))
                mapped.setdefault(sku, []).append({
                    "regular_price": round(regular_price, 2),
                    "shelf_life_days": shelf_life
                })
            return mapped

        except Exception as e:
            print(f"❌ Bakery POS ingestion error: {e}")
            return {}      

    def get_sales_history(self):
        """Get sales history (from spreadsheet or fallback)"""
        return self.get_sales_history_from_spreadsheet()
    
    def get_stock_placeholder(self):
        """Get stock data (from spreadsheet or fallback)"""
        return self.get_stock_from_spreadsheet()
    
    def get_user_notes(self):
        """Get user notes"""
        return [
            "Customers complain tomatoes are getting spoiled quickly.",
            "Festival season coming next week, expect demand surge."
        ]
    
    def get_inventory_history(self):
        return self.get_historical_inventory()

    # def run(self):
    #     """Run the ingestion agent"""
    #     current_weather = self.get_weather()
    #     forecast = self.get_weather_forecast()
    #     festivals = self.get_festival_events()
    #     sales = self.get_sales_history()
    #     stock = self.get_stock_placeholder()
    #     notes = self.get_user_notes()
    #     footfall_current = self.get_current_foot_fall_index()
    #     footfall_history = self.get_historical_foot_fall()
    #     historical_inventory=self.get_inventory_history()
    #     return {
    #         "weather": current_weather,
    #         "weather_forecast": forecast,
    #         "calendar_events": festivals,
    #         "sales_history": sales,
    #         "stock": stock,
    #         "user_notes": notes,
    #         "footfall_current": footfall_current,
    #         "footfall_history": footfall_history,
    #         "historical_inventory":historical_inventory
    #     }

    def run(self):
        """Run the ingestion agent"""
        current_weather = self.get_weather()
        forecast = self.get_weather_forecast()
        festivals = self.get_festival_events()
        sales = self.get_sales_history()
        stock = self.get_stock_placeholder()
        notes = self.get_user_notes()
        footfall_current = self.get_current_foot_fall_index()
        footfall_history = self.get_historical_foot_fall()
        historical_inventory = self.get_inventory_history()

        # ✅ Bakery ingestion
        bakery_sales = self.get_bakery_sales_history()
        bakery_inventory = self.get_bakery_inventory()
        bakery_stock = self.get_bakery_current_stock()
        bakery_pos = self.get_bakery_pos_data()

        return {
            "weather": current_weather,
            "weather_forecast": forecast,
            "calendar_events": festivals,
            "sales_history": sales,
            "stock": stock,
            "user_notes": notes,
            "footfall_current": footfall_current,
            "footfall_history": footfall_history,
            "historical_inventory": historical_inventory,
            "bakery": {
                "sales_history": bakery_sales,
                "historical_inventory": bakery_inventory,
                "stock": bakery_stock,
                "pos": bakery_pos
            }
        }


if __name__ == "__main__":
    agent = DataIngestionAgent()
    out = agent.run()
    import json
    print(json.dumps(out, indent=2))
