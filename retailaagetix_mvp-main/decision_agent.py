
# #####################################################################################################
import json
import time
import random
import requests

class DecisionAgent:
    def __init__(self, retry_base_delay=2.0, retry_max=6, per_item_delay=1.0):
        """
        retry_base_delay: starting delay (seconds) for exponential backoff
        retry_max: maximum number of retries per request
        per_item_delay: delay between processing each item to avoid rate limits
        """
        self.retry_base_delay = retry_base_delay
        self.retry_max = retry_max
        self.per_item_delay = per_item_delay

    # ───────────────────────────────
    # 🔹 Safe LLM call with retries and exponential backoff
    # ───────────────────────────────
    def call_ollama_safe(self, prompt, max_tokens=200):
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": "Bearer ************************",
            "Content-Type": "application/json"
        }
        body = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": "You are a helpful retail analyst."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens
        }

        for attempt in range(self.retry_max):
            try:
                response = requests.post(url, headers=headers, json=body, timeout=30)
                if response.status_code == 429:
                    # Too Many Requests → exponential backoff
                    delay = self.retry_base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"⚠️ Rate limit hit (429). Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    continue

                response.raise_for_status()
                return response.json()

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                delay = self.retry_base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"⚠️ Network error ({type(e).__name__}). Retrying in {delay:.1f}s...")
                time.sleep(delay)

            except requests.exceptions.RequestException as e:
                print(f"❌ Request failed: {e}")
                break

        print("❌ Failed after maximum retries.")
        return {"text": '{"suggested_price": null, "rationale": "fallback", "confidence": 0.0}'}

    # ───────────────────────────────
    # 🔹 Price prediction via LLM
    # ───────────────────────────────
    def get_price_from_llm(self, item, item_info, prediction):
        stock = item_info.get("stock_qty", 0)
        avg_price = item_info.get("avg_price", 10.0)
        avg_price=float(avg_price)
        future_demand = prediction.get("future_demand", 0)
        spoilage_risk = item_info.get("spoilage_risk", 0.05)

        prompt = (
            f"You are an expert retail pricing analyst.\n\n"
            f"Suggest an optimal selling price for the item to maximize profit while minimizing spoilage.\n\n"
            f"Item: {item}\n"
            f"Average price: {avg_price}\n"
            f"Stock quantity: {stock}\n"
            f"Predicted future demand: {future_demand}\n"
            f"Spoilage risk (0–1): {spoilage_risk}\n\n"
            f"Respond ONLY in JSON:\n"
            f"{{'suggested_price': float, 'rationale': str, 'confidence': float}}"
        )

        resp = self.call_ollama_safe(prompt, max_tokens=150)
        try:
            # Groq API returns in "choices[0].message.content"
            raw_text = (
                resp.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
                .strip("```json")
                .strip("```")
            )
            parsed = json.loads(raw_text)
            if isinstance(parsed.get("suggested_price"), (int, float)):
                return parsed
        except Exception as e:
            print(f"⚠️ JSON parsing failed for {item}: {e}")

        return None

    # ───────────────────────────────
    # 🔹 Manual fallback numeric price computation
    # ───────────────────────────────
    def compute_price_action(self, item_info, prediction):
        stock = item_info["stock_qty"]
        avg_price = item_info.get("avg_price", 10.0)
        pred = prediction.get("future_demand", 0)
        spoilage = item_info.get("spoilage_risk", 0.05)

        if pred > stock:
            action = "increase_price"
            delta_pct = min(0.25, (pred - stock) / max(1, stock) * 0.1 + 0.05)
        elif stock > pred * 1.5 or spoilage > 0.3:
            action = "decrease_price"
            delta_pct = min(0.4, max(0.05, spoilage * 0.5 + (stock - pred) / max(1, stock) * 0.05))
        else:
            action = "keep_price"
            delta_pct = 0.0

        suggested_price = round(avg_price * (1 + delta_pct if action == "increase_price" else 1 - delta_pct), 2)
        return {"price_action": action, "delta_pct": round(delta_pct, 3), "suggested_price": suggested_price}

    # ───────────────────────────────
    # 🔹 Rationale via LLM (safe)
    # ───────────────────────────────
    def rationale_via_llm(self, item, item_info, prediction, decision):
        prompt = (
            f"You are a senior retail pricing analyst.\n\n"
            f"Item: {item}\n"
            f"Item info: {json.dumps(item_info)}\n"
            f"Prediction: {json.dumps(prediction)}\n"
            f"Decision: {json.dumps(decision)}\n\n"
            "Respond ONLY with JSON: {'rationale': str, 'confidence': float, 'risks': str}"
        )

        resp = self.call_ollama_safe(prompt, max_tokens=150)
        try:
            raw_text = (
                resp.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
                .strip("```json")
                .strip("```")
            )
            parsed = json.loads(raw_text)
            return parsed
        except Exception:
            return {"rationale": f"Auto decision: {decision['price_action']}.", "confidence": 0.6, "risks": "Model uncertainty"}

    # ───────────────────────────────
    # 🔹 Main runner for all items
    # ───────────────────────────────
    def run(self, analysis_output, prediction_output):
        analysis = analysis_output["analysis"]
        predictions = prediction_output["predictions"]
        decisions = {}

        for item, item_info in analysis.items():
            pred = predictions.get(item, {"future_demand": 0})

            # 1️⃣ Try LLM price suggestion
            llm_price_data = self.get_price_from_llm(item, item_info, pred)
            time.sleep(self.per_item_delay)

            if llm_price_data and isinstance(llm_price_data.get("suggested_price"), (int, float)):
                suggested_price=llm_price_data.get("suggested_price")
                avg_price = item_info.get("avg_price", 0)

                if suggested_price > avg_price:
                    price_action = "Increase"
                elif suggested_price < avg_price:
                    price_action = "Decrease"
                else:
                    price_action = "No Change"
                decision = {
                    "source": "LLM",
                    "price_action": price_action,
                    "delta_pct": round((suggested_price - item_info.get("avg_price", 10.0)) / max(0.01, item_info.get("avg_price", 10.0)), 3),
                    "suggested_price": suggested_price,
                }
            else:
                # Fallback to manual computation
                decision = self.compute_price_action(item_info, pred)
                decision["source"] = "manual"

            # 2️⃣ Always get rationale (safe)
            rationale = self.rationale_via_llm(item, item_info, pred, decision)
            # rationale=None
            # if llm_price_data and isinstance(llm_price_data.get("rationale"),(str)):
            #     rationale = llm_price_data["rationale"]

            time.sleep(self.per_item_delay)

            decisions[item] = {
                "decision": decision,
                "rationale": rationale,
            }

        return {"decisions": decisions}
