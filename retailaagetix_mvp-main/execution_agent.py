
# execution_agent.py

import json
from utils.ollama_client import call_ollama
from datetime import datetime


class ExecutionAgent:
    def __init__(self):
        self.price_table = {}  # item -> current price
        self.procurement_orders = []

    def apply_price_change(self, item, decision):
        new_price = decision.get("suggested_price", 0)
        self.price_table[item] = {
            "price": new_price,
            "updated_at": datetime.utcnow().isoformat(),
        }

    def create_procurement_order(self, item, qty):
        order = {
            "item": item,
            "qty": qty,
            "created_at": datetime.utcnow().isoformat(),
        }
        self.procurement_orders.append(order)
        return order

    def build_notification(self, item, decision, rationale):
        prompt = (
            f"Create a short notification for store managers about the action on '{item}'.\n"
            f"Decision: {json.dumps(decision)}\n"
            f"Rationale: {json.dumps(rationale)}\n"
            "Keep it one short paragraph."
        )
        resp = call_ollama(prompt, max_tokens=120)
        return resp.get("text", "").strip()

    def run(self, decision_output, analysis_output=None, prediction_output=None):
        logs = []

        # 🔐 Safety check to prevent KeyError
        if not decision_output or "decisions" not in decision_output:
            return {
                "execution_log": [],
                "price_table": self.price_table,
                "procurement_orders": self.procurement_orders,
                "warning": "No decisions provided to ExecutionAgent",
            }

        for item, info in decision_output["decisions"].items():
            decision = info.get("decision", {})
            rationale = info.get("rationale", {})

            self.apply_price_change(item, decision)

            # procurement rule
            pred = (
                prediction_output.get("predictions", {}).get(item, {"future_demand": 0})
                if prediction_output
                else {"future_demand": 0}
            )
            stock = (
                analysis_output.get("analysis", {}).get(item, {}).get("stock_qty", 0)
                if analysis_output
                else 0
            )

            purchase_qty = max(0, pred["future_demand"] - stock)
            order = None
            if purchase_qty > 0:
                order = self.create_procurement_order(item, purchase_qty)

            # notif = self.build_notification(item, decision, rationale)
            # print('notify values',notif)
            logs.append(
                {
                    "item": item,
                    "price_action": decision.get("price_action"),
                    "new_price": self.price_table[item]["price"],
                    "procurement_order": order,

                }
            )

        return {
            "execution_log": logs,
            "price_table": self.price_table,
            "procurement_orders": self.procurement_orders,
        }
