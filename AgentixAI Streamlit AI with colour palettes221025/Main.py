"""
RetailAgentix – Enhanced Edition
All original behaviour preserved, only additive UI / I18n layer added.
Author: <your-name>
"""

import json
import os
import zoneinfo
from datetime import datetime
from typing import Dict, Any
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import time
from datetime import datetime
# -------------- NEW EXTERNAL HELPERS --------------
from utils.css_loader import load_css
from components.settings_dialog import settings_dialog
from i18n.formats import localize, UICONF
from i18n.formats import tz_now
import streamlit.components.v1 as components
from datetime import datetime
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

LABELS: Dict[str, str] = json.load(open("i18n/ui_text.json",encoding="utf-8"))

# -------------- BACKEND --------------
BACKEND_URL = "http://localhost:8001"
import matplotlib.pyplot as plt
# -------------- SESSION DEFAULTS --------------
if "palette" not in st.session_state:
    st.session_state.update({
        "palette": "blue",
        "theme": "light",
        "timezone": "UTC",
        "currency": "INR",
        "currency_symbol": "₹",
        "datefmt": "DD/MM/YYYY",
        "units": "metric"
    })

# Initialize human-in-the-loop states
if "awaiting_approval" not in st.session_state:
    st.session_state.awaiting_approval = False
if "current_run_id" not in st.session_state:
    st.session_state.current_run_id = None
if "modified_decisions" not in st.session_state:
    st.session_state.modified_decisions = {}
if "original_decisions" not in st.session_state:
    st.session_state.original_decisions = {}

# -------------- CSS + THEME --------------
load_css()
palette_map = {
    "blue":   ("#1a42f8", "#4ba29c"),
    "green":  ("#10b981", "#34d399"),
    "orange": ("#f59e0b", "#fbbf24"),
    "slate":  ("#475569", "#94a3b8")
}
st.markdown(f"""
<style>
/* ---------- palette & theme vars ---------- */
:root{{
  --grad-start : {palette_map[st.session_state.palette][0]};
  --grad-end   : {palette_map[st.session_state.palette][1]};
  --accent     : {palette_map[st.session_state.palette][0]};
  --accent-dark: {palette_map[st.session_state.palette][1]};
  --bg         : {'#f8fafc' if st.session_state.theme=='light' else '#0f172a'};
  --card       : {'#ffffff' if st.session_state.theme=='light' else '#1e293b'};
  --text       : {'#1f2937' if st.session_state.theme=='light' else '#e2e8f0'};
  --border     : {'#e5e7eb' if st.session_state.theme=='light' else '#334155'};
}}

/* ---------- buttons ---------- */
.stButton > button {{
    background:var(--accent) !important;
    border-color:var(--accent) !important;
}}
.stButton > button:hover {{
    background:var(--accent-dark) !important;
    border-color:var(--accent-dark) !important;
}}

/* ---------- tabs (shadow-root) ---------- */
/* selected tab background */
.stTabs [data-baseweb="tab"][aria-selected="true"] {{
    background-color:var(--accent) !important;
    color:#fff !important;
}}
/* inactive tab hover */
.stTabs [data-baseweb="tab"]:hover {{
    background-color:var(--soft) !important;
    color:var(--text) !important;
}}
/* bottom indicator bar */
.stTabs [role="tablist"]::after {{
    background:linear-gradient(90deg, var(--grad-start) 0%, var(--grad-end) 100%) !important;
}}
/* links, radio, select highlights */
.st-dk, .st-eb, a, .st-bb, [data-testid="stRadio"] label:has(input:checked),
[data-testid="stSelectbox"] svg {{
    color:var(--accent) !important;
}}
section[data-testid="stSidebar"] [data-baseweb="button"] {{
    background:var(--accent) !important;
    border-color:var(--accent) !important;
}}
section[data-testid="stSidebar"] [data-baseweb="button"]:hover {{
    background:var(--accent-dark) !important;
    border-color:var(--accent-dark) !important;
}}

.exec-band {{
    background:linear-gradient(135deg, var(--grad-start) 0%, var(--grad-end) 100%);
    color:white;
    font-size:1.5rem;
    font-weight:700;
    margin:0 -2rem 1.5rem -2rem;   /* bleed into gutters */
    padding:1rem 2rem;
    text-align:center;
    box-shadow:0 4px 6px -1px rgba(0,0,0,.1);
}}
.exec-sep {{
    height: 6px;
    background: linear-gradient(90deg, var(--grad-start) 0%, var(--grad-end) 100%);
    margin: -1rem -2rem 1.5rem -2rem;   /* full-bleed, hug the band above */
}}

agent-summary {{
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-radius: 12px;
        padding: 16px;
        margin: 8px;
        text-align: center;
    }}

.exec-rainbow {{
    height: 15px;
    margin: -1rem -2rem 1.5rem -2rem;   /* full bleed */
    display: flex;
}}
.exec-rainbow span {{
    flex: 1 1 20%;
}}
/* palette colours */
.exec-rainbow span:nth-child(1) {{ background: var(--grad-start); }}
.exec-rainbow span:nth-child(2) {{ background: color-mix(in srgb, var(--grad-start) 75%, var(--grad-end)); }}
.exec-rainbow span:nth-child(3) {{ background: color-mix(in srgb, var(--grad-start) 50%, var(--grad-end)); }}
.exec-rainbow span:nth-child(4) {{ background: color-mix(in srgb, var(--grad-start) 25%, var(--grad-end)); }}
.exec-rainbow span:nth-child(5) {{ background: var(--grad-end); }}

/* mobile gutters */
@media (max-width: 768px) {{
    .exec-rainbow {{ margin: -1rem -1rem 1rem -1rem; }}
}}

section[data-testid="stSidebar"] .stButton > button,
section[data-testid="stSidebar"] button[class*="st-emotion"] {{
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    color: #fff !important;
}}
section[data-testid="stSidebar"] .stButton > button:hover,
section[data-testid="stSidebar"] button[class*="st-emotion"]:hover {{
    background: var(--accent-dark) !important;
    border-color: var(--accent-dark) !important;
}}

/* Human approval styles */
.approval-banner {{
    background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
    color: white;
    padding: 20px;
    border-radius: 10px;
    margin: 20px 0;
    text-align: center;
    font-size: 18px;
    font-weight: 600;
    box-shadow: 0 4px 12px rgba(251, 191, 36, 0.3);
    animation: pulse 2s infinite;
}}

@keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.8; }}
}}

.editable-cell {{
    background: #fffbeb;
    border: 2px solid #fbbf24;
    padding: 4px 8px;
    border-radius: 4px;
}}

</style>
""", unsafe_allow_html=True)

# -------------- SETTINGS GEAR (TOP-RIGHT) --------------
settings_dialog()

class RetailAgentixClient:

    """Identical to the uploaded version – only comments removed for brevity."""
    def __init__(self):
            self.current_run_id = None
    def start_pipeline_sync(self, initial_state=None):
            if initial_state is None:
                initial_state = {}
            try:
                response = requests.post(
                    f"{BACKEND_URL}/agui/run",
                    json=initial_state,
                    headers={
                        'Accept': 'text/event-stream',
                        'Cache-Control': 'no-cache'
                    },
                    stream=True,
                    timeout=(10, 3600)  # Increased timeout for human approval
                )
                if response.status_code == 200:
                    return self.process_sse_stream_simple(response)
                else:
                    return {'error': f"HTTP {response.status_code}: {response.text}"}
            except requests.exceptions.Timeout:
                return {'error': 'Request timed out - backend may still be processing'}
            except requests.exceptions.ConnectionError as e:
                return {'error': f"Connection failed: {e}"}
            except Exception as e:
                return {'error': f"Unexpected error: {e}"}

    def process_sse_stream_simple(self, response):
        events, final_state, status, error_message = [], None, 'running', None
        awaiting_human_approval = False
        
        try:
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                line = line.strip()
                if line.startswith('data: '):
                    data_content = line[6:].strip()
                    if not data_content:
                        continue
                    try:
                        event_data = json.loads(data_content)
                        events.append({'timestamp': datetime.now(), 'data': event_data})
                        event_type = event_data.get('type')
                        
                        # Handle human approval required
                        if event_type == 'HUMAN_APPROVAL_REQUIRED':
                            awaiting_human_approval = True
                            st.session_state.awaiting_approval = True
                            st.session_state.original_decisions = event_data.get('data', {})
                            # Store run_id for approval endpoint
                            if 'run_id' in event_data:
                                st.session_state.current_run_id = event_data['run_id']
                            # Don't break - return immediately to show UI
                            print(f"[Frontend] HUMAN_APPROVAL_REQUIRED received, returning to show UI")
                            break  # Break to show UI, but mark as awaiting
                            
                        elif event_type == 'HUMAN_APPROVAL_RECEIVED':
                            awaiting_human_approval = False
                            st.session_state.awaiting_approval = False
                            
                        elif event_type == 'RUN_FINISHED':
                            final_state = event_data.get('result', {})
                            status = 'completed'
                            break
                            
                        elif event_type == 'RUN_ERROR':
                            status = 'error'
                            error_message = event_data.get('error', 'Unknown error')
                            break
                            
                    except json.JSONDecodeError as e:
                        events.append({'timestamp': datetime.now(),
                                      'data': {'type': 'PARSE_ERROR', 'error': str(e),
                                               'raw_content': data_content[:100]}})
                        continue
        except Exception as e:
            status = 'error'
            error_message = f"Stream processing failed: {str(e)}"
        
        # Build partial state if awaiting approval
        if awaiting_human_approval:
            partial_state = {}
            for event in events:
                event_data = event.get('data', {})
                if event_data.get('type') == 'STEP_FINISHED':
                    step = event_data.get('step')
                    data = event_data.get('data')
                    if step and data:
                        partial_state[step] = data
            final_state = partial_state
            status = 'awaiting_approval'
            
        return {
            'events': events,
            'final_state': final_state,
            'status': status,
            'error_message': error_message,
            'awaiting_human_approval': awaiting_human_approval
        }

    def approve_decisions(self, run_id: str, decisions: dict):
        """Send approved decisions to backend"""
        try:
            response = requests.post(
                f"{BACKEND_URL}/agui/run/{run_id}/approve",
                json={"decisions": decisions},
                timeout=10
            )
            if response.status_code == 200:
                return {'status': 'success'}
            else:
                return {'error': f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {'error': f"Failed to approve decisions: {e}"}

    def get_thread_state(self, run_id: str):
        """Fetch latest thread/run state from backend."""
        try:
            response = requests.get(f"{BACKEND_URL}/agui/run/threads/{run_id}/state", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return None
        except Exception:
            return None

        """Send approved decisions to backend"""
        try:
            response = requests.post(
                f"{BACKEND_URL}/agui/run/{run_id}/approve",
                json={"decisions": decisions},
                timeout=10
            )
            if response.status_code == 200:
                return {'status': 'success'}
            else:
                return {'error': f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {'error': f"Failed to approve decisions: {e}"}



def display_ingestion_data(ingestion_data: Dict[str, Any]):
    # [KEEP YOUR ORIGINAL CODE - NO CHANGES]
    if not ingestion_data:
        st.warning(LABELS["msg_no_data"])
        return
    
    category = st.session_state.get("category", "perishable")
    print("catagory",category)
    # 🔹 Extract the right dataset
    if category == "bakery" and "bakery" in ingestion_data:
        ingestion_data = ingestion_data["bakery"]
        st.info("🧁 Displaying **Bakery** ingestion data")
    elif category == "perishable":
        st.info("🥦 Displaying **Perishables** ingestion data")
    else:
        st.warning("⚠️ Unknown category – showing default data")

    st.markdown('<div class="section-header">📥 Data Ingestion Results</div>', unsafe_allow_html=True)
    tabs = st.tabs([LABELS["tab_sales_stock"], LABELS["tab_weather"],
                   LABELS["tab_calendar"], LABELS["tab_notes"]])
    with tabs[0]:
        st.markdown('<div class="subsection-header">📊 Sales History & Stock Data</div>', unsafe_allow_html=True)
        sales_data, stock_data = [], []
        if 'sales_history' in ingestion_data:
            for item_name, records in ingestion_data['sales_history'].items():
                if isinstance(records, list):
                    for r in records:
                        if isinstance(r, dict):
                            r_with_item = r.copy()
                            r_with_item['item'] = item_name
                            sales_data.append(r_with_item)
        if 'stock' in ingestion_data:
            for item_name, records in ingestion_data['stock'].items():
                if isinstance(records, list):
                    for r in records:
                        if isinstance(r, dict):
                            r_with_item = r.copy()
                            r_with_item['item'] = item_name
                            stock_data.append(r_with_item)
        col1, col2 = st.columns(2)
        with col1:
            if sales_data:
                st.markdown(f"**📈 {LABELS['tbl_item']}**")
                df_sales = pd.DataFrame(sales_data)
                st.dataframe(df_sales, use_container_width=True, hide_index=True)
            else:
                st.info(LABELS["msg_no_data"])
        with col2:
            if stock_data:
                st.markdown(f"**📦 {LABELS['tbl_item']}**")
                df_stock = pd.DataFrame(stock_data)
                st.dataframe(df_stock, use_container_width=True, hide_index=True)
            else:
                st.info(LABELS["msg_no_data"])
    with tabs[1]:
        st.markdown('<div class="subsection-header">🌤️ Weather Information</div>', unsafe_allow_html=True)
        if 'weather' in ingestion_data:
            weather = ingestion_data['weather']
            col1, col2, col3 = st.columns(3)
            unit_sym = UICONF["units"][st.session_state.units]["temp"]
            with col1:
                st.metric(LABELS["metric_temperature"], f"{weather.get('temperature', 'N/A')}{unit_sym}")
            with col2:
                st.metric(LABELS["metric_wind"], f"{weather.get('windspeed', 'N/A')} {UICONF['units'][st.session_state.units]['speed']}")
            with col3:
                weather_code = weather.get('weathercode', 'N/A')
                weather_desc = {0: "Clear", 1: "Mostly Clear", 2: "Partly Cloudy", 3: "Overcast"}.get(weather_code, "Unknown")
                st.metric(LABELS["metric_weather"], weather_desc)
        if 'weather_forecast' in ingestion_data:
            st.markdown("**📈 Weather Forecast**")
            forecast_data = []
            for forecast in ingestion_data['weather_forecast']:
                forecast_data.append({
                    'Date': forecast.get('date', 'N/A'),
                    f'Max Temp ({unit_sym})': forecast.get('temp_max', 'N/A'),
                    f'Min Temp ({unit_sym})': forecast.get('temp_min', 'N/A'),
                })
            if forecast_data:
                df_forecast = pd.DataFrame(forecast_data)
                st.dataframe(df_forecast, use_container_width=True, hide_index=True)
    with tabs[2]:
        st.markdown('<div class="subsection-header">📅 Upcoming Events</div>', unsafe_allow_html=True)
        if 'calendar_events' in ingestion_data:
            events_data = []
            for event in ingestion_data['calendar_events'][:20]:
                events_data.append({
                    'Date': (
                        datetime.strptime(event.get('date', ''), "%Y-%m-%dT%H:%M:%S%z").strftime(UICONF["datefmt"][st.session_state.datefmt])
                        if 'T' in event.get('date', '') and ('+' in event.get('date', '') or '-' in event.get('date', ''))
                        else datetime.strptime(event.get('date', ''), "%Y-%m-%d").strftime(UICONF["datefmt"][st.session_state.datefmt])
                    ) if event.get('date', '') else 'N/A',
                    'Event': event.get('event', 'N/A'),
                    'Impact': event.get('impact', 'N/A'),
                    })
            if events_data:
                df_events = pd.DataFrame(events_data)
                st.dataframe(df_events, use_container_width=True, hide_index=True)
                if len(ingestion_data['calendar_events']) > 20:
                    st.info(f"Showing first 20 of {len(ingestion_data['calendar_events'])} total events")
        else:
            st.info(LABELS["msg_no_data"])
    with tabs[3]:
        st.markdown('<div class="subsection-header">📝 User Notes & Insights</div>', unsafe_allow_html=True)
        if 'user_notes' in ingestion_data:
            notes = ingestion_data['user_notes']
            if isinstance(notes, list) and len(notes) > 0:
                for note in notes:
                    st.write(f"- {note}")
            else:
                st.write('No notes available')
        else:
            st.info(LABELS["msg_no_data"])


def display_analysis_data(analysis_data: Dict[str, Any]):
    # [KEEP YOUR ORIGINAL CODE - NO CHANGES]
    if not analysis_data:
        st.warning(LABELS["msg_no_data"])
        return
    st.markdown('<div class="section-header">📊 Analysis Results</div>', unsafe_allow_html=True)
    if 'analysis' in analysis_data:
        analysis = analysis_data['analysis']
        analysis_table = []
        for item, data in analysis.items():
            analysis_table.append({
                LABELS["tbl_item"]: item.capitalize(),
                LABELS["tbl_stock_qty"]: data.get('stock_qty', 0),
                LABELS["tbl_avg_daily"]: f"{data.get('avg_daily_sales', 0):.2f}",
                LABELS["tbl_spoilage"]: f"{data.get('spoilage_risk', 0) * 100:.1f}%",
                LABELS["tbl_price"]: f"{st.session_state.currency_symbol}{float(data.get('avg_price', 0)):.2f}",
            })
        if analysis_table:
            df_analysis = pd.DataFrame(analysis_table)
            st.dataframe(df_analysis, use_container_width=True, hide_index=True)
            col1, col2 = st.columns(2)
            with col1:
                items = [row[LABELS["tbl_item"]] for row in analysis_table]
                stock_qty = [data.get('stock_qty', 0) for data in analysis.values()]
                daily_sales = [data.get('avg_daily_sales', 0) for data in analysis.values()]
                fig1 = go.Figure()
                fig1.add_trace(go.Bar(name='Stock Quantity', x=items, y=stock_qty, marker_color='#3b82f6'))
                fig1.add_trace(go.Bar(name='Avg Daily Sales', x=items, y=daily_sales, marker_color='#10b981'))
                fig1.update_layout(title="Stock vs Daily Sales", xaxis_title="Items", yaxis_title="Quantity",
                                  barmode='group', template='plotly_white')
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                spoilage_risks = [data.get('spoilage_risk', 0) * 100 for data in analysis.values()]
                fig2 = px.pie(values=spoilage_risks, names=items, title="Spoilage Risk Distribution")
                st.plotly_chart(fig2, use_container_width=True)
                st.markdown("<div style='text-align: center; color: #888;'>Spoilage risk percentage. By Simple heuristic method</div>", unsafe_allow_html=True)
    else:
        st.info(LABELS["msg_no_data"])

def display_prediction_data(prediction_data: Dict[str, Any]):
    # [KEEP YOUR ORIGINAL CODE - NO CHANGES]
    if not prediction_data:
        st.warning(LABELS["msg_no_data"])
        return
    st.markdown('<div class="section-header">🔮 Prediction Results</div>', unsafe_allow_html=True)
    if 'predictions' in prediction_data:
        predictions = prediction_data['predictions']
        prediction_table = []
        for category, data in predictions.items():
            prediction_table.append({
                'Category': category.replace('_', ' ').title(),
                'Future Demand': data.get('future_demand', 0),
                'Prediction Method': data.get('method', 'N/A'),
            })
        if prediction_table:
            df_predictions = pd.DataFrame(prediction_table)
            st.dataframe(df_predictions, use_container_width=True, hide_index=True)
            categories = [row['Category'] for row in prediction_table]
            demands = [row['Future Demand'] for row in prediction_table]
            if any(d > 0 for d in demands):
                fig = px.bar(x=categories, y=demands, title="Predicted Future Demand",
                            labels={'x': 'Category', 'y': 'Demand'},
                            color=demands, color_continuous_scale='viridis')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(LABELS["msg_no_data"])
    else:
        st.info(LABELS["msg_no_data"])

def display_decision_data_with_approval(decision_data: Dict[str, Any], analysis_data: Dict[str, Any]):
    """Display decision data with editable inputs for human-in-the-loop approval"""
    if (not decision_data or not decision_data.get("decisions")):
        final_state = st.session_state.get("pipeline_result", {}).get("final_state", {})
        human_decisions = final_state.get("human_approved_decisions", {})
        if human_decisions:
            decision_data = {"decisions": human_decisions}
    if not decision_data or not decision_data.get("decisions"):
            st.warning(LABELS["msg_no_data"])
            return
    
    st.markdown('<div class="section-header">🎯 Decision Agent Results</div>', unsafe_allow_html=True)
    
    # Show approval banner if awaiting approval
    if st.session_state.awaiting_approval:
        st.markdown("""
        <div class="approval-banner">
            ⚠️ HUMAN APPROVAL REQUIRED - Please review and modify pricing decisions below
        </div>
        """, unsafe_allow_html=True)
    
    decisions = decision_data['decisions']
    analysis = analysis_data.get('analysis', {})

    # Initialize modified_decisions if not present
    if not st.session_state.modified_decisions:
        st.session_state.modified_decisions = {}
        for item in decisions.keys():
            st.session_state.modified_decisions[item] = {
                'new_price': float(decisions[item]['decision'].get('suggested_price', 0)),
                'action': decisions[item]['decision'].get('price_action', 'no_change')
            }

    decision_table = []

    for item, decision_info in decisions.items():
        decision = decision_info.get('decision', {})
        rationale_raw = decision_info.get('rationale', '')
        
        # Parse rationale
        confidence = "N/A"
        reasoning = "No reasoning provided"
        risks = "No risks identified"

        if isinstance(rationale_raw, str):
            rationale_raw = rationale_raw.strip()
            if rationale_raw.startswith("{") and rationale_raw.endswith("}"):
                try:
                    rationale_data = json.loads(rationale_raw)
                    confidence = rationale_data.get("confidence", confidence)
                    reasoning = rationale_data.get("rationale", reasoning)
                    risks = rationale_data.get("risks", risks)
                except Exception:
                    reasoning = rationale_raw
            else:
                reasoning = rationale_raw
        elif isinstance(rationale_raw, dict):
            confidence = rationale_raw.get("confidence", confidence)
            reasoning = rationale_raw.get("rationale", reasoning)
            risks = rationale_raw.get("risks", risks)

        confidence = float(confidence) if confidence != "N/A" else 0
        reasoning = str(reasoning)
        risks = str(risks)

        current_price = float(analysis.get(item, {}).get('avg_price', 0.0))
        
        # Use modified price if available
        if st.session_state.awaiting_approval:
            suggested_price = st.session_state.modified_decisions.get(item, {}).get('new_price', 
                                float(decision.get('suggested_price', 0)))
            action = st.session_state.modified_decisions.get(item, {}).get('action',
                        decision.get('price_action', 'no_change'))
        else:
            suggested_price = float(decision.get('suggested_price', 0))
            action = decision.get('price_action', 'no_change')
        
        price_change = abs(current_price - suggested_price)
        
        raw_action = action.lower().strip()
        if 'increase' in raw_action:
            action_clean = 'Increase'
        elif 'decrease' in raw_action:
            action_clean = 'Decrease'
        else:
            action_clean = raw_action.title() or 'No Action'

        decision_table.append({
            LABELS["tbl_item"]: item.capitalize(),
            'Current Price': f"{st.session_state.currency_symbol}{current_price:.2f}",
            'Suggested Price': suggested_price,
            'Price Change': f"{st.session_state.currency_symbol}{price_change:.2f}",
            'Action': action_clean,
            'Change %': f"{decision.get('delta_pct', 0) * 100:.1f}%",
            'AI Confidence': confidence,
            'Reasoning': reasoning,
            'Risk Assessment': risks
        })
    
    if decision_table:
        # Display editable table if awaiting approval
        if st.session_state.awaiting_approval:
            st.markdown("### ✏️ Edit Pricing Decisions Below")
            st.info("💡 Modify the suggested prices and actions as needed, then click 'Submit Approved Decisions'")
            
            # Create editable inputs for each item
            for idx, row in enumerate(decision_table):
                with st.expander(f"📦 {row[LABELS['tbl_item']]} - Edit Decision", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    
                    item_key = list(decisions.keys())[idx]
                    
                    with col1:
                        st.metric("Current Price", row['Current Price'])
                    
                    with col2:
                        new_price = st.number_input(
                            "New Price",
                            min_value=0.0,
                            value=float(row['Suggested Price']),
                            step=0.01,
                            key=f"price_{item_key}",
                            help="Modify the suggested price"
                        )
                        st.session_state.modified_decisions[item_key]['new_price'] = new_price
                    
                    with col3:
                        action_options = ['Increase', 'Decrease', 'No Change']
                        current_action = row['Action']
                        default_idx = action_options.index(current_action) if current_action in action_options else 2
                        
                        selected_action = st.selectbox(
                            "Action",
                            action_options,
                            index=default_idx,
                            key=f"action_{item_key}",
                            help="Choose the pricing action"
                        )
                        st.session_state.modified_decisions[item_key]['action'] = selected_action
                    
                    st.markdown("**🤖 AI Reasoning:**")
                    st.write(row['Reasoning'])
                    st.markdown("**⚠️ Risk Assessment:**")
                    st.write(row['Risk Assessment'])
            
            # FIXED: Proper indentation for submit button
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("✅ Submit Approved Decisions", type="primary", use_container_width=True):
                    # Prepare approved decisions
                    approved_decisions = {'decisions': {}}
                    
                    for item_key, modifications in st.session_state.modified_decisions.items():
                        original_decision = decisions[item_key]
                        approved_decisions['decisions'][item_key] = {
                            'decision': {
                                'suggested_price': modifications['new_price'],
                                'price_action': modifications['action'],
                                'delta_pct': (
                                    (modifications['new_price'] - float(analysis.get(item_key, {}).get('avg_price', 0))) / 
                                    float(analysis.get(item_key, {}).get('avg_price', 1))
                                ) if float(analysis.get(item_key, {}).get('avg_price', 0)) > 0 else 0
                            },
                            'rationale': original_decision.get('rationale', '')
                        }
                    
                    # Send to backend
                    client = RetailAgentixClient()
                    result = client.approve_decisions(st.session_state.current_run_id, approved_decisions['decisions'])
                    
                    if 'error' not in result:
                        st.success("✅ Decisions approved! Continuing pipeline to execution...")
                        st.session_state.awaiting_approval = False
                        st.session_state.approved_decisions_display = approved_decisions['decisions']

                        # ✅ Poll backend for updated execution results
                        client = RetailAgentixClient()
                        final_state = None
                        timeout_seconds = 45
                        poll_interval = 2.0
                        elapsed = 0.0

                        with st.spinner("⚡ Waiting for execution results from backend..."):
                            while elapsed < timeout_seconds:
                                try:
                                    thread_resp = client.get_thread_state(st.session_state.current_run_id)
                                    if thread_resp and 'state' in thread_resp:
                                        state = thread_resp['state']
                                        # Wait until execution appears
                                        if 'execution' in state and state['execution']:
                                            final_state = state
                                            break
                                except Exception as e:
                                    print(f"Polling error: {e}")
                                time.sleep(poll_interval)
                                elapsed += poll_interval

                        if final_state:
                            # ✅ Update the state where main() expects it
                            st.session_state['pipeline_result'] = {
                                'status': 'completed',
                                'final_state': final_state
                            }
                            st.session_state.awaiting_approval = False
                            st.success("⚡ Execution completed successfully! Refreshing dashboard...")
                            st.rerun()
                        else:
                            st.warning("⚠️ Backend did not return execution results yet. Please refresh manually in a few seconds.")
        else:
            # Display non-editable table
            display_df = pd.DataFrame(decision_table)
            display_df['Suggested Price'] = display_df['Suggested Price'].apply(
                lambda x: f"{st.session_state.currency_symbol}{float(x):.2f}"
            )
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Charts
            col1, col2 = st.columns(2)
            with col1:
                items = [row[LABELS["tbl_item"]] for row in decision_table]
                current_prices = [float(row['Current Price'].replace(st.session_state.currency_symbol, '')) for row in decision_table]
                suggested_prices = [float(row['Suggested Price']) if isinstance(row['Suggested Price'], (int, float)) 
                                   else float(str(row['Suggested Price']).replace(st.session_state.currency_symbol, '')) 
                                   for row in decision_table]
                fig1 = go.Figure()
                fig1.add_trace(go.Bar(name='Current Price', x=items, y=current_prices, marker_color='#ef4444'))
                fig1.add_trace(go.Bar(name='Suggested Price', x=items, y=suggested_prices, marker_color='#10b981'))
                fig1.update_layout(title="Price Comparison", xaxis_title="Items", yaxis_title=f"Price ({st.session_state.currency})",
                                  barmode='group', template='plotly_white')
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                actions = [row['Action'] for row in decision_table]
                action_counts = {}
                for action in actions:
                    action_counts[action] = action_counts.get(action, 0) + 1
                fig2 = px.pie(values=list(action_counts.values()), names=list(action_counts.keys()), title="Recommended Actions")
                st.plotly_chart(fig2, use_container_width=True)


def display_decision_data(decision_data: Dict[str, Any], analysis_data: Dict[str, Any]):
    """Wrapper for backward compatibility"""
    display_decision_data_with_approval(decision_data, analysis_data)


#################Working with two type of data##################

import plotly.express as px
import plotly.graph_objects as go

def display_execution_data(execution_data: Dict[str, Any],
                           ingestion_data: Dict[str, Any],
                           analysis_data: Dict[str, Any],
                           prediction_data: Dict[str, Any],
                           decisions: Dict[str, Any]):

    if not execution_data:
        st.warning(LABELS["msg_no_data"])
        return

    st.markdown('<div class="section-header">⚡ Execution Results</div>', unsafe_allow_html=True)

    # =====================================================================
    # DATA SOURCE SELECTION BASED ON CATEGORY
    # =====================================================================
    # Determine which data source to use based on category
    is_bakery = st.session_state.get("category") == "bakery"
    
    if is_bakery:
        # Use bakery-specific data from ingestion_data["bakery"]
        # data_source = ingestion_data.get("bakery", {})
        data_source = ingestion_data.get("bakery", ingestion_data)
        print('data source',data_source)
        stock_data = data_source.get("stock", {})
        sales_history = data_source.get("sales_history", {})
        historical_inventory = data_source.get("historical_inventory", {})
    else:
        # Use regular perishables data structure
        data_source = ingestion_data
        stock_data = ingestion_data.get("stock", {})
        sales_history = ingestion_data.get("sales_history", {})
        historical_inventory = ingestion_data.get("historical_inventory", {})


    # =====================================================================
    # 🔧 Summary and Execution Log Display (Original Section)
    # =====================================================================
    st.markdown("<br>", unsafe_allow_html=True)
    if 'summary' in execution_data:
        summary = execution_data['summary']
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Decisions", summary.get('total_decisions_reviewed', 0))
        with col2:
            st.metric("Executed", summary.get('approved_and_executed', 0))
        with col3:
            st.metric("Success Rate", f"{summary.get('execution_success_rate', 0) * 100:.1f}%")
        with col4:
            st.metric("Modified", summary.get('human_modification_count', 0))
    
    if 'execution_log' in execution_data:
        st.markdown('<div class="subsection-header">📋 Execution Log</div>', unsafe_allow_html=True)

        log_data = []
        for entry in execution_data['execution_log']:
            new_price = f"{st.session_state.currency_symbol} {entry.get('new_price', 'N/A')}"
            status_badge = entry.get('status', 'unknown')
            if status_badge == 'success':
                badge_html = '<span class="status-badge-success">✅ Success</span>'
            elif status_badge == 'error':
                badge_html = '<span class="status-badge-error">❌ Error</span>'
            else:
                badge_html = '<span class="status-badge-warning">⚠️ Warning</span>'

            log_data.append({
                LABELS["tbl_item"]: entry.get('item', 'N/A').capitalize(),
                'Action': entry.get('price_action', 'N/A').replace('_', ' ').title(),
                'New Price': new_price,
                'Procurement Order': entry.get('procurement_order', 'N/A')
            })

        if log_data:
            df_log = pd.DataFrame(log_data)

            st.markdown("""
                <style>
                table {
                    width: 100% !important;
                    border-collapse: collapse !important;
                    background-color: white !important;
                    color: black !important;
                    font-family: Arial, sans-serif !important;
                    font-size: 14px !important;
                }
                th, td {
                    border: 1px solid #333 !important;
                    padding: 8px 10px !important;
                    text-align: left !important;
                    color: black !important;
                }
                th {
                    background-color: #f2f2f2 !important;
                    font-weight: 600 !important;
                }
                tr:nth-child(even) {
                    background-color: #fafafa !important;
                }
                </style>
            """, unsafe_allow_html=True)

            st.markdown(df_log.to_html(escape=False, index=False), unsafe_allow_html=True)

            # Display Bar
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**🏪 Display Board: Pushed Display data - Click to view display board:**")

            num_items = len(execution_data['execution_log'])
            num_cols = min(4, num_items)
            
            st.markdown("""
                <style>
                div.stButton > button[kind="secondary"],
                div.stButton > button {
                    color: #fff !important;
                    border: 1px solid #222 !important;
                    border-radius: 6px !important;
                    padding: 0.5rem 1rem !important;
                    font-weight: 600 !important;
                    box-shadow: none !important;
                    transition: background-color 0.2s ease;
                }
                div.stButton > button:hover {
                    background-color: #555 !important;
                    color: #fff !important;
                    border-color: #111 !important;
                }
                </style>
            """, unsafe_allow_html=True)
            
            for row_start in range(0, num_items, num_cols):
                cols = st.columns(num_cols)
                for col_idx, idx in enumerate(range(row_start, min(row_start + num_cols, num_items))):
                    entry = execution_data['execution_log'][idx]
                    unique_key = f"display_btn_{row_start}_{col_idx}"
                    with cols[col_idx]:
                        if st.button(
                            f"🏪 {entry.get('item', 'Item').capitalize()}",
                            key=unique_key,
                            use_container_width=True
                        ):
                            show_led_display_popup(entry, execution_data)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**POS data display**")
        if st.button("📡 Send Execution Results to POS", use_container_width=True, key="send_pos_btn"):
            st.session_state.show_pos_popup = True

        if st.session_state.get("show_pos_popup", False):
            show_pos_push_popup(execution_data)
            st.session_state.show_pos_popup = False

        if st.button("💼 CFO Dashboard", use_container_width=True, key="cfo_dash_btn"):
            st.session_state.show_cfo_dashboard = True

        if st.session_state.get("show_cfo_dashboard", False):
            show_cfo_dashboard_popup(execution_data,ingestion_data,analysis_data,prediction_data,decisions)
            st.session_state.show_cfo_dashboard = False

        if st.session_state.get("category", "perishable") == "bakery":
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**📱 Mobile Notification Simulation (Bakery Deals)**")

            if st.button("📲 Push Bakery Notifications", use_container_width=True, key="push_bakery_notif_btn"):
                st.session_state.show_bakery_notification = True

            if st.session_state.get("show_bakery_notification", False):
                print("execution data",stock_data)
                show_bakery_notification_popup(execution_data,stock_data)
                st.session_state.show_bakery_notification = False


    if 'price_updates' in execution_data:
        st.markdown('<div class="subsection-header">💰 Price Updates</div>', unsafe_allow_html=True)
        price_data = []
        for item, update in execution_data['price_updates'].items():
            price_data.append({
                LABELS["tbl_item"]: item.capitalize(),
                'Old Price': f"{st.session_state.currency_symbol}{update.get('old_price', 0):.2f}",
                'New Price': f"{st.session_state.currency_symbol}{update.get('new_price', 0):.2f}",
                'Human Modified': 'Yes' if update.get('human_modified', False) else 'No',
                'Execution Time': datetime.fromtimestamp(update.get('execution_timestamp', 0)).strftime(
                    UICONF["datefmt"][st.session_state.datefmt] + " %H:%M:%S") if update.get('execution_timestamp') else 'N/A',
                'Notes': update.get('human_notes', 'N/A')[:50] + "..." if len(update.get('human_notes', '')) > 50 else update.get('human_notes', 'N/A')
            })
        if price_data:
            df_prices = pd.DataFrame(price_data)
            st.dataframe(df_prices, use_container_width=True, hide_index=True)

    if not ingestion_data:
        st.warning(LABELS["msg_no_data"])
        return


#############################################################################################

@st.dialog("🏪 Retail Display Board", width="large")
def show_led_display_popup(entry: Dict[str, Any], execution_data: Dict[str, Any]):
    """
    Display a LED-style retail board showing the product price
    """
    item_name = entry.get('item', 'Unknown Product').upper()
    new_price = entry.get('new_price', 0)
    price_action = entry.get('price_action', 'no_change')
    
    # Get old price for comparison if available
    old_price = None
    if 'price_updates' in execution_data:
        old_price = execution_data['price_updates'].get(
            entry.get('item'), {}
        ).get('old_price')
    
    # Determine LED color based on price action
    if price_action == 'Increase':
        led_color = '#00ff00'  # Green for price increase
        action_text = '▲ PRICE INCREASED'
    elif price_action == 'Decrease':
        led_color = '#ff0000'  # Red for price decrease
        action_text = '▼ PRICE REDUCED - SPECIAL OFFER'
    else:
        led_color = '#00ffff'  # Cyan for no change
        action_text = '● CURRENT PRICE'
    
    # Calculate price difference
    price_diff_text = ''
    if old_price and old_price != new_price:
        price_diff = new_price - old_price
        price_diff_percent = ((new_price - old_price) / old_price * 100) if old_price > 0 else 0
        if price_diff > 0:
            price_diff_text = f'    |    WAS: {st.session_state.currency_symbol}{old_price:.2f}    |    INCREASED: +{price_diff_percent:.1f}%'
        else:
            price_diff_text = f'    |    WAS: {st.session_state.currency_symbol}{old_price:.2f}    |    SAVE: {abs(price_diff_percent):.1f}%'
    
    # Create scrolling text
    scroll_speed = 15
    text_line = f'{action_text}    |||    {item_name}    |||    NOW: {st.session_state.currency_symbol}{new_price:.2f}{price_diff_text}    |||    '
    text_line = text_line * 3  # Repeat for smooth scrolling
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # LED Display Board
    st.markdown(f"""
        <style>
        @keyframes scroll {{
            0% {{ transform: translateX(0%); }}
            100% {{ transform: translateX(-33.33%); }}
        }}
        .led {{
            background: black;
            color: {led_color};
            font-family: 'Courier New', monospace;
            font-size: 38px;
            font-weight: bold;
            padding: 20px;
            border: 6px solid #222;
            border-radius: 8px;
            box-shadow: 0 0 30px {led_color}, inset 0 0 20px rgba(0,0,0,0.8);
            overflow: hidden;
            white-space: nowrap;
            text-transform: uppercase;
            margin: 20px 0;
        }}
        .scroll {{
            display: inline-block;
            animation: scroll {scroll_speed}s linear infinite;
        }}
        .led-info {{
            background: #1a1a1a;
            color: #888;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            padding: 10px;
            border: 2px solid #333;
            border-radius: 4px;
            margin-top: 10px;
        }}
        </style>
        
        <div class='led'>
            <div class='scroll'>{text_line}</div>
        </div>
        <div class='led-info'>
            ⏰ Last updated: {timestamp}
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Additional details in a simple format
    # col1, col2, col3 = st.columns(3)
    
    # with col1:
    #     st.markdown("### 📦 Procurement")
    #     procurement = entry.get('procurement_order')
    #     if procurement and isinstance(procurement, dict):
    #         st.info(f"**Order Qty:** {procurement.get('qty', 0)} units")
    #     else:
    #         st.info("No order")
    
    # with col2:
    #     st.markdown("### 📊 Status")
    #     status = entry.get('status', 'pending')
    #     if status == 'success':
    #         st.success("✅ Executed")
    #     elif status == 'error':
    #         st.error("❌ Error")
    #     else:
    #         st.warning("⚠️ Pending")
    
    # with col3:
    #     st.markdown("### 💰 Price Action")
    #     st.info(f"**Action:** {price_action.replace('_', ' ').title()}")
    
    # Close button
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("✖ Close Display", type="primary", use_container_width=True):
        st.rerun()


@st.dialog("POS Data Push", width="large")
def show_pos_push_popup(execution_data):
    """Enhanced popup to preview and send execution results to POS system"""

    # ----- Modern Header with Icon -----
    st.markdown("""
        <style>
        .pos-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            margin-bottom: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .pos-header h1 {
            margin: 0;
            font-size: 28px;
            font-weight: 700;
        }
        .pos-header p {
            margin: 8px 0 0 0;
            font-size: 14px;
            opacity: 0.9;
        }
        .info-card {
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
        }
        .info-card strong {
            color: #667eea;
        }
        .pos-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .pos-table thead {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .pos-table th {
            padding: 15px;
            text-align: left;
            font-weight: 600;
            font-size: 14px;
        }
        .pos-table td {
            padding: 12px 15px;
            border-bottom: 1px solid #e9ecef;
            color: #333;
        }
        .pos-table tbody tr:hover {
            background-color: #f8f9fa;
        }
        .pos-table tbody tr:last-child td {
            border-bottom: none;
        }
        .json-container {
            background: #1e1e1e;
            color: #d4d4d4;
            padding: 20px;
            border-radius: 12px;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            line-height: 1.6;
            max-height: 450px;
            overflow-y: auto;
            border: 2px solid #333;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }
        .json-container::-webkit-scrollbar {
            width: 8px;
        }
        .json-container::-webkit-scrollbar-track {
            background: #2d2d2d;
        }
        .json-container::-webkit-scrollbar-thumb {
            background: #667eea;
            border-radius: 4px;
        }
        .status-badge {
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }
        .status-success {
            background: #d4edda;
            color: #155724;
        }
        .status-warning {
            background: #fff3cd;
            color: #856404;
        }
        .status-error {
            background: #f8d7da;
            color: #721c24;
        }
        </style>
        
        <div class="pos-header">
            <h1>🧾 POS Data Push Preview</h1>
            <p>Review and confirm data before sending to Point of Sale system</p>
        </div>
    """, unsafe_allow_html=True)

    # ----- Summary Info Cards -----
    summary = execution_data.get("summary", {})
    if summary:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Total Items", summary.get('total_decisions_reviewed', 0))
        with col2:
            st.metric("✅ Executed", summary.get('approved_and_executed', 0))
        with col3:
            success_rate = summary.get('execution_success_rate', 0) * 100
            st.metric("📈 Success Rate", f"{success_rate:.1f}%")

    # ----- POS Metadata Card -----
    timestamp = datetime.utcnow().isoformat() + "Z"
    st.markdown(f"""
        <div class="info-card">
            <strong>📍 Store ID:</strong> STORE_101 &nbsp;&nbsp;|&nbsp;&nbsp;
            <strong>💱 Currency:</strong> {st.session_state.currency_symbol} &nbsp;&nbsp;|&nbsp;&nbsp;
            <strong>🤖 Executed By:</strong> RetailAgentix_AI &nbsp;&nbsp;|&nbsp;&nbsp;
            <strong>⏰ Timestamp:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    """, unsafe_allow_html=True)

    # ----- Construct Payload -----
    payload = {
        "store_id": "STORE_101",
        "execution_timestamp": timestamp,
        "currency": f"{st.session_state.currency_symbol}",
        "executed_by": "RetailAgentix_AI",
        "execution_summary": summary,
        "price_updates": execution_data.get("execution_log", [])
    }

    st.markdown("---")

    # ----- Toggle between Table & JSON -----
    view_mode = st.radio(
        "📊 **Display Mode:**",
        ["📋 Table View", "💻 JSON View"],
        horizontal=True,
        key="pos_view_toggle"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if view_mode == "📋 Table View":
        # Enhanced table view with better styling
        if "execution_log" in execution_data and execution_data["execution_log"]:
            # Build the complete HTML with styles included
            table_html = """
            <style>
            .pos-table-container {
                width: 100%;
                overflow-x: auto;
            }
            .pos-table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                background: white;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            .pos-table thead {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .pos-table th {
                padding: 15px;
                text-align: left;
                font-weight: 600;
                font-size: 14px;
            }
            .pos-table td {
                padding: 12px 15px;
                border-bottom: 1px solid #e9ecef;
                color: #333;
            }
            .pos-table tbody tr:hover {
                background-color: #f8f9fa;
            }
            .pos-table tbody tr:last-child td {
                border-bottom: none;
            }
            .status-badge {
                padding: 4px 12px;
                border-radius: 12px;
                font-size: 12px;
                font-weight: 600;
                display: inline-block;
            }
            .status-success {
                background: #d4edda;
                color: #155724;
            }
            .status-warning {
                background: #fff3cd;
                color: #856404;
            }
            .status-error {
                background: #f8d7da;
                color: #721c24;
            }
            </style>
            <div class="pos-table-container">
            <table class="pos-table">
                <thead>
                    <tr>
                        <th>Item Name</th>
                        <th>Action</th>
                        <th>New Price</th>
                        
                        <th>Procurement</th>
                    </tr>
                </thead>
                <tbody>
            """
            
            for entry in execution_data["execution_log"]:
                item_name = entry.get("item", "N/A").capitalize()
                action = entry.get("price_action", "N/A").replace('_', ' ').title()
                new_price = f"{st.session_state.currency_symbol}{entry.get('new_price', 'N/A')}"
                
                # Status badge
                status = entry.get("status", "unknown")
                if status == "success":
                    status_html = '<span class="status-badge status-success">✓ Success</span>'
                elif status == "error":
                    status_html = '<span class="status-badge status-error">✗ Error</span>'
                else:
                    status_html = '<span class="status-badge status-warning">⚠ Warning</span>'
                
                # Procurement info
                procurement = entry.get("procurement_order")
                if isinstance(procurement, dict):
                    proc_qty = procurement.get("qty", 0)
                    procurement_html = f"📦 Qty: {proc_qty}"
                else:
                    procurement_html = "—"
                
                table_html += f"""
                    <tr>
                        <td><strong>{item_name}</strong></td>
                        <td>{action}</td>
                        <td><strong>{new_price}</strong></td>
                        
                        <td>{procurement_html}</td>
                    </tr>
                """
            
            table_html += """
                </tbody>
            </table>
            </div>
            """
            
            components.html(table_html, height=500, scrolling=True)
        else:
            st.info("ℹ️ No POS items available to display.")

    elif view_mode == "💻 JSON View":
        # Formatted JSON with syntax highlighting
        json_str = json.dumps(payload, indent=2, ensure_ascii=False)
        
        st.markdown(
            f'<div class="json-container">{json_str}</div>',
            unsafe_allow_html=True
        )
        
        # Copy to clipboard button
        if st.button("📋 Copy JSON to Clipboard", use_container_width=True, type="secondary"):
            st.code(json_str, language="json")

    # ----- Action Buttons -----
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🚀 Confirm & Push to POS", use_container_width=True, type="primary", key="push_pos_btn"):
            with st.spinner("Sending data to POS system..."):
                # Simulate API call delay
                import time
                time.sleep(1)
                st.success("✅ Data successfully sent to POS API!")
                st.balloons()



@st.dialog("💼 CFO Dashboard — Financial Impact Overview",width="large")
def show_cfo_dashboard_popup(execution_data: Dict[str, Any],
                           ingestion_data: Dict[str, Any],
                           analysis_data: Dict[str, Any],
                           prediction_data: Dict[str, Any],
                           decisions: Dict[str, Any]):
    """Popup view for CFO metrics and visual summary"""
    
    is_bakery = st.session_state.get("category") == "bakery"
    
    if is_bakery:
        # Use bakery-specific data from ingestion_data["bakery"]
        # data_source = ingestion_data.get("bakery", {})
        data_source = ingestion_data.get("bakery", ingestion_data)
        stock_data = data_source.get("stock", {})
        sales_history = data_source.get("sales_history", {})
        historical_inventory = data_source.get("historical_inventory", {})
    else:
        # Use regular perishables data structure
        data_source = ingestion_data
        stock_data = ingestion_data.get("stock", {})
        sales_history = ingestion_data.get("sales_history", {})
        historical_inventory = ingestion_data.get("historical_inventory", {})

    # =====================================================================
    # 🎛️ INTERACTIVE SIMULATION CONTROLS (ENHANCED WITH TOOLTIPS & LEVELS)
    # =====================================================================
    st.markdown('<div class="subsection-header">🎛️ Multi-Level Simulation Controls</div>', unsafe_allow_html=True)
    
    st.markdown("""
        <style>
        .slider-label {
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 4px;
            font-size: 14px;
        }
        .tab-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 10px 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            font-weight: 600;
        }
        .metrics-container {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            border: 2px solid #e0e0e0;
            height: 100%;
            position: sticky;
            top: 20px;
        }
        .metric-card {
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metric-title {
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }
        .metric-delta {
            font-size: 14px;
            margin-top: 5px;
        }
        .delta-positive {
            color: #28a745;
        }
        .delta-negative {
            color: #dc3545;
        }
        .margin-card {
            background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
            border-left: 4px solid #667eea;
            padding: 12px;
            border-radius: 8px;
            margin-top: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Tooltip definitions
    TOOLTIP_DEFINITIONS = {
        "cost_factor": """
        **Cost Factor (Default Cost Multiplier)**
        
        Represents the cost-to-price ratio used when actual cost data is unavailable.
        
        • **Range**: 0.1 to 1.0
        • **Typical Values**: 
          - Fresh produce: 0.6-0.7
          - Packaged goods: 0.5-0.6
          - Premium items: 0.4-0.5
        • **Impact**: Higher values mean higher costs relative to selling price, affecting margin calculations
        • **Formula**: Cost = Selling_Price × Cost_Factor
        """,
        
        "price_elasticity": """
        **Price Elasticity of Demand**
        
        Measures how sensitive customer demand is to price changes. Always negative for normal goods.
        
        • **Range**: -2.0 to -0.1
        • **Interpretation**:
          - **-0.2 to -0.5**: Inelastic (essentials like milk, bread)
          - **-0.5 to -1.0**: Moderately elastic (common groceries)
          - **-1.0 to -2.0**: Highly elastic (luxury/premium items)
        • **Example**: Elasticity of -1.2 means a 10% price cut increases demand by 12%
        • **Formula**: %ΔQuantity = Elasticity × %ΔPrice
        """,
        
        "weather_multiplier": """
        **Weather Impact Multiplier**
        
        Adjusts demand based on weather conditions affecting shopping behavior.
        
        • **Range**: 0.7 to 1.5
        • **Scenarios**:
          - **1.3-1.5**: Extreme heat (beverages ↑, fresh produce ↑)
          - **1.0**: Normal weather (baseline)
          - **0.8-0.9**: Heavy rain/storm (footfall ↓, delivery ↑)
          - **1.1-1.2**: Pleasant weather (general shopping ↑)
        • **Impact**: Multiplies baseline demand to reflect weather-driven buying patterns
        """,
        
        "event_multiplier": """
        **Calendar Event / Seasonal Multiplier**
        
        Accounts for festivals, holidays, and special events driving demand spikes.
        
        • **Range**: 0.7 to 1.5
        • **Common Events**:
          - **1.4-1.5**: Major festivals (Diwali, Christmas, Eid)
          - **1.2-1.3**: Weekends, payday periods
          - **1.0**: Regular days
          - **0.7-0.8**: Post-festival lull, mid-week slump
        • **Application**: Combines with base forecast to model event-driven surges
        """,
        
        "footfall_multiplier": """
        **Store Footfall Multiplier**
        
        Reflects actual vs. expected customer traffic based on store analytics.
        
        • **Range**: 0.5 to 1.5
        • **Drivers**:
          - **1.3-1.5**: Promotions, new store openings, viral campaigns
          - **1.0**: Normal footfall (baseline)
          - **0.7-0.9**: Competitive pressure, maintenance closures
          - **0.5-0.6**: Pandemic/lockdown scenarios
        • **Data Source**: POS systems, door counters, heat maps
        """,
        
        "shelf_life_sim": """
        **Shelf-Life Based Simulation Mode**
        
        Enables day-by-day spoilage modeling over product shelf life.
        
        • **When Enabled**: 
          - Simulates each day of remaining shelf life
          - Applies daily sales and computes incremental spoilage
          - More accurate for perishables with varying decay rates
        
        • **When Disabled**: 
          - Uses aggregate formula (faster computation)
          - Suitable for longer shelf-life items
        
        • **Recommended For**: Fresh produce, dairy, bakery items
        • **Formula**: Daily_Shrink = (Stock - Daily_Sales) × (Spoilage_Risk / Shelf_Life)
        """
    }

    def create_slider_with_tooltip(label, tooltip_key, min_val, max_val, default, step, key, format_str=None):
        # Display label above the slider
        st.markdown(f'<div class="slider-label">{label}</div>', unsafe_allow_html=True)

        # Create the slider with tooltip
        if format_str:
            return st.slider(
                "",
                min_value=min_val,
                max_value=max_val,
                value=default,
                step=step,
                key=key,
                format=format_str,
                help=TOOLTIP_DEFINITIONS[tooltip_key],
            )
        else:
            return st.slider(
                "",
                min_value=min_val,
                max_value=max_val,
                value=default,
                step=step,
                key=key,
                help=TOOLTIP_DEFINITIONS[tooltip_key],
            )

    def calculate_shrinkage_metrics(target_skus, params_dict, use_shelf_life_sim, 
                                   is_optimized=False, exec_price_map=None):
        """Calculate shrinkage AND gross margin metrics for a set of SKUs with given parameters"""
        analysis = analysis_data.get("analysis", {})
        predictions = prediction_data.get("predictions", {})
        
        total_sales = 0
        total_spoilage = 0
        total_cost = 0
        total_revenue = 0
        sku_details = []
        
        for sku in target_skus:
            if sku not in stock_data:
                continue
                
            stock_entry = stock_data[sku][0] if isinstance(stock_data[sku], list) else stock_data[sku]
            current_qty = stock_entry.get("available_qty", 0)
            current_price = stock_entry.get("avg_price", 0)
            
            params = params_dict.get(sku, params_dict.get('default', {}))
            cost_per_unit = stock_entry.get("cost", current_price * params["cost_factor"])
            spoilage_risk = float(analysis.get(sku, {}).get("spoilage_risk", 0))
            future_demand = predictions.get(sku, {}).get("future_demand", 0)
            shelf_life_days = stock_entry.get("shelf_life_days", 5)
            avg_daily_sales = stock_entry.get("avg_daily_sales", future_demand / max(shelf_life_days, 1))
            
            # Price and elasticity effect
            if is_optimized and exec_price_map and sku in exec_price_map:
                new_price = float(exec_price_map[sku])
                price_change_pct = (new_price - current_price) / current_price if current_price else 0
                demand_multiplier = max(0.05, 1 + (params["elasticity"] * price_change_pct))
                price = new_price
                cost = new_price * params["cost_factor"]
            else:
                demand_multiplier = 1.0
                price = current_price
                cost = cost_per_unit
            
            adjusted_demand = future_demand * demand_multiplier * params["weather"] * params["event"] * params["footfall"]
            
            # Calculate sales and spoilage
            if use_shelf_life_sim and shelf_life_days > 0:
                Q = current_qty
                total_shrink_qty = 0.0
                total_sales_qty = 0.0
                daily_risk = spoilage_risk / max(shelf_life_days, 1)
                
                for day in range(1, int(shelf_life_days) + 1):
                    daily_sales = avg_daily_sales * demand_multiplier * params["weather"] * params["event"] * params["footfall"]
                    daily_sales = min(daily_sales, Q)
                    residual = Q - daily_sales
                    daily_shrink = residual * daily_risk
                    total_shrink_qty += daily_shrink
                    total_sales_qty += daily_sales
                    Q = max(0.0, residual)
                
                sales_value = total_sales_qty * price
                spoilage_value = total_shrink_qty * cost
                cost_value = total_sales_qty * cost
            else:
                sales_value = adjusted_demand * price
                spoilage_units = max(0, current_qty - adjusted_demand)
                spoilage_value = spoilage_units * cost * spoilage_risk
                cost_value = adjusted_demand * cost
            
            # Gross margin calculations
            gross_margin_value = sales_value - cost_value - spoilage_value
            gross_margin_pct = (gross_margin_value / sales_value * 100) if sales_value > 0 else 0
            
            total_sales += sales_value
            total_spoilage += spoilage_value
            total_cost += cost_value
            total_revenue += sales_value
            inventory_value_at_cost = current_qty * cost

            shrink_pct = (spoilage_value / sales_value * 100) if sales_value > 0 else 0
            # shrink_pct = (spoilage_value / inventory_value_at_cost * 100) if inventory_value_at_cost > 0 else 0
            sku_details.append({
                "sku": sku,
                "sales_value": sales_value,
                "spoilage_value": spoilage_value,
                "cost_value": cost_value,
                "gross_margin_value": gross_margin_value,
                "gross_margin_pct": gross_margin_pct,
                "shrink_pct": shrink_pct
            })
        
        avg_shrink = (total_spoilage / total_sales * 100) if total_sales > 0 else 0
        # avg_shrink = (total_spoilage / total_inventory_cost * 100) if total_inventory_cost > 0 else 0
        total_gross_margin = total_revenue - total_cost - total_spoilage
        avg_gross_margin_pct = (total_gross_margin / total_revenue * 100) if total_revenue > 0 else 0
        
        return {
            "total_sales": total_sales,
            "total_spoilage": total_spoilage,
            "total_cost": total_cost,
            "total_revenue": total_revenue,
            "total_gross_margin": total_gross_margin,
            "avg_gross_margin_pct": avg_gross_margin_pct,
            "avg_shrink": avg_shrink,
            "sku_details": sku_details
        }

    # =====================================================================
    # TAB-BASED CONTROL SYSTEM: Overall, Category, SKU
    # =====================================================================
    tab1, tab2, tab3 = st.tabs([
        "🌍 Overall Parameters (All Products)",
        "📦 Category-Level Parameters",
        "🎯 SKU-Level Parameters"
    ])

    # ----------------------------------------------------------------------
    # TAB 1: OVERALL PARAMETERS (applies to all products)
    # ----------------------------------------------------------------------

    category_overrides = st.session_state.get("category_overrides", {})
    sku_overrides = st.session_state.get("sku_overrides", {})

    def get_effective_params(sku: str, category: str) -> dict:
        """Returns final param for ONE SKU — Global → Category → SKU"""
        params = {
            "cost_factor": cost_factor_global,
            "elasticity": price_elasticity_global,
            "weather": F_weather_global,
            "event": F_event_global,
            "footfall": F_footfall_global,
        }
        # Category override
        if category in category_overrides:
            params.update(category_overrides[category])
        # SKU override (highest priority)
        if sku in sku_overrides:
            params.update(sku_overrides[sku])
        return params

    with tab1:
        st.markdown('<div class="tab-header">🌍 Global Parameters - Applied to All Products</div>', unsafe_allow_html=True)
        st.info("💡 These sliders affect the entire inventory across all categories and SKUs.")
        

        left_col, right_col = st.columns([1, 1])
        with left_col:
            st.markdown("### 🎚️ Adjust Parameters")
            
            cost_factor_global = create_slider_with_tooltip(
                "Cost Factor", "cost_factor",
                0.1, 1.0, 0.65, 0.01, "cost_factor_global"
            )
            
            price_elasticity_global = create_slider_with_tooltip(
                "Price Elasticity", "price_elasticity",
                -2.0, -0.1, -1.2, 0.05, "elasticity_global"
            )
            
            F_weather_global = create_slider_with_tooltip(
                "Weather Multiplier", "weather_multiplier",
                0.7, 1.5, 1.0, 0.05, "weather_global"
            )
            
            F_event_global = create_slider_with_tooltip(
                "Event Multiplier", "event_multiplier",
                0.7, 1.5, 1.0, 0.05, "event_global"
            )
            
            F_footfall_global = create_slider_with_tooltip(
                "Footfall Multiplier", "footfall_multiplier",
                0.5, 1.5, 1.0, 0.05, "footfall_global"
            )
            
            use_shelf_life_sim = st.checkbox(
                "Enable Shelf-Life Simulation",
                value=True,
                key="shelf_life_sim_global"
            )
        
        with right_col:
            st.markdown("### 📊 Overall Financial Metrics")
            st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
            
            # Calculate overall metrics with current parameters
            all_skus = list(stock_data.keys())
            global_params = {
                "default": {
                    "cost_factor": cost_factor_global,
                    "elasticity": price_elasticity_global,
                    "weather": F_weather_global,
                    "event": F_event_global,
                    "footfall": F_footfall_global
                }
            }
            
            # Predictive (no optimization)
            pred_metrics = calculate_shrinkage_metrics(
                all_skus, global_params, use_shelf_life_sim, is_optimized=False
            )
            
            # Optimized (with execution prices)
            execution_log = execution_data.get("execution_log", [])
            exec_price_map = {}
            if execution_log:
                exec_df = pd.DataFrame(execution_log)
                if "item" in exec_df.columns and "new_price" in exec_df.columns:
                    exec_price_map = exec_df.groupby("item")["new_price"].last().to_dict()
            
            opt_metrics = calculate_shrinkage_metrics(
                all_skus, global_params, use_shelf_life_sim, 
                is_optimized=True, exec_price_map=exec_price_map
            )
            
            delta_shrink = pred_metrics["avg_shrink"] - opt_metrics["avg_shrink"]
            delta_margin = opt_metrics["avg_gross_margin_pct"] - pred_metrics["avg_gross_margin_pct"]
            margin_value_improvement = opt_metrics["total_gross_margin"] - pred_metrics["total_gross_margin"]
            
            # Display metrics
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Predictive Shrinkage %</div>
                <div class="metric-value" style="color: #ff4b4b;">{pred_metrics["avg_shrink"]:.2f}%</div>
                <div class="metric-delta">Sales: ₹{pred_metrics["total_sales"]:,.0f}</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">Optimized Shrinkage %</div>
                <div class="metric-value" style="color: #00b386;">{opt_metrics["avg_shrink"]:.2f}%</div>
                <div class="metric-delta">Sales: ₹{opt_metrics["total_sales"]:,.0f}</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">Shrinkage Improvement</div>
                <div class="metric-value delta-positive">↓ {delta_shrink:.2f}%</div>
                <div class="metric-delta">Spoilage Saved: ₹{(pred_metrics["total_spoilage"] - opt_metrics["total_spoilage"]):,.0f}</div>
            </div>
            
            <div class="margin-card">
                <div class="metric-title" style="color: #667eea;">💰 GROSS MARGIN ANALYSIS</div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px;">
                    <div>
                        <div style="font-size: 11px; color: #666;">Predictive</div>
                        <div style="font-size: 20px; font-weight: bold; color: #ff6b6b;">{pred_metrics["avg_gross_margin_pct"]:.2f}%</div>
                        <div style="font-size: 12px; color: #888;">₹{pred_metrics["total_gross_margin"]:,.0f}</div>
                    </div>
                    <div>
                        <div style="font-size: 11px; color: #666;">Optimized</div>
                        <div style="font-size: 20px; font-weight: bold; color: #51cf66;">{opt_metrics["avg_gross_margin_pct"]:.2f}%</div>
                        <div style="font-size: 12px; color: #888;">₹{opt_metrics["total_gross_margin"]:,.0f}</div>
                    </div>
                </div>
                <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #ddd;">
                    <div style="font-size: 13px; color: #667eea; font-weight: 600;">
                        Margin Improvement: <span style="color: #51cf66;">↑ {delta_margin:.2f}%</span>
                    </div>
                    <div style="font-size: 12px; color: #888; margin-top: 3px;">
                        Additional Profit: ₹{margin_value_improvement:,.0f}
                    </div>
                </div>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">Total SKUs Analyzed</div>
                <div class="metric-value">{len(all_skus)}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        with st.expander("ℹ️ What does this mean?", expanded=False):
            st.markdown(f'<div class="custom-help-text">{TOOLTIP_DEFINITIONS["shelf_life_sim"]}</div>', unsafe_allow_html=True)

    # ----------------------------------------------------------------------
    # TAB 2: CATEGORY-LEVEL PARAMETERS
    # ----------------------------------------------------------------------

    with tab2:
        st.markdown('<div class="tab-header">Category-Level Parameters</div>', unsafe_allow_html=True)
        st.info("Click any category → tweak sliders → **see live shrinkage & margin instantly** → collapse")

        # Extract categories
        categories = sorted({
            (e[0] if isinstance(e, list) else e).get("category", "Uncategorized")
            for e in stock_data.values()
        })

        if not categories:
            st.warning("No categories found.")
            st.stop()

        search = st.text_input("Search Category", placeholder="e.g. Fruit, Bakery...", key="cat_search")
        filtered = [c for c in categories if search.lower() in c.lower()] if search else categories
        st.caption(f"**{len(filtered)} categories** | Click to expand")

        # Pre-load execution prices
        exec_price_map = {}
        if execution_data.get("execution_log"):
            df = pd.DataFrame(execution_data["execution_log"])
            if {"item", "new_price"}.issubset(df.columns):
                exec_price_map = df.groupby("item")["new_price"].last().to_dict()

        for cat in filtered:
            cat_skus = [
                sku for sku, e in stock_data.items()
                if (e[0] if isinstance(e, list) else e).get("category") == cat
            ]
            sku_count = len(cat_skus)
            total_stock = sum(
                (e[0] if isinstance(e, list) else e).get("available_qty", 0)
                for sku, e in stock_data.items()
                if (e[0] if isinstance(e, list) else e).get("category") == cat
            )

            # ——— EXPANDER: ONE ROW PER CATEGORY ———
            with st.expander(f"{cat} ({sku_count} SKUs, {total_stock:.0f} units)", expanded=False):
                left, right = st.columns([1.3, 1])

                with left:
                    st.markdown("**Override Parameters**")
                    enabled = st.checkbox("Enable Category Override", key=f"cat_on_{cat}", value=cat in category_overrides)

                    if enabled:
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            cost = create_slider_with_tooltip("Cost Factor", "cost_factor", 0.1, 1.0, 0.65, 0.01, f"cc_{cat}")
                            elast = create_slider_with_tooltip("Price Elasticity", "price_elasticity", -2.0, -0.1, -1.2, 0.05, f"ce_{cat}")
                        with c2:
                            weather = create_slider_with_tooltip("Weather Multiplier", "weather_multiplier", 0.7, 1.5, 1.0, 0.05, f"cw_{cat}")
                            event = create_slider_with_tooltip("Event Multiplier", "event_multiplier", 0.7, 1.5, 1.0, 0.05, f"cv_{cat}")
                        with c3:
                            foot = create_slider_with_tooltip("Footfall Multiplier", "footfall_multiplier", 0.5, 1.5, 1.0, 0.05, f"cf_{cat}")

                        # Save override
                        category_overrides[cat] = {
                            "cost_factor": cost, "elasticity": elast,
                            "weather": weather, "event": event, "footfall": foot
                        }
                    else:
                        st.success("Using Global values")

                with right:
                    st.markdown("**Live Category Performance**")
                    # Build params using hierarchy
                    sku_params = {sku: get_effective_params(sku, cat) for sku in cat_skus}

                    pred = calculate_shrinkage_metrics(cat_skus, sku_params, use_shelf_life_sim, False)
                    opt  = calculate_shrinkage_metrics(cat_skus, sku_params, use_shelf_life_sim, True, exec_price_map)

                    delta = pred["avg_shrink"] - opt["avg_shrink"]
                    saved = pred["total_spoilage"] - opt["total_spoilage"]
                    margin_delta = opt["avg_gross_margin_pct"] - pred["avg_gross_margin_pct"]
                    margin_saved = opt["total_gross_margin"] - pred["total_gross_margin"]

                    shrink_text = "Saved" if saved >= 0 else "Lost"
                    shrink_color = '#4CAF50' if saved >= 0 else '#ff4b4b'
                    margin_text = "Improved" if margin_saved >= 0 else "Declined"
                    margin_color = '#51cf66' if margin_saved >= 0 else '#ff6b6b'

                    st.html(f"""
                    <div style="background:#1e1e1e; padding:18px; border-radius:14px; text-align:center; box-shadow:0 4px 12px rgba(0,0,0,0.3);">
                        <div style="font-size:0.95em; color:#aaa; margin-bottom:8px;">
                            📉 SHRINKAGE ANALYSIS
                        </div>
                        <div style="font-size:2.3em; font-weight:bold; color:#ff4b4b; line-height:1;">
                            {pred['avg_shrink']:.2f}%
                        </div>
                        <div style="font-size:2.1em; color:#00b386; line-height:1;">
                            {opt['avg_shrink']:.2f}%
                        </div>
                        <div style="font-size:1.2em; color:{shrink_color}; margin-top:10px; font-weight:600;">
                            ↓ {delta:.2f}%  |  {shrink_text}: ₹{abs(round(saved)):,}
                        </div>
                        
                        <div style="margin: 15px 0; border-top: 1px solid #444;"></div>
                        
                        <div style="font-size:0.95em; color:#aaa; margin-bottom:8px;">
                            💰 GROSS MARGIN ANALYSIS
                        </div>
                        <div style="font-size:2.1em; font-weight:bold; color:#ff6b6b; line-height:1;">
                            {pred['avg_gross_margin_pct']:.2f}%
                        </div>
                        <div style="font-size:2.0em; color:#51cf66; line-height:1;">
                            {opt['avg_gross_margin_pct']:.2f}%
                        </div>
                        <div style="font-size:1.2em; color:{margin_color}; margin-top:10px; font-weight:600;">
                            ↑ {margin_delta:.2f}%  |  {margin_text}: ₹{abs(round(margin_saved)):,}
                        </div>
                        
                        <div style="font-size:0.8em; color:#888; margin-top:12px;">
                            Updated: {pd.Timestamp.now().strftime('%H:%M:%S')}
                        </div>
                    </div>
                    """)

            st.markdown("---")

        # Auto-save
        st.session_state.category_overrides = category_overrides
        st.session_state.sku_overrides = sku_overrides

    #-----------------------------------------------------------------------
    # TAB 3: SKU-LEVEL PARAMETERS
    # ----------------------------------------------------------------------

    with tab3:
        st.markdown('<div class="tab-header">SKU-Level Fine-Tuning</div>', unsafe_allow_html=True)
        st.info("Click any SKU → sliders + **live shrinkage & margin results appear together** → collapse when done")

        all_skus = list(stock_data.keys())
        search = st.text_input("Search SKU", key="sku_search3", placeholder="Filter instantly...")
        filtered = [s for s in all_skus if search.lower() in s.lower()] if search else all_skus
        st.caption(f"**{len(filtered)} SKUs** | Click to expand")

        # Pre-load execution prices
        exec_price_map = {}
        if execution_data.get("execution_log"):
            df = pd.DataFrame(execution_data["execution_log"])
            if {"item","new_price"}.issubset(df.columns):
                exec_price_map = df.groupby("item")["new_price"].last().to_dict()

        for sku in filtered:
            entry = stock_data[sku][0] if isinstance(stock_data[sku], list) else stock_data[sku]
            name = entry.get("name", sku)
            cat = entry.get("category", "Unknown")
            qty = entry.get("available_qty", 0)

            # ——— EXPANDER: ONE CLEAN ROW PER SKU ———
            with st.expander(f"{name} ({qty:.0f} units)", expanded=False):
                left, right = st.columns([1.3, 1])

                with left:
                    st.markdown("**Override Parameters**")
                    enabled = st.checkbox("Enable SKU Override", key=f"sku_on_{sku}", value=sku in sku_overrides)

                    if enabled:
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            cost = create_slider_with_tooltip("Cost factor", "cost_factor", 0.1, 1.0, 0.65, 0.01, f"c_{sku}")
                            elast = create_slider_with_tooltip("Price Elasticity", "price_elasticity", -2.0, -0.1, -1.2, 0.05, f"e_{sku}")
                        with c2:
                            weather = create_slider_with_tooltip("Weather Multiplier", "weather_multiplier", 0.7, 1.5, 1.0, 0.05, f"w_{sku}")
                            event = create_slider_with_tooltip("Event Multiplier", "event_multiplier", 0.7, 1.5, 1.0, 0.05, f"v_{sku}")
                        with c3:
                            foot = create_slider_with_tooltip("Footfall Multiplier", "footfall_multiplier", 0.5, 1.5, 1.0, 0.05, f"f_{sku}")

                        # Save instantly
                        sku_overrides[sku] = {
                            "cost_factor": cost, "elasticity": elast,
                            "weather": weather, "event": event, "footfall": foot
                        }
                    else:
                        st.success("Using Category/Global values")

                with right:
                    st.markdown("**Live Simulation**")
                    params = {sku: get_effective_params(sku, cat)}

                    pred = calculate_shrinkage_metrics([sku], params, use_shelf_life_sim, False)
                    opt  = calculate_shrinkage_metrics([sku], params, use_shelf_life_sim, True, exec_price_map)

                    delta = pred["avg_shrink"] - opt["avg_shrink"]
                    saved = pred["total_spoilage"] - opt["total_spoilage"]
                    margin_delta = opt["avg_gross_margin_pct"] - pred["avg_gross_margin_pct"]
                    margin_saved = opt["total_gross_margin"] - pred["total_gross_margin"]
                    
                    shrink_text = "Saved" if saved >= 0 else "Lost"
                    shrink_color = '#4CAF50' if saved >= 0 else '#ff4b4b'
                    margin_text = "Improved" if margin_saved >= 0 else "Declined"
                    margin_color = '#51cf66' if margin_saved >= 0 else '#ff6b6b'

                    st.html(f"""
                    <div style="background:#1e1e1e; padding:15px; border-radius:12px; text-align:center;">
                        <div style="font-size:0.9em; color:#aaa;">📉 SHRINKAGE</div>
                        <div style="font-size:2em; font-weight:bold; color:#ff4b4b;">
                            {pred['avg_shrink']:.2f}%
                        </div>
                        <div style="font-size:1.8em; color:#00b386;">
                            {opt['avg_shrink']:.2f}%
                        </div>
                        <div style="font-size:1.1em; color:{shrink_color}; margin-top:8px;">
                            ↓ {delta:.2f}%  |  {shrink_text} <b>₹{abs(int(saved)):,}</b>
                        </div>
                        
                        <div style="margin: 12px 0; border-top: 1px solid #444;">
                        </div>
                        
                        <div style="font-size:0.9em; color:#aaa;">💰 GROSS MARGIN</div>
                        <div style="font-size:1.8em; font-weight:bold; color:#ff6b6b;">
                            {pred['avg_gross_margin_pct']:.2f}%
                        </div>
                        <div style="font-size:1.7em; color:#51cf66;">
                            {opt['avg_gross_margin_pct']:.2f}%
                        </div>
                        <div style="font-size:1.1em; color:{margin_color}; margin-top:8px;">
                            ↑ {margin_delta:.2f}%  |  {margin_text} <b>₹{abs(int(margin_saved)):,}</b>
                        </div>
                    </div>
                    """)

                    st.caption(f"Updated: {pd.Timestamp.now().strftime('%H:%M:%S')}")

        # Auto-save
        st.session_state.sku_overrides = sku_overrides
        st.session_state.category_overrides = category_overrides
    
    # =====================================================================
    # HELPER FUNCTION: Get effective parameters for a SKU
    # =====================================================================
    

    # =====================================================================
    # 📉 A. INTERACTIVE Predictive Baseline Shrinkage (Before Optimization)
    # =====================================================================
    
    try:
        analysis = analysis_data.get("analysis", {})
        predictions = prediction_data.get("predictions", {})

        predictive_records = []

        for sku, stock_entries in stock_data.items():
            if not stock_entries:
                continue

            stock_entry = stock_entries[0] if isinstance(stock_entries, list) else stock_entries
            current_qty = stock_entry.get("available_qty", 0)
            current_price = stock_entry.get("avg_price", 0)
            category = stock_entry.get("category", "Uncategorized")
            
            # Get effective parameters for this SKU
            params = get_effective_params(sku, category)
            
            cost_per_unit = stock_entry.get("cost", current_price * params["cost_factor"])
            spoilage_risk = float(analysis.get(sku, {}).get("spoilage_risk", 0))
            future_demand = predictions.get(sku, {}).get("future_demand", 0)
            
            shelf_life_days = stock_entry.get("shelf_life_days", 5)
            avg_daily_sales = stock_entry.get("avg_daily_sales", future_demand / shelf_life_days)

            # Apply effective multipliers
            adjusted_future_demand = future_demand * params["weather"] * params["event"] * params["footfall"]

            if use_shelf_life_sim and shelf_life_days > 0:
                Q = current_qty
                total_shrink_qty = 0.0
                total_sales_qty = 0.0
                daily_risk = spoilage_risk / max(shelf_life_days, 1)
                
                for day in range(1, int(shelf_life_days) + 1):
                    daily_sales = avg_daily_sales * params["weather"] * params["event"] * params["footfall"]
                    daily_sales = min(daily_sales, Q)
                    residual = Q - daily_sales
                    daily_shrink = residual * daily_risk
                    total_shrink_qty += daily_shrink
                    total_sales_qty += daily_sales
                    Q = max(0.0, residual)
                
                expected_sales_value = total_sales_qty * current_price
                expected_spoilage_value = total_shrink_qty * cost_per_unit
            else:
                expected_sales_value = adjusted_future_demand * current_price
                expected_spoilage_units = max(0, current_qty - adjusted_future_demand)
                expected_spoilage_value = expected_spoilage_units * cost_per_unit * spoilage_risk

            shrink_pct_predicted = (
                (expected_spoilage_value / expected_sales_value) * 100
                if expected_sales_value > 0 else 0
            )

            predictive_records.append({
                "sku": sku,
                "category": category,
                "available_qty": current_qty,
                "current_price": current_price,
                "future_demand": adjusted_future_demand,
                "expected_sales_value": expected_sales_value,
                "expected_spoilage_value": expected_spoilage_value,
                "shrink_pct_predicted": shrink_pct_predicted,
                "shelf_life_days": shelf_life_days,
                "params_source": "SKU" if sku in sku_overrides else ("Category" if category in category_overrides else "Global")
            })

        if predictive_records:
            pred_df = pd.DataFrame(predictive_records)
            st.subheader("📉 Predictive Baseline Shrinkage (Before Optimization)")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Sales Value", f"₹{pred_df['expected_sales_value'].sum():,.2f}")
            with col2:
                st.metric("Total Spoilage Value", f"₹{pred_df['expected_spoilage_value'].sum():,.2f}")
            with col3:
                avg_shrink = (pred_df['expected_spoilage_value'].sum() / pred_df['expected_sales_value'].sum() * 100) if pred_df['expected_sales_value'].sum() > 0 else 0
                st.metric("Avg Shrink %", f"{avg_shrink:.2f}%")
            with col4:
                st.metric("SKUs Analyzed", len(pred_df))
            
            st.dataframe(pred_df.round(3), use_container_width=True)

            # fig, ax = plt.subplots(figsize=(12, 5))
            # pred_df.plot(x="sku", y="shrink_pct_predicted", kind="bar", ax=ax, color="#ff4b4b")
            # ax.set_ylabel("%")
            # ax.set_title("Shrink % (Predicted Baseline with Multi-Level Parameters)")
            # ax.axhline(y=avg_shrink, color='gray', linestyle='--', label=f'Avg: {avg_shrink:.2f}%')
            # ax.legend()
            # plt.xticks(rotation=45, ha='right')
            # plt.tight_layout()
            # st.pyplot(fig)

            fig = px.bar(
                pred_df,
                x="sku",
                y="shrink_pct_predicted",
                title="Shrink % (Predicted Baseline)",
                color_discrete_sequence=["#ff4b4b"],
                text="shrink_pct_predicted",
                hover_data={
                    "sku": True,
                    "shrink_pct_predicted": ":.2f%",
                    "expected_spoilage_value": ":,.0f",
                    "expected_sales_value": ":,.0f"
                }
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.add_hline(y=avg_shrink, line_dash="dash", line_color="white",
                        annotation_text=f"Avg: {avg_shrink:.2f}%", annotation_position="top left")
            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                font_color="black",
                title_font_size=20,
                xaxis_tickangle=45,
                hovermode="x unified",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ No current stock data found for predictive shrinkage calculation.")
    except Exception as e:
        st.error(f"Error calculating Predictive Baseline Shrinkage: {e}")
        import traceback
        st.code(traceback.format_exc())


    # =====================================================================
    # 📈 B. INTERACTIVE Prescriptive / Optimized Shrinkage (After Pricing Action)
    # =====================================================================
    
    try:
        analysis = analysis_data.get("analysis", {})
        predictions = prediction_data.get("predictions", {})
        execution_log = execution_data.get("execution_log", [])

        exec_df = pd.DataFrame(execution_log) if execution_log else pd.DataFrame(columns=["item", "new_price"])
        exec_price_map = {}
        if not exec_df.empty and "item" in exec_df.columns and "new_price" in exec_df.columns:
            exec_price_map = exec_df.groupby("item")["new_price"].last().to_dict()

        optimized_records = []

        for sku, a_vals in analysis.items():
            if sku not in exec_price_map:
                continue

            stock_entry = stock_data.get(sku, [{}])[0] if isinstance(stock_data.get(sku), list) else stock_data.get(sku, {})
            pred_entry = predictions.get(sku, {})

            current_qty = stock_entry.get("available_qty", 0)
            current_price = stock_entry.get("avg_price", 0)
            category = stock_entry.get("category", "Uncategorized")
            inventory_value = current_qty * cost_per_unit
            # total_inventory_cost_opt += inventory_value
            # Get effective parameters
            params = get_effective_params(sku, category)
            
            cost_per_unit = stock_entry.get("cost", current_price * params["cost_factor"])
            spoilage_risk = a_vals.get("spoilage_risk", 0)
            future_demand = pred_entry.get("future_demand", 0)
            new_price = float(exec_price_map[sku])
            new_cost_per_unit = new_price * params["cost_factor"]
            
            shelf_life_days = stock_entry.get("shelf_life_days", 5)
            avg_daily_sales = stock_entry.get("avg_daily_sales", future_demand / shelf_life_days)

            price_change_pct = (new_price - current_price) / current_price if current_price else 0
            demand_multiplier = 1 + (params["elasticity"] * price_change_pct)
            demand_multiplier = max(0.05, demand_multiplier)

            future_demand_after = future_demand * demand_multiplier * params["weather"] * params["event"] * params["footfall"]

            if use_shelf_life_sim and shelf_life_days > 0:
                Q = current_qty
                total_shrink_qty = 0.0
                total_sales_qty = 0.0
                daily_risk = spoilage_risk / max(shelf_life_days, 1)
                
                for day in range(1, int(shelf_life_days) + 1):
                    daily_sales = avg_daily_sales * demand_multiplier * params["weather"] * params["event"] * params["footfall"]
                    daily_sales = min(daily_sales, Q)
                    residual = Q - daily_sales
                    daily_shrink = residual * daily_risk
                    total_shrink_qty += daily_shrink
                    total_sales_qty += daily_sales
                    Q = max(0.0, residual)
                
                expected_sales_value_after = total_sales_qty * new_price
                expected_spoilage_value_after = total_shrink_qty * new_cost_per_unit
            else:
                expected_sales_value_after = future_demand_after * new_price
                expected_spoilage_units_after = max(0, current_qty - future_demand_after)
                expected_spoilage_value_after = expected_spoilage_units_after * new_cost_per_unit * spoilage_risk

            shrink_pct_optimized = (
                (expected_spoilage_value_after / expected_sales_value_after) * 100
                if expected_sales_value_after > 0 else 0
            )
            # shrink_pct_optimized = (expected_spoilage_value_after / inventory_value * 100) if inventory_value > 0 else 0

            # Baseline comparison
            base_demand = future_demand * params["weather"] * params["event"] * params["footfall"]
            expected_sales_value = base_demand * current_price
            expected_spoilage_value = max(0, (current_qty - base_demand)) * cost_per_unit * spoilage_risk
            shrink_pct_predicted = (
                (expected_spoilage_value / expected_sales_value) * 100
                if expected_sales_value > 0 else 0
            )

            delta_shrink_pct = shrink_pct_predicted - shrink_pct_optimized

            revenue_baseline = expected_sales_value
            revenue_after_action = expected_sales_value_after
            gross_margin_baseline = revenue_baseline - (current_qty * cost_per_unit)
            gross_margin_after = revenue_after_action - (current_qty * new_cost_per_unit)
            delta_gross_margin_pct = (
                ((gross_margin_after - gross_margin_baseline) / revenue_baseline) * 100
                if revenue_baseline > 0 else 0
            )

            optimized_records.append({
                "sku": sku,
                "category": category,
                "available_qty": current_qty,
                "current_price": current_price,
                "optimized_price": new_price,
                "price_change_%": price_change_pct * 100,
                "demand_multiplier": demand_multiplier,
                "future_demand": base_demand,
                "future_demand_after_action": future_demand_after,
                "expected_sales_value_after": expected_sales_value_after,
                "expected_spoilage_value_after": expected_spoilage_value_after,
                "shrink_pct_optimized": shrink_pct_optimized,
                "Δ Shrink %": delta_shrink_pct,
                "Δ Gross Margin %": delta_gross_margin_pct,
                "shelf_life_days": shelf_life_days,
                "params_source": "SKU" if sku in sku_overrides else ("Category" if category in category_overrides else "Global")
            })

        if optimized_records:
            opt_df = pd.DataFrame(optimized_records)
            st.subheader("📈 Prescriptive Shrinkage (After Optimization)")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Optimized Sales", f"₹{opt_df['expected_sales_value_after'].sum():,.2f}")
            with col2:
                st.metric("Total Spoilage (Optimized)", f"₹{opt_df['expected_spoilage_value_after'].sum():,.2f}")
            with col3:
                avg_opt_shrink = (opt_df['expected_spoilage_value_after'].sum() / opt_df['expected_sales_value_after'].sum() * 100) if opt_df['expected_sales_value_after'].sum() > 0 else 0
                st.metric("Avg Shrink % (Optimized)", f"{avg_opt_shrink:.2f}%")
            with col4:
                avg_delta = opt_df['Δ Shrink %'].mean()
                st.metric("Avg Shrink Reduction", f"{avg_delta:.2f}%", delta=f"{avg_delta:.2f}%")
            
            st.dataframe(opt_df.round(3), use_container_width=True)

            # fig, ax = plt.subplots(figsize=(12, 5))
            # opt_df.plot(x="sku", y="shrink_pct_optimized", kind="bar", ax=ax, color="#00b386")
            # ax.set_ylabel("%")
            # ax.set_title("Shrink % (Optimized with Multi-Level Parameters)")
            # ax.axhline(y=avg_opt_shrink, color='gray', linestyle='--', label=f'Avg: {avg_opt_shrink:.2f}%')
            # ax.legend()
            # plt.xticks(rotation=45, ha='right')
            # plt.tight_layout()
            # st.pyplot(fig)
            fig = px.bar(
                opt_df,
                x="sku",
                y="shrink_pct_optimized",
                title="Shrink % (After AI Pricing)",
                color_discrete_sequence=["#00b386"],
                text="shrink_pct_optimized",
                hover_data={
                    "optimized_price": ":,.0f",
                    "Δ Shrink %": ":+.2f",
                    "expected_sales_value_after": ":,.0f"
                }
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.add_hline(y=avg_opt_shrink, line_dash="dash", line_color="white",
                        annotation_text=f"Avg: {avg_opt_shrink:.2f}%")
            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                font_color="black",
                xaxis_tickangle=45,
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- Comparison chart ---
            if 'pred_df' in locals() and not pred_df.empty:
                comparison_df = pd.merge(pred_df, opt_df, on="sku", how="outer")
                comparison_df["Δ Shrink %"] = (
                    comparison_df["shrink_pct_predicted"].fillna(0) - comparison_df["shrink_pct_optimized"].fillna(0)
                )

                st.subheader("📊 Shrinkage Improvement (Predicted → Optimized)")
                
                # Enhanced summary with parameter usage breakdown
                total_shrink_saved = comparison_df['Δ Shrink %'].sum()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info(f"💰 **Total Shrinkage Reduction**: {total_shrink_saved:.2f}% across all SKUs")
                with col2:
                    sku_param_count = len([r for r in optimized_records if r["params_source"] == "SKU"])
                    st.info(f"🎯 **SKU-Level Params**: {sku_param_count} SKUs")
                with col3:
                    cat_param_count = len([r for r in optimized_records if r["params_source"] == "Category"])
                    st.info(f"📦 **Category-Level Params**: {cat_param_count} SKUs")
                
                st.dataframe(
                    comparison_df[["sku", "category_x", "shrink_pct_predicted", "shrink_pct_optimized", "Δ Shrink %", "params_source_x"]].rename(
                        columns={"category_x": "category", "params_source_x": "params_source"}
                    ).round(3),
                    use_container_width=True
                )

                # Multi-series comparison chart
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                # Chart 1: Side-by-side comparison
                comparison_df_plot = comparison_df.set_index("sku")
                comparison_df_plot[["shrink_pct_predicted", "shrink_pct_optimized"]].plot(
                    kind="bar", ax=ax1, color=["#ff4b4b", "#00b386"]
                )
                ax1.set_ylabel("Shrink %")
                ax1.set_title("Predicted vs Optimized Shrink %")
                ax1.legend(["Predicted", "Optimized"])
                ax1.grid(axis='y', alpha=0.3)
                ax1.tick_params(axis='x', rotation=45)
                
                # Chart 2: Improvement delta
                comparison_df_plot["Δ Shrink %"].plot(kind="bar", ax=ax2, color="#4CAF50")
                ax2.set_ylabel("Improvement (%)")
                ax2.set_title("Shrinkage Reduction per SKU")
                ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                ax2.grid(axis='y', alpha=0.3)
                ax2.tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                # st.pyplot(fig)
                col1, col2 = st.columns(2)

                with col1:
                    fig1 = px.bar(
                        comparison_df_plot,
                        x=comparison_df_plot.index,
                        y=["shrink_pct_predicted", "shrink_pct_optimized"],
                        barmode="group",
                        title="Predicted vs Optimized",
                        color_discrete_sequence=["#ff4b4b", "#00b386"],
                        text_auto=".1f"
                    )
                    fig1.update_layout(plot_bgcolor="white", paper_bgcolor="white", font_color="black", height=500)
                    st.plotly_chart(fig1, use_container_width=True)

                with col2:
                    fig2 = px.bar(
                        comparison_df_plot,
                        x=comparison_df_plot.index,
                        y="Δ Shrink %",
                        title="Shrink Saved per SKU",
                        color="Δ Shrink %",
                        color_continuous_scale="Geyser",
                        text="Δ Shrink %"
                    )
                    fig2.add_hline(y=0, line_color="white")
                    fig2.update_traces(texttemplate="%{text:+.1f}%")
                    fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white", font_color="black", height=500)
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Category-wise aggregation
                if "category_x" in comparison_df.columns:
                    st.markdown("### 📦 Category-Wise Performance")
                    cat_summary = comparison_df.groupby("category_x").agg({
                        "shrink_pct_predicted": "mean",
                        "shrink_pct_optimized": "mean",
                        "Δ Shrink %": "mean",
                        "sku": "count"
                    }).rename(columns={
                        "shrink_pct_predicted": "Avg Predicted Shrink %",
                        "shrink_pct_optimized": "Avg Optimized Shrink %",
                        "Δ Shrink %": "Avg Improvement %",
                        "sku": "SKU Count"
                    }).round(3)
                    
                    st.dataframe(cat_summary, use_container_width=True)
                    
                    # Category comparison chart
                    fig, ax = plt.subplots(figsize=(10, 5))
                    cat_summary[["Avg Predicted Shrink %", "Avg Optimized Shrink %"]].plot(
                        kind="bar", ax=ax, color=["#ff4b4b", "#00b386"]
                    )
                    ax.set_ylabel("Average Shrink %")
                    ax.set_title("Category-Wise Shrinkage Comparison")
                    ax.legend(["Predicted", "Optimized"])
                    ax.grid(axis='y', alpha=0.3)
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    # st.pyplot(fig)

                    fig = px.bar(
                        cat_summary,
                        x=cat_summary.index,
                        y=["Avg Predicted Shrink %", "Avg Optimized Shrink %"],
                        barmode="group",
                        title="Category Performance",
                        color_discrete_sequence=["#ff4b4b", "#00b386"],
                        text_auto=True
                    )
                    fig.update_layout(
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        font_color="black",
                        xaxis_tickangle=45,
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)



                
                # Download enhanced results
                csv = comparison_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Simulation Results (CSV)",
                    data=csv,
                    file_name="shrinkage_simulation_results_multilevel.csv",
                    mime="text/csv"
                )

        else:
            st.info("ℹ️ Not enough data to calculate optimized shrinkage.")

    except Exception as e:
        st.error(f"Error calculating Prescriptive Shrinkage: {e}")
        import traceback
        st.code(traceback.format_exc())

    # =====================================================================
    # 📘 Formula Explanation Section
    # =====================================================================
    with st.expander("📘 View Calculation Formulas & Parameter Hierarchy", expanded=False):
        st.markdown("""
        ### Parameter Hierarchy System
        
        The system uses a **three-tier parameter hierarchy**:
        
        1. **🌍 Global Parameters**: Default values applied to all products
        2. **📦 Category Parameters**: Override global for specific categories
        3. **🎯 SKU Parameters**: Highest priority, override both global and category
        
        **Resolution Order**: SKU → Category → Global
        
        ---
        
        ### Formulas Used in Simulation
        
        **1. Adjusted Future Demand (Predicted)**
        ```
        Adjusted_Demand = Base_Demand × F_weather × F_event × F_footfall
        
        Where each multiplier is resolved via hierarchy:
        - F_weather = SKU_weather OR Category_weather OR Global_weather
        - F_event = SKU_event OR Category_event OR Global_event
        - F_footfall = SKU_footfall OR Category_footfall OR Global_footfall
        ```
        
        **2. Price Elasticity Effect**
        ```
        Price_Change_% = (New_Price - Current_Price) / Current_Price
        Elasticity = Effective_Elasticity_For_SKU (via hierarchy)
        Demand_Multiplier = 1 + (Elasticity × Price_Change_%)
        Demand_Multiplier = max(0.05, Demand_Multiplier)  // Floor at 5%
        ```
        
        **3. Day-by-Day Simulation (if shelf-life mode enabled)**
        ```
        Daily_Risk = Spoilage_Risk / Shelf_Life_Days
        
        For each day in [1..Shelf_Life_Days]:
            Daily_Sales = Avg_Daily_Sales × Demand_Multiplier × F_weather × F_event × F_footfall
            Daily_Sales = min(Daily_Sales, Remaining_Qty)
            Residual = Remaining_Qty - Daily_Sales
            Daily_Shrink = Residual × Daily_Risk
            
            Total_Shrink += Daily_Shrink
            Total_Sales += Daily_Sales
            Remaining_Qty = max(0, Residual)
        ```
        
        **4. Shrinkage Percentage**
        ```
        Spoilage_Value = Total_Shrink_Qty × Cost_Per_Unit
        Sales_Value = Total_Sales_Qty × Price
        Shrink_% = (Spoilage_Value / Sales_Value) × 100
        ```
        
        **5. Gross Margin Calculation**
        ```
        Cost_Per_Unit = SKU_Cost OR (Price × Cost_Factor_Effective)
        
        Baseline_Margin = Revenue_Baseline - (Qty × Cost_Per_Unit)
        Optimized_Margin = Revenue_Optimized - (Qty × Cost_Per_Unit_New)
        
        Δ_Margin_% = ((Optimized_Margin - Baseline_Margin) / Revenue_Baseline) × 100
        ```
        
        **6. Cost Factor Resolution**
        ```
        If SKU has actual cost:
            Use actual cost
        Else:
            Cost_Factor = Effective_Cost_Factor_For_SKU (via hierarchy)
            Cost = Price × Cost_Factor
        ```
        
        ---
        
        ### Multi-Level Impact Example
        
        Consider SKU "Tomato-Fresh" in category "Vegetables":
        
        - **Global**: Weather=1.2, Elasticity=-1.0
        - **Category (Vegetables)**: Weather=1.4 (rainy season boost)
        - **SKU (Tomato-Fresh)**: Elasticity=-1.5 (highly elastic)
        
        **Effective Parameters**:
        - Weather = 1.4 (from Category, overrides Global)
        - Elasticity = -1.5 (from SKU, overrides both)
        - Event/Footfall = Global values (no overrides)
        
        **Impact**: Tomato's demand is highly weather-sensitive and price-elastic, 
        benefiting from granular control vs. one-size-fits-all approach.
        """)



from datetime import datetime, timedelta, date

@st.dialog("💼 bakery notification — Financial Impact Overview")
def show_bakery_notification_popup(execution_data,stock_data):
    """
    Simulated mobile notification pop-up showing bakery items nearing expiry
    and their discounted (optimized) prices.
    """
    # 1. Inject custom CSS for styling
    st.html("""
        <style>
        /* ... (Your existing CSS here) ... */
        .notif-container {
            background: linear-gradient(145deg, #fffaf0, #ffe6cc);
            border-radius: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            padding: 20px;
            width: 380px;
            margin: 0 auto;
            font-family: 'Segoe UI', sans-serif;
        }
        .notif-header {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            font-size: 20px;
            font-weight: 600;
            color: #d35400;
            margin-bottom: 10px;
        }
        .notif-card {
            background-color: #fff;
            border-radius: 12px;
            padding: 10px 14px;
            margin-bottom: 12px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.1);
        }
        .notif-card h4 {
            margin: 0;
            color: #333;
            font-size: 16px;
            font-weight: 600;
        }
        .notif-card p {
            margin: 4px 0;
            font-size: 14px;
            color: #666;
        }
        .notif-footer {
            text-align: center;
            font-size: 13px;
            color: #999;
            margin-top: 8px;
        }
        </style>
    """)

    # Set currency symbol with a fallback just in case
    currency_symbol = st.session_state.get("currency_symbol", "₹") 

    # 🔹 Extract bakery items nearing expiry
    notif_items = []
    expirey_days={}
    for sku,items in stock_data.items():
        for item in items:
            today = date.today()
            received_date= datetime.strptime(item['date'],"%Y-%m-%d").date()
            shelf_life_days = int(item['shelf_life_days'])
            expiry_date = received_date + timedelta(days=shelf_life_days)
            days_remaining = (expiry_date - today).days
            expirey_days[sku] = days_remaining

    for entry in execution_data.get("execution_log", []):
        item = entry.get("item", "").capitalize()
        new_price = entry.get("new_price", "N/A")
        days_remaining = expirey_days.get(item.lower())  # fetch from dict

        if days_remaining is not None and days_remaining <= 1:
            notif_items.append({
                "item": item,
                "new_price": new_price,
                "days_left": days_remaining
            })

    # 🔸 Render notification popup
    if notif_items:
        notif_html = """
        <div class='notif-container'>
            <div class='notif-header'>📱 Bakery Deals Alert</div>
        """
        for n in notif_items:
            notif_html += f"""
                <div class='notif-card'>
                    <h4>{n['item']}</h4>
                    <p>🕒 Only {n['days_left']} day(s) left before expiry!</p>
                    <p>💸 Grab it now at <b>{currency_symbol} {n['new_price']}</b></p>
                </div>
            """
        notif_html += """
            <div class='notif-footer'>🧁 These offers auto-refresh based on real-time stock & expiry</div>
        </div>
        """
    else:
        notif_html = """
        <div class='notif-container'>
            <div class='notif-header'>📱 No Expiry Alerts</div>
            <p style='text-align:center;'>🎉 All bakery items are fresh and well-stocked!</p>
        </div>
        """
    
    # 2. Render the final HTML content
    st.html(notif_html)




################################################################################################



def check_backend_health():
    try:
        response = requests.post(f"{BACKEND_URL}/agui/run/assistants/search", timeout=5)
        return response.status_code == 200
    except:
        return False

# -------------- SESSION STATE --------------
if 'pipeline_result' not in st.session_state:
    st.session_state['pipeline_result'] = None
if 'is_running' not in st.session_state:
    st.session_state['is_running'] = False


def main():
    st.set_page_config(layout="wide")
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, {palette_map[st.session_state.palette][0]} 0%, {palette_map[st.session_state.palette][1]} 100%);
                padding: 30px; border-radius: 20px; margin-bottom: 30px; text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.2);'>
        <h1 style='color: white; margin: 0; font-size: 36px; font-weight: 700;'>🎯 {LABELS["app_title"]}</h1>
        <p style='color: #e8f2ff; margin: -5px 0px 0px; font-size: 18px;'>{LABELS["app_subtitle"]}</p>
    </div>
    """, unsafe_allow_html=True)

    backend_healthy = check_backend_health()

    with st.sidebar:
        st.markdown('<div class="section-header">🎛️ Control Center</div>', unsafe_allow_html=True)
        if backend_healthy:
            st.success("🟢 Backend Connected", icon="✅")
        else:
            st.error("🔴 Backend Offline", icon="❌")
            st.error(f"Cannot connect to {BACKEND_URL}")
            return
        
        st.markdown("### 🏬 Choose Product Category to Analyze")
        category = st.radio(
            "Select a category:",
            ["🥦 Perishables", "🧁 Bakery"],
            horizontal=True,
            key="category_choice"
        )

        if st.session_state.awaiting_approval:
            st.warning("⏸️ Pipeline Paused", icon="⏳")
            st.info("Awaiting human approval on Decision tab")
        
        launch_disabled = st.session_state['is_running'] or st.session_state.awaiting_approval
        
        if st.button(LABELS["btn_launch"], disabled=launch_disabled):
            st.session_state['is_running'] = True
            st.session_state.awaiting_approval = False
            st.session_state.modified_decisions = {}
            st.session_state.original_decisions = {}

            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            timeline_placeholder = st.empty()

            with progress_placeholder.container():
                progress_bar = st.progress(0)
                with status_placeholder.container():
                    st.info(f"🚀 Initializing {category} analytics pipeline...")
                with timeline_placeholder.container():
                    st.markdown("**🔄 Progress Timeline:** Starting...")

            selected_category = "bakery" if "Bakery" in category else "perishables"
            st.session_state["category"] = selected_category
            
            client = RetailAgentixClient()
            progress_bar.progress(20)
            status_placeholder.info("📡 Connecting to analytics engine...")

            result = client.start_pipeline_sync(initial_state={"category": selected_category})

            if 'error' in result:
                progress_bar.progress(100)
                status_placeholder.error(f"❌ {result['error']}")
                timeline_placeholder.markdown("**⚠️ Pipeline Status:** Failed")
                st.session_state['pipeline_result'] = {
                    'status': 'timeout' if "timed out" in result['error'].lower() else 'error',
                    'error_message': result['error']
                }
            elif result.get('awaiting_human_approval'):
                progress_bar.progress(75)
                status_placeholder.warning("⏸️ Awaiting Human Approval")
                timeline_placeholder.markdown("**⏳ Pipeline Status:** Waiting for decision approval")
                
                st.session_state['pipeline_result'] = {
                    'status': 'awaiting_approval',
                    'final_state': result.get('final_state', {}),
                    'events': result.get('events', [])
                }
                st.session_state.awaiting_approval = True
            else:
                events = result.get('events', [])
                completed_steps, timeline_steps = 0, []
                for event in events:
                    event_type = event['data'].get('type')
                    if event_type == 'STEP_FINISHED':
                        completed_steps += 1
                        progress = min(20 + (completed_steps * 15), 95)
                        progress_bar.progress(progress)
                        step = event['data'].get('step', '').title()
                        status_placeholder.success(f"✅ {step} completed")
                        timeline_steps.append(f"- ✅ {step} ({tz_now().strftime('%H:%M:%S')})")
                        timeline_placeholder.markdown("**🔄 Progress Timeline:**\n" + "\n".join(timeline_steps))
                        time.sleep(0.2)

                progress_bar.progress(100)
                status_placeholder.success("🎉 Analytics pipeline completed!")
                timeline_steps.append(f"- 🎉 All agents completed ({tz_now().strftime('%H:%M:%S')})")
                timeline_placeholder.markdown("**🔄 Progress Timeline:**\n" + "\n".join(timeline_steps))
                st.session_state['pipeline_result'] = result

            st.session_state['is_running'] = False
            time.sleep(1)
            st.rerun()

        if st.button(LABELS["btn_reset"]):
            st.session_state['pipeline_result'] = None
            st.session_state['is_running'] = False
            st.session_state.awaiting_approval = False
            st.session_state.modified_decisions = {}
            st.session_state.original_decisions = {}
            st.session_state.current_run_id = None
            st.rerun()

        if st.session_state.get('pipeline_result'):
            st.markdown("---")
            st.markdown("**🤖 Agent Status Overview**")
            status = st.session_state['pipeline_result'].get('status', 'unknown')
            if st.session_state['is_running']:
                st.info("🔄 Processing Pipeline...", icon="⏳")
            elif st.session_state.awaiting_approval:
                st.warning("⏸️ Awaiting Approval", icon="✋")
            elif status == 'completed':
                st.success("✅ Process Completed", icon="🎉")
            elif status == 'error':
                st.error("❌ Pipeline Error", icon="⚠️")
            elif status == 'timeout':
                st.warning("⚠️ Pipeline Timeout", icon="⏰")
            else:
                st.info("🟡 Ready to Launch", icon="🚀")
            
            final_state = st.session_state['pipeline_result'].get('final_state', {})
            agent_names = ['ingestion', 'analysis', 'prediction', 'decision', 'execution']
            agent_labels = ['📥 Ingestion', '📊 Analysis', '🔮 Prediction', '🎯 Decision', '⚡ Execution']
            for i, (agent_key, label) in enumerate(zip(agent_names, agent_labels)):
                if agent_key in final_state:
                    st.success(f"{label} ✔", icon="✅")
                elif st.session_state.awaiting_approval and i < 4:
                    st.success(f"{label} ✔", icon="✅")
                elif st.session_state.awaiting_approval and i == 4:
                    st.warning(f"{label} ⏳", icon="⏸️")
                else:
                    st.info(f"{label} ⏳", icon="⏳")

    # -------------- MAIN CONTENT AREA --------------
    result = st.session_state.get('pipeline_result')
    if not result:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown('<div class="data-card">', unsafe_allow_html=True)
            st.markdown('<div class="subsection-header">🚀 Getting Started</div>', unsafe_allow_html=True)
            st.markdown(f"""
            Welcome to the **{LABELS["app_title"]}**! This intelligent system provides comprehensive retail insights through a multi-agent pipeline with human-in-the-loop decision making.

            **📋 What the system analyzes:**
            - **📥 Data Ingestion** - Sales history, stock levels, weather data, calendar events
            - **📊 Analysis** - Trend analysis, spoilage risk, pricing patterns
            - **🔮 Predictions** - Future demand forecasting using multiple methods
            - **🎯 Decisions** - AI-powered pricing recommendations with confidence scores
            - **✋ Human Approval** - Review and modify AI decisions before execution
            - **⚡ Execution** - Automated price updates with approved decisions
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="data-card">', unsafe_allow_html=True)
            st.markdown('<div class="subsection-header">📊 Quick Stats</div>', unsafe_allow_html=True)
            st.markdown('<div class="agent-summary">', unsafe_allow_html=True)
            st.markdown("**5 AI Agents**<br/>Working together", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="agent-summary">', unsafe_allow_html=True)
            st.markdown("**Real-time Analysis**<br/>Live data processing", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="agent-summary">', unsafe_allow_html=True)
            st.markdown("**Human-in-Loop**<br/>You control final decisions", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    elif result.get('status') == 'error':
        st.markdown('<div class="data-card">', unsafe_allow_html=True)
        st.error("❌ Pipeline execution failed", icon="⚠️")
        st.error(result.get('error_message', 'Unknown error'))
        st.markdown('</div>', unsafe_allow_html=True)
    elif result.get('status') == 'timeout':
        st.markdown('<div class="data-card">', unsafe_allow_html=True)
        st.warning("⚠️ Pipeline timed out on frontend", icon="⏰")
        st.info("The backend may still be processing. Check backend logs for completion status.")
        st.info(result.get('error_message', ''))
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        final_state = result.get('final_state', {})
        if not final_state:
            st.markdown('<div class="data-card">', unsafe_allow_html=True)
            st.warning("🔄 Pipeline completed but no final state received", icon="ℹ️")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="exec-band">📈 Executive Dashboard</div>', unsafe_allow_html=True)
            st.markdown('<div class="exec-rainbow"><span></span><span></span><span></span><span></span><span></span></div>', unsafe_allow_html=True)
            
            # Agent summary cards
            available_agents = list(final_state.keys())
            agent_cols = st.columns(len(available_agents) if available_agents else 5)
            
            agent_display_map = {
                'ingestion': ('📥', 'Ingestion'),
                'analysis': ('📊', 'Analysis'),
                'prediction': ('🔮', 'Prediction'),
                'decision': ('🎯', 'Decision'),
                'execution': ('⚡', 'Execution')
            }
            
            for idx, agent_name in enumerate(available_agents):
                with agent_cols[idx]:
                    st.markdown('<div class="agent-summary">', unsafe_allow_html=True)
                    emoji, title = agent_display_map.get(agent_name, ('🤖', agent_name.title()))
                    st.markdown(f"**{emoji} {title}**<br/>Completed", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Show approval status if waiting
            if st.session_state.awaiting_approval:
                st.markdown("""
                <div class="approval-banner">
                    ⏸️ PIPELINE PAUSED - Navigate to Decision tab to review and approve pricing changes
                </div>
                """, unsafe_allow_html=True)
            
            # Build tabs based on available agents
            tab_names, tab_data = [], []
            # for agent_name, agent_data in final_state.items():
            #     if agent_data:
            #         if agent_name == 'ingestion':
            #             tab_names.append("📥 Data Ingestion")
            #             tab_data.append(('ingestion', agent_data))
            #         elif agent_name == 'analysis':
            #             tab_names.append("📊 Analysis")
            #             tab_data.append(('analysis', agent_data))
            #         elif agent_name == 'prediction':
            #             tab_names.append("🔮 Predictions")
            #             tab_data.append(('prediction', agent_data))
            #         elif agent_name == 'decision':
            #             tab_names.append("🎯 Decisions")
            #             tab_data.append(('decision', agent_data))
            #         elif agent_name == 'execution':
            #             tab_names.append("⚡ Execution")
            #             tab_data.append(('execution', agent_data))

            for agent_name in ['ingestion', 'analysis', 'prediction', 'decision', 'execution']:
                agent_data = final_state.get(agent_name, {})

                # ✅ If decision data missing but human_approved_decisions exist, use them
                if agent_name == 'decision' and not agent_data:
                    human_decisions = final_state.get("human_approved_decisions", {})
                    if human_decisions:
                        agent_data = {"decisions": human_decisions}

                if agent_data:
                    if agent_name == 'ingestion':
                        tab_names.append("📥 Data Ingestion")
                        tab_data.append(('ingestion', agent_data))
                    elif agent_name == 'analysis':
                        tab_names.append("📊 Analysis")
                        tab_data.append(('analysis', agent_data))
                    elif agent_name == 'prediction':
                        tab_names.append("🔮 Predictions")
                        tab_data.append(('prediction', agent_data))
                    elif agent_name == 'decision':
                        tab_names.append("🎯 Decisions")
                        tab_data.append(('decision', agent_data))
                    elif agent_name == 'execution':
                        tab_names.append("⚡ Execution")
                        tab_data.append(('execution', agent_data))
            
            if tab_names:
                tabs = st.tabs(tab_names)
                for i, (agent_type, agent_data) in enumerate(tab_data):
                    with tabs[i]:
                        if agent_type == 'ingestion':
                            root_category = st.session_state.get("category", "perishable")
                            if root_category == "bakery":
                                display_ingestion_data({"category": "bakery", **agent_data})
                            else:
                                display_ingestion_data({"category": "perishable", **agent_data})
                        elif agent_type == 'analysis':
                            display_analysis_data(agent_data)
                        elif agent_type == 'prediction':
                            display_prediction_data(agent_data)
                        elif agent_type == 'decision':
                            display_decision_data(agent_data, final_state.get('analysis', {}))
                        elif agent_type == 'execution':
                            # Copy your display_execution_data and all popup functions here
                            # from your original code
                            display_execution_data(agent_data,final_state["ingestion"],final_state["analysis"],final_state["prediction"],final_state["decision"])
                            # st.info("Execution display - add your original display_execution_data() function here")
            
            with st.expander("🔍 Complete Raw Data (Developer View)"):
                st.json(final_state)


if __name__ == "__main__":
    main()