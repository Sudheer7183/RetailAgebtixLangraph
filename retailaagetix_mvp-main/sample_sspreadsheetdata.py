
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
from datetime import datetime
import numpy as np
spreadsheet_api = FastAPI()

# Enable CORS
spreadsheet_api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Spreadsheet file paths
SPREADSHEET_CONFIG = {
    "inventory": "data/Historic Inventory data daily_inventory_30_skus.csv",
    "sales_history": "data/Historic_Sales_with_price_fluctuations_v2.csv",
    "current_stock": "data/current_inventory_with_prices_copy.csv",
    "pos_data":"data/pos_pricing_current_day.csv",
    "foot_fall":"data/footfall_index_report.csv",
    "historical_foot_fall":"data/historical_footfall_reference.csv"
    
}

BAKERY_SPREADSHEET_CONFIG = {
    "bakery_inventory": "data/bakery/bakery_inventory_365.csv",
    "bakery_sales_history": "data/bakery/bakery_sales_365.csv",
    "backery_current_stock": "data/bakery/bakery_current_inventory.csv",
    "backery_pos_data":"data/bakery/bakery_pos_pricing_today.csv",
    # "foot_fall":"data/bakery/footfall_index_report.csv",
    # "historical_foot_fall":"data/bakery/historical_footfall_reference.csv"
    
}

# ───────────────────────────────────────────────
# 🔹 Utility: Load spreadsheet (CSV or Excel)
# ───────────────────────────────────────────────
def load_spreadsheet(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    if file_path.endswith(('.xlsx', '.xls')):
        return pd.read_excel(file_path)
    else:
        return pd.read_csv(file_path)


# ───────────────────────────────────────────────
# 🔹 /api/spreadsheet/inventory
# ───────────────────────────────────────────────
@spreadsheet_api.get("/api/spreadsheet/inventory")
async def get_inventory_data():
    """Fetch inventory data (flat list)."""
    try:
        df = load_spreadsheet(SPREADSHEET_CONFIG["inventory"])
        df.columns = df.columns.str.lower().str.strip()
        return {
            "status": "success",
            "data": df.to_dict("records"),
            "total_items": len(df),
            "timestamp": datetime.now().isoformat()
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Inventory file not found: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading inventory: {str(e)}")


# ───────────────────────────────────────────────
# 🔹 /api/spreadsheet/sales-history
# ───────────────────────────────────────────────
@spreadsheet_api.get("/api/spreadsheet/sales-history")
async def get_sales_history():
    """Fetch flat sales history data with consistent columns."""
    try:
        df = load_spreadsheet(SPREADSHEET_CONFIG["sales_history"])
        df.columns = df.columns.str.lower().str.strip()

        # ✅ Normalize key columns for Analysis Agent
        rename_map = {
            "sku_name": "sku_name",
            "date": "date",
            "units_sold": "units_sold",
            "price_per_unit_usd": "price_per_unit_usd",
            "revenue_usd": "revenue_usd"
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        return {
            "status": "success",
            "data": df.to_dict("records"),  # ✅ flat list, not nested per product
            "total_records": len(df),
            "timestamp": datetime.now().isoformat()
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Sales history file not found: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading sales history: {str(e)}")


# ───────────────────────────────────────────────
# 🔹 /api/spreadsheet/current-stock
# ───────────────────────────────────────────────
@spreadsheet_api.get("/api/spreadsheet/current-stock")
async def get_current_stock():
    """Fetch current stock data as flat list."""
    try:
        df = load_spreadsheet(SPREADSHEET_CONFIG["current_stock"])
        df.columns = df.columns.str.lower().str.strip()

        rename_map = {
            "sku_name": "sku_name",
            "total_weight_lbs": "total_weight_lbs",
            "spoiled_overripe_lbs": "spoiled_overripe_lbs",
            "ripening_aging_softening_lbs": "ripening_aging_softening_lbs",
            "avg_price": "avg_price",
            "fresh_lbs":"fresh_lbs",
            "rancid_lbs":"rancid_lbs",
            "category":"category"
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        return {
            "status": "success",
            "data": df.to_dict("records"),
            "total_items": len(df),
            "timestamp": datetime.now().isoformat()
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Current stock file not found: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading current stock: {str(e)}")


# ───────────────────────────────────────────────
# 🔹 /api/spreadsheet/pos-pricing-data
# ───────────────────────────────────────────────
@spreadsheet_api.get("/api/spreadsheet/pos-data")
async def get_pos_data():
    try:
        df=load_spreadsheet(SPREADSHEET_CONFIG["pos_data"])
        df.columns=df.columns.str.lower().str.strip()

        return{
            "status":"success",
            "data":df.to_dict("records"),
            "total_items":len(df),
            "timestamp":datetime.now().isoformat()
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404,detail=f"pos file not found{e}")

    except Exception as e:
        raise HTTPException(status_code=404,detail=f"error fetching the pos-data {str(e)}")

# ───────────────────────────────────────────────
# 🔹 /api/spreadsheet/foot-fall-index
# ───────────────────────────────────────────────
@spreadsheet_api.get("/api/spreadsheet/foot-fall-index")
async def get_foot_fall_index():
    try:
        df=load_spreadsheet(SPREADSHEET_CONFIG["foot_fall"])
        df.columns=df.columns.str.lower().str.strip()

        return {
            "status":"success",
            "data":df.to_dict("records"),
            "total_items":len(df),
            "timestamp":datetime.now().isoformat()
        }

    except FileNotFoundError as e:
        raise HTTPException (status_code=404,detail=f"foot fall index file not found {e}")

    except Exception as e:
        raise HTTPException(status_code=500,detail=f"Unable to read the foot fall dara {str(e)}")

# ───────────────────────────────────────────────
# 🔹 /api/spreadsheet/historical-foot-fall
# ───────────────────────────────────────────────

@spreadsheet_api.get("/api/spreadsheet/historical-foot-fall")
async def get_historical_foot_fall():
    """
    Fetch historical footfall data (daily averages, events, weather etc.)
    Used for model training and analysis.
    """
    try:
        # Load the configured spreadsheet (e.g. CSV path from settings)
        df = load_spreadsheet(SPREADSHEET_CONFIG["historical_foot_fall"])
        df = df.replace([np.nan, np.inf, -np.inf], None)
        df.columns = df.columns.str.lower().str.strip()

        # Optional: keep only relevant columns
        expected_cols = [
            "store_id",
            "date",
            "day_of_week",
            "avg_footfall_estimated",
            "event_flag",
            "weather_condition",
        ]
        # Filter only existing columns to avoid KeyErrors
        df = df[[col for col in expected_cols if col in df.columns]]

        # Clean/format the data
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

        df = df.replace([np.nan, np.inf, -np.inf], None)
        # Sort by date for consistency
        df = df.sort_values("date")

        return {
            "status": "success",
            "data": df.to_dict("records"),
            "total_items": len(df),
            "timestamp": datetime.now().isoformat(),
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Historical footfall file not found: {e}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unable to read historical footfall data: {str(e)}")

# ───────────────────────────────────────────────
# 🔹 /api/spreadsheet/bakcery-inventory
# ───────────────────────────────────────────────

@spreadsheet_api.get("/api/spreadsheet/bakcery-inventory")
async def get_backery_inventory():
    """Fetch the backery inventory data from csv files"""

    try:
        df= load_spreadsheet(BAKERY_SPREADSHEET_CONFIG["bakery_inventory"])
        df.columns = df.columns.str.lower().str.strip()
        return {
            "status": "success",
            "data": df.to_dict("records"),
            "total_items": len(df),
            "timestamp": datetime.now().isoformat()
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Inventory file not found: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading inventory: {str(e)}")
# ───────────────────────────────────────────────
# 🔹 /api/spreadsheet/bakcery-sales
# ───────────────────────────────────────────────

@spreadsheet_api.get("/api/spreadsheet/backer-sales")
async def get_backert_sales():
    """Fetch the backery sales data historical from csv files"""
    try:
        df = load_spreadsheet(BAKERY_SPREADSHEET_CONFIG["bakery_sales_history"])
        df.columns = df.columns.str.lower().str.strip()

        rename_map={
            "sku_name": "sku_name",
            "date": "date",
            "units_sold": "units_sold",
            "price_per_unit_usd": "unit_price_usd",
            "revenue_usd": "revenue_usd"
        }
        df= df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        return {
            "status": "success",
            "data": df.to_dict("records"),  # ✅ flat list, not nested per product
            "total_records": len(df),
            "timestamp": datetime.now().isoformat()
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Sales history file not found: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading sales history: {str(e)}")

# ───────────────────────────────────────────────
# 🔹 /api/spreadsheet/bakery-current-stock
# ───────────────────────────────────────────────

@spreadsheet_api.get("/api/spreadsheet/backery-current-stock")
async def get_backery_current_stock():
    """Fetch bakery current stock details"""

    try:
        df=load_spreadsheet(BAKERY_SPREADSHEET_CONFIG["backery_current_stock"])
        df.columns= df.columns.str.lower().str.strip()

        rename_map={
            "sku_name": "sku_name",
            "total_weight_lbs": "total_units",
            "spoiled_overripe_lbs": "spoiled_units",
            "ripening_aging_softening_lbs": "stale_units",
            # "avg_price": "avg_price",
            "fresh_lbs":"fresh_units",
            # "rancid_lbs":"rancid_lbs",
            "category":"category"
        }

        df=df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns })

        return {
            "status": "success",
            "data": df.to_dict("records"),  # ✅ flat list, not nested per product
            "total_records": len(df),
            "timestamp": datetime.now().isoformat()
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Sales history file not found: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading sales history: {str(e)}")

# ───────────────────────────────────────────────
# 🔹 /api/spreadsheet/bakery-pos-data
# ───────────────────────────────────────────────

@spreadsheet_api.get("/api/spreadsheet/backer-pos-data")
async def get_backer_pos_data():
    """Fetch pos data for the backery item from csv files"""

    try:
        df=load_spreadsheet(BAKERY_SPREADSHEET_CONFIG["backery_pos_data"])
        df.columns=df.columns.str.lower().str.strip()

        rename_map={
            "sku_name":"sku_name",
            "avg_price":"regular_price_usd",
            "shelf_life":"expiry_days"
        }

        df=df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        return {
            "status":"success",
            "data":df.to_dict("records"),
            "total_records": len(df),
            "timestamp": datetime.now().isoformat()

        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404,detail=f"pos data file is not found {e}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server issue, can't able to read the pos data {e}")
        


# ───────────────────────────────────────────────
# 🔹 /api/spreadsheet/all
# ───────────────────────────────────────────────
@spreadsheet_api.get("/api/spreadsheet/all")
async def get_all_data():
    """Fetch all sheets in one call."""
    try:
        inventory = await get_inventory_data()
        sales = await get_sales_history()
        stock = await get_current_stock()
        return {
            "status": "success",
            "inventory": inventory["data"],
            "sales_history": sales["data"],
            "current_stock": stock["data"],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching all data: {str(e)}")



@spreadsheet_api.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "spreadsheet-api"}
