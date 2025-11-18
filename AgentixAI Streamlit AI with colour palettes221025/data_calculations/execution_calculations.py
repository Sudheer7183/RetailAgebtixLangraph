import pandas as pd
import streamlit as st
import plotly.express as px
from typing import Dict, Any, Callable

def calculate_shrinkage_metrics_v2(target_skus, params_dict, use_shelf_life_sim,analysis_data, prediction_data, is_optimized=False, exec_price_map=None):
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
        # adjusted_demand = future_demand  * params["weather"] * params["event"] * params["footfall"]

        
        # Calculate sales and spoilage
        if use_shelf_life_sim and shelf_life_days > 0:
            Q = current_qty
            total_shrink_qty = 0.0
            total_sales_qty = 0.0
            daily_risk = spoilage_risk / max(shelf_life_days, 1)
            
            for day in range(1, int(shelf_life_days) + 1):
                daily_sales = avg_daily_sales * demand_multiplier * params["weather"] * params["event"] * params["footfall"]
                # daily_sales = avg_daily_sales  * params["weather"] * params["event"] * params["footfall"]

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


def calculate_predictive_shrinkage(
    stock_data: Dict[str, Any],
    analysis_data: Dict[str, Any],
    prediction_data: Dict[str, Any],
    get_effective_params: Callable,
    use_shelf_life_sim: bool = True
) -> pd.DataFrame:
    """
    Calculate predictive baseline shrinkage (before optimization).
    
    Parameters
    ----------
    stock_data : dict
        Current stock information for all SKUs
    analysis_data : dict
        Analysis results containing spoilage risk
    prediction_data : dict
        Future demand predictions
    get_effective_params : callable
        Function to get effective parameters for a SKU: get_effective_params(sku, category) -> dict
    use_shelf_life_sim : bool
        Whether to use day-by-day shelf-life simulation
    
    Returns
    -------
    pd.DataFrame
        DataFrame with predictive shrinkage metrics per SKU
    """
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
        avg_daily_sales = stock_entry.get("avg_daily_sales", future_demand / shelf_life_days if shelf_life_days > 0 else 0)

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
            "params_source": params.get("source", "Unknown")
        })

    return pd.DataFrame(predictive_records) if predictive_records else pd.DataFrame()


def display_predictive_shrinkage_section(pred_df: pd.DataFrame):
    """
    Display the predictive shrinkage section with metrics and visualizations.
    
    Parameters
    ----------
    pred_df : pd.DataFrame
        DataFrame from calculate_predictive_shrinkage()
    """
    if pred_df.empty:
        st.warning("⚠️ No current stock data found for predictive shrinkage calculation.")
        return
    
    st.subheader("📉 Predictive Baseline Shrinkage (Before Optimization)")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Sales Value", f"₹{pred_df['expected_sales_value'].sum():,.2f}")
    with col2:
        st.metric("Total Spoilage Value", f"₹{pred_df['expected_spoilage_value'].sum():,.2f}")
    with col3:
        avg_shrink = (
            (pred_df['expected_spoilage_value'].sum() / pred_df['expected_sales_value'].sum() * 100)
            if pred_df['expected_sales_value'].sum() > 0 else 0
        )
        st.metric("Avg Shrink %", f"{avg_shrink:.2f}%")
    with col4:
        st.metric("SKUs Analyzed", len(pred_df))
    
    # Data table
    st.dataframe(pred_df.round(3), use_container_width=True)

    # Visualization
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
    fig.add_hline(
        y=avg_shrink, 
        line_dash="dash", 
        line_color="white",
        annotation_text=f"Avg: {avg_shrink:.2f}%", 
        annotation_position="top left"
    )
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



def calculate_prescriptive_shrinkage(
    stock_data: Dict[str, Any],
    analysis_data: Dict[str, Any],
    prediction_data: Dict[str, Any],
    execution_data: Dict[str, Any],
    get_effective_params: Callable,
    use_shelf_life_sim: bool = True
) -> pd.DataFrame:
    """
    Calculate prescriptive/optimized shrinkage (after pricing optimization).
    
    Parameters
    ----------
    stock_data : dict
        Current stock information
    analysis_data : dict
        Analysis results
    prediction_data : dict
        Demand predictions
    execution_data : dict
        Execution log with optimized prices
    get_effective_params : callable
        Function to get effective parameters: get_effective_params(sku, category) -> dict
    use_shelf_life_sim : bool
        Whether to use shelf-life simulation
    
    Returns
    -------
    pd.DataFrame
        DataFrame with optimized shrinkage metrics per SKU
    """
    analysis = analysis_data.get("analysis", {})
    predictions = prediction_data.get("predictions", {})
    execution_log = execution_data.get("execution_log", [])

    # Build execution price map
    exec_df = pd.DataFrame(execution_log) if execution_log else pd.DataFrame(columns=["item", "new_price"])
    exec_price_map = {}
    if not exec_df.empty and "item" in exec_df.columns and "new_price" in exec_df.columns:
        exec_price_map = exec_df.groupby("item")["new_price"].last().to_dict()

    optimized_records = []

    for sku, a_vals in analysis.items():
        if sku not in exec_price_map:
            continue

        stock_entry = (
            stock_data.get(sku, [{}])[0] 
            if isinstance(stock_data.get(sku), list) 
            else stock_data.get(sku, {})
        )
        pred_entry = predictions.get(sku, {})

        current_qty = stock_entry.get("available_qty", 0)
        current_price = stock_entry.get("avg_price", 0)
        category = stock_entry.get("category", "Uncategorized")
        
        # Get effective parameters
        params = get_effective_params(sku, category)
        
        cost_per_unit = stock_entry.get("cost", current_price * params["cost_factor"])
        spoilage_risk = a_vals.get("spoilage_risk", 0)
        future_demand = pred_entry.get("future_demand", 0)
        new_price = float(exec_price_map[sku])
        new_cost_per_unit = new_price * params["cost_factor"]
        
        shelf_life_days = stock_entry.get("shelf_life_days", 5)
        avg_daily_sales = stock_entry.get(
            "avg_daily_sales", 
            future_demand / shelf_life_days if shelf_life_days > 0 else 0
        )

        # Price elasticity effect
        price_change_pct = (new_price - current_price) / current_price if current_price else 0
        demand_multiplier = 1 + (params["elasticity"] * price_change_pct)
        demand_multiplier = max(0.05, demand_multiplier)

        future_demand_after = (
            future_demand * demand_multiplier * 
            params["weather"] * params["event"] * params["footfall"]
        )

        # Shelf-life simulation
        if use_shelf_life_sim and shelf_life_days > 0:
            Q = current_qty
            total_shrink_qty = 0.0
            total_sales_qty = 0.0
            daily_risk = spoilage_risk / max(shelf_life_days, 1)
            
            for day in range(1, int(shelf_life_days) + 1):
                daily_sales = (
                    avg_daily_sales * demand_multiplier * 
                    params["weather"] * params["event"] * params["footfall"]
                )
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
            expected_spoilage_value_after = (
                expected_spoilage_units_after * new_cost_per_unit * spoilage_risk
            )

        shrink_pct_optimized = (
            (expected_spoilage_value_after / expected_sales_value_after) * 100
            if expected_sales_value_after > 0 else 0
        )

        # Baseline comparison
        base_demand = future_demand * params["weather"] * params["event"] * params["footfall"]
        expected_sales_value = base_demand * current_price
        expected_spoilage_value = max(0, (current_qty - base_demand)) * cost_per_unit * spoilage_risk
        shrink_pct_predicted = (
            (expected_spoilage_value / expected_sales_value) * 100
            if expected_sales_value > 0 else 0
        )

        delta_shrink_pct = shrink_pct_predicted - shrink_pct_optimized

        # Margin calculations
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
            "params_source": params.get("source", "Unknown")
        })

    return pd.DataFrame(optimized_records) if optimized_records else pd.DataFrame()


def display_prescriptive_shrinkage_section(opt_df: pd.DataFrame):
    """
    Display the prescriptive shrinkage section with metrics and visualizations.
    
    Parameters
    ----------
    opt_df : pd.DataFrame
        DataFrame from calculate_prescriptive_shrinkage()
    """
    if opt_df.empty:
        st.info("ℹ️ Not enough data to calculate optimized shrinkage.")
        return
    
    st.subheader("📈 Prescriptive Shrinkage (After Optimization)")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Optimized Sales", f"₹{opt_df['expected_sales_value_after'].sum():,.2f}")
    with col2:
        st.metric("Total Spoilage (Optimized)", f"₹{opt_df['expected_spoilage_value_after'].sum():,.2f}")
    with col3:
        avg_opt_shrink = (
            (opt_df['expected_spoilage_value_after'].sum() / 
             opt_df['expected_sales_value_after'].sum() * 100)
            if opt_df['expected_sales_value_after'].sum() > 0 else 0
        )
        st.metric("Avg Shrink % (Optimized)", f"{avg_opt_shrink:.2f}%")
    with col4:
        avg_delta = opt_df['Δ Shrink %'].mean()
        st.metric("Avg Shrink Reduction", f"{avg_delta:.2f}%", delta=f"{avg_delta:.2f}%")
    
    # Data table
    st.dataframe(opt_df.round(3), use_container_width=True)

    # Visualization
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
    fig.add_hline(
        y=avg_opt_shrink, 
        line_dash="dash", 
        line_color="white",
        annotation_text=f"Avg: {avg_opt_shrink:.2f}%"
    )
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_color="black",
        xaxis_tickangle=45,
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)


def display_comparison_analysis(pred_df: pd.DataFrame, opt_df: pd.DataFrame):
    """
    Display comparison between predicted and optimized shrinkage with detailed analysis.
    
    Parameters
    ----------
    pred_df : pd.DataFrame
        Predictive shrinkage results
    opt_df : pd.DataFrame
        Prescriptive shrinkage results
    """
    if pred_df.empty or opt_df.empty:
        st.warning("⚠️ Both predictive and prescriptive data required for comparison.")
        return
    
    comparison_df = pd.merge(pred_df, opt_df, on="sku", how="outer")
    comparison_df["Δ Shrink %"] = (
        comparison_df["shrink_pct_predicted"].fillna(0) - 
        comparison_df["shrink_pct_optimized"].fillna(0)
    )

    st.subheader("📊 Shrinkage Improvement (Predicted → Optimized)")
    
    # Summary metrics
    total_shrink_saved = comparison_df['Δ Shrink %'].sum()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"💰 **Total Shrinkage Reduction**: {total_shrink_saved:.2f}% across all SKUs")
    with col2:
        sku_param_count = len(comparison_df[comparison_df["params_source_y"] == "SKU"])
        st.info(f"🎯 **SKU-Level Params**: {sku_param_count} SKUs")
    with col3:
        cat_param_count = len(comparison_df[comparison_df["params_source_y"] == "Category"])
        st.info(f"📦 **Category-Level Params**: {cat_param_count} SKUs")
    
    # Comparison table
    st.dataframe(
        comparison_df[[
            "sku", "category_x", "shrink_pct_predicted", 
            "shrink_pct_optimized", "Δ Shrink %", "params_source_x"
        ]].rename(columns={
            "category_x": "category", 
            "params_source_x": "params_source"
        }).round(3),
        use_container_width=True
    )

    # Side-by-side charts
    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.bar(
            comparison_df.set_index("sku"),
            x=comparison_df.set_index("sku").index,
            y=["shrink_pct_predicted", "shrink_pct_optimized"],
            barmode="group",
            title="Predicted vs Optimized",
            color_discrete_sequence=["#ff4b4b", "#00b386"],
            text_auto=".1f"
        )
        fig1.update_layout(
            plot_bgcolor="white", 
            paper_bgcolor="white", 
            font_color="black", 
            height=500
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.bar(
            comparison_df.set_index("sku"),
            x=comparison_df.set_index("sku").index,
            y="Δ Shrink %",
            title="Shrink Saved per SKU",
            color="Δ Shrink %",
            color_continuous_scale="Geyser",
            text="Δ Shrink %"
        )
        fig2.add_hline(y=0, line_color="white")
        fig2.update_traces(texttemplate="%{text:+.1f}%")
        fig2.update_layout(
            plot_bgcolor="white", 
            paper_bgcolor="white", 
            font_color="black", 
            height=500
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Category-wise analysis
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
    
    # Download option
    csv = comparison_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Simulation Results (CSV)",
        data=csv,
        file_name="shrinkage_simulation_results_multilevel.csv",
        mime="text/csv"
    )
