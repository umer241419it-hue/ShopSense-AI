"""
Streamlit Web Application: Online Shopper Purchase Intention Predictor.
Provides an interactive dashboard for predicting e-commerce visitor purchasing intention.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

from src.predict import ShopperPurchasePredictor
from src.preprocessing import RAW_NUMERICAL_FEATURES, RAW_CATEGORICAL_FEATURES

st.set_page_config(
    page_title="Online Shopper Purchase Intention Predictor",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .purchase-yes {
        background-color: #DEF7EC;
        border: 2px solid #31C48D;
        color: #03543F;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
    .purchase-no {
        background-color: #FDE8E8;
        border: 2px solid #F98080;
        color: #9B1C1C;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    """Load and cache the trained inference pipeline."""
    model_path = PROJECT_ROOT / "models" / "best_model.joblib"
    return ShopperPurchasePredictor(model_path)


# Predefined realistic customer personas
PERSONAS = {
    "Select a Preset (Optional)...": None,
    "🎯 High-Intent Shopper (High PageValues, Active Product Browsing)": {
        "Administrative": 3, "Administrative_Duration": 95.0,
        "Informational": 1, "Informational_Duration": 35.0,
        "ProductRelated": 32, "ProductRelated_Duration": 1280.0,
        "BounceRates": 0.005, "ExitRates": 0.015, "PageValues": 42.5,
        "SpecialDay": 0.0, "Month": "Nov", "OperatingSystems": 2,
        "Browser": 2, "Region": 1, "TrafficType": 2,
        "VisitorType": "Returning_Visitor", "Weekend": False
    },
    "🏃 Quick Bouncer (Single page visit, High bounce rate)": {
        "Administrative": 0, "Administrative_Duration": 0.0,
        "Informational": 0, "Informational_Duration": 0.0,
        "ProductRelated": 1, "ProductRelated_Duration": 0.0,
        "BounceRates": 0.20, "ExitRates": 0.20, "PageValues": 0.0,
        "SpecialDay": 0.0, "Month": "Feb", "OperatingSystems": 1,
        "Browser": 1, "Region": 1, "TrafficType": 1,
        "VisitorType": "Returning_Visitor", "Weekend": False
    },
    "🔍 Window Shopper (Many product views, Zero page values)": {
        "Administrative": 1, "Administrative_Duration": 12.0,
        "Informational": 0, "Informational_Duration": 0.0,
        "ProductRelated": 45, "ProductRelated_Duration": 1500.0,
        "BounceRates": 0.012, "ExitRates": 0.035, "PageValues": 0.0,
        "SpecialDay": 0.0, "Month": "May", "OperatingSystems": 2,
        "Browser": 2, "Region": 3, "TrafficType": 3,
        "VisitorType": "Returning_Visitor", "Weekend": True
    },
    "🎁 Holiday New Visitor (SpecialDay high, High engagement)": {
        "Administrative": 2, "Administrative_Duration": 50.0,
        "Informational": 2, "Informational_Duration": 80.0,
        "ProductRelated": 25, "ProductRelated_Duration": 920.0,
        "BounceRates": 0.01, "ExitRates": 0.02, "PageValues": 18.0,
        "SpecialDay": 0.6, "Month": "May", "OperatingSystems": 3,
        "Browser": 2, "Region": 2, "TrafficType": 4,
        "VisitorType": "New_Visitor", "Weekend": True
    }
}


def main():
    st.markdown('<div class="main-header">🛍️ Online Shopper Purchase Intention Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Autonomous Machine Learning Decision Support System for E-Commerce Revenue Prediction</div>', unsafe_allow_html=True)
    
    try:
        predictor = load_predictor()
    except Exception as e:
        st.error(f"Error loading trained model: {e}. Please ensure train.py has been executed.")
        return
        
    # Navigation Tabs
    tab_pred, tab_model, tab_eda = st.tabs(["🔮 Live Prediction", "📊 Model Performance", "📈 Dataset Insights"])
    
    with tab_pred:
        col_preset, col_info = st.columns([2, 1])
        with col_preset:
            selected_persona = st.selectbox(
                "⚡ Quick Load Customer Persona:",
                list(PERSONAS.keys())
            )
            
        preset_values = PERSONAS.get(selected_persona) or {}
        
        st.markdown("---")
        st.subheader("Customer Session Attributes")
        
        # Form layout
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("##### 📄 Page Navigation Activity")
                admin = st.number_input("Administrative Pages Visited", min_value=0, max_value=50, value=int(preset_values.get("Administrative", 2)))
                admin_dur = st.number_input("Administrative Duration (seconds)", min_value=0.0, max_value=5000.0, value=float(preset_values.get("Administrative_Duration", 45.0)), step=10.0)
                info = st.number_input("Informational Pages Visited", min_value=0, max_value=30, value=int(preset_values.get("Informational", 0)))
                info_dur = st.number_input("Informational Duration (seconds)", min_value=0.0, max_value=3000.0, value=float(preset_values.get("Informational_Duration", 0.0)), step=10.0)
                prod = st.number_input("Product-Related Pages Visited", min_value=0, max_value=500, value=int(preset_values.get("ProductRelated", 20)))
                prod_dur = st.number_input("Product-Related Duration (seconds)", min_value=0.0, max_value=50000.0, value=float(preset_values.get("ProductRelated_Duration", 650.0)), step=50.0)
                
            with col2:
                st.markdown("##### 📉 Engagement & Quality Metrics")
                bounce = st.slider("Bounce Rate (Proportion)", min_value=0.0, max_value=0.20, value=float(preset_values.get("BounceRates", 0.01)), step=0.005, format="%.3f")
                exit_rate = st.slider("Exit Rate (Proportion)", min_value=0.0, max_value=0.20, value=float(preset_values.get("ExitRates", 0.03)), step=0.005, format="%.3f")
                page_val = st.number_input("PageValues (E-commerce Page Value)", min_value=0.0, max_value=400.0, value=float(preset_values.get("PageValues", 12.0)), step=1.0)
                special = st.slider("Special Day Closeness (e.g. Valentine / Mother's Day)", min_value=0.0, max_value=1.0, value=float(preset_values.get("SpecialDay", 0.0)), step=0.2)
                month = st.selectbox("Month of Session", ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], index=["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"].index(preset_values.get("Month", "Nov")))
                
            with col3:
                st.markdown("##### 👤 User Demographics & Environment")
                visitor = st.selectbox("Visitor Type", ["Returning_Visitor", "New_Visitor", "Other"], index=["Returning_Visitor", "New_Visitor", "Other"].index(preset_values.get("VisitorType", "Returning_Visitor")))
                weekend = st.checkbox("Session Occurred on Weekend", value=bool(preset_values.get("Weekend", False)))
                os_val = st.selectbox("Operating System Code", list(range(1, 9)), index=int(preset_values.get("OperatingSystems", 2)) - 1)
                browser = st.selectbox("Browser Code", list(range(1, 14)), index=int(preset_values.get("Browser", 2)) - 1)
                region = st.selectbox("Region Code", list(range(1, 10)), index=int(preset_values.get("Region", 1)) - 1)
                traffic = st.selectbox("Traffic Type Code", list(range(1, 21)), index=int(preset_values.get("TrafficType", 2)) - 1)
                
            submit_btn = st.form_submit_button("⚡ Predict Purchasing Intention", use_container_width=True, type="primary")
            
        if submit_btn or selected_persona != "Select a Preset (Optional)...":
            input_dict = {
                "Administrative": admin,
                "Administrative_Duration": admin_dur,
                "Informational": info,
                "Informational_Duration": info_dur,
                "ProductRelated": prod,
                "ProductRelated_Duration": prod_dur,
                "BounceRates": bounce,
                "ExitRates": exit_rate,
                "PageValues": page_val,
                "SpecialDay": special,
                "Month": month,
                "OperatingSystems": os_val,
                "Browser": browser,
                "Region": region,
                "TrafficType": traffic,
                "VisitorType": visitor,
                "Weekend": weekend
            }
            
            result = predictor.predict(input_dict)
            
            st.markdown("---")
            st.subheader("🎯 Prediction Results & Decision Intelligence")
            
            res_col1, res_col2 = st.columns([1, 1.2])
            
            with res_col1:
                if result["is_purchase"]:
                    st.markdown(f"""
                    <div class="purchase-yes">
                        <h2>✅ PURCHASE INTENDED</h2>
                        <h4>Predicted Class: Revenue = True</h4>
                        <p style="font-size: 1.1rem; margin-top: 10px;">
                            <strong>Customer Intent Level:</strong> {result['intent_level']}<br>
                            <strong>Purchase Probability:</strong> {result['purchase_probability']:.2%}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="purchase-no">
                        <h2>❌ NO PURCHASE PREDICTED</h2>
                        <h4>Predicted Class: Revenue = False</h4>
                        <p style="font-size: 1.1rem; margin-top: 10px;">
                            <strong>Customer Intent Level:</strong> {result['intent_level']}<br>
                            <strong>No-Purchase Probability:</strong> {result['no_purchase_probability']:.2%}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                st.markdown("<br>", unsafe_allow_html=True)
                st.info(f"💡 **Session Diagnostic:** PageValues = {page_val} | BounceRate = {bounce*100:.1f}% | Total Time = {admin_dur + info_dur + prod_dur:.0f}s")
                
            with res_col2:
                # Plotly Probability Bar
                fig = go.Figure(data=[
                    go.Bar(
                        name="No Purchase",
                        x=["Probability"],
                        y=[result["no_purchase_probability"]],
                        marker_color="#4575b4",
                        text=[f"{result['no_purchase_probability']:.1%}"],
                        textposition="auto"
                    ),
                    go.Bar(
                        name="Purchase",
                        x=["Probability"],
                        y=[result["purchase_probability"]],
                        marker_color="#d73027",
                        text=[f"{result['purchase_probability']:.1%}"],
                        textposition="auto"
                    )
                ])
                fig.update_layout(
                    barmode="stack",
                    title="Conversion Likelihood Breakdown",
                    yaxis=dict(range=[0, 1.0], tickformat=".0%"),
                    height=280,
                    margin=dict(l=20, r=20, t=40, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
                
            # E-Commerce Actionable Recommendations
            st.markdown("##### 📌 E-Commerce Action Recommendations")
            if result["is_purchase"]:
                st.success("🎯 **High Conversion Action:** Deploy streamlined 1-click checkout, express payment options, and subtle upsell suggestions without friction.")
            elif result["purchase_probability"] > 0.25:
                st.warning("⚠️ **Hesitant Shopper Action:** High churn risk. Trigger personalized limited-time discount pop-up, free shipping guarantee, or live chat assistance.")
            else:
                st.error("📉 **Low Intent Visitor Action:** Focus on re-engagement: show trending products, customer reviews, or trigger an exit-intent email newsletter subscription.")

    with tab_model:
        st.subheader("Model Evaluation & Algorithm Comparison")
        
        comp_csv = PROJECT_ROOT / "reports" / "model_comparison.csv"
        if comp_csv.exists():
            df_comp = pd.read_csv(comp_csv)
            st.dataframe(df_comp.style.format({
                "Accuracy": "{:.2%}",
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1-Score": "{:.4f}",
                "ROC-AUC": "{:.4f}",
                "PR-AUC": "{:.4f}"
            }).highlight_max(subset=["F1-Score", "ROC-AUC", "Recall"], color="#D1FAE5"), use_container_width=True)
        else:
            st.warning("Model comparison data not found.")
            
        mcol1, mcol2 = st.columns(2)
        with mcol1:
            roc_img = PROJECT_ROOT / "reports" / "figures" / "roc_curves_comparison.png"
            if roc_img.exists():
                st.image(str(roc_img), caption="Receiver Operating Characteristic (ROC) Comparison")
                
        with mcol2:
            pr_img = PROJECT_ROOT / "reports" / "figures" / "precision_recall_curves_comparison.png"
            if pr_img.exists():
                st.image(str(pr_img), caption="Precision-Recall Curve Comparison")
                
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            fi_img = PROJECT_ROOT / "reports" / "figures" / "feature_importance.png"
            if fi_img.exists():
                st.image(str(fi_img), caption="Top Predictive Features (Random Forest)")
                
        with fcol2:
            cm_img = PROJECT_ROOT / "reports" / "figures" / "confusion_matrices_comparison.png"
            if cm_img.exists():
                st.image(str(cm_img), caption="Confusion Matrices Comparison")

    with tab_eda:
        st.subheader("Exploratory Data Analysis Highlights")
        
        ecol1, ecol2 = st.columns(2)
        with ecol1:
            t_dist = PROJECT_ROOT / "reports" / "figures" / "target_distribution.png"
            if t_dist.exists():
                st.image(str(t_dist), caption="Class Distribution (Revenue)")
                
            m_dist = PROJECT_ROOT / "reports" / "figures" / "purchase_rate_by_month.png"
            if m_dist.exists():
                st.image(str(m_dist), caption="Seasonality: Monthly Session Volume & Conversion")
                
        with ecol2:
            pv_dist = PROJECT_ROOT / "reports" / "figures" / "pagevalues_vs_revenue.png"
            if pv_dist.exists():
                st.image(str(pv_dist), caption="PageValues vs Purchase Conversion")
                
            v_dist = PROJECT_ROOT / "reports" / "figures" / "purchase_rate_by_visitor_type.png"
            if v_dist.exists():
                st.image(str(v_dist), caption="Visitor Type Conversion Comparison")


if __name__ == "__main__":
    main()
