# ============================================================
# app.py
# TELCO CUSTOMER CHURN PREDICTION - STREAMLIT (Redesigned UI)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# GLOBAL STYLES
# ============================================================

st.markdown(
    """
    <style>
        /* ---------- General ---------- */
        .stApp {
            background: linear-gradient(180deg, #f7f9fc 0%, #eef1f8 100%);
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }

        /* ---------- Header ---------- */
        .hero {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 60%, #a855f7 100%);
            padding: 2.2rem 2.5rem;
            border-radius: 18px;
            color: white;
            margin-bottom: 1.8rem;
            box-shadow: 0 10px 30px rgba(79, 70, 229, 0.25);
        }
        .hero h1 {
            margin: 0;
            font-size: 2.1rem;
            font-weight: 800;
        }
        .hero p {
            margin-top: 0.4rem;
            font-size: 1.02rem;
            opacity: 0.92;
        }

        /* ---------- Section cards ---------- */
        .section-card {
            background: white;
            border-radius: 16px;
            padding: 1.4rem 1.6rem 0.6rem 1.6rem;
            box-shadow: 0 4px 18px rgba(15, 23, 42, 0.06);
            border: 1px solid rgba(15, 23, 42, 0.04);
            margin-bottom: 1.3rem;
        }
        .section-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #312e81;
            margin-bottom: 0.6rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {
            background: #1e1b4b;
        }
        section[data-testid="stSidebar"] * {
            color: #e5e7eb !important;
        }
        section[data-testid="stSidebar"] hr {
            border-color: rgba(255,255,255,0.15);
        }

        /* ---------- Buttons ---------- */
        div.stButton > button {
            background: linear-gradient(135deg, #4f46e5, #7c3aed);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.75rem 1rem;
            font-weight: 700;
            font-size: 1.02rem;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        div.stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 20px rgba(124, 58, 237, 0.35);
            color: white;
        }

        /* ---------- Result cards ---------- */
        .result-card {
            border-radius: 16px;
            padding: 1.6rem;
            text-align: center;
            font-weight: 700;
            font-size: 1.3rem;
            margin-bottom: 1rem;
        }
        .result-churn {
            background: linear-gradient(135deg, #fee2e2, #fecaca);
            color: #991b1b;
            border: 1px solid #fca5a5;
        }
        .result-safe {
            background: linear-gradient(135deg, #dcfce7, #bbf7d0);
            color: #166534;
            border: 1px solid #86efac;
        }

        .risk-badge {
            display: inline-block;
            padding: 0.35rem 0.9rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.95rem;
        }
        .risk-high { background: #fee2e2; color: #b91c1c; }
        .risk-medium { background: #fef3c7; color: #b45309; }
        .risk-low { background: #dcfce7; color: #15803d; }

        /* ---------- Metric tiles ---------- */
        .metric-tile {
            background: #f8fafc;
            border-radius: 14px;
            padding: 1rem;
            text-align: center;
            border: 1px solid #eef0f5;
        }
        .metric-tile .label {
            font-size: 0.8rem;
            color: #64748b;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }
        .metric-tile .value {
            font-size: 1.25rem;
            font-weight: 800;
            color: #1e1b4b;
            margin-top: 0.15rem;
        }

        footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("customer_churn_pipeline.pkl")


try:
    model = load_model()
    model_loaded = True
except Exception as load_error:
    model = None
    model_loaded = False
    model_load_error = str(load_error)


# ============================================================
# HERO HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>📊 Telco Customer Churn Prediction</h1>
        <p>Fill in the customer's profile and get an instant, data-driven churn risk assessment.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not model_loaded:
    st.error(
        f"⚠️ Could not load the model file `customer_churn_pipeline.pkl`. "
        f"Make sure it's in the same folder as this app.\n\nDetails: {model_load_error}"
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 📡 Churn Predictor")
    st.markdown(
        "Use this tool to estimate the probability that a telecom "
        "customer will churn, based on their account and service profile."
    )
    st.divider()
    st.markdown("### 🧭 How it works")
    st.markdown(
        "1. Enter customer details in the form\n"
        "2. Click **Predict Churn**\n"
        "3. Review the risk level & probability"
    )
    st.divider()
    st.markdown("### ℹ️ About the model")
    st.markdown(
        "Trained on historical telecom customer data using engineered "
        "features such as tenure group, service usage, and monthly "
        "spending patterns."
    )
    st.divider()
    st.caption("Built with Streamlit · Model served via scikit-learn pipeline")


# ============================================================
# CUSTOMER INPUT FORM
# ============================================================

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">👤 Customer Demographics & Account</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["👤 Profile", "🌐 Services", "💳 Billing & Contract"])

# ------------------------------------------------------------
# TAB 1: PROFILE
# ------------------------------------------------------------
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
    with c2:
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])

    multiple_lines = st.selectbox(
        "Multiple Lines", ["No", "Yes", "No phone service"]
    )

# ------------------------------------------------------------
# TAB 2: SERVICES
# ------------------------------------------------------------
with tab2:
    c1, c2 = st.columns(2)
    with c1:
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    with c2:
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

# ------------------------------------------------------------
# TAB 3: BILLING & CONTRACT
# ------------------------------------------------------------
with tab3:
    c1, c2 = st.columns(2)
    with c1:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
    with c2:
        monthly_charges = st.number_input("Monthly Charges (₹)", min_value=0.0, value=70.0, step=1.0)
        total_charges = st.number_input("Total Charges (₹)", min_value=0.0, value=1000.0, step=10.0)
        cltv = st.number_input("CLTV (Customer Lifetime Value)", min_value=0.0, value=4000.0, step=100.0)

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_churn(customer_data):
    customer_df = pd.DataFrame([customer_data])

    # --------------------------------------------------------
    # Feature Engineering
    # --------------------------------------------------------

    # Feature 1: Tenure Group
    customer_df["Tenure_Group"] = pd.cut(
        customer_df["Tenure Months"],
        bins=[-1, 6, 12, 24, 48, 72],
        labels=["0-6 Months", "7-12 Months", "13-24 Months", "25-48 Months", "49-72 Months"],
    )

    # Feature 2: Average Monthly Spending
    customer_df["Average_Monthly_Spending"] = (
        customer_df["Total Charges"] / customer_df["Tenure Months"].replace(0, np.nan)
    )

    # Feature 3: CLTV
    customer_df["Customer_Lifetime_Value"] = customer_df["CLTV"]

    # Feature 4: Total Service Count
    service_columns = [
        "Phone Service",
        "Multiple Lines",
        "Online Security",
        "Online Backup",
        "Device Protection",
        "Tech Support",
        "Streaming TV",
        "Streaming Movies",
    ]

    customer_df["Total_Service_Count"] = (
        customer_df[service_columns].apply(lambda row: sum(value == "Yes" for value in row), axis=1)
    )

    # Feature 5: Service Usage Indicator
    customer_df["Service_Usage_Indicator"] = (customer_df["Total_Service_Count"] >= 3).astype(int)

    # Feature 6: Monthly Charge Category
    customer_df["Monthly_Charge_Category"] = pd.cut(
        customer_df["Monthly Charges"],
        bins=[-np.inf, 40, 70, np.inf],
        labels=["Low", "Medium", "High"],
    )

    # --------------------------------------------------------
    # Remove columns not used during training
    # --------------------------------------------------------
    remove_columns = [
        "CustomerID", "Count", "Country", "State", "City", "Zip Code",
        "Lat Long", "Latitude", "Longitude",
        "Churn Value", "Churn Score", "Churn Reason",
    ]

    customer_df = customer_df.drop(columns=remove_columns, errors="ignore")

    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------
    prediction = model.predict(customer_df)[0]
    probability = model.predict_proba(customer_df)[0, 1]

    return prediction, probability


# ============================================================
# GAUGE CHART HELPER
# ============================================================

def make_gauge(probability):
    if probability >= 0.70:
        bar_color = "#dc2626"
    elif probability >= 0.40:
        bar_color = "#f59e0b"
    else:
        bar_color = "#16a34a"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 40, "color": "#1e1b4b"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
                "bar": {"color": bar_color, "thickness": 0.32},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "#dcfce7"},
                    {"range": [40, 70], "color": "#fef3c7"},
                    {"range": [70, 100], "color": "#fee2e2"},
                ],
            },
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#1e1b4b"},
    )
    return fig


# ============================================================
# PREDICT BUTTON
# ============================================================

st.write("")
predict_clicked = st.button("🔮 Predict Churn", type="primary", use_container_width=True, disabled=not model_loaded)

if predict_clicked:
    customer_data = {
        "Gender": gender,
        "Senior Citizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "Tenure Months": tenure,
        "Phone Service": phone_service,
        "Multiple Lines": multiple_lines,
        "Internet Service": internet_service,
        "Online Security": online_security,
        "Online Backup": online_backup,
        "Device Protection": device_protection,
        "Tech Support": tech_support,
        "Streaming TV": streaming_tv,
        "Streaming Movies": streaming_movies,
        "Contract": contract,
        "Paperless Billing": paperless_billing,
        "Payment Method": payment_method,
        "Monthly Charges": monthly_charges,
        "Total Charges": total_charges,
        "CLTV": cltv,
    }

    try:
        prediction, probability = predict_churn(customer_data)

        if probability >= 0.70:
            risk, risk_class = "High Risk", "risk-high"
        elif probability >= 0.40:
            risk, risk_class = "Medium Risk", "risk-medium"
        else:
            risk, risk_class = "Low Risk", "risk-low"

        st.markdown("---")
        st.markdown("## 🎯 Prediction Result")

        result_col, gauge_col = st.columns([1, 1])

        with result_col:
            if prediction == 1:
                st.markdown(
                    '<div class="result-card result-churn">⚠️ This customer is likely to churn</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="result-card result-safe">✅ This customer is unlikely to churn</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                f'<p style="text-align:center; margin-top:-0.5rem;">'
                f'<span class="risk-badge {risk_class}">{risk}</span></p>',
                unsafe_allow_html=True,
            )

            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(
                    f'<div class="metric-tile"><div class="label">Churn Probability</div>'
                    f'<div class="value">{probability:.1%}</div></div>',
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f'<div class="metric-tile"><div class="label">Tenure</div>'
                    f'<div class="value">{tenure} mo</div></div>',
                    unsafe_allow_html=True,
                )
            with m3:
                st.markdown(
                    f'<div class="metric-tile"><div class="label">Monthly Charges</div>'
                    f'<div class="value">₹{monthly_charges:.0f}</div></div>',
                    unsafe_allow_html=True,
                )

        with gauge_col:
            st.plotly_chart(make_gauge(probability), use_container_width=True, config={"displayModeBar": False})

        # ----------------------------------------------------
        # CUSTOMER SUMMARY TABLE
        # ----------------------------------------------------
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📋 Customer Summary</div>', unsafe_allow_html=True)

        summary = pd.DataFrame(
            {
                "Parameter": [
                    "Tenure Months", "Contract", "Internet Service",
                    "Monthly Charges", "Total Charges", "CLTV",
                    "Churn Probability", "Risk Level",
                ],
                "Value": [
                    tenure, contract, internet_service,
                    f"₹{monthly_charges:.2f}", f"₹{total_charges:.2f}", f"₹{cltv:.2f}",
                    f"{probability:.2%}", risk,
                ],
            }
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ----------------------------------------------------
        # SIMPLE RECOMMENDATION
        # ----------------------------------------------------
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">💡 Suggested Next Step</div>', unsafe_allow_html=True)
        if risk == "High Risk":
            st.write(
                "Consider proactive retention outreach: offer a loyalty discount, "
                "a contract upgrade incentive, or a dedicated support check-in."
            )
        elif risk == "Medium Risk":
            st.write(
                "Monitor this customer and consider a satisfaction survey or "
                "a bundled service offer to increase engagement."
            )
        else:
            st.write("No immediate action needed — this customer shows a healthy retention profile.")
        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction error: {e}")

elif not model_loaded:
    st.info("Fix the model loading issue above, then try again.")
else:
    st.info("👆 Fill in the customer details above and click **Predict Churn** to see the result.")
